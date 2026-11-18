"""OpenAlex API usage management. NO retrieval logic lives here.

Why this exists: OpenAlex bills per request (usage-based pricing; a search
costs 10 credits, a free key gets 10,000 credits = ~1,000 searches per UTC
day). Once the day's budget is gone every request returns HTTP 429, and the
pipeline's own error handling swallows that (search_and_dedupe never raises;
Hop-1 returns empty), so a benchmark run silently turns "out of budget" into
"the retrieval found nothing". This module gives the client and the benchmark
harness the pieces needed to prevent that:

  * request-level cache  -- every search_openalex() caller (including
    diagnostic scripts that bypass search_and_dedupe) shares the existing
    cgscout.cache. Only successful (HTTP 200) responses are ever stored.
  * usage ledger         -- one JSON line per real HTTP attempt, plus an
    in-process per-run-label counter, so a benchmark run can report exactly
    how many requests / cache hits / credits it consumed.
  * budget state/latch   -- the first budget-exhausted 429 is remembered
    (in-process and in a small file, so other processes see it too) until
    OpenAlex's reported reset time; later cache-miss requests fail fast with
    no HTTP call and no retry sleeps.
  * strict mode          -- OPENALEX_STRICT_BUDGET=1 makes exhaustion raise
    OpenAlexBudgetExhausted, a BaseException on purpose: `except Exception`
    handlers elsewhere in the pipeline cannot swallow it, so a benchmark stops
    instead of recording failed requests as retrieval misses. Off by default.
  * cross-process semantic throttle and benchmark slots -- OpenAlex limits
    search.semantic to 1 request/second; the old throttle only held inside one
    process.

The API key is read by openalex.py from OPENALEX_API_KEY (unchanged) and is
never logged, stored or returned by anything here.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

from cgscout import cache as _cache

try:  # POSIX only; the throttle / slot lock degrade to in-process behaviour elsewhere
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

REQUEST_CACHE_NAMESPACE = "openalex_request"
# Same drift argument as search.py's _SEARCH_CACHE_TTL_SECONDS: OpenAlex ranking
# and indexing change over time, so a cached query answer expires after a day.
REQUEST_CACHE_TTL_S = 24 * 60 * 60
CREDITS_PER_SEARCH_DEFAULT = 10     # documented price of one search; used only when the response omits it
SEMANTIC_MIN_INTERVAL_S = 1.05      # OpenAlex hard-limits search.semantic to 1 req/s

_lock = threading.Lock()
_run_label: Optional[str] = None
_counters: dict[str, dict[str, Any]] = {}
_latch: Optional[dict[str, Any]] = None


class OpenAlexBudgetExhausted(BaseException):
    """The OpenAlex daily budget is exhausted (strict mode only).

    Derives from BaseException deliberately: the retrieval pipeline wraps
    searches in `except Exception`, which would turn this into an empty result
    and a false "retrieval miss". Only code that explicitly catches this class
    (the benchmark harness) should handle it."""

    def __init__(self, message: str, *, headers: Optional[dict] = None, reset_s: Optional[int] = None):
        super().__init__(message)
        self.headers = headers or {}
        self.reset_s = reset_s


class OpenAlexBudgetInsufficient(RuntimeError):
    """Pre-flight check failed: the remaining budget cannot cover the planned run."""


# ------------------------------------------------------------------ paths
def _dir() -> Path:
    d = _cache._default_cache_dir()
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:  # noqa: BLE001 -- never let bookkeeping crash a real computation
        pass
    return d


def ledger_path() -> Path:
    return _dir() / "openalex_usage.jsonl"


def _state_path() -> Path:
    return _dir() / "openalex_budget_state.json"


def strict_budget() -> bool:
    return os.environ.get("OPENALEX_STRICT_BUDGET", "").strip().lower() in ("1", "true", "yes")


def request_cache_enabled() -> bool:
    return os.environ.get("OPENALEX_REQUEST_CACHE", "1").strip().lower() not in ("0", "false", "no")


# ------------------------------------------------------- request-level cache
def request_cache_key(query: str, *, limit: int, mode: str, select: str) -> str:
    return _cache.content_hash("openalex_request", mode, str(limit), select, query)


def request_cache_get(key: str) -> Optional[list]:
    if not request_cache_enabled():
        return None
    return _cache.get(REQUEST_CACHE_NAMESPACE, key, ttl_seconds=REQUEST_CACHE_TTL_S)


def request_cache_set(key: str, papers: list) -> None:
    if request_cache_enabled():
        _cache.set(REQUEST_CACHE_NAMESPACE, key, papers)


# --------------------------------------------------------- accounting
def _label_counters(label: str) -> dict[str, Any]:
    return _counters.setdefault(label, {
        "http_requests": 0, "http_200": 0, "http_429": 0, "http_other_failures": 0,
        "cache_hits": 0, "credits_charged": 0, "credits_assumed_requests": 0,
        "remaining_first": None, "remaining_last": None,
        "_unresolved": set(),     # (mode, limit, query) whose attempts have NOT yet produced an HTTP 200
    })


def set_run_label(label: Optional[str]) -> None:
    """Attribute subsequent requests to `label` (a benchmark run / case)."""
    global _run_label
    _run_label = label
    if label:
        with _lock:
            _label_counters(label)


def reset_run(label: str) -> None:
    with _lock:
        _counters.pop(label, None)
        _label_counters(label)


def run_usage(label: Optional[str] = None) -> dict[str, Any]:
    """Snapshot of what the labelled run actually did against OpenAlex."""
    label = label or _run_label or "default"
    with _lock:
        c = dict(_label_counters(label))
        unresolved = len(c.pop("_unresolved"))
    c["label"] = label
    # transient_failed_attempts: HTTP attempts that failed but whose query later got a 200 (the client's own
    # retry worked -- harmless). failed_requests: queries that NEVER got a 200 -- the only failures that can
    # make a run's retrieval results untrustworthy.
    c["transient_failed_attempts"] = c["http_429"] + c["http_other_failures"] - unresolved
    c["failed_requests"] = unresolved
    return c


def count_cache_hit() -> None:
    with _lock:
        _label_counters(_run_label or "default")["cache_hits"] += 1


def _int(v: Any) -> Optional[int]:
    try:
        return int(str(v))
    except (TypeError, ValueError):
        return None


def record_request(*, mode: str, limit: int, status: Optional[int], headers: Optional[dict],
                   elapsed_s: float, query: str, error: Optional[str] = None) -> None:
    """Account one REAL HTTP attempt (cache hits are not requests). Credits are charged
    only on HTTP 200; the price comes from the response when it states it."""
    headers = headers or {}
    remaining = _int(headers.get("x-ratelimit-remaining"))
    required = _int(headers.get("x-ratelimit-credits-required"))
    charged = 0
    assumed = False
    if status == 200:
        charged = required if required is not None else CREDITS_PER_SEARCH_DEFAULT
        assumed = required is None
    with _lock:
        c = _label_counters(_run_label or "default")
        c["http_requests"] += 1
        req_key = (mode, limit, query)
        if status == 200:
            c["_unresolved"].discard(req_key)
        else:
            c["_unresolved"].add(req_key)
        if status == 200:
            c["http_200"] += 1
            c["credits_charged"] += charged
            c["credits_assumed_requests"] += int(assumed)
        elif status == 429:
            c["http_429"] += 1
        else:
            c["http_other_failures"] += 1
        if remaining is not None:
            if c["remaining_first"] is None:
                c["remaining_first"] = remaining
            c["remaining_last"] = remaining
    line = {"ts": round(time.time(), 3), "pid": os.getpid(), "label": _run_label, "mode": mode, "limit": limit,
            "status": status, "elapsed_s": round(elapsed_s, 3), "credits_charged": charged,
            "credits_assumed": assumed, "remaining": remaining, "query": query[:100]}
    if error:
        line["error"] = error[:160]
    try:
        with open(ledger_path(), "a") as f:
            f.write(json.dumps(line) + "\n")
    except Exception:  # noqa: BLE001
        pass


def used_today_utc() -> dict[str, Any]:
    """Credits this machine has spent since 00:00 UTC, summed over ALL processes (ledger)."""
    midnight = time.time() - (time.time() % 86400)
    n = credits = 0
    try:
        with open(ledger_path()) as f:
            for raw in f:
                try:
                    e = json.loads(raw)
                except ValueError:
                    continue
                if e.get("ts", 0) >= midnight and e.get("status") == 200:
                    n += 1
                    credits += int(e.get("credits_charged") or 0)
    except FileNotFoundError:
        pass
    return {"requests": n, "credits": credits}


# ------------------------------------------------------ budget state / latch
def is_budget_exhausted_response(headers: dict) -> bool:
    return headers.get("x-ratelimit-remaining") == "0" or headers.get("x-ratelimit-remaining-usd") == "0"


def mark_exhausted(headers: dict, message: str = "") -> dict[str, Any]:
    reset_s = _int(headers.get("retry-after")) or _int(headers.get("x-ratelimit-reset")) or 3600
    state = {"since": time.time(), "reset_at": time.time() + reset_s, "reset_s": reset_s,
             "headers": {k: v for k, v in headers.items()}, "message": message[:200]}
    global _latch
    with _lock:
        _latch = state
    try:
        _state_path().write_text(json.dumps(state))
    except Exception:  # noqa: BLE001
        pass
    return state


def exhausted() -> Optional[dict[str, Any]]:
    """The exhaustion state if the budget is currently known to be exhausted (in-process or
    recorded by another process), else None. Clears itself once OpenAlex's reset time passes."""
    global _latch
    with _lock:
        st = _latch
    if st is None:
        p = _state_path()
        if p.exists():
            try:
                st = json.loads(p.read_text())
            except Exception:  # noqa: BLE001
                st = None
    if st and time.time() >= st.get("reset_at", 0):
        clear_exhausted()
        return None
    return st


def clear_exhausted() -> None:
    global _latch
    with _lock:
        _latch = None
    try:
        _state_path().unlink()
    except FileNotFoundError:
        pass
    except Exception:  # noqa: BLE001
        pass


def exhausted_message(st: dict) -> str:
    left = max(0, int(st.get("reset_at", 0) - time.time()))
    return (f"OpenAlex daily budget exhausted; it resets at midnight UTC (~{left}s from now). "
            f"No more requests will be attempted until then; cached queries are still served.")


# --------------------------------------------------------------- pre-flight
def budget_status(max_age_s: int = 300, force: bool = False) -> dict[str, Any]:
    """Current budget. Order: known-exhausted latch (no request) -> a recent response seen by this
    process (no request) -> one minimal live request (costs 10 credits if it succeeds, free if the
    budget is already gone). The API key is never returned."""
    import requests  # local import: only the pre-flight needs it here

    ex = None if force else exhausted()   # force=True ignores the remembered state (e.g. after topping up funds)
    if ex:
        return {"state": "exhausted", "source": "latch", "http_status": 429, "remaining_credits": 0,
                "reset_s": int(max(0, ex["reset_at"] - time.time())), "authenticated": None}
    key = os.environ.get("OPENALEX_API_KEY")
    if not force:
        with _lock:
            for c in _counters.values():
                if c.get("remaining_last") is not None:
                    return {"state": "ok", "source": "recent_response", "http_status": 200,
                            "remaining_credits": c["remaining_last"], "reset_s": None, "authenticated": bool(key)}
    params = {"search": "budget check", "per_page": 1, "select": "id"}
    if key:
        params["api_key"] = key
    t0 = time.time()
    r = requests.get("https://api.openalex.org/works", params=params, timeout=20)
    h = {k.lower(): v for k, v in r.headers.items()}
    record_request(mode="lexical", limit=1, status=r.status_code, headers=h, elapsed_s=time.time() - t0, query="<budget check>")
    if r.status_code == 429 and is_budget_exhausted_response(h):
        mark_exhausted(h, "budget check")
        return {"state": "exhausted", "source": "request", "http_status": 429, "remaining_credits": 0,
                "reset_s": _int(h.get("retry-after")) or _int(h.get("x-ratelimit-reset")), "authenticated": bool(key)}
    rem = _int(h.get("x-ratelimit-remaining"))
    if r.status_code == 200:
        clear_exhausted()   # a live success proves the remembered exhaustion is over (reset, or funds added)
    return {"state": "ok" if r.status_code == 200 else "error", "source": "request", "http_status": r.status_code,
            "limit_credits": _int(h.get("x-ratelimit-limit")), "remaining_credits": rem,
            "reset_s": _int(h.get("x-ratelimit-reset")), "authenticated": bool(key)}


def require_budget(needed_searches: int, credits_per_search: int = CREDITS_PER_SEARCH_DEFAULT) -> dict[str, Any]:
    """Refuse to start work that the remaining budget cannot cover. If a successful response does not
    state the remaining credits, the run is allowed (a 200 proves the budget is not zero) and
    `remaining_credits` stays None so the caller can say it is unverified."""
    b = budget_status()
    need = needed_searches * credits_per_search
    if b["state"] == "exhausted":
        raise OpenAlexBudgetInsufficient(
            f"OpenAlex budget is exhausted (resets in ~{b.get('reset_s')}s, i.e. midnight UTC). "
            f"Not starting: this run needs ~{needed_searches} uncached searches.")
    if b["state"] == "error":
        raise OpenAlexBudgetInsufficient(f"Could not verify the OpenAlex budget (HTTP {b['http_status']}); not starting.")
    if b["remaining_credits"] is not None and b["remaining_credits"] < need:
        raise OpenAlexBudgetInsufficient(
            f"OpenAlex budget too low: {b['remaining_credits']} credits left, run needs ~{need} "
            f"({needed_searches} uncached searches x {credits_per_search}).")
    return b


# ---------------------------------------------------- throttle and parallelism
def throttle_semantic(min_interval_s: float = SEMANTIC_MIN_INTERVAL_S) -> None:
    """Hold search.semantic to OpenAlex's 1 request/second ACROSS PROCESSES (flock on a timestamp file)."""
    if fcntl is None:  # pragma: no cover
        time.sleep(min_interval_s)
        return
    p = _dir() / "openalex_semantic.ts"
    with open(p, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            try:
                last = float(f.read().strip() or 0)
            except ValueError:
                last = 0.0
            wait = min_interval_s - (time.time() - last)
            if wait > 0:
                time.sleep(wait)
            f.seek(0)
            f.truncate()
            f.write(str(time.time()))
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


_held_slots: list = []


def acquire_benchmark_slot(max_parallel: int = 1):
    """Take one of `max_parallel` benchmark slots for this process's lifetime, or return None if all
    are busy (another benchmark is running). Prevents several harness processes hammering the API."""
    if fcntl is None:  # pragma: no cover
        return True
    for i in range(max_parallel):
        f = open(_dir() / f"benchmark_slot_{i}.lock", "a+")
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            f.close()
            continue
        _held_slots.append(f)
        return f
    return None


def _reset_for_tests() -> None:
    global _latch, _run_label
    with _lock:
        _latch = None
        _run_label = None
        _counters.clear()
    for f in _held_slots:
        try:
            f.close()
        except Exception:  # noqa: BLE001
            pass
    _held_slots.clear()


# ----------------------------------------------------------------- CLI
def _main() -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m cgscout.sources.openalex_usage")
    ap.add_argument("cmd", choices=["status", "today"])
    ap.add_argument("--force", action="store_true", help="make a live request instead of using the latch/recent response")
    a = ap.parse_args()
    if a.cmd == "today":
        print(json.dumps(used_today_utc()))
        return 0
    b = budget_status(force=a.force)
    print(json.dumps({**b, "used_today_by_this_machine": used_today_utc()}))
    return 0 if b["state"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(_main())
