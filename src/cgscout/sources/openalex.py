"""Thin client for the OpenAlex Works API (search only).

Docs: https://docs.openalex.org/api-entities/works/search-works

No API key required and no registration needed. OpenAlex asks callers to send
a contact email via the `mailto` param to get into their faster "polite pool" --
that's opt-in here via the OPENALEX_MAILTO env var (unset by default) rather than
hardcoded, since it's your contact info going to a third-party service.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

import requests

from cgscout.models import ExternalIds, Paper
from cgscout.sources import openalex_usage as usage

_BASE_URL = "https://api.openalex.org/works"
_SELECT = ",".join(
    [
        "id",
        "doi",
        "title",
        "publication_year",
        "authorships",
        "primary_location",
        "open_access",
        "abstract_inverted_index",
        "type",
    ]
)

# Usage-instrumentation only (added this session, diagnosing a real daily-
# budget exhaustion) -- does NOT change retry count, backoff, per_page/
# limit, or when a retry happens; purely records what already occurs.
# Header names per OpenAlex's actual (observed) 2026 usage-based-pricing
# response, not guessed: x-ratelimit-limit / -remaining are CREDITS (1000 =
# $0.10 anonymous-tier daily budget, confirmed live via a real 429 response
# this session); -limit-usd / -remaining-usd mirror the same in dollars;
# -credits-required / -cost-required-usd are what THIS request would have
# cost. A successful (non-429) response is expected to carry the used-side
# equivalents (x-ratelimit-credits-used / -remaining after debit) -- names
# kept generic/lower-cased-lookup below so whichever subset a given response
# actually includes is captured, rather than assuming a fixed set.
_RATE_LIMIT_HEADER_KEYS = (
    "x-ratelimit-limit", "x-ratelimit-remaining",
    "x-ratelimit-limit-usd", "x-ratelimit-remaining-usd",
    "x-ratelimit-credits-required", "x-ratelimit-credits-used",
    "x-ratelimit-cost-required-usd", "x-ratelimit-cost-usd",
    "x-ratelimit-reset", "retry-after",
)


def _extract_rate_limit_headers(headers) -> dict[str, str]:
    return {k: headers[k] for k in _RATE_LIMIT_HEADER_KEYS if k in headers}


class OpenAlexRateLimitExceeded(RuntimeError):
    """Raised when OpenAlex's usage-based-pricing budget is exhausted (HTTP
    429 on every retry) -- distinct from a generic request failure so a
    caller (or usage report) can tell "we are out of budget" apart from
    "OpenAlex/the network had a transient problem" without string-matching
    a generic RuntimeError's message. `headers` carries whatever rate-limit
    response headers were present on the LAST 429 seen (see
    _RATE_LIMIT_HEADER_KEYS) for exact accounting -- never fabricated when
    a header is absent."""

    def __init__(self, message: str, *, headers: Optional[dict[str, str]] = None):
        super().__init__(message)
        self.headers = headers or {}


search_openalex_last_calls: list[dict[str, Any]] = []

# search.semantic= (OpenAlex, added 2026-08): title+abstract embedded with
# GTE-Large-EN, cosine-similarity retrieval, max 50 results, hard-limited
# to 1 request/second by OpenAlex itself -- unlike lexical `search=`, which
# has no such per-second cap (only the usage-based credit budget above).
# Enforced here with a simple last-call-timestamp throttle -- correct for
# this codebase's actual concurrency (every real caller of search_openalex
# in the semantic-mode call sites runs its retrieval loop on the main
# thread, sequentially; Hop-1's ThreadPoolExecutor -- see related_system_
# discovery.py -- parallelizes the LLM extraction step, never the search
# calls). A module-level global is sufficient; this is not meant to be
# thread-safe beyond that.
_SEMANTIC_MIN_INTERVAL_S = 1.05
_last_semantic_call_ts = 0.0


def _throttle_semantic() -> None:
    """Hold search.semantic to 1 request/second ACROSS processes (see openalex_usage.throttle_semantic)."""
    usage.throttle_semantic(_SEMANTIC_MIN_INTERVAL_S)


def _raise_budget_exhausted(query: str, headers: dict, detail: str):
    """Out-of-budget error. Default: OpenAlexRateLimitExceeded (a RuntimeError, as before; its message still
    contains "rate_limit_exceeded"). With OPENALEX_STRICT_BUDGET=1: OpenAlexBudgetExhausted, a BaseException
    that `except Exception` blocks cannot swallow, so a benchmark stops instead of logging a false miss."""
    msg = f"rate_limit_exceeded: {detail} (query={query!r}, headers={headers})"
    if usage.strict_budget():
        reset = headers.get("retry-after") or headers.get("x-ratelimit-reset")
        raise usage.OpenAlexBudgetExhausted(msg, headers=headers, reset_s=int(reset) if str(reset).isdigit() else None)
    raise OpenAlexRateLimitExceeded(msg, headers=headers)


def search_openalex(
    query: str,
    limit: int = 20,
    *,
    mode: str = "lexical",
    max_retries: int = 3,
    timeout: float = 15.0,
) -> list[Paper]:
    """Query OpenAlex's search-works endpoint and return normalized Papers.

    `mode`: "lexical" (default, unchanged behavior -- `search=`, full-text/
    stemmed keyword matching, no per-request rate cap beyond the shared
    credit budget) or "semantic" (`search.semantic=`, GTE-Large-EN
    embedding + cosine similarity, capped at 50 results, throttled to
    1 req/s). Every OTHER existing caller (nothing passes `mode` yet) sees
    byte-identical behavior to before this parameter existed.

    Every real HTTP attempt (success, 429, or network error) is appended to
    module-level `search_openalex_last_calls` -- {query, attempt, status,
    elapsed_s, headers, mode} -- for exact usage accounting; also exposed as
    `search_openalex.last_calls` (same list object) matching this file's
    existing `.last_errors`-style introspection convention. Persistent 429
    across every retry raises OpenAlexRateLimitExceeded explicitly (message
    contains "rate_limit_exceeded") instead of a generic RuntimeError --
    this is the ONLY behavior change; retry count/backoff/coverage are
    unchanged."""
    if mode not in ("lexical", "semantic"):
        raise ValueError(f"search_openalex: unknown mode {mode!r} (expected 'lexical' or 'semantic')")
    search_param = "search" if mode == "lexical" else "search.semantic"
    per_page_cap = 200 if mode == "lexical" else 50  # semantic's own documented max
    params: dict[str, Any] = {
        search_param: query,
        "per_page": min(limit, per_page_cap),  # API cap per request -- ONE request, no pagination loop
        "select": _SELECT,
    }
    mailto = os.environ.get("OPENALEX_MAILTO")
    if mailto:
        params["mailto"] = mailto
    # OpenAlex's current (2026) usage-based pricing: keyless/anonymous callers
    # get a $0.10/day budget (~10 search requests/day at 10 credits each); a
    # free API key raises that 10x to $1/day. Opt-in via env var, same
    # reasoning as OPENALEX_MAILTO above -- never hardcoded, since it's the
    # user's own key.
    api_key = os.environ.get("OPENALEX_API_KEY")
    if api_key:
        params["api_key"] = api_key

    # --- usage management (no retrieval logic) ---------------------------------
    # Request-level cache shared by EVERY caller, including diagnostic scripts that
    # bypass search_and_dedupe. Only HTTP-200 responses are ever stored, so a failed
    # request can never be replayed as "this query has no results".
    cache_key = usage.request_cache_key(query, limit=params["per_page"], mode=mode, select=_SELECT)
    cached = usage.request_cache_get(cache_key)
    if cached is not None:
        usage.count_cache_hit()
        return [Paper.model_validate(d) for d in cached]
    # Budget already known to be exhausted (this process or another): fail fast -- no HTTP, no retry sleeps.
    known = usage.exhausted()
    if known:
        _raise_budget_exhausted(query, known.get("headers", {}), usage.exhausted_message(known))

    last_error: Exception | None = None
    last_rate_limit_headers: dict[str, str] = {}
    for attempt in range(max_retries):
        if mode == "semantic":
            _throttle_semantic()
        t0 = time.time()
        try:
            resp = requests.get(_BASE_URL, params=params, timeout=timeout)
            rl_headers = _extract_rate_limit_headers(resp.headers)
            call_record = {
                "query": query, "mode": mode, "attempt": attempt, "status": resp.status_code,
                "elapsed_s": round(time.time() - t0, 3), "headers": rl_headers,
            }
            search_openalex_last_calls.append(call_record)
            usage.record_request(mode=mode, limit=params["per_page"], status=resp.status_code,
                                 headers=rl_headers, elapsed_s=time.time() - t0, query=query)
            if resp.status_code == 429:
                last_rate_limit_headers = rl_headers
                if usage.is_budget_exhausted_response(rl_headers):
                    # Out of budget: retrying cannot succeed, so do not burn time on backoff sleeps.
                    usage.mark_exhausted(rl_headers, f"query={query[:60]!r}")
                    break
                time.sleep(2**attempt)
                continue
            resp.raise_for_status()
            data = resp.json()
            papers = [_to_paper(item) for item in data.get("results", [])]
            usage.request_cache_set(cache_key, [p.model_dump(mode="json") for p in papers])
            return papers
        except requests.RequestException as exc:  # noqa: PERF203
            search_openalex_last_calls.append({
                "query": query, "mode": mode, "attempt": attempt, "status": None,
                "elapsed_s": round(time.time() - t0, 3), "headers": {}, "error": str(exc),
            })
            if not isinstance(exc, requests.HTTPError):   # an HTTP status was already accounted for above
                usage.record_request(mode=mode, limit=params["per_page"], status=None, headers={},
                                     elapsed_s=time.time() - t0, query=query, error=str(exc))
            last_error = exc
            time.sleep(min(2**attempt, 8))

    if usage.is_budget_exhausted_response(last_rate_limit_headers):
        _raise_budget_exhausted(query, last_rate_limit_headers,
                                f"OpenAlex budget exhausted (HTTP 429, remaining=0)")
    raise RuntimeError(f"OpenAlex search failed after {max_retries} attempts") from last_error


search_openalex.last_calls = search_openalex_last_calls  # type: ignore[attr-defined]


def _to_paper(item: dict[str, Any]) -> Paper:
    authorships = item.get("authorships") or []
    authors = [
        a.get("author", {}).get("display_name", "")
        for a in authorships
        if a.get("author", {}).get("display_name")
    ]

    primary_location = item.get("primary_location") or {}
    source = primary_location.get("source") or {}
    venue = source.get("display_name")

    open_access = item.get("open_access") or {}
    doi = item.get("doi")
    if doi:
        doi = doi.removeprefix("https://doi.org/")

    openalex_id = item.get("id")
    if openalex_id:
        openalex_id = openalex_id.removeprefix("https://openalex.org/")

    return Paper(
        title=item.get("title") or "(untitled)",
        authors=authors,
        year=item.get("publication_year"),
        doi=doi,
        abstract=_reconstruct_abstract(item.get("abstract_inverted_index")),
        venue=venue,
        sources=["openalex"],
        external_ids=ExternalIds(openalex_id=openalex_id),
        oa_type=item.get("type"),
        is_open_access=open_access.get("is_oa"),
        oa_pdf_url=open_access.get("oa_url"),
        url=primary_location.get("landing_page_url"),
    )


def _reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str | None:
    """OpenAlex stores abstracts as {word: [positions]} to dodge copyright issues.

    Reassemble into plain text; return None if there's nothing to reconstruct.
    """
    if not inverted_index:
        return None
    positions: dict[int, str] = {}
    for word, idxs in inverted_index.items():
        for idx in idxs:
            positions[idx] = word
    if not positions:
        return None
    return " ".join(positions[i] for i in sorted(positions))
