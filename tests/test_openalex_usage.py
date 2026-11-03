"""OpenAlex API usage management: request cache, ledger/accounting, budget latch,
strict mode, pre-flight, throttle and benchmark slots. requests.get is always faked --
no test here touches the network or the project's real .cache/ (conftest isolates it)."""
import json
import time

import pytest
import requests

from beam_agent.sources import openalex
from beam_agent.sources import openalex_usage as usage

_REAL_SLEEP = time.sleep
KEY = "test-secret-key-123"
OK_HEADERS = {"x-ratelimit-limit": "10000", "x-ratelimit-remaining": "9990", "x-ratelimit-credits-required": "10"}
EXHAUSTED_HEADERS = {"x-ratelimit-limit": "10000", "x-ratelimit-remaining": "0", "x-ratelimit-remaining-usd": "0",
                     "x-ratelimit-credits-required": "10", "x-ratelimit-reset": "5000", "retry-after": "5000"}


class FakeResp:
    def __init__(self, status=200, results=None, headers=None):
        self.status_code = status
        self._results = results if results is not None else [{"id": "https://openalex.org/W1", "title": "A paper",
                                                              "publication_year": 2024, "doi": "https://doi.org/10.1/x"}]
        self.headers = headers or {}

    def json(self):
        return {"results": self._results}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    usage._reset_for_tests()
    monkeypatch.setenv("OPENALEX_API_KEY", KEY)
    monkeypatch.delenv("OPENALEX_STRICT_BUDGET", raising=False)
    monkeypatch.delenv("OPENALEX_REQUEST_CACHE", raising=False)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    yield
    usage._reset_for_tests()


def fake_get(monkeypatch, responses):
    calls = []
    it = iter(responses)

    def _get(url, params=None, timeout=None):
        calls.append(dict(params or {}))
        try:
            r = next(it)
        except StopIteration:
            r = responses[-1]
        if isinstance(r, Exception):
            raise r
        return r
    monkeypatch.setattr(openalex.requests, "get", _get)
    return calls


def test_cache_hit_makes_no_http_request_and_is_counted(monkeypatch):
    calls = fake_get(monkeypatch, [FakeResp(200, headers=OK_HEADERS)])
    usage.set_run_label("t1")
    a = openalex.search_openalex("beta helix folding", 20)
    b = openalex.search_openalex("beta helix folding", 20)
    assert len(calls) == 1                       # second call served from the cache
    assert [p.title for p in a] == [p.title for p in b]
    u = usage.run_usage("t1")
    assert (u["http_requests"], u["http_200"], u["cache_hits"], u["credits_charged"]) == (1, 1, 1, 10)


def test_every_caller_shares_the_cache_even_with_other_limits_keyed_separately(monkeypatch):
    calls = fake_get(monkeypatch, [FakeResp(200, headers=OK_HEADERS)])
    openalex.search_openalex("q", 20)
    openalex.search_openalex("q", 50)            # different per_page -> a genuinely different request
    assert len(calls) == 2


def test_failed_requests_are_never_cached(monkeypatch):
    calls = fake_get(monkeypatch, [FakeResp(500), FakeResp(500), FakeResp(500), FakeResp(200, headers=OK_HEADERS)])
    with pytest.raises(RuntimeError):
        openalex.search_openalex("q", 20, max_retries=3)
    assert len(calls) == 3
    again = openalex.search_openalex("q", 20)    # nothing was cached from the failures -> real request
    assert len(calls) == 4 and again


def test_budget_exhausted_stops_retrying_and_latches(monkeypatch):
    calls = fake_get(monkeypatch, [FakeResp(429, headers=EXHAUSTED_HEADERS)])
    with pytest.raises(openalex.OpenAlexRateLimitExceeded) as ei:
        openalex.search_openalex("q1", 20)
    assert "rate_limit_exceeded" in str(ei.value)   # substring other modules key on
    assert len(calls) == 1                          # no pointless retries
    assert usage.exhausted() is not None
    with pytest.raises(openalex.OpenAlexRateLimitExceeded):
        openalex.search_openalex("q2", 20)           # a different, uncached query
    assert len(calls) == 1                           # fail fast: no HTTP at all


def test_cached_queries_still_served_when_budget_is_exhausted(monkeypatch):
    fake_get(monkeypatch, [FakeResp(200, headers=OK_HEADERS)])
    openalex.search_openalex("cached q", 20)
    usage.mark_exhausted(EXHAUSTED_HEADERS)
    assert openalex.search_openalex("cached q", 20)  # cache hit, no exception


def test_strict_mode_raises_something_except_exception_cannot_swallow(monkeypatch):
    monkeypatch.setenv("OPENALEX_STRICT_BUDGET", "1")
    fake_get(monkeypatch, [FakeResp(429, headers=EXHAUSTED_HEADERS)])
    swallowed = False
    with pytest.raises(usage.OpenAlexBudgetExhausted) as ei:
        try:
            openalex.search_openalex("q", 20)
        except Exception:  # noqa: BLE001 -- exactly what the retrieval pipeline does
            swallowed = True
    assert not swallowed
    assert not isinstance(ei.value, Exception)
    assert ei.value.reset_s == 5000


def test_non_budget_429_is_retried_normally(monkeypatch):
    calls = fake_get(monkeypatch, [FakeResp(429, headers={"x-ratelimit-remaining": "50"}), FakeResp(200, headers=OK_HEADERS)])
    assert openalex.search_openalex("q", 20)
    assert len(calls) == 2 and usage.exhausted() is None


def test_latch_clears_after_the_reset_time(monkeypatch):
    usage.mark_exhausted({**EXHAUSTED_HEADERS, "retry-after": "100", "x-ratelimit-reset": "100"})
    assert usage.exhausted() is not None
    t0 = time.time()
    monkeypatch.setattr(time, "time", lambda: t0 + 101)
    assert usage.exhausted() is None


def test_exhaustion_is_visible_to_a_second_process_via_the_state_file():
    usage.mark_exhausted(EXHAUSTED_HEADERS)
    usage._latch = None                              # simulate a fresh process: no in-memory latch
    assert usage.exhausted() is not None


def test_ledger_records_each_real_attempt_and_never_the_key(monkeypatch):
    fake_get(monkeypatch, [FakeResp(200, headers=OK_HEADERS), FakeResp(429, headers=EXHAUSTED_HEADERS)])
    usage.set_run_label("ledger")
    openalex.search_openalex("good", 20)
    with pytest.raises(openalex.OpenAlexRateLimitExceeded) as ei:
        openalex.search_openalex("bad", 20)
    lines = [json.loads(l) for l in usage.ledger_path().read_text().splitlines()]
    assert [(e["status"], e["credits_charged"]) for e in lines] == [(200, 10), (429, 0)]
    assert lines[0]["remaining"] == 9990 and lines[0]["label"] == "ledger"
    assert KEY not in usage.ledger_path().read_text()
    assert KEY not in str(ei.value)
    assert KEY not in (usage._state_path().read_text() if usage._state_path().exists() else "")
    assert usage.used_today_utc() == {"requests": 1, "credits": 10}


def test_credits_default_is_flagged_as_assumed_when_response_omits_price(monkeypatch):
    fake_get(monkeypatch, [FakeResp(200, headers={})])
    usage.set_run_label("assumed")
    openalex.search_openalex("q", 20)
    u = usage.run_usage("assumed")
    assert u["credits_charged"] == usage.CREDITS_PER_SEARCH_DEFAULT and u["credits_assumed_requests"] == 1


def test_budget_status_parses_headers(monkeypatch):
    calls = fake_get(monkeypatch, [FakeResp(200, headers=OK_HEADERS)])
    b = usage.budget_status(force=True)
    assert b["state"] == "ok" and b["remaining_credits"] == 9990 and b["authenticated"] is True
    assert calls[0].get("api_key") == KEY            # key is sent ...
    assert KEY not in json.dumps(b)                   # ... but never returned


def test_budget_status_reports_exhausted_and_latches(monkeypatch):
    fake_get(monkeypatch, [FakeResp(429, headers=EXHAUSTED_HEADERS)])
    b = usage.budget_status(force=True)
    assert b["state"] == "exhausted" and b["remaining_credits"] == 0 and b["reset_s"] == 5000
    assert usage.exhausted() is not None


def test_require_budget_refuses_when_exhausted_or_too_low(monkeypatch):
    usage.mark_exhausted(EXHAUSTED_HEADERS)
    with pytest.raises(usage.OpenAlexBudgetInsufficient, match="exhausted"):
        usage.require_budget(10)
    usage.clear_exhausted()
    fake_get(monkeypatch, [FakeResp(200, headers={**OK_HEADERS, "x-ratelimit-remaining": "300"})])
    with pytest.raises(usage.OpenAlexBudgetInsufficient, match="too low"):
        usage.require_budget(100)                    # needs 1000 credits, 300 left
    assert usage.require_budget(20)["remaining_credits"] == 300   # 200 needed -> fine


def test_require_budget_allows_when_remaining_header_is_absent(monkeypatch):
    fake_get(monkeypatch, [FakeResp(200, headers={})])
    assert usage.require_budget(50)["remaining_credits"] is None  # unverified, but a 200 proves budget > 0


def test_semantic_mode_uses_the_cross_process_throttle(monkeypatch):
    seen = []
    monkeypatch.setattr(usage, "throttle_semantic", lambda interval=None: seen.append(interval))
    fake_get(monkeypatch, [FakeResp(200, headers=OK_HEADERS)])
    openalex.search_openalex("q", 20, mode="semantic")
    assert seen == [openalex._SEMANTIC_MIN_INTERVAL_S]


def test_throttle_spaces_requests_across_calls(monkeypatch):
    monkeypatch.setattr(time, "sleep", _REAL_SLEEP)  # this test needs the real sleep
    t0 = time.time()
    usage.throttle_semantic(0.25)
    usage.throttle_semantic(0.25)
    assert time.time() - t0 >= 0.24


def test_benchmark_slots_limit_parallel_runs():
    first = usage.acquire_benchmark_slot(1)
    assert first
    assert usage.acquire_benchmark_slot(1) is None   # second benchmark must not start
    assert usage.acquire_benchmark_slot(2)           # a second slot, if explicitly allowed


def test_request_cache_can_be_disabled(monkeypatch):
    monkeypatch.setenv("OPENALEX_REQUEST_CACHE", "0")
    calls = fake_get(monkeypatch, [FakeResp(200, headers=OK_HEADERS)])
    openalex.search_openalex("q", 20)
    openalex.search_openalex("q", 20)
    assert len(calls) == 2


def test_forced_status_check_clears_a_stale_exhaustion_state(monkeypatch):
    usage.mark_exhausted(EXHAUSTED_HEADERS)               # remembered exhaustion
    assert usage.budget_status()["state"] == "exhausted"   # default: trust the latch, no request
    fake_get(monkeypatch, [FakeResp(200, headers=OK_HEADERS)])
    b = usage.budget_status(force=True)                    # e.g. funds were added
    assert b["state"] == "ok" and usage.exhausted() is None


def test_transient_429_followed_by_success_is_not_an_unresolved_failure(monkeypatch):
    fake_get(monkeypatch, [FakeResp(429, headers={"x-ratelimit-remaining": "9000"}), FakeResp(200, headers=OK_HEADERS)])
    usage.set_run_label("transient")
    assert openalex.search_openalex("q", 20)
    u = usage.run_usage("transient")
    assert (u["http_429"], u["failed_requests"], u["transient_failed_attempts"]) == (1, 0, 1)


def test_a_query_that_never_succeeds_is_an_unresolved_failure(monkeypatch):
    fake_get(monkeypatch, [FakeResp(500)])
    usage.set_run_label("hard-fail")
    with pytest.raises(RuntimeError):
        openalex.search_openalex("q", 20, max_retries=2)
    u = usage.run_usage("hard-fail")
    assert u["failed_requests"] == 1 and u["transient_failed_attempts"] == 1
