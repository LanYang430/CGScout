"""Unit tests for search_and_dedupe()'s dev-speed retrieval cache
(beam_agent/search.py). No real network -- search_openalex/
search_semantic_scholar are monkeypatched with call-counting fakes.
tests/conftest.py isolates BEAM_CACHE_DIR per test automatically.
"""
from __future__ import annotations

import time

from beam_agent import cache
import beam_agent.search as search
from beam_agent.models import Paper


def _fake_source(papers_by_query: dict, calls: list):
    def _fn(query: str, limit: int = 20):
        calls.append(query)
        return list(papers_by_query.get(query, []))
    return _fn


def test_second_identical_call_is_a_cache_hit_no_second_source_call(monkeypatch):
    calls_oa: list = []
    calls_s2: list = []
    papers = {"beta-helix": [Paper(title="Beta-Helix Review", doi="10.1/x", sources=["openalex"])]}
    monkeypatch.setattr(search, "search_openalex", _fake_source(papers, calls_oa))
    monkeypatch.setattr(search, "search_semantic_scholar", _fake_source(papers, calls_s2))

    first = search.search_and_dedupe("beta-helix", limit=5, sources=("openalex",))
    second = search.search_and_dedupe("beta-helix", limit=5, sources=("openalex",))

    assert len(calls_oa) == 1  # NOT 2 -- the second call must be a cache hit
    assert [p.title for p in first] == [p.title for p in second]
    assert first[0].doi == second[0].doi == "10.1/x"
    assert search.search_and_dedupe.last_errors == []


def test_different_query_or_limit_or_sources_is_a_cache_miss(monkeypatch):
    calls_oa: list = []
    papers = {
        "beta-helix": [Paper(title="A", doi="10.1/a", sources=["openalex"])],
        "coiled-coil": [Paper(title="B", doi="10.1/b", sources=["openalex"])],
    }
    monkeypatch.setattr(search, "search_openalex", _fake_source(papers, calls_oa))
    monkeypatch.setattr(search, "search_semantic_scholar", _fake_source({}, []))

    search.search_and_dedupe("beta-helix", limit=5, sources=("openalex",))
    search.search_and_dedupe("coiled-coil", limit=5, sources=("openalex",))   # different query
    search.search_and_dedupe("beta-helix", limit=8, sources=("openalex",))    # different limit
    search.search_and_dedupe("beta-helix", limit=5, sources=("openalex", "semantic_scholar"))  # different sources

    assert len(calls_oa) == 4  # every one of the 4 calls above is a genuine miss


def test_a_source_error_is_never_cached(monkeypatch):
    calls_oa: list = []

    def _flaky_openalex(query, limit=20):
        calls_oa.append(query)
        if len(calls_oa) == 1:
            raise RuntimeError("simulated rate limit")
        return [Paper(title="Recovered", doi="10.1/r", sources=["openalex"])]

    monkeypatch.setattr(search, "search_openalex", _flaky_openalex)
    monkeypatch.setattr(search, "search_semantic_scholar", _fake_source({}, []))

    first = search.search_and_dedupe("beta-helix", limit=5, sources=("openalex",))
    assert first == []
    assert search.search_and_dedupe.last_errors != []

    second = search.search_and_dedupe("beta-helix", limit=5, sources=("openalex",))
    assert len(calls_oa) == 2  # the errored first call must NOT have been cached
    assert [p.title for p in second] == ["Recovered"]
    assert search.search_and_dedupe.last_errors == []


def test_cache_hit_preserves_paper_order_and_fields(monkeypatch):
    calls_oa: list = []
    ordered = [
        Paper(title="First", doi="10.1/1", sources=["openalex", "semantic_scholar"]),
        Paper(title="Second", doi="10.1/2", sources=["openalex"]),
    ]
    monkeypatch.setattr(search, "search_openalex", _fake_source({"q": ordered}, calls_oa))
    monkeypatch.setattr(search, "search_semantic_scholar", _fake_source({}, []))

    first = search.search_and_dedupe("q", limit=5, sources=("openalex",))
    second = search.search_and_dedupe("q", limit=5, sources=("openalex",))

    assert [p.doi for p in first] == [p.doi for p in second] == ["10.1/1", "10.1/2"]
    assert [p.sources for p in second] == [["openalex", "semantic_scholar"], ["openalex"]]


# --- 24h TTL -------------------------------------------------------------

def test_expired_entry_is_re_fetched_not_reused(monkeypatch):
    import json

    calls_oa: list = []
    monkeypatch.setattr(search, "search_openalex",
                         _fake_source({"q": [Paper(title="Fresh", doi="10.1/f", sources=["openalex"])]}, calls_oa))
    monkeypatch.setattr(search, "search_semantic_scholar", _fake_source({}, []))

    # Directly write a backdated entry (past the 24h TTL search_and_dedupe
    # uses) for the exact key this call will compute -- simulates
    # yesterday's cache, without needing to actually wait 25 hours.
    key = cache.content_hash("q", "5", "openalex")
    path = cache._path_for("search_and_dedupe", key, None)  # default cache dir -- BEAM_CACHE_DIR from conftest
    path.parent.mkdir(parents=True, exist_ok=True)
    stale_cached_at = time.time() - (25 * 60 * 60)  # 25h ago
    path.write_text(json.dumps({
        "_cached_at": stale_cached_at,
        "value": [{"title": "Stale", "doi": "10.1/stale", "sources": ["openalex"]}],
    }))

    result = search.search_and_dedupe("q", limit=5, sources=("openalex",))

    assert len(calls_oa) == 1  # the stale entry was NOT reused -- a real call happened
    assert [p.title for p in result] == ["Fresh"]


def test_entry_within_24h_is_still_reused(monkeypatch):
    calls_oa: list = []
    monkeypatch.setattr(search, "search_openalex",
                         _fake_source({"q": [Paper(title="Original", doi="10.1/o", sources=["openalex"])]}, calls_oa))
    monkeypatch.setattr(search, "search_semantic_scholar", _fake_source({}, []))

    search.search_and_dedupe("q", limit=5, sources=("openalex",))  # populates the cache, 1 real call

    result = search.search_and_dedupe("q", limit=5, sources=("openalex",))  # well within 24h

    assert len(calls_oa) == 1  # still just the one real call
    assert [p.title for p in result] == ["Original"]
