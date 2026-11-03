"""Unit test for this session's v1 decision: search_and_dedupe()'s default
`sources` is OpenAlex-only, Semantic Scholar removed from the default (not
just deprioritized) -- diagnosed live against the canonical case: every
flat-route query failed against S2's unauthenticated rate limit,
contributing 0 papers while costing ~10.5s/query in retry/backoff.
Semantic Scholar itself is untouched -- passing sources=("semantic_scholar",
"openalex") explicitly still works exactly as before.

No real network -- same monkeypatch convention as test_search_cache.py.
"""
from __future__ import annotations

import beam_agent.search as search
from beam_agent.models import Paper


def _fake_source(papers_by_query: dict, calls: list):
    def _fn(query: str, limit: int = 20, *, mode: str = "lexical"):
        calls.append(query)
        return list(papers_by_query.get(query, []))
    return _fn


def test_default_sources_is_openalex_only():
    import inspect
    sig = inspect.signature(search.search_and_dedupe)
    assert sig.parameters["sources"].default == ("openalex",)


def test_default_call_never_touches_semantic_scholar(monkeypatch):
    calls_oa: list = []
    calls_s2: list = []
    papers = {"beta-helix": [Paper(title="Beta-Helix Review", doi="10.1/x", sources=["openalex"])]}
    monkeypatch.setattr(search, "search_openalex", _fake_source(papers, calls_oa))
    monkeypatch.setattr(search, "search_semantic_scholar", _fake_source(papers, calls_s2))

    result = search.search_and_dedupe("beta-helix", limit=5)  # no sources= -- exercises the default

    assert calls_oa == ["beta-helix"]
    assert calls_s2 == []  # never called at all under the new default
    assert len(result) == 1
    assert result[0].title == "Beta-Helix Review"


def test_explicit_sources_can_still_re_enable_semantic_scholar(monkeypatch):
    """Semantic Scholar itself is unchanged -- an explicit sources= still works."""
    calls_oa: list = []
    calls_s2: list = []
    papers = {"beta-helix": [Paper(title="OA paper", doi="10.1/oa", sources=["openalex"])]}
    s2_papers = {"beta-helix": [Paper(title="S2 paper", doi="10.1/s2", sources=["semantic_scholar"])]}
    monkeypatch.setattr(search, "search_openalex", _fake_source(papers, calls_oa))
    monkeypatch.setattr(search, "search_semantic_scholar", _fake_source(s2_papers, calls_s2))

    result = search.search_and_dedupe("beta-helix", limit=5, sources=("semantic_scholar", "openalex"))

    assert calls_oa == ["beta-helix"]
    assert calls_s2 == ["beta-helix"]
    titles = {p.title for p in result}
    assert titles == {"OA paper", "S2 paper"}
