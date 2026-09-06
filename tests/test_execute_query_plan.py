"""Unit tests for search.execute_query_plan -- no real network calls.

Patches beam_agent.search.search_openalex/search_semantic_scholar (the names
as imported into search.py, not their own module) with fakes so behavior is
deterministic and fast.
"""

from unittest.mock import patch

from beam_agent import search as search_module
from beam_agent.models import Paper
from beam_agent.query_planner import PlannedQuery, QueryPlan
from beam_agent.concept_resolver import ScientificConceptProfile


def _plan(*planned_queries: PlannedQuery) -> QueryPlan:
    return QueryPlan(profile=ScientificConceptProfile(raw_query="stub"), queries=list(planned_queries))


def test_execute_query_plan_stamps_provenance_and_dedupes():
    plan = _plan(
        PlannedQuery(query="beta-helix folding", category="system_specific"),
        PlannedQuery(query="coarse-grained model folding", category="general_discovery"),
    )

    def fake_openalex(query, limit):
        return [Paper(title="Shared Paper", doi="10.1/shared", sources=["openalex"])]

    def fake_s2(query, limit):
        return [Paper(title="Shared Paper", doi="10.1/shared", sources=["semantic_scholar"])]

    with patch.object(search_module, "search_openalex", side_effect=fake_openalex), \
         patch.object(search_module, "search_semantic_scholar", side_effect=fake_s2):
        result = search_module.execute_query_plan(plan, per_source_limit=5)

    assert len(result) == 1  # same DOI from both sources, both queries -- merges into one
    merged = result[0]
    assert set(merged.sources) == {"openalex", "semantic_scholar"}
    assert set(merged.found_by_queries) == {"beta-helix folding", "coarse-grained model folding"}
    assert set(merged.query_categories) == {"system_specific", "general_discovery"}


def test_execute_query_plan_passes_per_source_limit_through():
    plan = _plan(PlannedQuery(query="q", category="system_specific"))
    seen_limits = []

    def fake_openalex(query, limit):
        seen_limits.append(("openalex", limit))
        return []

    def fake_s2(query, limit):
        seen_limits.append(("semantic_scholar", limit))
        return []

    with patch.object(search_module, "search_openalex", side_effect=fake_openalex), \
         patch.object(search_module, "search_semantic_scholar", side_effect=fake_s2):
        search_module.execute_query_plan(plan, per_source_limit=3)

    assert ("openalex", 3) in seen_limits
    assert ("semantic_scholar", 3) in seen_limits


def test_execute_query_plan_one_source_fails_other_still_returns():
    plan = _plan(PlannedQuery(query="q", category="system_specific"))

    def failing_s2(query, limit):
        raise RuntimeError("Semantic Scholar unavailable")

    def working_openalex(query, limit):
        return [Paper(title="OpenAlex-only paper", doi="10.1/oa-only", sources=["openalex"])]

    with patch.object(search_module, "search_openalex", side_effect=working_openalex), \
         patch.object(search_module, "search_semantic_scholar", side_effect=failing_s2):
        result = search_module.execute_query_plan(plan, per_source_limit=5)

    assert len(result) == 1
    assert result[0].title == "OpenAlex-only paper"
    assert len(search_module.execute_query_plan.last_errors) == 1
    assert "semantic_scholar" in search_module.execute_query_plan.last_errors[0]


def test_execute_query_plan_both_sources_fail_returns_empty_not_raises():
    plan = _plan(PlannedQuery(query="q", category="system_specific"))

    def failing(query, limit):
        raise RuntimeError("down")

    with patch.object(search_module, "search_openalex", side_effect=failing), \
         patch.object(search_module, "search_semantic_scholar", side_effect=failing):
        result = search_module.execute_query_plan(plan, per_source_limit=5)

    assert result == []
    assert len(search_module.execute_query_plan.last_errors) == 2


def test_execute_query_plan_last_raw_papers_exposes_pre_dedup_counts():
    plan = _plan(PlannedQuery(query="q", category="system_specific"))

    def fake_openalex(query, limit):
        return [Paper(title="Same", doi="10.1/x", sources=["openalex"])]

    def fake_s2(query, limit):
        return [Paper(title="Same", doi="10.1/x", sources=["semantic_scholar"])]

    with patch.object(search_module, "search_openalex", side_effect=fake_openalex), \
         patch.object(search_module, "search_semantic_scholar", side_effect=fake_s2):
        result = search_module.execute_query_plan(plan, per_source_limit=5)

    assert len(result) == 1  # deduped
    assert len(search_module.execute_query_plan.last_raw_papers) == 2  # pre-dedup: one per source
