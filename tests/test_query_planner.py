"""Unit tests for the Stage 1 baseline query planner. No network calls."""

from __future__ import annotations

from beam_agent.concept_resolver import BaselineConceptResolver, ConceptTerm, ScientificConceptProfile
from beam_agent.query_planner import BaselineQueryPlanner


def _resolve(query: str) -> ScientificConceptProfile:
    return BaselineConceptResolver().resolve(query)


def test_first_system_specific_query_is_original_verbatim():
    profile = _resolve("200-aa beta-helix protein folding coarse-grained model")
    plan = BaselineQueryPlanner().plan(profile)

    system_specific = [pq for pq in plan.queries if pq.category == "system_specific"]
    assert system_specific[0].query == "200-aa beta-helix protein folding coarse-grained model"


def test_produces_both_categories():
    profile = _resolve("200-aa beta-helix protein folding coarse-grained model")
    plan = BaselineQueryPlanner().plan(profile)

    categories = {pq.category for pq in plan.queries}
    assert categories == {"system_specific", "general_discovery"}


def test_general_discovery_drops_size_constraint():
    profile = _resolve("200-aa beta-helix protein folding coarse-grained model")
    plan = BaselineQueryPlanner().plan(profile)

    general = [pq for pq in plan.queries if pq.category == "general_discovery"]
    assert general  # non-empty
    assert all("200-aa" not in pq.query for pq in general)


def test_general_discovery_covers_structural_term_without_system_specifics():
    profile = _resolve("200-aa beta-helix protein folding coarse-grained model")
    plan = BaselineQueryPlanner().plan(profile)

    general_queries = [pq.query for pq in plan.queries if pq.category == "general_discovery"]
    assert any("beta-helix" in q for q in general_queries)


def test_general_discovery_uses_related_families_when_present():
    # Baseline resolver never populates related_families itself, but the
    # planner must still consume it correctly once an LLM resolver does.
    profile = ScientificConceptProfile(
        raw_query="beta-helix protein folding",
        related_families=[ConceptTerm(term="autotransporter passenger domains")],
    )
    plan = BaselineQueryPlanner().plan(profile)

    general_queries = [pq.query for pq in plan.queries if pq.category == "general_discovery"]
    assert any("autotransporter passenger domains" in q for q in general_queries)


def test_general_discovery_falls_back_to_raw_query_when_nothing_resolved():
    profile = ScientificConceptProfile(raw_query="xyz unresolved query")
    plan = BaselineQueryPlanner().plan(profile)

    general = [pq for pq in plan.queries if pq.category == "general_discovery"]
    assert len(general) == 1
    assert general[0].query == "xyz unresolved query"
    assert general[0].rationale == "fallback: no concepts resolved"


def test_unique_query_strings_dedupes_across_categories():
    profile = _resolve("folding")
    plan = BaselineQueryPlanner().plan(profile)

    strings = plan.unique_query_strings()
    lowered = [s.lower() for s in strings]
    assert len(lowered) == len(set(lowered))


def test_respects_max_queries_per_category():
    profile = _resolve(
        "200-aa beta-helix protein folding coarse-grained model machine learning molecular dynamics"
    )
    planner = BaselineQueryPlanner(max_queries_per_category=2)
    plan = planner.plan(profile)

    for category in ("system_specific", "general_discovery"):
        count = sum(1 for pq in plan.queries if pq.category == category)
        assert count <= 2


def test_profile_embedded_in_plan_for_provenance():
    profile = _resolve("folding kinetics")
    plan = BaselineQueryPlanner().plan(profile)

    assert plan.profile.raw_query == "folding kinetics"
    assert plan.planner == "BaselineQueryPlanner"
