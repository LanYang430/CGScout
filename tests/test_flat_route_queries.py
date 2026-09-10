"""Unit tests for the flat-route query builders after this session's
refactor: direct_system deleted, system_specific_cg added, conversational
wrappers stripped from every flat-route query. No LLM, no network.

    .venv/bin/python -m pytest tests/test_flat_route_queries.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent.parent / "benchmarks" / "concept_resolver_reasoning"
sys.path.insert(0, str(BENCH))

from evidence_route_planner import EvidenceRoutePlan, RouteTerm  # noqa: E402
from route_retrieval_plan import (  # noqa: E402
    FLAT_ROUTES,
    RouteRetrievalBudgets,
    _strip_conversational_wrapper,
    build_context_queries,
    build_process_queries,
    build_route_retrieval_plan,
    build_system_specific_cg_queries,
)

_B = RouteRetrievalBudgets()


def _plan(*, system="", objective=None, context=None):
    return EvidenceRoutePlan(
        scientific_objective=objective or "",
        system_description=system,
        process_terms=[RouteTerm(term=objective, route_type="process", source="USER_GROUNDED")]
        if objective else [],
        context_terms=[RouteTerm(term=context, route_type="context", source="USER_GROUNDED")]
        if context else [],
    )


# ---- direct_system is gone --------------------------------------------------

def test_direct_system_route_no_longer_exists():
    assert "direct_system" not in FLAT_ROUTES
    assert "system_specific_cg" in FLAT_ROUTES
    import route_retrieval_plan as m
    assert not hasattr(m, "build_direct_system_queries")


def test_build_route_retrieval_plan_emits_no_direct_system_query():
    plan = build_route_retrieval_plan(_plan(system="EspP", objective="EspP folding"))
    routes = {q.route for q in plan.route_queries}
    assert "direct_system" not in routes
    assert routes <= set(FLAT_ROUTES)


# ---- system_specific_cg ---------------------------------------------------

def test_system_specific_cg_query_is_system_plus_cg_suffix():
    qs = build_system_specific_cg_queries(_plan(system="EspP"), _B)
    assert len(qs) == 1
    assert qs[0].route == "system_specific_cg"
    assert qs[0].query == "EspP coarse-grained model"


def test_system_specific_cg_empty_when_no_system():
    assert build_system_specific_cg_queries(_plan(system=""), _B) == []


def test_system_specific_cg_strips_wrapper():
    qs = build_system_specific_cg_queries(_plan(system="I want to study EspP"), _B)
    assert qs[0].query == "EspP coarse-grained model"


# ---- process: {system} {process} coarse-grained model --------------------

def test_process_query_shape_and_wrapper_strip():
    # objective span carries the wrapper AND the system name inline (the
    # real "I want to study EspP folding." grounding)
    qs = build_process_queries(_plan(system="EspP", objective="I want to study EspP folding"), _B)
    assert len(qs) == 1
    assert qs[0].query == "EspP folding coarse-grained model"


def test_process_query_prepends_system_when_objective_lacks_it():
    qs = build_process_queries(_plan(system="alpha-synuclein", objective="want to study aggregation"), _B)
    assert qs[0].query == "alpha-synuclein aggregation coarse-grained model"


def test_process_query_no_system_doubling():
    qs = build_process_queries(_plan(system="pertactin", objective="pertactin folding"), _B)
    assert qs[0].query == "pertactin folding coarse-grained model"


# ---- context: grounded-context logic unchanged, wrapper stripped ---------

def test_context_empty_without_grounded_context_term():
    assert build_context_queries(_plan(system="EspP", objective="EspP folding"), _B) == []


def test_context_query_strips_wrapper_on_both_terms():
    qs = build_context_queries(
        _plan(system="tau", objective="I want to study aggregation", context="in a crowded environment"),
        _B,
    )
    joined = " || ".join(q.query for q in qs)
    assert "I want to study" not in joined
    assert "in a crowded environment aggregation" in joined  # context + stripped process
    assert "in a crowded environment coarse-grained model" in joined  # context + CG fallback


# ---- the widened conversational wrapper ----------------------------------

def test_wrapper_strips_the_listed_prefixes():
    cases = {
        "I want to study EspP folding": "EspP folding",
        "I want to simulate alpha-synuclein": "alpha-synuclein",  # article-eating bug fixed
        "I am interested in tau aggregation": "tau aggregation",
        "I'm studying the conformational ensemble": "conformational ensemble",
        "Could you recommend a CG model for pertactin folding": "pertactin folding",
        "we want to model amyloid-beta aggregation": "amyloid-beta aggregation",
    }
    for raw, want in cases.items():
        assert _strip_conversational_wrapper(raw) == want, raw


def test_wrapper_leaves_clean_scientific_terms_untouched():
    for s in ["EspP folding", "beta-helix protein", "alpha-synuclein",
              "an intrinsically disordered protein", "adenylate kinase open-to-closed transition"]:
        assert _strip_conversational_wrapper(s) == s


def test_wrapper_never_returns_empty():
    assert _strip_conversational_wrapper("I want to study") == "I want to study"
    assert _strip_conversational_wrapper("") == ""
