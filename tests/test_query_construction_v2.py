"""Unit tests for Query Construction v2 (cgscout/retrieval/direct/query_construction_v2.py).  No LLM, no network.
The fixtures are generic planner-style objectives; nothing here (or in the module) is specific to one system or benchmark case."""
from __future__ import annotations

import re


import cgscout.retrieval.direct.query_construction_v2 as QC  # noqa: E402
from cgscout.retrieval.direct.evidence_plan import EvidenceRoutePlan, RouteTerm  # noqa: E402
from cgscout.retrieval.direct.route_plan import RouteRetrievalBudgets, build_route_retrieval_plan  # noqa: E402

_LEGACY = RouteRetrievalBudgets(flat_anchor_short_context=True, flat_query_hygiene=True, need_semantic_enabled=True)
_V2 = RouteRetrievalBudgets(flat_anchor_short_context=True, flat_query_hygiene=True, need_semantic_enabled=True, query_construction_v2=True)
_FUNC = set("a an the of and or in on with to for by using via at from as into through during while between across among than that which when how whether".split())

OBJ = ("Use coarse-grained simulations to model moderate flexibility during protein–protein association, identify plausible bound orientations, "
       "and distinguish native-like interfaces from nonproductive docking arrangements.")
AXES = ["protein–protein coarse-grained docking", "flexible protein–protein association", "bound-orientation and encounter-complex sampling",
        "native-like interface versus nonproductive docking discrimination", "interface scoring and decoy ranking", "coarse-grained model benchmarks for protein–protein complexes"]


def _plan(system="protein–protein complex", objective=OBJ, axes=AXES, context=()):
    return EvidenceRoutePlan(
        scientific_objective=objective, system_description=system,
        process_terms=[RouteTerm(term=objective, route_type="process", source="USER_GROUNDED")],
        context_terms=[RouteTerm(term=c, route_type="context", source="USER_GROUNDED") for c in context], search_axes=list(axes))


def _queries(plan, budgets):
    return build_route_retrieval_plan(plan, budgets).route_queries


# ---- the switch is independent: OFF reproduces the legacy builders ----------------------------------------------------------------------------
def test_flag_off_is_the_legacy_plan_and_ignores_search_axes():
    with_axes = [(q.route, q.query) for q in _queries(_plan(), _LEGACY)]
    without_axes = [(q.route, q.query) for q in _queries(_plan(axes=[]), _LEGACY)]
    assert with_axes == without_axes
    assert any(q.query == "Use coarse-grained simulations to model moderate flexibility during protein–protein" for q in _queries(_plan(), _LEGACY))   # the legacy cut is still there


# ---- objective -> goals -----------------------------------------------------------------------------------------------------------------------------
def test_method_frame_is_removed_and_all_goals_are_kept():
    goals = QC.split_goals(OBJ)
    assert goals == ["moderate flexibility during protein–protein association", "plausible bound orientations", "native-like interfaces from nonproductive docking arrangements"]


def test_question_word_is_kept_and_leading_verb_dropped():
    assert QC.split_goals("Determine how confinement reshapes the architecture of composite networks") == ["how confinement reshapes the architecture of composite networks"]


def test_trailing_method_clause_and_leading_method_wrapper_are_removed():
    assert QC.split_goals("Simulate sequence-dependent kinetics using coarse-grained methods") == ["sequence-dependent kinetics"]
    assert QC.split_goals("Coarse-grained modeling of association among capsids, including electrostatic screening and explicit ion association") == \
        ["association among capsids", "electrostatic screening and explicit ion association"]


def test_noun_lists_are_not_split():
    assert QC.split_goals("Model the binding of A, B and C to the membrane") == ["binding of A, B and C to the membrane"]


# ---- search_axes -------------------------------------------------------------------------------------------------------------------------------------
def test_axes_drop_category_labels_and_list_axes_and_strip_method_words():
    axes = ["system-specific coarse-grained model precedent", "structurally related system transfer", "mesoscopic filament, rod, beam, or elastic-network model",
            "protein–protein coarse-grained docking", "topology-aware coarse-grained modeling of loop closure", "sequence-dependent folding kinetics in coarse-grained protein models"]
    assert QC.axis_concepts(axes) == ["protein–protein docking", "loop closure", "sequence-dependent folding kinetics"]


# ---- flat lexical + need_semantic -------------------------------------------------------------------------------------------------------------------
def test_v2_flat_queries_use_the_axis_task_complete_phrases_and_no_method_duplication():
    qs = _queries(_plan(), _V2)
    flat = [q.query for q in qs if q.route != "need_semantic"]
    assert "protein–protein docking coarse-grained model" in flat
    assert "protein–protein complex coarse-grained model" in flat                      # system-specific query unchanged
    assert len(flat) <= 6
    for q in flat:
        assert len(re.findall(r"coarse[- ]grain", q, re.I)) == 1, q
        assert not q.lower().startswith("use "), q
        assert re.findall(r"[A-Za-z–\-]+", q.lower())[-1] not in _FUNC, q
    assert len(set(flat)) == len(flat)


def test_v2_keeps_the_system_and_context_entity_queries_unchanged():
    legacy = [q.query for q in _queries(_plan(system="tetra-nucleosome array with HP1α and a truncated linker protein", context=("HP1α", "truncated linker protein")), _LEGACY)
              if q.route in ("system_specific_cg", "context")]
    v2 = [q.query for q in _queries(_plan(system="tetra-nucleosome array with HP1α and a truncated linker protein", context=("HP1α", "truncated linker protein")), _V2)
          if q.route in ("system_specific_cg", "context")]
    assert legacy == v2 and "HP1α coarse-grained model" in v2


def test_v2_semantic_query_is_one_natural_sentence_without_repeated_method_words():
    ns = [q.query for q in _queries(_plan(), _V2) if q.route == "need_semantic"]
    assert len(ns) == 1
    q = ns[0]
    assert q.startswith("Coarse-grained modeling of protein–protein complex:") and q.endswith(".")
    assert len(re.findall(r"coarse[- ]grain", q, re.I)) == 1 and not re.search(r"\buse\b", q, re.I)
    for g in QC.split_goals(OBJ):
        assert g in q


def test_v2_flat_budget_never_exceeds_the_legacy_ceiling():
    ctx = ("alpha one", "beta two", "gamma three", "delta four")
    flat = [q for q in _queries(_plan(system="a long multi component assembly with several named partners", context=ctx), _V2) if q.route != "need_semantic"]
    assert len(flat) <= 6


# ---- 3A ------------------------------------------------------------------------------------------------------------------------------------------------
def test_v2_3a_has_a_complete_need_query_and_distinct_emphases():
    an = QC.analyze(_plan(), _V2)
    qs = QC.render_direct_concept_queries_v2("flexible protein–protein association complexes", an)
    assert 2 <= len(qs) <= 3 and len(set(qs)) == len(qs)
    assert all(g in qs[0] for g in QC.split_goals(OBJ))                               # the first query expresses the complete need
    assert "protein–protein docking" in " ".join(qs)
    for q in qs:
        assert len(re.findall(r"coarse[- ]grain", q, re.I)) == 1 and not re.search(r"\buse\b", q, re.I), q
    assert QC.render_direct_concept_queries_v2("", an) == []


def test_v2_works_without_search_axes():
    qs = [q.query for q in _queries(_plan(axes=[]), _V2)]
    assert "moderate flexibility during protein–protein association coarse-grained model" in qs
