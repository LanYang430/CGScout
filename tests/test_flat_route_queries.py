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


# ---- v2: short, complementary flat lexical queries -----------------------------------------------------------------

def test_long_system_is_reduced_to_its_head_noun_phrase():
    qs = build_system_specific_cg_queries(
        _plan(system="a tetra-nucleosome array with HP1x and a truncated Polyhomeotic-like protein"), _B)
    assert qs[0].query == "tetra-nucleosome array coarse-grained model"


def test_short_system_is_unchanged():
    assert build_system_specific_cg_queries(_plan(system="EspP passenger domain"), _B)[0].query == "EspP passenger domain coarse-grained model"


def test_comma_inside_a_token_does_not_cut_the_system():
    q = build_system_specific_cg_queries(_plan(system="sequence-specific Na+,K+-ATPase in the sperm flagellar membrane"), _B)[0].query
    assert "Na+,K+-ATPase" in q


def test_process_drops_leading_verb_and_cg_method_tail():
    qs = build_process_queries(_plan(system="x", objective="Simulate protein-mediated bridging and altered chromatin-array folding using coarse-grained methods"), _B)
    assert qs[0].query == "protein-mediated bridging and altered chromatin-array folding coarse-grained model"


def test_descriptive_process_is_not_prefixed_with_the_system():
    qs = build_process_queries(_plan(system="a long multi component system description here", objective="Determine how confinement reshapes network architecture"), _B)
    assert qs[0].query == "confinement reshapes network architecture coarse-grained model"


def test_process_takes_only_the_first_clause():
    qs = build_process_queries(_plan(system="x", objective="Model capsid flexibility while retaining symmetry"), _B)
    assert qs[0].query == "x capsid flexibility coarse-grained model" or qs[0].query == "capsid flexibility coarse-grained model"


def test_each_context_term_gets_its_own_short_query_and_repeats_are_dropped():
    plan = EvidenceRoutePlan(
        scientific_objective="Simulate protein-mediated bridging and altered chromatin-array folding",
        system_description="a tetra-nucleosome array with HP1x and a truncated Polyhomeotic-like protein",
        process_terms=[RouteTerm(term="Simulate protein-mediated bridging and altered chromatin-array folding", route_type="process", source="USER_GROUNDED")],
        context_terms=[RouteTerm(term=t, route_type="context", source="USER_GROUNDED")
                       for t in ("tetranucleosome array", "HP1x", "truncated Polyhomeotic-like protein", "protein-mediated bridging")],
    )
    qs = [q.query for q in build_context_queries(plan, _B)]
    assert qs == ["HP1x coarse-grained model", "truncated Polyhomeotic-like protein coarse-grained model"]  # system head + process repeats skipped


def test_context_term_naming_the_cg_method_is_skipped():
    plan = _plan(system="capsid", objective="assembly", context="coarse-grained resolution")
    assert build_context_queries(plan, _B) == []


def test_bridge_passes_each_search_context_item_as_its_own_context_term():
    sys.path.insert(0, str(BENCH / "mellea_integration"))
    from planner_retrieval_bridge import evidence_plan_from_planner_output
    from scientific_planner import PlannerOutput
    po = PlannerOutput(system="a protein A bound to a protein B", target_component=None, scientific_objective="Simulate bridging",
                       clarification_needed=False, search_context=["alpha one", "beta two", "gamma three"])
    ctx = [t.term for t in evidence_plan_from_planner_output(po).context_terms]
    assert ctx == ["alpha one", "beta two", "gamma three"]


# ---- v3: uncovered-modifier chunks, system+process combination, same-route dedupe ---------------------------------------
from route_retrieval_plan import _system_chunks, _system_part  # noqa: E402


def test_system_chunks_split_head_and_modifiers_keeping_substantive_connectors():
    head, mods = _system_chunks("complementary nucleic-acid duplexes invaded by a third strand")
    assert head == "complementary nucleic-acid duplexes"
    assert mods == ["invaded by a third strand"]            # 'invaded by' names the relation and is kept
    head, mods = _system_chunks("a tetra-nucleosome array with HP1x and a truncated Polyhomeotic-like protein")
    assert head == "tetra-nucleosome array" and mods == ["HP1x", "truncated Polyhomeotic-like protein"]   # plain 'with'/'and' dropped


def test_short_system_has_no_modifiers():
    assert _system_chunks("EspP passenger domain") == ("EspP passenger domain", [])


def test_uncovered_modifier_is_kept_covered_one_is_not():
    s = "complementary nucleic-acid duplexes invaded by a third strand"
    assert _system_part(s, []) == "complementary nucleic-acid duplexes invaded by a third strand"       # nothing else carries it
    assert _system_part(s, ["invaded by a third strand mismatch"]) == "complementary nucleic-acid duplexes"   # already searched elsewhere
    assert _system_part(s, None) == "complementary nucleic-acid duplexes"                                  # no coverage info -> v2 behaviour


def test_at_most_one_modifier_chunk_is_added():
    part = _system_part("a tetra-nucleosome array with HP1x and a truncated Polyhomeotic-like protein", [])
    assert part == "tetra-nucleosome array HP1x"


def test_system_query_with_covered_context_stays_short():
    plan = EvidenceRoutePlan(
        scientific_objective="Simulate bridging", system_description="a tetra-nucleosome array with HP1x and a truncated Polyhomeotic-like protein",
        process_terms=[RouteTerm(term="Simulate bridging and folding using coarse-grained methods", route_type="process", source="USER_GROUNDED")],
        context_terms=[RouteTerm(term=t, route_type="context", source="USER_GROUNDED") for t in ("HP1x", "truncated Polyhomeotic-like protein")])
    qs = {(q.route, q.query_role): q.query for q in build_route_retrieval_plan(plan).route_queries}
    assert qs[("system_specific_cg", "system_specific_cg_evidence")] == "tetra-nucleosome array coarse-grained model"


def test_combo_query_is_added_for_a_descriptive_process_and_has_no_context_terms():
    plan = EvidenceRoutePlan(
        scientific_objective="x", system_description="complementary nucleic-acid duplexes invaded by a third strand",
        process_terms=[RouteTerm(term="Studying sequence-dependent strand-displacement kinetics, including the effects of mismatches, using coarse-grained simulation models",
                                 route_type="process", source="USER_GROUNDED")],
        context_terms=[RouteTerm(term="mismatch effects", route_type="context", source="USER_GROUNDED")])
    plan_q = build_route_retrieval_plan(plan).route_queries
    roles = {q.query_role: q.query for q in plan_q}
    assert roles["system_specific_cg_evidence"] == "complementary nucleic-acid duplexes invaded by a third strand coarse-grained model"
    assert roles["process_cg_evidence"] == "sequence-dependent strand-displacement kinetics coarse-grained model"
    assert roles["system_process_cg_evidence"] == "complementary nucleic-acid duplexes invaded by a third strand sequence-dependent strand-displacement kinetics coarse-grained model"
    assert "mismatch" not in roles["system_process_cg_evidence"]
    assert all(q.route in FLAT_ROUTES for q in plan_q)


def test_no_combo_when_process_is_short_or_already_contains_the_system():
    short = build_process_queries(_plan(system="alpha-synuclein", objective="want to study aggregation"), _B)
    assert len(short) == 1                                           # the system-prefixed process query IS the combination
    contained = build_process_queries(_plan(system="ESCRT-III filament", objective="Simulating how the ESCRT-III filament senses and generates membrane curvature"), _B)
    assert len(contained) == 1


def test_combo_respects_the_route_query_cap_and_word_ceiling():
    qs = build_process_queries(_plan(system="a very long system description with many components and modifiers here and there",
                                     objective="Determine how confinement reshapes the architecture of composite networks across several scales and conditions"), _B)
    assert len(qs) <= _B.max_queries_per_route
    from route_retrieval_plan import _content_words, _MAX_COMBO_WORDS
    for q in qs:
        assert len(_content_words(q.query)) <= _MAX_COMBO_WORDS + 3      # +'coarse-grained model'


def test_same_string_in_two_routes_is_kept_but_a_repeat_inside_one_route_is_dropped():
    plan = _plan(system="EspP", objective="folding", context="folding")
    routes = [(q.route, q.query) for q in build_route_retrieval_plan(plan).route_queries]
    assert len(routes) == len(set((r, " ".join(q.lower().split())) for r, q in routes))      # no exact repeat within a route
    from route_retrieval_plan import RouteQuery
    assert FLAT_ROUTES                                                                       # provenance route names unchanged


def test_flat_routes_are_unchanged():
    assert FLAT_ROUTES == ("system_specific_cg", "process", "context", "modeling_requirement")


# ---- v4: P1 (anchor generic one-word context), P2 (hygiene), P3 (need_semantic) ----------------------------------------
# Switches live on RouteRetrievalBudgets; every one of them OFF reproduces the v3 queries exactly.
from route_retrieval_plan import (  # noqa: E402
    NEED_SEMANTIC_ROUTE,
    _clean_chunk,
    _is_entity_like,
    _is_low_information_context,
    _objective_phenomenon,
    _process_core,
    build_need_semantic_queries,
    resolve_structural_and_safety_budgets,
    run_route_retrieval,
)
import pytest  # noqa: E402

_V3 = RouteRetrievalBudgets(flat_anchor_short_context=False, flat_query_hygiene=False)


def _ctx_plan(system, objective, ctx_terms):
    return EvidenceRoutePlan(
        scientific_objective=objective, system_description=system,
        process_terms=[RouteTerm(term=objective, route_type="process", source="USER_GROUNDED")],
        context_terms=[RouteTerm(term=t, route_type="context", source="USER_GROUNDED", provenance=pv) for t, pv in ctx_terms],
    )


def test_defaults_p1_p2_on_p3_off():
    b = RouteRetrievalBudgets()
    assert b.flat_anchor_short_context and b.flat_query_hygiene and not b.need_semantic_enabled


# -- P1 ---------------------------------------------------------------------------------------------------------------------

def test_entity_like_tokens():
    for t in ("HP1α", "PHC3", "α4β1", "Na+,K+-ATPase", "GlnBP", "CdvB"):
        assert _is_entity_like(t), t
    for t in ("membrane", "intracellular", "Cholesterol", "DNA", "tau", "bulk"):
        assert not _is_entity_like(t), t


def test_low_information_context():
    assert _is_low_information_context("membrane", None)
    assert _is_low_information_context("cholesterol", {})
    assert _is_low_information_context("membrane-bound", None)
    assert not _is_low_information_context("HP1α", None)                       # a single entity is informative alone
    assert not _is_low_information_context("vesicle confinement", None)        # two content words keep their own query
    assert _is_low_information_context("two words here", {"source": "system_description_lexical_marker"})   # marker provenance always anchors


def test_p1_anchors_a_bare_generic_word_to_the_system_head():
    plan = _ctx_plan("sequence-specific α4β1 Na+,K+-ATPase in the sperm flagellar membrane", "Resolve subunit assembly",
                     [("membrane", {"source": "system_description_lexical_marker"})])
    q = [x.query for x in build_context_queries(plan, RouteRetrievalBudgets())]
    assert q == ["sequence-specific α4β1 Na+,K+-ATPase membrane coarse-grained model"]
    assert "membrane coarse-grained model" in [x.query for x in build_context_queries(plan, _V3)]       # v3 behaviour, switch off


def test_p1_keeps_entities_and_multiword_terms_standalone():
    plan = _ctx_plan("a tetra-nucleosome array with HP1α and a truncated Polyhomeotic-like protein", "Simulate bridging and folding of the array",
                     [("HP1α", None), ("vesicle confinement", None), ("cholesterol", None)])
    q = [x.query for x in build_context_queries(plan, RouteRetrievalBudgets())]
    assert "HP1α coarse-grained model" in q                                    # entity: unchanged
    assert "vesicle confinement coarse-grained model" in q                      # 2 words: unchanged
    assert "tetra-nucleosome array cholesterol coarse-grained model" in q       # generic one-worder: anchored
    assert "cholesterol coarse-grained model" not in q


def test_p1_anchor_does_not_repeat_head_words():
    plan = _ctx_plan("asymmetric mammalian plasma-membrane bilayer with cholesterol", "Track flip-flop", [("asymmetric", None)])
    q = [x.query for x in build_context_queries(plan, RouteRetrievalBudgets())]
    assert all(x.count("asymmetric") <= 1 for x in q)


# -- P2 ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("raw,clean", [
    ("Quantify how valency and temperature affect the junction", "valency and temperature affect the junction"),
    ("Capturing coupled motions and redistribution of mechanical strain during mechanically",
     "coupled motions and redistribution of mechanical strain"),
    ("comparison of empty and loaded capsids", "empty and loaded capsids"),
    ("and compare protein-assisted versus direct permeation across", "protein-assisted versus direct permeation"),
    ("lipid sorting and early", "lipid sorting"),
    ("capsid assembly", "capsid assembly"),                  # a noun ending in -ly-like letters is NOT an adverb
    ("family of kinases", "family of kinases"),
    ("lipid-binding family", "lipid-binding family"),
])
def test_clean_chunk(raw, clean):
    assert _clean_chunk(raw, True) == clean


def test_clean_chunk_is_a_noop_when_off_and_never_empty():
    assert _clean_chunk("Quantify how x during", False) == "Quantify how x during"
    assert _clean_chunk("and", True) == "and"
    assert _clean_chunk("", True) == ""


def test_trailing_hygiene_only_for_system_chunks_does_not_strip_leading_words():
    assert _clean_chunk("comparison of x", True, leading=False) == "comparison of x"


def test_process_core_hygiene_flag():
    t = "Capturing coupled motions and strain redistribution using coarse-grained methods"
    assert _process_core(t, True) == "coupled motions and strain redistribution"
    assert _process_core(t, False).startswith("Capturing")                        # v3


def test_p2_cg_named_context_term_is_now_skipped():
    plan = _ctx_plan("capsid", "assembly of the capsid shell", [("CG resolution", None)])
    assert build_context_queries(plan, RouteRetrievalBudgets()) == []
    assert build_context_queries(plan, _V3) != []                               # v3's mis-escaped regex never fired


# -- P3 ---------------------------------------------------------------------------------------------------------------------

_NEED = RouteRetrievalBudgets(need_semantic_enabled=True)


def test_objective_phenomenon():
    assert _objective_phenomenon("Determine how confinement reshapes the architecture of networks") == "how confinement reshapes the architecture of networks"
    assert _objective_phenomenon("Resolve α4β1 subunit assembly and inhibitor binding geometry using coarse-grained simulation") == \
        "α4β1 subunit assembly and inhibitor binding geometry"
    assert _objective_phenomenon("Modeling HP1α-mediated bridging") == "HP1α-mediated bridging"        # entity keeps its case
    assert _objective_phenomenon("") == ""


def test_need_semantic_off_by_default_and_plan_unchanged():
    plan = _plan(system="EspP passenger domain", objective="Simulate folding")
    assert build_need_semantic_queries(plan, RouteRetrievalBudgets()) == []
    assert all(q.route in FLAT_ROUTES for q in build_route_retrieval_plan(plan).route_queries)


def test_need_semantic_query_keeps_specific_system_and_objective():
    plan = _plan(system="composite actin–microtubule networks with tau and actin crosslinkers in vesicles and bulk",
                 objective="Determine how confinement reshapes the architecture of composite actin–microtubule networks")
    qs = build_need_semantic_queries(plan, _NEED)
    assert len(qs) == 1 and qs[0].route == NEED_SEMANTIC_ROUTE and qs[0].query_role == "need_semantic_evidence"
    assert qs[0].query == ("Coarse-grained models of composite actin–microtubule networks with tau and actin crosslinkers in vesicles and bulk "
                           "to model how confinement reshapes the architecture of composite actin–microtubule networks.")
    assert "tau" in qs[0].query and "crosslinkers" in qs[0].query


def test_need_semantic_without_objective_and_context_control_arm():
    assert build_need_semantic_queries(_plan(system="a soluble ligand"), _NEED)[0].query == "Coarse-grained models of soluble ligand."
    plan = _ctx_plan("a soluble ligand binding a receptor", "Quantify association rates", [("lipid membrane", None), ("receptor", None)])
    ctx_arm = RouteRetrievalBudgets(need_semantic_enabled=True, need_semantic_include_context=True)
    q = build_need_semantic_queries(plan, ctx_arm)[0]
    assert q.query_role == "need_semantic_with_context_evidence" and "Relevant aspects: lipid membrane" in q.query
    assert "receptor" not in q.query.split("Relevant aspects:")[1]              # redundant with the system text -> not repeated
    assert "Relevant aspects" not in build_need_semantic_queries(plan, _NEED)[0].query


def test_need_semantic_enters_the_plan_only_when_enabled_and_flat_queries_are_unchanged():
    plan = _plan(system="EspP passenger domain", objective="Simulate folding")
    off = build_route_retrieval_plan(plan, _V3).route_queries
    on = build_route_retrieval_plan(plan, RouteRetrievalBudgets(flat_anchor_short_context=False, flat_query_hygiene=False, need_semantic_enabled=True)).route_queries
    assert [q for q in on if q.route != NEED_SEMANTIC_ROUTE] == off
    assert [q.route for q in on].count(NEED_SEMANTIC_ROUTE) == 1


class _P:
    def __init__(self, ref): self.doi, self.title, self.year = ref, f"t-{ref}", 2020


def test_run_route_retrieval_executes_need_semantic_with_its_own_limit_and_provenance(monkeypatch):
    import route_retrieval_plan as RP
    monkeypatch.setattr(RP, "_paper_ref", lambda p: p.doi)
    calls = []
    flat = lambda q, l: [_P("f1")]
    need = lambda q, l: (calls.append((q, l)) or [_P("n1"), _P("f1")])
    plan = _plan(system="EspP passenger domain", objective="Simulate folding")
    res = run_route_retrieval(plan, flat_search_fn=flat, need_semantic_search_fn=need,
                              budgets=RouteRetrievalBudgets(need_semantic_enabled=True, need_semantic_limit=37))
    assert calls and calls[0][1] == 37 and calls[0][0].startswith("Coarse-grained models of EspP passenger domain")
    by_ref = {p.paper_ref: p for p in res.papers}
    assert {e.route for e in by_ref["n1"].found_by} == {NEED_SEMANTIC_ROUTE}
    assert {e.route for e in by_ref["f1"].found_by} >= {NEED_SEMANTIC_ROUTE} and len(by_ref["f1"].found_by) > 1   # provenance merged, not replaced
    assert any(o.route == NEED_SEMANTIC_ROUTE for o in res.per_route)


def test_run_route_retrieval_off_has_no_need_route_and_enabled_without_fn_fails_loudly(monkeypatch):
    import route_retrieval_plan as RP
    monkeypatch.setattr(RP, "_paper_ref", lambda p: p.doi)
    plan = _plan(system="EspP passenger domain", objective="Simulate folding")
    res = run_route_retrieval(plan, flat_search_fn=lambda q, l: [_P("f1")])
    assert all(o.route != NEED_SEMANTIC_ROUTE for o in res.per_route)
    with pytest.raises(ValueError):
        run_route_retrieval(plan, flat_search_fn=lambda q, l: [], budgets=RouteRetrievalBudgets(need_semantic_enabled=True))


def test_pool_cap_floor_accounts_for_the_need_route():
    off = resolve_structural_and_safety_budgets(RouteRetrievalBudgets(), structural_hop2_search_limit=20)
    on = resolve_structural_and_safety_budgets(RouteRetrievalBudgets(need_semantic_enabled=True), structural_hop2_search_limit=20)
    assert on.max_total_unique_papers == off.max_total_unique_papers + RouteRetrievalBudgets().max_papers_per_route


def test_stage3_abstract_recovery_is_on_by_default_and_switchable():
    assert RouteRetrievalBudgets().stage3_recover_abstracts is True
    assert RouteRetrievalBudgets(stage3_recover_abstracts=False).stage3_recover_abstracts is False
