"""The structural-branch guard of run_route_retrieval, pinned before it is rewritten (zero API, zero network).

Written for Phase 5 / C2b-0 (a tests-only commit, no change to src/).  `run_route_retrieval` raises NotImplementedError when the evidence plan carries structural terms: the
Hop-1/Hop-2 branch behind them was removed and the live path (from_planner) always clears them.  C2b replaces the way the condition is computed (today:
`plan.structural_branch_enabled`, a field of RouteRetrievalPlan that `select_primary_structural_term` fills; after: `evidence_plan.structural_terms` itself) and deletes
those plan fields, so every test here talks only about what must NOT change: which inputs raise, with which exception and message, and WHEN (after the flat routes and the
need_semantic route have run).  None of them reads a field that C2b removes, so all pass before and after."""
import pytest

import cgscout.retrieval.direct.from_planner as FP
import cgscout.retrieval.direct.route_plan as RP
from cgscout.planning.scientific_planner import PlannerOutput
from cgscout.models import Paper
from cgscout.retrieval.direct.evidence_plan import EvidenceRoutePlan, RouteTerm, plan_evidence_routes
from cgscout.retrieval.direct.route_plan import FLAT_ROUTES, RouteRetrievalBudgets

RUN, BUILD = RP.run_route_retrieval, RP.build_route_retrieval_plan                 # bound at collection time, like the other consolidation-era tests
EVIDENCE_PLAN_FROM_PLANNER_OUTPUT = FP.evidence_plan_from_planner_output

MESSAGE = ("the structural (Hop-1/Hop-2) branch of run_route_retrieval() was removed (historical code, git tag `legacy-pre-cgscout`); "
           "the live path never enables it -- from_planner always clears structural_terms.")


def term(text, source="USER_GROUNDED"):
    return RouteTerm(term=text, route_type="structural", source=source, supporting_span=text if source != "PLANNER_DERIVED" else None,
                     confidence="exploratory" if source == "PLANNER_DERIVED" else None, justification="x" if source == "PLANNER_DERIVED" else None)


# every non-empty list of terms raises, whatever the terms look like (the old condition was "a primary term was found", i.e. the list is not empty)
CASES = {
    "one USER term": [term("beta-helix")],
    "one LITERATURE term": [term("solenoid", "LITERATURE_GROUNDED")],
    "LITERATURE first, USER second": [term("solenoid", "LITERATURE_GROUNDED"), term("beta-helix")],
    "USER first, LITERATURE second": [term("beta-helix"), term("solenoid", "LITERATURE_GROUNDED")],
    "empty-string term": [term("")],
    "whitespace term": [term("   ")],
    "two empty-string terms": [term(""), term("")],
    "PLANNER_DERIVED-only term": [term("beta-helix", "PLANNER_DERIVED")],
}


def evidence_plan(structural_terms=()):
    return EvidenceRoutePlan(
        scientific_objective="folding pathway of the beta-helix", system_description="EspP passenger domain",
        process_terms=[RouteTerm(term="folding pathway", route_type="process", source="USER_GROUNDED")],
        context_terms=[RouteTerm(term=t, route_type="context", source="USER_GROUNDED") for t in ("membrane", "thermal stability")],
        structural_terms=list(structural_terms),
    )


class Search:
    def __init__(self):
        self.calls = []

    def __call__(self, query, limit):
        self.calls.append((query, limit))
        return [Paper(title=f"p{len(self.calls)}", year=2001, doi=f"10.1/{len(self.calls)}", abstract="x" * 80)]


def flat_queries(plan):
    return [q.query for q in plan.route_queries if q.route in FLAT_ROUTES]


@pytest.mark.parametrize("name", CASES)
def test_any_structural_term_makes_run_route_retrieval_raise_with_the_exact_message(name):
    with pytest.raises(NotImplementedError) as exc:
        RUN(evidence_plan(CASES[name]), flat_search_fn=Search())
    assert str(exc.value) == MESSAGE


def test_without_structural_terms_nothing_is_raised_and_per_route_has_no_structural_entry():
    res = RUN(evidence_plan([]), flat_search_fn=Search())
    assert [o.route for o in res.per_route] == list(FLAT_ROUTES)


@pytest.mark.parametrize("name", ["one USER term", "empty-string term", "LITERATURE first, USER second"])
def test_the_guard_fires_only_after_every_flat_route_has_searched(name):
    ep, search = evidence_plan(CASES[name]), Search()
    with pytest.raises(NotImplementedError):
        RUN(ep, flat_search_fn=search)
    assert [q for q, _ in search.calls] == flat_queries(BUILD(evidence_plan([])))      # all the flat queries ran, in plan order, and nothing else


def test_the_guard_fires_after_the_need_semantic_route_too():
    flat, need = Search(), Search()
    with pytest.raises(NotImplementedError):
        RUN(evidence_plan(CASES["one USER term"]), flat_search_fn=flat, need_semantic_search_fn=need, budgets=RouteRetrievalBudgets(need_semantic_enabled=True))
    assert flat.calls and need.calls


def test_the_guard_does_not_depend_on_the_need_semantic_function_being_missing():
    # need_semantic enabled without its function is a ValueError that comes BEFORE the guard (that route is built and run first)
    with pytest.raises(ValueError, match="need_semantic_search_fn"):
        RUN(evidence_plan(CASES["one USER term"]), flat_search_fn=Search(), budgets=RouteRetrievalBudgets(need_semantic_enabled=True))


@pytest.mark.parametrize("name", CASES)
def test_building_the_plan_never_raises_and_structural_terms_never_enter_a_query(name):
    plain = BUILD(evidence_plan([]))
    with_terms = BUILD(evidence_plan(CASES[name]))            # the guard is in run_route_retrieval, not here
    assert [(q.route, q.query) for q in with_terms.route_queries] == [(q.route, q.query) for q in plain.route_queries]
    assert with_terms.empty_routes == plain.empty_routes


def test_an_explicit_grounded_structural_feature_reaches_the_guard():
    ep = plan_evidence_routes("EspP passenger domain", "folding pathway", grounded_structural_feature="beta-helix")
    assert ep.structural_terms
    with pytest.raises(NotImplementedError):
        RUN(ep, flat_search_fn=Search())


def test_literature_structural_mentions_reach_the_guard():
    ep = plan_evidence_routes("EspP passenger domain", "folding pathway", literature_structural_mentions=["solenoid"])
    assert ep.structural_terms
    with pytest.raises(NotImplementedError):
        RUN(ep, flat_search_fn=Search())


def test_the_planner_bridge_clears_structural_terms_so_the_live_path_never_reaches_the_guard(monkeypatch):
    real = FP.plan_evidence_routes

    def with_structural_terms(*a, **kw):
        ep = real(*a, **kw)
        ep.structural_terms = [term("beta-helix")]             # as if plan_evidence_routes had derived one
        return ep
    monkeypatch.setattr(FP, "plan_evidence_routes", with_structural_terms)
    ep = EVIDENCE_PLAN_FROM_PLANNER_OUTPUT(PlannerOutput(system="EspP passenger domain", target_component=None, scientific_objective="folding pathway",
                                                         clarification_needed=False, search_context=["membrane"], search_axes=[]))
    assert ep.structural_terms == []
    RUN(ep, flat_search_fn=Search())                             # does not raise
