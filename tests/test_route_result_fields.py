"""The fields of RouteOutcome and RouteRetrievalResult after the structural scaffolding is removed (zero API, zero network).

Phase 5 / C2c-1: run_route_retrieval no longer appends a placeholder outcome for the structural route, so `per_route` lists exactly the routes that were run and
`routes_with_zero_results` names only those; `RouteOutcome.structural_cleanup_log` and `RouteRetrievalResult.structural_branch_result` (always empty / None, read by
nothing) are gone.  The behaviour of the remaining outcomes is pinned by tests/test_route_outcomes.py, which passed unchanged before and after."""
import pytest

import cgscout.retrieval.direct.route_plan as RP
from cgscout.retrieval.direct.evidence_plan import EvidenceRoutePlan, RouteTerm
from cgscout.retrieval.direct.route_plan import FLAT_ROUTES, NEED_SEMANTIC_ROUTE, RouteRetrievalBudgets

RUN = RP.run_route_retrieval


def evidence_plan():
    return EvidenceRoutePlan(scientific_objective="folding pathway of the beta-helix", system_description="EspP passenger domain",
                             process_terms=[RouteTerm(term="folding pathway", route_type="process", source="USER_GROUNDED")])


def test_route_outcome_has_exactly_the_remaining_fields():
    assert list(RP.RouteOutcome.model_fields) == ["route", "route_term", "queries_tried", "n_papers_found", "n_new_unique_contributed", "n_dropped_route_cap",
                                                  "n_dropped_global_cap", "empty", "empty_reason", "search_errors"]


def test_route_retrieval_result_has_exactly_the_remaining_fields():
    assert list(RP.RouteRetrievalResult.model_fields) == ["plan", "per_route", "papers", "n_unique_papers", "n_papers_multi_route", "routes_with_zero_results",
                                                          "retrieval_seconds"]


def test_a_removed_field_is_in_no_dump_no_schema_and_not_readable():
    res = RUN(evidence_plan(), flat_search_fn=lambda q, l: [])
    dump = res.model_dump()
    assert "structural_branch_result" not in dump and all("structural_cleanup_log" not in o for o in dump["per_route"])
    schema = RP.RouteRetrievalResult.model_json_schema()
    assert "structural_branch_result" not in schema["properties"]
    assert "structural_cleanup_log" not in schema["$defs"]["RouteOutcome"]["properties"]
    assert "Hop2Result" not in schema["$defs"]                        # the nested definition that only the removed field pulled in
    assert "structural_cleanup_log" not in RP.RouteOutcome.model_json_schema()["properties"]
    with pytest.raises(AttributeError):
        res.structural_branch_result
    with pytest.raises(AttributeError):
        res.per_route[0].structural_cleanup_log


def test_removed_fields_passed_to_the_constructors_are_ignored_not_stored():
    outcome = RP.RouteOutcome(route="x", structural_cleanup_log=["a"])
    assert not hasattr(outcome, "structural_cleanup_log") and "structural_cleanup_log" not in outcome.model_dump()
    res = RP.RouteRetrievalResult(plan=RP.build_route_retrieval_plan(evidence_plan()), structural_branch_result={"anything": 1})
    assert not hasattr(res, "structural_branch_result") and "structural_branch_result" not in res.model_dump()


@pytest.mark.parametrize("need", [False, True])
def test_per_route_lists_exactly_the_routes_that_ran_and_no_structural_entry(need):
    budgets = RouteRetrievalBudgets(need_semantic_enabled=need)
    res = RUN(evidence_plan(), flat_search_fn=lambda q, l: [], need_semantic_search_fn=lambda q, l: [], budgets=budgets)
    expected = [*FLAT_ROUTES, NEED_SEMANTIC_ROUTE] if need else list(FLAT_ROUTES)
    assert [o.route for o in res.per_route] == expected
    assert "structural" not in res.routes_with_zero_results and res.routes_with_zero_results == [r for r in expected]      # nothing was found anywhere
