"""What run_route_retrieval reports per route, pinned before the structural placeholder is removed (zero API, zero network).

DRAFT for Phase 5 / C2c-0 (tests only, no change to src/).  C2c deletes the placeholder outcome `RouteOutcome(route="structural", ...)` that run_route_retrieval always
appends to `per_route` (and with it the "structural" entry of `routes_with_zero_results`), the field `RouteOutcome.structural_cleanup_log` and the field
`RouteRetrievalResult.structural_branch_result`.  Nothing else about the outcomes may change, and until now almost nothing pinned it: no test read
`routes_with_zero_results`, the empty reasons, the search errors or the drop counters.  Every test below looks at the flat and need_semantic outcomes only (the
placeholder is filtered out by name, the removed fields are never read), so all of them pass before and after C2c."""
import pytest

import cgscout.retrieval.direct.route_plan as RP
from cgscout.models import Paper
from cgscout.retrieval.direct.evidence_plan import EvidenceRoutePlan, RouteTerm
from cgscout.retrieval.direct.route_plan import FLAT_ROUTES, NEED_SEMANTIC_ROUTE, RouteRetrievalBudgets

RUN, BUILD = RP.run_route_retrieval, RP.build_route_retrieval_plan        # bound at collection time, like the other consolidation-era tests


def evidence_plan():
    # three applicable routes (system_specific_cg: 1 query, process: 1, context: 3) and one that has nothing to search (modeling_requirement)
    return EvidenceRoutePlan(
        scientific_objective="folding pathway of the beta-helix", system_description="EspP passenger domain",
        process_terms=[RouteTerm(term="folding pathway", route_type="process", source="USER_GROUNDED")],
        context_terms=[RouteTerm(term=t, route_type="context", source="USER_GROUNDED") for t in ("membrane", "thermal stability")],
    )


def paper(n):
    return Paper(title=f"paper {n}", year=2001, doi=f"10.1/{n}", abstract="x" * 80)


def counting_search(per_call=1, fail=lambda n: False):
    """A search function returning `per_call` distinct new papers per call; raises RuntimeError("boom") when fail(call_number)."""
    state = {"n": 0, "papers": 0}

    def search(query, limit):
        state["n"] += 1
        if fail(state["n"]):
            raise RuntimeError("boom")
        out = [paper(state["papers"] + i) for i in range(per_call)]
        state["papers"] += per_call
        return out
    return search


def outcomes(res):
    """The per-route outcomes by route name, without the structural placeholder (that entry is what C2c removes)."""
    return {o.route: o for o in res.per_route if o.route != "structural"}


def planned(plan, route):
    return [q.query for q in plan.route_queries if q.route == route]


def test_the_flat_routes_come_first_in_order_then_need_semantic_when_enabled():
    flat = counting_search()
    res = RUN(evidence_plan(), flat_search_fn=flat)
    assert [o.route for o in res.per_route if o.route != "structural"] == list(FLAT_ROUTES)
    res = RUN(evidence_plan(), flat_search_fn=flat, need_semantic_search_fn=counting_search(), budgets=RouteRetrievalBudgets(need_semantic_enabled=True))
    assert [o.route for o in res.per_route if o.route != "structural"] == [*FLAT_ROUTES, NEED_SEMANTIC_ROUTE]


def test_each_applicable_route_reports_its_term_and_its_queries_and_the_other_one_is_the_not_applicable_stub():
    res = RUN(evidence_plan(), flat_search_fn=counting_search())
    out = outcomes(res)
    for route in ("system_specific_cg", "process", "context"):
        queries = [q for q in res.plan.route_queries if q.route == route]
        assert out[route].queries_tried == [q.query for q in queries] and out[route].route_term == queries[0].route_term
        assert not out[route].empty and out[route].empty_reason is None and out[route].search_errors == []
        assert out[route].n_papers_found == len(queries) and out[route].n_new_unique_contributed == len(queries)     # one new paper per call
        assert out[route].n_dropped_route_cap == 0 and out[route].n_dropped_global_cap == 0
    stub = out["modeling_requirement"]
    assert (stub.route_term, stub.queries_tried, stub.n_papers_found, stub.n_new_unique_contributed, stub.search_errors) == (None, [], 0, 0, [])
    assert stub.empty and stub.empty_reason == "not_applicable_no_grounded_term"


def test_routes_whose_queries_return_nothing_are_empty_with_that_reason():
    res = RUN(evidence_plan(), flat_search_fn=lambda q, l: [])
    out = outcomes(res)
    for route in ("system_specific_cg", "process", "context"):
        assert out[route].empty and out[route].empty_reason == "queries_returned_zero_papers" and out[route].queries_tried == planned(res.plan, route)
        assert out[route].n_papers_found == 0 and out[route].search_errors == []
    assert out["modeling_requirement"].empty_reason == "not_applicable_no_grounded_term"
    assert res.papers == [] and res.n_unique_papers == 0 and res.n_papers_multi_route == 0
    assert [r for r in res.routes_with_zero_results if r != "structural"] == list(FLAT_ROUTES)


def test_routes_whose_searches_all_fail_are_empty_with_search_error_and_keep_each_error():
    res = RUN(evidence_plan(), flat_search_fn=counting_search(fail=lambda n: True))
    out = outcomes(res)
    for route in ("system_specific_cg", "process", "context"):
        assert out[route].empty and out[route].empty_reason == "search_error"
        assert out[route].search_errors == [f"{q!r}: boom" for q in planned(res.plan, route)]
        assert out[route].queries_tried == planned(res.plan, route)       # a failed query is still a query that was tried
    assert res.papers == [] and [r for r in res.routes_with_zero_results if r != "structural"] == list(FLAT_ROUTES)


def test_one_failing_query_never_blocks_the_others_and_does_not_make_its_route_empty():
    # calls 1 and 2 are system_specific_cg and process; call 3 is the first query of the context route
    res = RUN(evidence_plan(), flat_search_fn=counting_search(fail=lambda n: n == 3))
    ctx = outcomes(res)["context"]
    queries = planned(res.plan, "context")
    assert len(queries) == 3 and ctx.queries_tried == queries
    assert ctx.search_errors == [f"{queries[0]!r}: boom"]
    assert not ctx.empty and ctx.empty_reason is None and ctx.n_papers_found == 2
    assert "context" not in res.routes_with_zero_results and res.n_unique_papers == 4


def test_the_same_paper_found_by_every_query_is_one_unique_paper_with_all_its_provenance():
    res = RUN(evidence_plan(), flat_search_fn=lambda q, l: [paper(0)])
    queries = [q for q in res.plan.route_queries if q.route in FLAT_ROUTES]
    assert res.n_unique_papers == 1 and res.n_papers_multi_route == 1 and len(res.papers) == 1
    assert [(e.route, e.query, e.retrieval_rank) for e in res.papers[0].found_by] == [(q.route, q.query, 0) for q in queries]
    out = outcomes(res)
    assert out["system_specific_cg"].n_new_unique_contributed == 1 and out["process"].n_new_unique_contributed == 0 and out["context"].n_new_unique_contributed == 0
    assert [out[r].n_papers_found for r in ("system_specific_cg", "process", "context")] == [1, 1, 3]       # re-merged provenance still counts as found


def test_the_per_route_cap_drops_new_papers_and_counts_them():
    res = RUN(evidence_plan(), flat_search_fn=counting_search(per_call=5), budgets=RouteRetrievalBudgets(max_papers_per_route=2))
    out = outcomes(res)
    # one query of 5 new papers: 2 accepted, 3 dropped; the context route has 3 queries: 2 accepted from the first, then everything is dropped by the route cap
    assert (out["system_specific_cg"].n_papers_found, out["system_specific_cg"].n_new_unique_contributed, out["system_specific_cg"].n_dropped_route_cap) == (2, 2, 3)
    assert (out["context"].n_papers_found, out["context"].n_new_unique_contributed, out["context"].n_dropped_route_cap) == (2, 2, 3 + 5 + 5)
    assert all(o.n_dropped_global_cap == 0 for o in out.values()) and res.n_unique_papers == 6


def test_the_global_cap_drops_new_papers_and_counts_them_separately():
    res = RUN(evidence_plan(), flat_search_fn=counting_search(per_call=2), budgets=RouteRetrievalBudgets(max_total_unique_papers=3))
    out = outcomes(res)
    assert (out["system_specific_cg"].n_papers_found, out["system_specific_cg"].n_dropped_global_cap) == (2, 0)
    assert (out["process"].n_papers_found, out["process"].n_new_unique_contributed, out["process"].n_dropped_global_cap) == (1, 1, 1)
    assert (out["context"].n_papers_found, out["context"].n_dropped_global_cap, out["context"].empty) == (0, 6, False)      # papers were returned, all dropped
    assert all(o.n_dropped_route_cap == 0 for o in out.values()) and res.n_unique_papers == 3


def test_a_paper_found_by_exactly_two_routes_counts_as_multi_route():
    n = {"calls": 0}

    def search(query, limit):
        n["calls"] += 1
        return [paper(0 if n["calls"] <= 2 else n["calls"])]       # calls 1 and 2 are system_specific_cg and process: the same paper
    res = RUN(evidence_plan(), flat_search_fn=search)
    assert res.n_papers_multi_route == 1 and res.n_unique_papers == 4
    assert {e.route for e in res.papers[0].found_by} == {"system_specific_cg", "process"}


def test_the_result_counters_and_the_paper_order():
    res = RUN(evidence_plan(), flat_search_fn=counting_search())
    assert [p.title for p in res.papers] == [f"paper {i}" for i in range(5)]            # first-seen order
    assert res.n_unique_papers == len(res.papers) == 5
    assert res.n_papers_multi_route == sum(len(p.found_by) > 1 for p in res.papers) == 0
    assert isinstance(res.retrieval_seconds, float) and res.retrieval_seconds >= 0
    assert res.plan.route_queries == BUILD(evidence_plan()).route_queries
    assert [r for r in res.routes_with_zero_results if r != "structural"] == ["modeling_requirement"]
