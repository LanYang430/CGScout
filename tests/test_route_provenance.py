"""How the direct route records which routes and queries found each paper, and how that reaches Stage 3 and Stage 4 (zero API, zero network).

Written in Phase 5 / C2d-0, before C2d-1 removed `RetrievedPaper.hop1_provenance` / `hop1_provenances` (the remnants of the structural branch's provenance, always None / []
on the direct route because the only call that could fill them passed None) and the accumulator parameter that fed them.  What stays is the route provenance:
`RetrievedPaper.found_by` (one RouteProvenanceEntry per query result a paper appeared in), `provenance_index`, `papers_for_stage3`, and `retrieve_direct` writing
`found_by_queries` / `query_categories` onto each Paper, which Stage 3 (per-query scoring, route-aware gate) and Stage 4 (routes, supporting-query count) read.  Until
then nothing pinned that chain in the test suite (only the consolidation mock world, which fakes it).  Every test here reads only what stays; they passed unchanged
before and after C2d-1, which only touched this docstring."""
import pytest

import cgscout.retrieval.direct.from_planner as FP
import cgscout.retrieval.direct.route_plan as RP
from cgscout.models import Paper
from cgscout.planning.scientific_planner import PlannerOutput
from cgscout.retrieval.direct.evidence_plan import EvidenceRoutePlan, RouteTerm
from cgscout.retrieval.direct.route_plan import FLAT_ROUTES, RouteProvenanceEntry, RouteRetrievalBudgets

# Bound at collection time on purpose: after the migration self-tests tests/consolidation_world.py leaves wrappers in some from_planner attributes (an isolation gap
# of that safety net), so the functions under test are taken now and put back with monkeypatch where a test goes through from_planner's module namespace.
RUN = RP.run_route_retrieval
RETRIEVE_DIRECT, RUN_RETRIEVAL_FROM_PLAN = FP.retrieve_direct, FP.run_retrieval_from_plan
PAPERS_FOR_STAGE3, PROVENANCE_INDEX = FP.papers_for_stage3, FP.provenance_index


def evidence_plan():
    return EvidenceRoutePlan(
        scientific_objective="folding pathway of the beta-helix", system_description="EspP passenger domain",
        process_terms=[RouteTerm(term="folding pathway", route_type="process", source="USER_GROUNDED")],
        context_terms=[RouteTerm(term=t, route_type="context", source="USER_GROUNDED") for t in ("membrane", "thermal stability")],
    )


def planner_output():
    return PlannerOutput(system="EspP passenger domain", target_component=None, scientific_objective="folding pathway of the beta-helix",
                         clarification_needed=False, search_context=["membrane", "thermal stability"], search_axes=[])


def paper(key, year=2001):
    return Paper(title=f"paper {key}", year=year, doi=f"10.1/{key}", abstract="x" * 80)


def planned_flat_queries(plan):
    return [q for q in plan.route_queries if q.route in FLAT_ROUTES]


def test_every_found_by_entry_carries_the_route_the_term_the_query_the_role_and_the_rank():
    n = {"calls": 0}

    def search(query, limit):
        n["calls"] += 1
        return [paper(n["calls"] * 10 + k, year=2000 + k) for k in range(3)]
    res = RUN(evidence_plan(), flat_search_fn=search)
    queries = planned_flat_queries(res.plan)
    assert len(res.papers) == 3 * len(queries)
    for i, q in enumerate(queries, start=1):
        for k in range(3):
            rp = next(p for p in res.papers if p.paper_ref.endswith(f"/{i * 10 + k}"))
            assert (rp.title, rp.year) == (f"paper {i * 10 + k}", 2000 + k)
            assert rp.found_by == [RouteProvenanceEntry(route=q.route, route_term=q.route_term, query=q.query, query_role=q.query_role, retrieval_rank=k)]


def test_a_paper_found_again_keeps_every_entry_in_discovery_order_each_with_its_own_rank():
    n = {"calls": 0}

    def search(query, limit):
        n["calls"] += 1
        c = n["calls"]
        if c == 1:
            return [paper("X"), paper("A")]           # X at rank 0 of the first query
        if c == 2:
            return [paper("B"), paper("X"), paper("X")]   # X at rank 1 AND again at rank 2 of the second query
        return [paper(f"n{c}")]
    res = RUN(evidence_plan(), flat_search_fn=search)
    queries = planned_flat_queries(res.plan)
    x = next(p for p in res.papers if p.paper_ref.endswith("/X"))
    assert [(e.route, e.query, e.retrieval_rank) for e in x.found_by] == [(queries[0].route, queries[0].query, 0), (queries[1].route, queries[1].query, 1),
                                                                         (queries[1].route, queries[1].query, 2)]
    assert [p.paper_ref.rsplit("/", 1)[1] for p in res.papers][:3] == ["X", "A", "B"]       # first-seen order; X is not moved by being found again


def test_a_paper_that_is_already_known_is_merged_even_when_the_caps_are_reached():
    n = {"calls": 0}

    def search(query, limit):
        n["calls"] += 1
        return [paper("X"), paper(f"new{n['calls']}")]            # X every time; one new paper per call, which the caps refuse
    res = RUN(evidence_plan(), flat_search_fn=search, budgets=RouteRetrievalBudgets(max_papers_per_route=1, max_total_unique_papers=1))
    queries = planned_flat_queries(res.plan)
    assert [p.paper_ref.rsplit("/", 1)[1] for p in res.papers] == ["X"]
    assert [(e.route, e.query) for e in res.papers[0].found_by] == [(q.route, q.query) for q in queries]       # every query's sighting of X was recorded
    assert res.n_unique_papers == 1 and res.n_papers_multi_route == 1


def test_provenance_index_maps_each_paper_ref_to_its_own_retrieved_paper():
    res = RUN(evidence_plan(), flat_search_fn=lambda q, l: [paper("X"), paper(q)])
    idx = PROVENANCE_INDEX(res)
    assert list(idx) == [rp.paper_ref for rp in res.papers] and all(idx[rp.paper_ref] is rp for rp in res.papers)


def test_papers_for_stage3_returns_the_cached_papers_in_result_order_and_warns_about_missing_ones(capsys):
    cache = {}

    def search(query, limit):
        out = [paper("X"), paper(query)]
        for p in out:
            cache.setdefault(FP._paper_ref(p), p)               # what the capturing search function of from_planner does
        return out
    res = RUN(evidence_plan(), flat_search_fn=search)
    assert [id(p) for p in PAPERS_FOR_STAGE3(res, cache)] == [id(cache[rp.paper_ref]) for rp in res.papers]       # the cached objects, in result order
    gone = res.papers[1].paper_ref
    del cache[gone]
    out = PAPERS_FOR_STAGE3(res, cache)
    assert [id(p) for p in out] == [id(cache[rp.paper_ref]) for rp in res.papers if rp.paper_ref != gone]
    assert capsys.readouterr().err == ("route_retrieval_live_adapter: WARNING -- 1 paper_ref(s) in RouteRetrievalResult had no cached Paper object: "
                                       f"[{gone!r}]\n")


class _FakeSearch:
    """Stands in for search_and_dedupe: the same paper for every query plus one paper per call."""
    def __init__(self):
        self.calls = []

    def __call__(self, query, limit=20, **kw):
        self.calls.append(query)
        return [paper("shared"), paper(f"own{len(self.calls)}")]


def test_retrieve_direct_writes_found_by_queries_and_query_categories_from_the_route_provenance(monkeypatch):
    monkeypatch.setattr(FP, "run_retrieval_from_plan", RUN_RETRIEVAL_FROM_PLAN)
    monkeypatch.setattr(FP, "papers_for_stage3", PAPERS_FOR_STAGE3)
    monkeypatch.setattr(FP, "provenance_index", PROVENANCE_INDEX)
    fake = _FakeSearch()
    monkeypatch.setattr(FP, "search_and_dedupe", fake)
    _, direct_papers = RETRIEVE_DIRECT(planner_output(), RouteRetrievalBudgets())
    queries = planned_flat_queries(RP.build_route_retrieval_plan(FP.evidence_plan_from_planner_output(planner_output())))
    assert fake.calls == [q.query for q in queries]
    assert [p.doi for p in direct_papers][0] == "10.1/shared" and len(direct_papers) == 1 + len(queries)
    shared = direct_papers[0]
    assert shared.found_by_queries == [q.query for q in queries]                           # every query that returned it, once, in order
    assert shared.query_categories == ["system_specific_cg", "process", "context"]          # every route that found it, once, in route order
    for i, (p, q) in enumerate(zip(direct_papers[1:], queries)):
        assert p.found_by_queries == [q.query] and p.query_categories == [q.route]


def test_the_direct_route_leaves_the_structural_discovery_fields_for_the_transfer_path(monkeypatch):
    monkeypatch.setattr(FP, "run_retrieval_from_plan", RUN_RETRIEVAL_FROM_PLAN)
    monkeypatch.setattr(FP, "papers_for_stage3", PAPERS_FOR_STAGE3)
    monkeypatch.setattr(FP, "provenance_index", PROVENANCE_INDEX)
    monkeypatch.setattr(FP, "search_and_dedupe", _FakeSearch())
    _, direct_papers = RETRIEVE_DIRECT(planner_output(), RouteRetrievalBudgets())
    assert direct_papers and all(p.structural_discovered_system is None and p.structural_discovered_systems == [] for p in direct_papers)
