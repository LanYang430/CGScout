"""The search functions the direct route hands to run_route_retrieval, and how the benchmark recorder classifies the requests they make (zero API, zero network).

Written for the removal of the two unused structural Hop-1/Hop-2 closures (Phase 5 / C1): every test here passes on the code BEFORE that removal and must keep
passing after it.  They never call `_make_capturing_search_fns` directly -- the search functions are taken from what `run_retrieval_from_plan` actually hands to
`run_route_retrieval` -- so they do not depend on how many closures that private factory returns."""
import pytest

import bench_support  # noqa: F401  (puts benchmark/evaluation on sys.path)
import recorder as RC

import cgscout.retrieval.direct.from_planner as FP
from cgscout.models import Paper
from cgscout.planning.scientific_planner import PlannerOutput
from cgscout.retrieval.direct.route_plan import RouteRetrievalBudgets
from cgscout.retrieval.transfer import _make_search_fn

SENTINEL = object()

# The function under test, bound at import (= collection) time ON PURPOSE: after the migration self-tests, tests/consolidation_world.py's `Patches.set` leaves a counting
# wrapper in the module attribute `from_planner.run_retrieval_from_plan` (an isolation gap of that safety net, which this change does not touch), so reading the
# attribute at test time would not give the real function in a full run.  Everything the real function looks up (run_route_retrieval, search_and_dedupe, ...) IS
# read from the module namespace at call time, so the tests below still patch those through `FP`.
RUN_RETRIEVAL_FROM_PLAN = FP.run_retrieval_from_plan


def _plan():
    return PlannerOutput(system="EspP passenger domain", target_component=None, scientific_objective="folding pathway of the beta-helix",
                         clarification_needed=False, search_context=["membrane"], search_axes=[])


def _paper(doi, title):
    return Paper(title=title, year=2001, doi=doi, abstract="x" * 80)


class _Fake:
    """Stands in for search_and_dedupe: records every call with exactly the keyword arguments it was given."""
    def __init__(self, papers_by_call=None):
        self.calls, self._papers = [], papers_by_call or (lambda n: [_paper(f"10.1/{n}", f"first {n}")])

    def __call__(self, query, limit=20, **kw):
        self.calls.append({"query": query, "limit": limit, **kw})
        return list(self._papers(len(self.calls)))


@pytest.fixture
def handed(monkeypatch):
    """run_retrieval_from_plan with run_route_retrieval replaced by a spy: -> (kwargs the spy received, returned triple)."""
    seen = {}

    def spy(evidence_plan, **kw):
        seen["kwargs"] = kw
        return SENTINEL
    monkeypatch.setattr(FP, "run_route_retrieval", spy)

    def go(**budget_spec):
        out = RUN_RETRIEVAL_FROM_PLAN(_plan(), RouteRetrievalBudgets(**budget_spec))
        return seen["kwargs"], out
    return go


def test_the_direct_route_hands_run_route_retrieval_only_the_flat_and_the_semantic_function(handed):
    kw, (ev, res, cache) = handed()
    assert set(kw) == {"flat_search_fn", "budgets", "need_semantic_search_fn"}      # no Hop-1 / Hop-2 function, no `call`
    assert callable(kw["flat_search_fn"]) and kw["need_semantic_search_fn"] is None   # default budgets: the semantic route is off
    assert res is SENTINEL and cache == {} and ev.structural_terms == []
    kw, _ = handed(need_semantic_enabled=True)
    assert set(kw) == {"flat_search_fn", "budgets", "need_semantic_search_fn"} and callable(kw["need_semantic_search_fn"])


def test_flat_search_function_searches_with_the_limit_only_and_keeps_the_first_paper_seen_per_ref(handed, monkeypatch):
    fake = _Fake(lambda n: [_paper("10.1/a", f"title from call {n}"), _paper(f"10.1/new{n}", "other")])
    monkeypatch.setattr(FP, "search_and_dedupe", fake)
    kw, (_, _, cache) = handed()
    flat = kw["flat_search_fn"]
    first = flat("q one", 7)
    second = flat("q two", 9)
    assert fake.calls == [{"query": "q one", "limit": 7}, {"query": "q two", "limit": 9}]   # no `sources`, no `mode`: the lexical OpenAlex default
    assert [p.title for p in first] == ["title from call 1", "other"] and [p.title for p in second] == ["title from call 2", "other"]
    assert list(cache) == ["10.1/a", "10.1/new1", "10.1/new2"]                              # keyed by paper_ref, insertion order
    assert cache["10.1/a"].title == "title from call 1"                                      # setdefault: the first object stays
    assert cache["10.1/a"] is first[0]


def test_semantic_search_function_uses_semantic_mode_and_caches_a_tagged_copy_without_replacing_a_cached_paper(handed, monkeypatch):
    fake = _Fake(lambda n: [_paper("10.1/a", f"title from call {n}")] + ([_paper("10.1/s", "semantic only")] if n == 2 else []))   # call 1 (flat): a;  call 2 (semantic): a, s
    monkeypatch.setattr(FP, "search_and_dedupe", fake)
    kw, (_, _, cache) = handed(need_semantic_enabled=True)
    flat, sem = kw["flat_search_fn"], kw["need_semantic_search_fn"]
    flat("lexical q", 5)
    out = sem("semantic q", 11)
    assert fake.calls[1] == {"query": "semantic q", "limit": 11, "mode": "semantic"}        # no `sources`
    assert [p.retrieval_modes for p in out] != [["semantic"], ["semantic"]]                  # the returned objects are the untouched ones
    assert cache["10.1/a"].title == "title from call 1" and cache["10.1/a"].retrieval_modes != ["semantic"]   # already cached by the flat route: kept
    assert cache["10.1/s"].retrieval_modes == ["semantic"]                                   # new from the semantic route: a copy tagged "semantic"
    assert cache["10.1/s"] is not out[1]


def test_recorder_classifies_the_direct_route_requests_as_flat_and_need_semantic_and_the_3b_closure_by_mode(handed):
    kw, _ = handed(need_semantic_enabled=True)
    assert RC.classify(kw["flat_search_fn"].__qualname__, "lexical") == "flat"
    assert RC.classify(kw["need_semantic_search_fn"].__qualname__, "semantic") == "need_semantic"
    closure = _make_search_fn({}, 15)
    assert RC.classify(closure.__qualname__, "semantic") == "hop1" and RC.classify(closure.__qualname__, "lexical") == "hop2"
    assert RC.classify("_run_direct_concept_search", "lexical") == "3A"
    assert RC.classify("_default_flat_search", "lexical") == "flat"
    assert RC.classify("something_else", "lexical") == "other:something_else"


def test_a_real_direct_run_under_the_recorder_makes_only_flat_and_need_semantic_requests(monkeypatch):
    monkeypatch.setattr(FP, "search_and_dedupe", _Fake(lambda n: [_paper(f"10.1/{n}", f"p{n}")]))
    rec = RC.Recorder(lambda: {"http_requests": 0, "cache_hits": 0}, lambda p: False, {})
    rec.install()
    try:
        RUN_RETRIEVAL_FROM_PLAN(_plan(), RouteRetrievalBudgets(need_semantic_enabled=True))
    finally:
        rec.uninstall()
    kinds = {r["kind"] for r in rec.records}
    assert kinds == {"flat", "need_semantic"}, kinds
    assert all(r["mode"] == ("semantic" if r["kind"] == "need_semantic" else "lexical") for r in rec.records)
