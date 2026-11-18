"""The parameters of run_route_retrieval (zero API, zero network).

Written for Phase 5 / C2a, which removes the five keyword parameters `call`, `hop1_search_fn`, `hop1_extract_fn`, `hop2_search_fn` and `hop1_progress_fn`: they were
accepted "for signature compatibility" and never used.

* The `retained_*` tests pin what stays -- `evidence_plan` positional, `flat_search_fn` / `budgets` / `need_semantic_search_fn` keyword-only and doing what they did.
  They pass on the code BEFORE the removal and must pass, unchanged, after it.
* The `removed_*` tests describe the new signature: exactly the four remaining parameters, and a TypeError (at binding time, before any search runs) for each
  removed one.  They are meaningful only after the removal (on the parent commit they fail, which is the point of the change)."""
import inspect

import pytest

import cgscout.retrieval.direct.route_plan as RP
import cgscout.retrieval.search as SEARCH
from cgscout.models import Paper
from cgscout.retrieval.direct.evidence_plan import EvidenceRoutePlan, RouteTerm
from cgscout.retrieval.direct.route_plan import FLAT_ROUTES, RouteRetrievalBudgets

# Bound at import (= collection) time on purpose, like the other consolidation-era tests: a module attribute that an earlier test file wrapped and did not restore
# must not change which function these tests call.
RUN = RP.run_route_retrieval

REMOVED = ("call", "hop1_search_fn", "hop1_extract_fn", "hop2_search_fn", "hop1_progress_fn")


def _plan():
    return EvidenceRoutePlan(
        scientific_objective="folding pathway of the beta-helix", system_description="EspP passenger domain",
        process_terms=[RouteTerm(term="folding pathway", route_type="process", source="USER_GROUNDED")],
        context_terms=[RouteTerm(term=t, route_type="context", source="USER_GROUNDED") for t in ("membrane", "thermal stability")],   # one query each: a route with 2 queries
    )


def _paper(n):
    return Paper(title=f"paper {n}", year=2001, doi=f"10.1/{n}", abstract="x" * 80)


class _Search:
    """A search function that records (query, limit) and returns one distinct paper per call."""
    def __init__(self):
        self.calls = []

    def __call__(self, query, limit):
        self.calls.append((query, limit))
        return [_paper(len(self.calls))]


# ---- retained behaviour (passes before AND after the removal) ----------------------------------------------------------------

def test_retained_only_the_evidence_plan_is_positional():
    search = _Search()
    with pytest.raises(TypeError):
        RUN(_plan(), search)                                   # flat_search_fn must be given by keyword
    with pytest.raises(TypeError):
        RUN(_plan(), None, RouteRetrievalBudgets())            # ... and so must budgets
    assert search.calls == []


def test_retained_flat_search_fn_serves_every_flat_query_with_the_budgeted_limit():
    search = _Search()
    res = RUN(_plan(), flat_search_fn=search, budgets=RouteRetrievalBudgets(max_papers_per_query=7))
    flat_queries = [q.query for q in res.plan.route_queries if q.route in FLAT_ROUTES]
    assert len(flat_queries) >= 4 and max(sum(q.route == r for q in res.plan.route_queries) for r in FLAT_ROUTES) >= 2   # some route has several queries
    assert [q for q, _ in search.calls] == flat_queries        # one call per planned flat query, in plan order, nothing else
    assert {lim for _, lim in search.calls} == {7}
    assert res.n_unique_papers == len(search.calls)            # every call returned a new paper
    assert [o.route for o in res.per_route] == list(FLAT_ROUTES)   # no need_semantic route unless it is switched on, and no structural entry


def test_retained_budgets_default_when_omitted():
    search = _Search()
    RUN(_plan(), flat_search_fn=search)
    assert search.calls and {lim for _, lim in search.calls} == {RouteRetrievalBudgets().max_papers_per_query}


def test_retained_without_flat_search_fn_the_default_flat_search_runs(monkeypatch):
    seen = []

    def fake_search_and_dedupe(query, limit=20, **kw):
        seen.append((query, limit, kw))
        return [_paper(len(seen))]
    monkeypatch.setattr(SEARCH, "search_and_dedupe", fake_search_and_dedupe)    # the default function imports it lazily from this module
    res = RUN(_plan(), budgets=RouteRetrievalBudgets(max_papers_per_query=5))
    flat_queries = [q.query for q in res.plan.route_queries if q.route in FLAT_ROUTES]
    assert [q for q, _, _ in seen] == flat_queries
    assert {lim for _, lim, _ in seen} == {5} and all(kw == {} for _, _, kw in seen)
    assert res.n_unique_papers == len(seen)


def test_retained_need_semantic_enabled_without_its_search_fn_fails_loudly():
    with pytest.raises(ValueError, match="need_semantic_search_fn"):
        RUN(_plan(), flat_search_fn=_Search(), budgets=RouteRetrievalBudgets(need_semantic_enabled=True))


def test_retained_need_semantic_search_fn_runs_with_its_own_limit():
    flat, need = _Search(), _Search()
    res = RUN(_plan(), flat_search_fn=flat, need_semantic_search_fn=need,
              budgets=RouteRetrievalBudgets(need_semantic_enabled=True, need_semantic_limit=37, max_papers_per_query=7))
    assert need.calls and {lim for _, lim in need.calls} == {37} and {lim for _, lim in flat.calls} == {7}
    assert res.per_route[-1].route == RP.NEED_SEMANTIC_ROUTE


# ---- the new signature (meaningful only after the removal) -------------------------------------------------------------------

def test_removed_signature_is_exactly_the_four_remaining_parameters():
    params = inspect.signature(RUN).parameters
    assert list(params) == ["evidence_plan", "flat_search_fn", "budgets", "need_semantic_search_fn"]
    assert params["evidence_plan"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert [params[n].kind for n in list(params)[1:]] == [inspect.Parameter.KEYWORD_ONLY] * 3
    assert all(params[n].default is None for n in list(params)[1:])


@pytest.mark.parametrize("name", REMOVED)
def test_removed_parameter_is_a_type_error_before_any_search_runs(name):
    search = _Search()
    with pytest.raises(TypeError, match=f"unexpected keyword argument '{name}'"):
        RUN(_plan(), flat_search_fn=search, **{name: lambda *a, **k: None})
    assert search.calls == []
