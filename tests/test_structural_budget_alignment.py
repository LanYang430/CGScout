"""Unit tests for this session's structural-branch budget alignment:
RouteRetrievalBudgets.structural_total_budget (a dedicated NEW-unique-paper
cap for the structural branch, passed as route_cap instead of the old
hardcoded None) + resolve_structural_and_safety_budgets() (computes that cap
as structural_hop2_max_systems * <Hop-2 per-system search depth> and raises
max_total_unique_papers, never lowers it, to at least the sum of every
route's own cap, so it can no longer bind before an individual route's own
cap does).

No LLM, no network -- search_fn/extract_fn are fakes, same convention as
tests/test_hop2_simplification.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks" / "concept_resolver_reasoning"))

from route_retrieval_plan import (  # noqa: E402
    FLAT_ROUTES,
    RouteRetrievalBudgets,
    RouteRetrievalPlan,
    _Accumulator,
    _run_structural_branch,
    resolve_structural_and_safety_budgets,
)
from beam_agent.models import Paper  # noqa: E402
from intake_contract import SystemExtraction, SystemMention  # noqa: E402


def _paper(title, abstract="", doi=None, year=2020):
    return Paper(title=title, abstract=abstract, doi=doi, year=year)


def _extract(mention_map):
    def _fn(title, abstract, structural_class, aka=None):
        return SystemExtraction(systems=[
            SystemMention(name=n, evidence_quote=q, structural_term_matched=s)
            for (n, q, s) in mention_map.get(title, [])
        ])
    return _fn


def _term_call(system_prompt, user_message, schema):
    import json
    return json.dumps({"original_term": "beta-helix", "expansion_terms": []})


# --- resolve_structural_and_safety_budgets(), pure-function behavior -------

def test_computes_structural_total_budget_from_max_systems_and_search_limit():
    b = RouteRetrievalBudgets(structural_hop2_max_systems=30)
    resolved = resolve_structural_and_safety_budgets(b, structural_hop2_search_limit=20)
    assert resolved.structural_total_budget == 600


def test_raises_global_cap_to_sum_of_every_route_own_cap():
    b = RouteRetrievalBudgets(max_papers_per_route=20, structural_hop2_max_systems=30,
                              max_total_unique_papers=120)
    resolved = resolve_structural_and_safety_budgets(b, structural_hop2_search_limit=20)
    expected_floor = len(FLAT_ROUTES) * 20 + 600  # 4*20 + 600 = 680
    assert resolved.max_total_unique_papers == expected_floor
    assert resolved.max_total_unique_papers > 120  # strictly raised from the old default


def test_never_lowers_an_already_higher_global_cap():
    b = RouteRetrievalBudgets(max_total_unique_papers=10_000)
    resolved = resolve_structural_and_safety_budgets(b, structural_hop2_search_limit=20)
    assert resolved.max_total_unique_papers == 10_000


def test_does_not_override_an_explicitly_set_structural_total_budget():
    b = RouteRetrievalBudgets(structural_total_budget=42)
    resolved = resolve_structural_and_safety_budgets(b, structural_hop2_search_limit=20)
    assert resolved.structural_total_budget == 42


def test_is_a_pure_function_original_budgets_untouched():
    b = RouteRetrievalBudgets()
    resolve_structural_and_safety_budgets(b, structural_hop2_search_limit=20)
    assert b.structural_total_budget is None
    assert b.max_total_unique_papers == 120


# --- end-to-end through _run_structural_branch: the diagnosed crowding case -

def _sysname(i: int) -> str:
    # Zero-padded so no name is ever a string-prefix of another
    # (System1/System19 would otherwise collide under a naive startswith
    # match) -- purely a test-fixture concern, not production code.
    return f"Sys{i:03d}"


def _make_many_systems_scenario(n_systems: int, papers_per_system: int):
    """n_systems discovered systems, RANKED in order 0..n_systems-1 (by
    first_seen_rank), each with its own Hop-2 query returning
    papers_per_system distinct papers. The LAST-ranked system's papers are
    the ones a shared/undersized global cap would crowd out first --
    exactly the diagnosed pertactin scenario (ranked 14th/20)."""
    hop1_papers = [_paper(f"{_sysname(i)} discovery paper", f"{_sysname(i)} is a beta-helix", doi=f"10/hop1-{i}")
                   for i in range(n_systems)]

    def hop1_search(q):
        return hop1_papers if "beta-helix" in q else []

    hop1_extract = _extract({
        p.title: [(_sysname(i), f"{_sysname(i)} is a beta-helix", "beta-helix")]
        for i, p in enumerate(hop1_papers)
    })

    hits_by_system = {
        _sysname(i): [_paper(f"{_sysname(i)} hop2 paper {j}", f"{_sysname(i)} is a beta-helix, paper {j}",
                             doi=f"10/hop2-{i}-{j}")
                      for j in range(papers_per_system)]
        for i in range(n_systems)
    }

    def hop2_search(q: str):
        for name, hits in hits_by_system.items():
            if q.startswith(name + " "):  # space-anchored: no name is a bare prefix of another anyway, belt-and-suspenders
                return hits
        return []

    return hop1_search, hop1_extract, hop2_search, hits_by_system


def test_late_ranked_system_is_crowded_out_by_shared_cap_when_route_cap_is_none():
    """Reproduces the diagnosed failure mode: with the OLD behavior
    (structural_total_budget unset -> route_cap=None, only a small shared
    max_total_unique_papers applies), a late-ranked system's papers get
    crowded out by earlier-ranked systems fully consuming the shared cap
    first."""
    n_systems, per_system = 20, 20
    hop1_search, hop1_extract, hop2_search, hits_by_system = _make_many_systems_scenario(n_systems, per_system)

    plan = RouteRetrievalPlan(
        system_description="beta-helix protein", scientific_objective="folding",
        structural_branch_enabled=True, structural_branch_structural_class="beta-helix",
        structural_branch_structural_class_source="USER_GROUNDED",
    )
    # OLD-style small shared cap, structural_total_budget left unset (None)
    budgets = RouteRetrievalBudgets(max_total_unique_papers=120)
    acc = _Accumulator(budgets)
    outcome, hop2_result = _run_structural_branch(
        plan, acc, budgets, call=_term_call,
        hop1_search_fn=hop1_search, hop1_extract_fn=hop1_extract, hop2_search_fn=hop2_search,
    )
    last_system_papers = hits_by_system[_sysname(n_systems - 1)]
    last_system_refs = {f"10/hop2-19-{j}" for j in range(per_system)}
    survived_refs = set(acc.papers.keys())
    # at least one of the last-ranked system's papers must be MISSING --
    # this assertion documents the bug this session's Part B fixes, not a
    # requirement of correct behavior.
    assert not last_system_refs.issubset(survived_refs)
    assert outcome.n_dropped_global_cap > 0


def test_late_ranked_system_fully_survives_with_dedicated_structural_budget():
    """Same scenario, but with structural_total_budget sized to the branch's
    real theoretical yield (as resolve_structural_and_safety_budgets() would
    set it live) -- the late-ranked system's full result must now survive
    regardless of processing order."""
    n_systems, per_system = 20, 20
    hop1_search, hop1_extract, hop2_search, hits_by_system = _make_many_systems_scenario(n_systems, per_system)

    plan = RouteRetrievalPlan(
        system_description="beta-helix protein", scientific_objective="folding",
        structural_branch_enabled=True, structural_branch_structural_class="beta-helix",
        structural_branch_structural_class_source="USER_GROUNDED",
    )
    budgets = RouteRetrievalBudgets(structural_hop2_max_systems=30)
    budgets = resolve_structural_and_safety_budgets(budgets, structural_hop2_search_limit=per_system)
    acc = _Accumulator(budgets)
    outcome, hop2_result = _run_structural_branch(
        plan, acc, budgets, call=_term_call,
        hop1_search_fn=hop1_search, hop1_extract_fn=hop1_extract, hop2_search_fn=hop2_search,
    )
    last_system_refs = {f"10/hop2-19-{j}" for j in range(per_system)}
    survived_refs = set(acc.papers.keys())
    assert last_system_refs.issubset(survived_refs)
    assert outcome.n_dropped_global_cap == 0
    assert outcome.n_dropped_route_cap == 0
    assert outcome.n_new_unique_contributed == n_systems * per_system
