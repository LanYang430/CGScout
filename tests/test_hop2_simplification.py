"""Unit tests for this session's Hop-2 simplification:
  1. one query per system (was 3 bounded roles), via build_hop2_queries()
     -- covered in tests/test_related_system_hop2.py; NOT repeated here.
  2. the raw objective span (e.g. "want to study folding") is cleaned to a
     scientific-task phrase (e.g. "folding") BEFORE it reaches Hop-2's query,
     via objective_normalization.normalize_objective() -- reused, not
     reimplemented -- called from route_retrieval_plan.py's
     _run_structural_branch().
  3. the structural branch's contribution to the merged RouteRetrievalResult
     is no longer capped by RouteRetrievalBudgets.max_papers_per_route (that
     budget now applies to flat routes only) -- only the genuinely global
     max_total_unique_papers still bounds it.

No LLM, no network -- call/search_fn/extract_fn are fakes, same convention
as run_route_retrieval_plan_standalone_validation.py's Case A/B/C.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks" / "concept_resolver_reasoning"))

from route_retrieval_plan import (  # noqa: E402
    RouteRetrievalBudgets,
    RouteRetrievalPlan,
    _Accumulator,
    _run_structural_branch,
)
from related_system_hop2 import Hop1Provenance  # noqa: E402
from route_retrieval_plan import RouteProvenanceEntry  # noqa: E402
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
    return json.dumps({"original_term": "beta-helix", "expansion_terms": []})


# --- 2. the objective is cleaned before it reaches Hop-2's query ---------

def test_raw_first_person_objective_is_cleaned_before_reaching_hop2_query():
    hop1_hits = [_paper("Pertactin beta-helix folding mechanism",
                        "The passenger domain of pertactin forms a right-handed parallel beta-helix.",
                        doi="10.1/pertactin")]
    hop1_search = lambda q: hop1_hits if "beta-helix" in q else []  # noqa: E731
    hop1_extract = _extract({
        "Pertactin beta-helix folding mechanism": [
            ("Pertactin", "pertactin forms a right-handed parallel beta-helix", "beta-helix"),
        ],
    })

    hop2_calls: list[str] = []

    def hop2_search(q: str):
        hop2_calls.append(q)
        return []

    plan = RouteRetrievalPlan(
        system_description="I have a beta-helix protein",
        scientific_objective="want to study folding",   # RAW, uncleaned -- exactly what the live fact span looks like
        structural_branch_enabled=True,
        structural_branch_structural_class="beta-helix",
        structural_branch_structural_class_source="USER_GROUNDED",
    )
    budgets = RouteRetrievalBudgets()
    acc = _Accumulator(budgets)

    outcome, hop2_result = _run_structural_branch(
        plan, acc, budgets, call=_term_call,
        hop1_search_fn=hop1_search, hop1_extract_fn=hop1_extract, hop2_search_fn=hop2_search,
    )

    assert len(hop2_calls) == 1
    query = hop2_calls[0]
    for banned in ("want to study", "want to", "I want", "I have"):
        assert banned not in query.lower(), f"un-cleaned objective text {banned!r} leaked into Hop-2 query {query!r}"
    assert query == "Pertactin folding coarse-grained model"


def test_clean_objective_passes_through_unmodified():
    """An objective that's already clean (e.g. Case A's "protein folding" in
    the standalone validation script) must not be altered by the new
    cleaning step."""
    hop1_hits = [_paper("Pertactin beta-helix folding mechanism",
                        "The passenger domain of pertactin forms a right-handed parallel beta-helix.",
                        doi="10.1/pertactin")]
    hop1_search = lambda q: hop1_hits if "beta-helix" in q else []  # noqa: E731
    hop1_extract = _extract({
        "Pertactin beta-helix folding mechanism": [
            ("Pertactin", "pertactin forms a right-handed parallel beta-helix", "beta-helix"),
        ],
    })
    hop2_calls: list[str] = []

    def hop2_search(q: str):
        hop2_calls.append(q)
        return []

    plan = RouteRetrievalPlan(
        system_description="beta-helix protein",
        scientific_objective="protein folding",   # already clean
        structural_branch_enabled=True,
        structural_branch_structural_class="beta-helix",
        structural_branch_structural_class_source="USER_GROUNDED",
    )
    budgets = RouteRetrievalBudgets()
    acc = _Accumulator(budgets)
    _run_structural_branch(
        plan, acc, budgets, call=_term_call,
        hop1_search_fn=hop1_search, hop1_extract_fn=hop1_extract, hop2_search_fn=hop2_search,
    )
    assert hop2_calls == ["Pertactin protein folding coarse-grained model"]


# --- 3. structural branch no longer bounded by max_papers_per_route ------

def test_accumulator_add_with_route_cap_none_ignores_the_route_cap():
    budgets = RouteRetrievalBudgets(max_papers_per_route=5, max_total_unique_papers=100)
    acc = _Accumulator(budgets)
    e = RouteProvenanceEntry(route="structural", route_term="x", query="q",
                             query_role="structural", retrieval_rank=0)
    accepted_count = 0
    for i in range(10):  # well past max_papers_per_route=5
        accepted, was_new = acc.add(f"paper-{i}", f"Paper {i}", 2020, e, None, accepted_count, None)
        assert accepted and was_new, f"paper-{i} should be accepted -- route_cap=None means no per-route limit"
        accepted_count += 1
    assert len(acc.papers) == 10


def test_accumulator_add_still_respects_the_route_cap_when_given():
    """Flat routes are UNCHANGED -- passing a real route_cap still gates them
    exactly as before."""
    budgets = RouteRetrievalBudgets(max_papers_per_route=5, max_total_unique_papers=100)
    acc = _Accumulator(budgets)
    e = RouteProvenanceEntry(route="process", route_term="x", query="q",
                             query_role="process_cg_evidence", retrieval_rank=0)
    accepted_count = 0
    results = []
    for i in range(10):
        accepted, was_new = acc.add(f"paper-{i}", f"Paper {i}", 2020, e, None, accepted_count, budgets.max_papers_per_route)
        results.append(accepted)
        if accepted and was_new:
            accepted_count += 1
    assert results == [True] * 5 + [False] * 5
    assert len(acc.papers) == 5


def test_structural_branch_can_carry_more_than_max_papers_per_route_papers():
    """End-to-end through _run_structural_branch(): with a LOW
    max_papers_per_route (well below what 3 systems' single queries
    produce), the structural branch's total contribution is still NOT
    capped at that number -- only the global max_total_unique_papers is."""
    systems_and_hits = {
        "SystemA": [_paper(f"SystemA paper {i}", f"SystemA is a beta-helix, paper {i}", doi=f"10/a{i}")
                    for i in range(8)],
        "SystemB": [_paper(f"SystemB paper {i}", f"SystemB is a beta-helix, paper {i}", doi=f"10/b{i}")
                    for i in range(8)],
        "SystemC": [_paper(f"SystemC paper {i}", f"SystemC is a beta-helix, paper {i}", doi=f"10/c{i}")
                    for i in range(8)],
    }
    hop1_papers = [systems_and_hits[s][0] for s in systems_and_hits]  # one Hop-1-discovery paper per system

    def hop1_search(q):
        return hop1_papers if "beta-helix" in q else []

    hop1_extract = _extract({
        p.title: [(name, f"{name} is a beta-helix", "beta-helix")]
        for name, hits in systems_and_hits.items() for p in [hits[0]]
    })

    def hop2_search(q: str):
        for name, hits in systems_and_hits.items():
            if q.startswith(name):
                return hits
        return []

    plan = RouteRetrievalPlan(
        system_description="beta-helix protein",
        scientific_objective="folding",
        structural_branch_enabled=True,
        structural_branch_structural_class="beta-helix",
        structural_branch_structural_class_source="USER_GROUNDED",
    )
    # max_papers_per_route=5 -- if this STILL applied to the structural
    # branch (the old bug), its total contribution would be capped at 5.
    budgets = RouteRetrievalBudgets(max_papers_per_route=5, max_total_unique_papers=100)
    acc = _Accumulator(budgets)

    outcome, hop2_result = _run_structural_branch(
        plan, acc, budgets, call=_term_call,
        hop1_search_fn=hop1_search, hop1_extract_fn=hop1_extract, hop2_search_fn=hop2_search,
    )

    # 3 systems x 8 papers each = 24 distinct papers, well above
    # max_papers_per_route=5 -- must all survive (well under
    # max_total_unique_papers=100 too).
    assert outcome.n_new_unique_contributed == 24
    assert outcome.n_dropped_route_cap == 0   # route cap no longer applies to this route at all
    assert outcome.n_dropped_global_cap == 0  # nowhere near the global cap either
    assert len(acc.papers) == 24
