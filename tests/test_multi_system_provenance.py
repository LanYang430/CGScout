"""Unit tests for this session's multi-system provenance fix: when the same
paper is independently retrieved by more than one discovered system's Hop-2
query, _Accumulator now keeps ALL of them (RetrievedPaper.hop1_provenances),
not just the first claimant (RetrievedPaper.hop1_provenance, unchanged,
still first-claimant-only for backward compat).

No LLM, no network -- direct _Accumulator.add() calls, same convention as
tests/test_hop2_simplification.py's accumulator tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks" / "concept_resolver_reasoning"))

from route_retrieval_plan import (  # noqa: E402
    RouteProvenanceEntry,
    RouteRetrievalBudgets,
    _Accumulator,
)
from related_system_hop2 import Hop1Provenance  # noqa: E402


def _prov(system: str) -> Hop1Provenance:
    return Hop1Provenance(structural_class="beta-helix", discovered_system=system,
                          n_support=1, first_seen_rank=0, evidence_quote=f"{system} evidence")


def _entry(system: str) -> RouteProvenanceEntry:
    return RouteProvenanceEntry(route="structural", route_term=system, query=f"{system} query",
                                query_role="structural", retrieval_rank=0)


def test_first_claimant_still_wins_the_singular_field():
    """Backward compat: hop1_provenance (singular) stays first-claimant-only,
    exactly as before this session's change."""
    budgets = RouteRetrievalBudgets()
    acc = _Accumulator(budgets)
    acc.add("ref-1", "T", 2020, _entry("EspP"), _prov("EspP"), 0, None)
    acc.add("ref-1", "T", 2020, _entry("Pertactin"), _prov("Pertactin"), 1, None)
    assert acc.papers["ref-1"].hop1_provenance.discovered_system == "EspP"


def test_all_claimants_are_kept_in_hop1_provenances():
    """The new list must contain BOTH systems, in claim order, regardless of
    which one 'won' the singular field."""
    budgets = RouteRetrievalBudgets()
    acc = _Accumulator(budgets)
    acc.add("ref-1", "T", 2020, _entry("EspP"), _prov("EspP"), 0, None)
    acc.add("ref-1", "T", 2020, _entry("Pertactin"), _prov("Pertactin"), 1, None)
    names = [p.discovered_system for p in acc.papers["ref-1"].hop1_provenances]
    assert names == ["EspP", "Pertactin"]


def test_a_paper_found_by_only_one_system_has_exactly_one_entry():
    budgets = RouteRetrievalBudgets()
    acc = _Accumulator(budgets)
    acc.add("ref-1", "T", 2020, _entry("EspP"), _prov("EspP"), 0, None)
    names = [p.discovered_system for p in acc.papers["ref-1"].hop1_provenances]
    assert names == ["EspP"]


def test_the_same_system_re_claiming_is_deduped_not_appended_twice():
    """Defensive: if the same system's own query somehow returns the same
    paper_ref twice (shouldn't happen given each system issues one Hop-2
    query, but the accumulator must not silently duplicate provenance if it
    ever does)."""
    budgets = RouteRetrievalBudgets()
    acc = _Accumulator(budgets)
    acc.add("ref-1", "T", 2020, _entry("EspP"), _prov("EspP"), 0, None)
    acc.add("ref-1", "T", 2020, _entry("EspP"), _prov("EspP"), 1, None)
    names = [p.discovered_system for p in acc.papers["ref-1"].hop1_provenances]
    assert names == ["EspP"]


def test_three_systems_all_kept_full_provenance_never_truncated():
    budgets = RouteRetrievalBudgets()
    acc = _Accumulator(budgets)
    for i, system in enumerate(("EspP", "Pertactin", "XadA")):
        acc.add("ref-1", "T", 2020, _entry(system), _prov(system), i, None)
    names = {p.discovered_system for p in acc.papers["ref-1"].hop1_provenances}
    assert names == {"EspP", "Pertactin", "XadA"}
    # found_by (route provenance entries) also keeps all three, unchanged behavior
    assert len(acc.papers["ref-1"].found_by) == 3


def test_a_paper_with_no_hop1_provenance_at_all_has_an_empty_list():
    """Flat-route papers (hop1_provenance=None) must not get a spurious
    entry."""
    budgets = RouteRetrievalBudgets()
    acc = _Accumulator(budgets)
    e = RouteProvenanceEntry(route="process", route_term="x", query="q",
                             query_role="process_cg_evidence", retrieval_rank=0)
    acc.add("ref-1", "T", 2020, e, None, 0, None)
    assert acc.papers["ref-1"].hop1_provenances == []
    assert acc.papers["ref-1"].hop1_provenance is None
