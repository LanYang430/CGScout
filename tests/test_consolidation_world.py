"""Consolidation safety net, Phase 0 -- trace equivalence.  Offline, no LLM, no network.

Each scenario of tests/consolidation_world.py runs the retrieval flow for real (only the network/LLM collaborators are fakes) and must reproduce the
trace stored from the BASELINE code (tag consol-0-baseline) exactly: execution order, collaborator arguments (expansion_terms order -> P5 message,
candidate order/truncation -> P6 key), the P6 cache keys, the exception boundaries and early returns, the tag write-back and the Stage-3 input order.
A difference means the refactor changed behaviour: stop and report, do not edit the fixture."""
import json

import pytest

import consolidation_layout as LY
import consolidation_world as W

BASELINE = json.loads(W.FIXTURE.read_text())


def test_fixture_covers_every_scenario():
    assert set(BASELINE) == set(W.SCENARIOS)


@pytest.mark.parametrize("name", sorted(W.SCENARIOS))
def test_trace_equals_baseline(name):
    got = W.run_scenario(name)
    # events: ordered collaborator/cache events; runs: results / exceptions; llm_calls; patch_calls: every patched collaborator called as often as in the
    # baseline, none never called (R-1/R-2); recorder: L/X/U; recorder_installed_in: the wrapped modules after the APPROVED module renames (consolidation_layout)
    why = LY.diff_against_baseline(got, BASELINE[name], LY.active_layout())
    assert not why, f"{name}: {why}"


def _events(name, kind):
    return [e for e in BASELINE[name]["events"] if e["kind"] == kind]


# ---- the baseline facts the traces pin (documented here so a reader sees WHAT is being protected) -------------------------------------------------

def test_baseline_facts_error_boundaries():
    err = lambda n: [l["error"] for l in BASELINE[n]["runs"][0]["result"]["axis_logs"] if l["executed"]]
    assert err("S03_p3_fails_rest_ok") == ["terminology_expansion: p3 boom"]          # P3 failure: the axis continues, the error persists on success
    assert err("S04_p3_fails_hop1_raises") == ["hop1: hop1 boom"]                       # a Hop-1 error replaces the P3 error
    assert err("S05_no_systems_keeps_p3_error") == ["terminology_expansion: p3 boom"]   # no systems: early return keeps the P3 error
    assert err("S06_no_candidates_identities_insufficient") == ["terminology_expansion: p3 boom"]
    assert err("S07_hop2_raises_replaces_p3_error") == ["hop2: hop2 boom"]              # a Hop-2 error replaces the P3 error
    assert err("S08_p4_raises_axis_level") == ["p4 boom"]                               # P4 is outside any try: the axis-level except, plain str(exc)
    assert BASELINE["S09_budget_exhaustion_propagates"]["runs"][0]["raised"].startswith("BudgetExhausted")   # BaseException is never swallowed


def test_baseline_facts_dataflow():
    ev = BASELINE["S01_3b_happy"]["events"]
    disc = next(e for e in ev if e["kind"] == "P5_discover")
    # combination terms first (class + each anchor, class + objective), then the P3 aliases: this ORDER is part of every P5 cache key
    assert disc["expansion_terms"] == ["alpha helix repeat repeat fold", "alpha helix repeat stacked strands", "alpha helix repeat folding mechanism",
                                      "alpha-solenoid", "repeat protein"]
    # (what the real P5 message contains is NOT pinned here: this family's discover_related_systems is a fake that passes the extractor only
    # expansion_terms[:3].  The aliases' way into the real P5 message and cache key is pinned in tests/test_consolidation_p5.py.)
    ext = next(e for e in ev if e["kind"] == "P5_extract")
    assert ext["basis"] == "distinctive_structure" and ext["backend"] == "mock" and ext["case_id"] == "T"       # threaded by the real orchestrator closure
    # the P4 sentence replaces whatever term Hop-1 asks for
    assert next(e for e in ev if e["kind"] == "search")["query"] == "semantic sentence about alpha helix repeat"
    # P6 sees candidates sorted by (-n_support, first_seen_rank), reindexed from 0
    p6 = next(e for e in BASELINE["S02_3b_truncation_ties"]["events"] if e["kind"] == "P6_in")
    assert [s[0] for s in p6["systems"]] == list(range(len(p6["systems"]))) and len(p6["systems"]) == 6        # HOP2_MAX_TO_NORMALIZE
    assert [(s[2], s[3]) for s in p6["systems"]] == sorted([(s[2], s[3]) for s in p6["systems"]], key=lambda t: (-t[0], t[1]))
    # ds_by_label keeps the LAST discovered system for two labels that differ only in case: both take its (n_support, first_seen_rank)
    sysa = [s for s in p6["systems"] if s[1].lower() == "sysa protein"]
    assert len(sysa) == 2 and {(s[2], s[3]) for s in sysa} == {(3, 20)}


def test_baseline_facts_cache_keys_and_warm_run():
    gets = [e for e in BASELINE["S15_warm_cache_second_run"]["events"] if e["kind"] == "cache.get"]
    assert [g["hit"] for g in gets] == [False, True]                                    # second run: the P6 answer comes from the cache, no second LLM call
    assert gets[0]["key"] == gets[1]["key"] and BASELINE["S15_warm_cache_second_run"]["llm_calls"] == 1


def test_baseline_facts_loop_and_merge():
    logs = BASELINE["S12_multi_axis_loop"]["runs"][0]["result"]["axis_logs"]
    assert [l["executed"] for l in logs] == [False, True, True, True, False, True, False]
    assert "semantic-query cap reached" in logs[4]["skip"] and "semantic-query cap reached" in logs[6]["skip"]
    assert "low confidence" in logs[0]["skip"]
    st3 = _events("S12_multi_axis_loop", "stage3")
    assert len(st3) == 1                                                                 # Stage 3 runs ONCE, on direct + transfer merged by paper_ref
    # direct papers are tagged in place from the route provenance
    direct = BASELINE["S12_multi_axis_loop"]["runs"][0]["result"]["direct_papers"]
    assert direct[0]["found_by_queries"] == ["dq-Direct A", "dq2-Direct A"] and direct[0]["query_categories"] == ["flat", "process"]


def test_recorder_acceptance_baseline_is_logical_records_vs_external_calls():
    """Safeguards addendum 4 / 8.1: L (recorder records) is NOT required to equal X (calls that reached the faked HTTP boundary).  What must hold, and is
    pinned for the baseline: no external call outside a record window (U), cold run 0 < X <= L, warm run X == 0 with every record served from the cache,
    the same logical records on both runs, every kind classified, and the wrapped modules."""
    rec = BASELINE["R01_real_search_stack_recorder"]
    L, X, U = rec["recorder"]["L"], rec["recorder"]["X"], rec["recorder"]["U"]
    cold, warm = rec["runs"]
    lc, lw = L[cold["l_from"]:cold["l_to"]], L[warm["l_from"]:warm["l_to"]]
    assert U == []                                                                                   # baseline: nothing external escapes the recorder
    assert 0 < cold["x_to"] - cold["x_from"] <= len(lc) and warm["x_to"] == warm["x_from"]
    assert {r["served_from"] for r in lc} == {"http"} and {r["served_from"] for r in lw} == {"cache"}
    assert [(r["kind"], r["mode"], r["limit"], r["query"]) for r in lc] == [(r["kind"], r["mode"], r["limit"], r["query"]) for r in lw]
    assert {r["kind"] for r in L} == {"3A", "hop1", "hop2"} and not any(r["kind"].startswith("other:") for r in L)
    assert rec["recorder_installed_in"] == ["cgscout.retrieval.orchestrator", "cgscout.retrieval.search_adapters"]
