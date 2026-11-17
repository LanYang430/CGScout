"""Consolidation safety net, Phase 3 preparation -- the traces the 3B L2 skeleton has to reproduce, written from the code of tag consol-2b-done (the OLD
_run_related_system_discovery), BEFORE that function was replaced.  Offline.

What the three scenarios add to the first two families: the REAL cleanup / identity / candidate assembly on data where cleanup merges IN PLACE (a suffix duplicate and an
interaction-shaped duplicate fold into a surviving system, a fragment on the blocklist is dropped) so that the P6 input depends on those in-place effects; the error semantics when
that path ends in "P3 failed, then Hop-2 raised" (the Hop-2 error replaces the P3 error); and two related-system axes in one run, which share ONE paper cache (the second axis's
tagged copies are derived from cache entries the first axis already changed, and `_merge_papers` accumulates the system labels across axes).
Whatever the next commit changes, these traces must stay as they are; a difference is a stop-and-report event."""
import json

import pytest

import consolidation_layout as LY
import consolidation_world as W

BASE = json.loads(W.FIXTURE_L2.read_text())


def test_fixture_covers_every_l2_scenario():
    assert set(BASE) == set(W.SCENARIOS_L2)


@pytest.mark.parametrize("name", sorted(W.SCENARIOS_L2))
def test_l2_trace_equals_baseline(name):
    got = W.run_scenario(name)
    why = LY.diff_against_baseline(got, BASE[name], LY.active_layout())
    assert not why, f"{name}: {why}"


def _ev(name, kind):
    return [e for e in BASE[name]["events"] if e["kind"] == kind]


def test_baseline_fact_cleanup_merges_in_place_and_changes_what_reaches_p6():
    (cl,), (ident,) = _ev("L2R01_cleanup_merges_in_place", "cleanup_in"), _ev("L2R01_cleanup_merges_in_place", "identities_in")
    assert [s["name"] for s in cl["systems"]] == ["EspP", "Hbp", "of the Escherichia coli Autotransporter EspP", "related", "EspP-BamA interaction"]
    assert [s["name"] for s in ident["systems"]] == ["EspP", "Hbp"]                       # suffix duplicate and interaction duplicate merged, fragment dropped
    (p6,) = _ev("L2R01_cleanup_merges_in_place", "P6_in")
    assert [(s[0], s[1]) for s in p6["systems"]] == [(0, "EspP"), (1, "Hbp")]              # candidates: stable order by (-n_support, first_seen_rank), re-indexed from 0


def test_baseline_fact_hop2_error_replaces_the_p3_error_and_keeps_identities():
    (log,) = [l for l in BASE["L2R02_cleanup_then_p3_fails_and_hop2_raises"]["runs"][0]["result"]["axis_logs"] if l["executed"]]
    assert log["error"] == "hop2: hop2 boom" and log["hop2_queries"] == [] and [i["label"] for i in log["identities"]] == ["EspP", "Hbp"]


def test_baseline_fact_two_axes_share_one_paper_cache():
    run = BASE["L2R03_two_3b_axes_share_the_paper_cache"]["runs"][0]["result"]
    assert [l["executed"] for l in run["axis_logs"]] == [True, True]
    shared = next(p for p in run["transfer_papers"] if p["title"] == "Shared coarse-grained study")
    assert shared["retrieval_modes"] == ["semantic", "lexical"]                              # the union keeps the order the routes found it in: Hop-1 (semantic) before Hop-2 (lexical)
    assert shared["sds"] == ["Aquila protein", "Borealis protein"]                            # labels accumulate, in the order of the systems that found it
    assert shared["sd"] == shared["sds"][0]                                                  # the singular field is the FIRST claimant
