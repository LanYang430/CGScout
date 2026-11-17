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


# ================================================================================================================ the three assembly steps on their own
# (the traces cannot see object identity, and the later _merge_papers unions the labels again, so the copy semantics of tag_hop2_papers are pinned here directly)
from cgscout.models import Paper
from cgscout.planning.expert_prior_planner import TransferAxis
from cgscout.retrieval.related_systems.discovery import DiscoveredSystem
from cgscout.retrieval.related_systems.hop2 import SystemInput
from cgscout.retrieval.related_systems.postprocess import RelatedSystemIdentity
from cgscout.retrieval.transfer import hop1_seed_terms, hop2_candidates, tag_hop2_papers


def _axis(anchors=("repeat fold", "stacked strands"), basis="distinctive_structure"):
    return TransferAxis(basis=basis, scientific_class="alpha helix repeat", discriminative_features=["f"], objective_relevance="o", retrieval_anchors=list(anchors),
                        rationale="r", confidence="high", execution_mode="related_system_discovery")


def test_hop1_seed_terms_are_combinations_first_then_the_aliases_in_order():
    axis = _axis(anchors=("a one", " b two "))
    assert hop1_seed_terms(axis, "cls", "obj", ["x", "y"]) == ["cls a one", "cls  b two".strip(), "cls obj", "x", "y"]
    assert hop1_seed_terms(axis, "cls", "obj", []) == ["cls a one", "cls  b two".strip(), "cls obj"]          # a failed P3 leaves only the combinations
    assert hop1_seed_terms(_axis(anchors=()), "cls", "obj", ["x"]) == ["cls obj", "x"]
    aliases = ["x", "y"]
    hop1_seed_terms(axis, "cls", "obj", aliases)
    assert aliases == ["x", "y"]                                                                              # the P3 list is not modified


def _ident(label, anchor="anchor"):
    return RelatedSystemIdentity(label=label, entity_type="specific_protein_or_system", identity_context=None, grounding="grounded_contextual", query_anchor=anchor,
                                 aliases=["al"], source_grounding=["q"], source_paper_titles=["t"], n_support=0, first_seen_rank=10 ** 9)


def _ds(name, support, rank):
    return DiscoveredSystem(name=name, norm_name=name.lower(), supporting_papers=[], n_support=support, first_seen_rank=rank)


def test_hop2_candidates_order_cut_reindex_and_label_matching():
    systems = [_ds("A", 1, 5), _ds("B", 3, 9), _ds("C", 3, 2), _ds("D", 2, 1)]
    idents = [_ident("A"), _ident("B"), _ident("C"), _ident("D"), _ident("NoSuchSystem"), _ident("D", anchor=None)]
    out = hop2_candidates(idents, systems, cap=3)
    assert [(c.index, c.label, c.n_support, c.first_seen_rank) for c in out] == [(0, "C", 3, 2), (1, "B", 3, 9), (2, "D", 2, 1)]     # -n_support, first_seen_rank; cut; reindexed
    assert [c.label for c in hop2_candidates(idents, systems, cap=10)] == ["C", "B", "D", "A"]            # identities without a query_anchor or without a system are not candidates
    assert all(isinstance(c, SystemInput) for c in out)


def test_hop2_candidates_duplicate_lowercased_names_take_the_last_systems_support():
    systems = [_ds("SysA", 3, 1), _ds("sysa", 1, 20)]                                                        # same lower-cased label: the LAST one wins for both identities
    out = hop2_candidates([_ident("SysA"), _ident("sysa")], systems, cap=5)
    assert [(c.label, c.n_support, c.first_seen_rank) for c in out] == [("SysA", 1, 20), ("sysa", 1, 20)]


def _p(title, ref, **kw):
    return Paper(title=title, doi=ref, abstract="a", year=2020, **kw)


def test_tag_hop2_papers_is_copy_on_write_shallow_and_derived_from_the_current_cache_entry():
    base = _p("P1", "10.1/p1", query_categories=["x"], found_by_queries=["old q"], structural_discovered_systems=["Earlier"], structural_discovered_system=None)
    cache = {"10.1/p1": base, "10.1/p2": _p("P2", "10.1/p2", structural_discovered_system="Claimed", structural_discovered_systems=["Claimed"])}
    found = {"SysA": {"10.1/p1": ["qa"], "10.1/p2": ["qa2"], "10.1/missing": ["zz"]}, "SysB": {"10.1/p1": ["qb"]}}
    out = tag_hop2_papers(found, cache, _axis(basis="physical_process"))
    assert [(t.title, t.structural_discovered_system, t.structural_discovered_systems) for t in out] == [        # found order x refs order; a ref the cache lacks is skipped
        ("P1", "SysA", ["Earlier", "SysA"]), ("P2", "Claimed", ["Claimed", "SysA"]), ("P1", "SysB", ["Earlier", "SysB"])]          # two copies of P1, each with ONLY its own label
    assert out[0].query_categories == ["x", "structural", "transfer:physical_process"] and out[0].found_by_queries == ["old q", "qa"] and out[2].found_by_queries == ["old q", "qb"]
    assert base.structural_discovered_systems == ["Earlier"] and base.structural_discovered_system is None and base.found_by_queries == ["old q"]      # the cache entry is untouched
    assert out[0] is not base and out[0].external_ids is base.external_ids                                    # shallow copy: nested objects stay shared with the cache entry
    assert out[0].structural_discovered_systems is not out[2].structural_discovered_systems                  # the lists written are new objects, one per copy
    assert out[0].found_by_queries is not base.found_by_queries
