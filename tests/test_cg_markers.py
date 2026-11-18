"""The coarse-grained marker vocabulary of Stage 3 (zero API, zero network).

Written for Phase 5 / C2e-0 (a tests-only commit, no change to src/).  C2e removes retrieval/types.py, whose only live content is the tuple `_CG_MARKERS`
(screening/filter.py imports it and extends it into `_STRUCTURAL_CG_MARKERS`); the tuple moves verbatim into filter.py.  Until now the markers were pinned only
indirectly: the Stage 3 tests build their fake classifier from `_STRUCTURAL_CG_MARKERS` itself, so a changed marker would change the test's expectation with it, and
nothing exercised the full-text override that reports the matched markers.  These tests pin the literal values, what the literal-marker check of the gate accepts and
rejects, and the order of the markers in the full-text override's output.  They read the markers through the filter module (where they stay), so they pass before and
after C2e."""
import pytest

import cgscout.screening.filter as F
import cgscout.sources.europepmc as EPMC
from cgscout.models import Paper

SCORE = F.score_paper_against_its_provenance
TRY_FULLTEXT = F._try_fulltext_cg_evidence

CG_MARKERS = ("coarse-grained", "coarse grained", "coarsegrained", "coarse-grain ", "coarse grain ")
STRUCTURAL_CG_MARKERS = CG_MARKERS + ("cg simulation", "cg simulations", "cg model", "cg models", "martini", "calvados", "aicg", "3spn", "hps")


def structural_paper(abstract, title="EspP folding"):
    return Paper(title=title, abstract=abstract, sources=["openalex"], found_by_queries=["espp folding coarse-grained model"],
                 query_categories=["structural"], structural_discovered_system="EspP", doi="10.1/x")


def test_the_marker_tuples_are_exactly_these_literals():
    assert F._CG_MARKERS == CG_MARKERS
    assert F._STRUCTURAL_CG_MARKERS == STRUCTURAL_CG_MARKERS
    assert F._STRUCTURAL_CG_MARKERS[:len(F._CG_MARKERS)] == F._CG_MARKERS


@pytest.mark.parametrize("marker", STRUCTURAL_CG_MARKERS)
def test_the_literal_marker_check_of_the_structural_gate_accepts_each_marker(marker):
    # classifier_score=None is the literal-substring path; the marker is followed by a word so the trailing-space forms ("coarse-grain ") match
    verdict = SCORE(structural_paper(f"EspP was studied with {marker}approaches."), classifier_score=None)
    assert verdict.keep, verdict.reason


@pytest.mark.parametrize("abstract", [
    "EspP was studied by coarse-graining.",          # not a listed form ("coarse-grain" is followed by "ing", not a space)
    "EspP was studied with all-atom molecular dynamics.",
    "EspP and the coarse-grain",                      # ends right after the marker: no trailing space in title + abstract
    "EspP was studied with a force field.",           # bare "force field" is deliberately not a marker
])
def test_the_literal_marker_check_rejects_text_without_a_listed_form(abstract):
    verdict = SCORE(structural_paper(abstract), classifier_score=None)
    assert not verdict.keep and "missing CG/simulation evidence" in verdict.reason


def test_the_literal_marker_check_looks_at_title_and_abstract_and_ignores_case():
    assert SCORE(structural_paper("EspP was studied with COARSE-GRAINED models."), classifier_score=None).keep
    assert SCORE(structural_paper("EspP was studied in detail.", title="Coarse-Grained EspP folding"), classifier_score=None).keep      # marker only in the title


def test_the_full_text_override_reports_the_matched_markers_in_vocabulary_order(monkeypatch):
    base = SCORE(structural_paper("EspP was studied with all-atom molecular dynamics."), classifier_score=None)
    assert not base.keep
    text = "Methods: we used Martini and coarse-grained CG models of EspP."
    monkeypatch.setattr(EPMC, "fetch_methods_text", lambda doi: text)           # the override imports the function from this module at call time
    out = TRY_FULLTEXT(structural_paper("EspP was studied with all-atom molecular dynamics."), base)
    assert out.keep
    assert out.matched_keywords == sorted(set(base.matched_keywords) | {"fulltext:coarse-grained", "fulltext:cg model", "fulltext:cg models", "fulltext:martini"})
    assert "Matched: ['coarse-grained', 'cg model', 'cg models', 'martini']." in out.reason       # vocabulary order, not text order
    assert f"Snippet: {text!r}" in out.reason


def test_the_full_text_override_ignores_case(monkeypatch):
    base = SCORE(structural_paper("EspP was studied with all-atom molecular dynamics."), classifier_score=None)
    monkeypatch.setattr(EPMC, "fetch_methods_text", lambda doi: "Methods: COARSE-GRAINED simulations of EspP.")
    out = TRY_FULLTEXT(structural_paper("x"), base)
    assert out.keep and "fulltext:coarse-grained" in out.matched_keywords


def test_the_full_text_override_leaves_the_verdict_alone_without_text_or_without_a_marker(monkeypatch):
    base = SCORE(structural_paper("EspP was studied with all-atom molecular dynamics."), classifier_score=None)
    monkeypatch.setattr(EPMC, "fetch_methods_text", lambda doi: None)
    assert TRY_FULLTEXT(structural_paper("x"), base) is base
    monkeypatch.setattr(EPMC, "fetch_methods_text", lambda doi: "Methods: all-atom simulations with a force field.")
    assert TRY_FULLTEXT(structural_paper("x"), base) is base
