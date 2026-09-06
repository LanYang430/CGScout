"""Unit tests for evidence_matching_handoff.py.

evidence_matching_handoff.py transitively imports RetrievalIntent from
query_planner_handoff.py -> concept_resolver_session.py -> mellea_adapter.py
-> `mellea`, so it (and this test file) requires beam_agent/.venv-mellea.
Under beam_agent/.venv (no mellea), this whole file is skipped -- not
failed/errored -- so the existing 106-test baseline under .venv stays
green; run under .venv-mellea to actually execute these:

    .venv-mellea/bin/python -m pytest tests/test_evidence_matching_handoff.py -v

No network, no real Ollama calls -- extraction is exercised via a fake
PaperEvidenceExtractor, matching evidence_extraction.py's own Protocol.
"""

import pytest

pytest.importorskip("mellea")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks" / "concept_resolver_reasoning" / "mellea_integration"))

from evidence_matching_handoff import (  # noqa: E402
    Stage4Result,
    _as_concept_terms,
    match_evidence_against_retrieval_intent,
    run_stage4,
)
from query_planner_handoff import RetrievalIntent  # noqa: E402

from beam_agent.evidence_matching import DimensionMatch  # noqa: E402
from beam_agent.models import MethodEvidence, Paper, PaperEvidence, ProcessEvidenceItem, SystemEvidence  # noqa: E402


def _paper(title="Some Paper", found_by=("q1",), categories=("methodology_evidence",)):
    return Paper(title=title, found_by_queries=list(found_by), query_categories=list(categories))


# ---------------------------------------------------------------------------
# _as_concept_terms
# ---------------------------------------------------------------------------

def test_as_concept_terms_no_synonyms_no_inference():
    terms = _as_concept_terms(["folding", "beta-helix protein"])
    assert [t.term for t in terms] == ["folding", "beta-helix protein"]
    assert all(t.synonyms == [] for t in terms)
    assert all(t.source == "retrieval_intent" for t in terms)


def test_as_concept_terms_empty_list():
    assert _as_concept_terms([]) == []


# ---------------------------------------------------------------------------
# match_evidence_against_retrieval_intent -- exact mapping
# ---------------------------------------------------------------------------

def test_scientific_objective_maps_to_processes_dimension():
    intent = RetrievalIntent(scientific_objective=["folding"])
    evidence = PaperEvidence(processes=[ProcessEvidenceItem(raw_text="protein folding kinetics", normalized_label="folding")])
    dims = match_evidence_against_retrieval_intent(intent, evidence)
    assert dims["task_process"].match == DimensionMatch.MATCHED


def test_methodology_intent_maps_to_method_model_class_dimension():
    intent = RetrievalIntent(methodology_intent=["coarse-grained model"])
    evidence = PaperEvidence(method=MethodEvidence(model_class=["coarse-grained"]))
    dims = match_evidence_against_retrieval_intent(intent, evidence)
    assert dims["method_class"].match == DimensionMatch.MATCHED


def test_grounded_system_description_maps_to_studied_systems_dimension():
    intent = RetrievalIntent(grounded_system_description=["beta-helix protein"])
    evidence = PaperEvidence(system=SystemEvidence(studied_systems=["a 200-residue beta-helix protein"]))
    dims = match_evidence_against_retrieval_intent(intent, evidence)
    assert dims["studied_systems"].match == DimensionMatch.MATCHED


def test_mismatch_when_paper_has_evidence_but_no_overlap():
    # "protein" deliberately avoided on both sides -- shared generic words
    # already cause a MATCH under the same high-recall stem-overlap design
    # Stage 3 uses (accepted, not re-litigated here); this isolates a
    # genuine zero-overlap MISMATCH instead.
    intent = RetrievalIntent(grounded_system_description=["beta-helix architecture"])
    evidence = PaperEvidence(system=SystemEvidence(studied_systems=["coronavirus envelope structure"]))
    dims = match_evidence_against_retrieval_intent(intent, evidence)
    assert dims["studied_systems"].match == DimensionMatch.MISMATCH


def test_studied_systems_not_matched_on_generic_word_protein_alone():
    """Regression: the grounded "200-aa beta-helix protein" must NOT come
    back MATCHED against "Coronavirus envelope (E) protein" purely on the
    shared generic word "protein" (observed live, stage4_test_results.json
    Case C). beta-helix vs coronavirus == genuine MISMATCH."""
    intent = RetrievalIntent(grounded_system_description=["200-aa beta-helix protein"])
    evidence = PaperEvidence(
        system=SystemEvidence(studied_systems=["Coronaviruses", "Coronavirus envelope (E) protein", "SARS-CoV E"])
    )
    dims = match_evidence_against_retrieval_intent(intent, evidence)
    assert dims["studied_systems"].match == DimensionMatch.MISMATCH


def test_studied_systems_still_matches_on_the_specific_term():
    """The guard strips only the generic word -- a real overlap on
    "beta-helix" itself still matches."""
    intent = RetrievalIntent(grounded_system_description=["200-aa beta-helix protein"])
    evidence = PaperEvidence(system=SystemEvidence(studied_systems=["a right-handed beta-helix protein"]))
    dims = match_evidence_against_retrieval_intent(intent, evidence)
    assert dims["studied_systems"].match == DimensionMatch.MATCHED


def test_studied_systems_generic_only_grounded_term_is_no_evidence():
    """A grounded system description that is *nothing but* generic domain
    words has nothing specific to match -> NO_EVIDENCE, never a confident
    MISMATCH."""
    intent = RetrievalIntent(grounded_system_description=["protein complex"])
    evidence = PaperEvidence(system=SystemEvidence(studied_systems=["Coronavirus envelope (E) protein"]))
    dims = match_evidence_against_retrieval_intent(intent, evidence)
    assert dims["studied_systems"].match == DimensionMatch.NO_EVIDENCE
    assert "generic domain words" in dims["studied_systems"].note


def test_method_class_still_matches_on_so_called_generic_words():
    """The guard is scoped to studied_systems only -- method_class must
    still match "coarse-grained model" <-> "coarse-grained", i.e. the word
    "model" being generic elsewhere must not weaken method matching."""
    intent = RetrievalIntent(methodology_intent=["coarse-grained model"])
    evidence = PaperEvidence(method=MethodEvidence(model_class=["coarse-grained"]))
    dims = match_evidence_against_retrieval_intent(intent, evidence)
    assert dims["method_class"].match == DimensionMatch.MATCHED


def test_no_evidence_when_intent_has_nothing_grounded_for_a_dimension():
    intent = RetrievalIntent(scientific_objective=["folding"])  # no methodology_intent at all
    evidence = PaperEvidence(method=MethodEvidence(model_class=["all-atom"]))
    dims = match_evidence_against_retrieval_intent(intent, evidence)
    assert dims["method_class"].match == DimensionMatch.NO_EVIDENCE


def test_no_evidence_when_paper_evidence_is_none():
    """The no-abstract case: evidence=None must degrade to NO_EVIDENCE on
    every dimension, never a fabricated MISMATCH."""
    intent = RetrievalIntent(scientific_objective=["folding"], methodology_intent=["coarse-grained"], grounded_system_description=["beta-helix protein"])
    dims = match_evidence_against_retrieval_intent(intent, None)
    assert all(d.match == DimensionMatch.NO_EVIDENCE for d in dims.values())


def test_family_fold_dimension_is_omitted_not_fabricated():
    """No RetrievalIntent equivalent exists for the old family_fold
    dimension -- must be absent from the output entirely, not present with
    a manufactured NO_EVIDENCE."""
    intent = RetrievalIntent(scientific_objective=["folding"])
    dims = match_evidence_against_retrieval_intent(intent, PaperEvidence())
    assert "family_fold" not in dims
    assert set(dims.keys()) == {"task_process", "method_class", "studied_systems"}


# ---------------------------------------------------------------------------
# run_stage4 -- orchestration, mixed success/failure
# ---------------------------------------------------------------------------

class _FakeExtractor:
    """Fake PaperEvidenceExtractor -- title 'FAIL' raises, everything else
    succeeds with a fixed PaperEvidence. Confirms run_stage4 never batches
    (always called with a length-1 list) and isolates failures."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def extract_many(self, papers):
        self.calls.append([p.title for p in papers])
        assert len(papers) == 1, "extract_many must be called one paper at a time, never batched"
        paper = papers[0]
        if paper.title == "FAIL":
            raise RuntimeError("simulated extraction failure")
        evidence = PaperEvidence(method=MethodEvidence(model_class=["coarse-grained"]), extractor="fake")
        return [paper.model_copy(update={"evidence": evidence})]


def test_run_stage4_mixed_success_failure_isolation():
    papers = [_paper("Paper A"), _paper("FAIL"), _paper("Paper C")]
    intent = RetrievalIntent(methodology_intent=["coarse-grained"])
    extractor = _FakeExtractor()

    results = run_stage4(papers, intent, extractor=extractor)

    assert len(results) == 3
    a, b, c = results

    assert a.paper.title == "Paper A" and not a.extraction_failed and a.error is None
    assert a.dimensions["method_class"].match == DimensionMatch.MATCHED

    assert b.paper.title == "FAIL" and b.extraction_failed is True
    assert "simulated extraction failure" in b.error
    assert b.dimensions == {}  # no matching attempted when extraction failed

    assert c.paper.title == "Paper C" and not c.extraction_failed and c.error is None
    assert c.dimensions["method_class"].match == DimensionMatch.MATCHED

    # Confirms B's failure did not stop C from being attempted.
    assert extractor.calls == [["Paper A"], ["FAIL"], ["Paper C"]]


def test_run_stage4_never_batches_multiple_papers_into_one_extraction_call():
    papers = [_paper("P1"), _paper("P2"), _paper("P3")]
    intent = RetrievalIntent()
    extractor = _FakeExtractor()

    run_stage4(papers, intent, extractor=extractor)

    assert extractor.calls == [["P1"], ["P2"], ["P3"]]


def test_run_stage4_preserves_provenance_on_result_paper():
    """Confirms found_by_queries/query_categories survive on the returned
    Paper unchanged -- run_stage4 never rewrites or discards provenance."""
    paper = _paper("Paper A", found_by=("query one", "query two"), categories=("system_with_methodology",))
    intent = RetrievalIntent()
    extractor = _FakeExtractor()

    [result] = run_stage4([paper], intent, extractor=extractor)

    assert result.paper.found_by_queries == ["query one", "query two"]
    assert result.paper.query_categories == ["system_with_methodology"]
