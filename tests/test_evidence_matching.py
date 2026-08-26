"""Unit tests for Stage 4b/5: deterministic evidence matching and ranking.
Pure functions, no LLM, no network.
"""

from __future__ import annotations

from beam_agent.concept_resolver import ConceptTerm, ScientificConceptProfile
from beam_agent.evidence_matching import (
    DimensionMatch,
    DimensionRequirement,
    FilterCriteria,
    match_evidence,
    rank_papers,
)
from beam_agent.models import MethodEvidence, Paper, PaperEvidence, ProcessEvidenceItem, SystemEvidence


def _profile(**overrides) -> ScientificConceptProfile:
    defaults = dict(raw_query="test query")
    defaults.update(overrides)
    return ScientificConceptProfile(**defaults)


def _evidence(**overrides) -> PaperEvidence:
    return PaperEvidence(**overrides)


# --- individual dimensions, via match_evidence -------------------------------


def test_process_matches_when_normalized_label_overlaps_profile_task():
    profile = _profile(scientific_tasks=[ConceptTerm(term="protein folding", synonyms=["folding"])])
    evidence = _evidence(processes=[ProcessEvidenceItem(raw_text="folding was demonstrated", normalized_label="folding")])

    result = match_evidence(profile, evidence)

    assert result.dimensions["task_process"].match == DimensionMatch.MATCHED


def test_process_mismatch_when_paper_studies_a_different_process():
    profile = _profile(scientific_tasks=[ConceptTerm(term="protein folding", synonyms=["folding"])])
    evidence = _evidence(processes=[ProcessEvidenceItem(raw_text="binding kinetics measured", normalized_label="binding")])

    result = match_evidence(profile, evidence)

    assert result.dimensions["task_process"].match == DimensionMatch.MISMATCH


def test_process_no_evidence_when_profile_has_no_target_task():
    profile = _profile()
    evidence = _evidence(processes=[ProcessEvidenceItem(raw_text="folding was demonstrated", normalized_label="folding")])

    result = match_evidence(profile, evidence)

    assert result.dimensions["task_process"].match == DimensionMatch.NO_EVIDENCE


def test_process_no_evidence_when_paper_states_no_process():
    profile = _profile(scientific_tasks=[ConceptTerm(term="protein folding")])
    evidence = _evidence()

    result = match_evidence(profile, evidence)

    assert result.dimensions["task_process"].match == DimensionMatch.NO_EVIDENCE


def test_studied_systems_matches_via_stem_overlap_with_structural_terms():
    profile = _profile(structural_terms=[ConceptTerm(term="beta-helix")])
    evidence = _evidence(system=SystemEvidence(studied_systems=["a 200-residue beta-helix protein"]))

    result = match_evidence(profile, evidence)

    assert result.dimensions["studied_systems"].match == DimensionMatch.MATCHED


def test_studied_systems_partial_when_only_some_profile_terms_covered():
    profile = _profile(
        structural_terms=[ConceptTerm(term="beta-helix"), ConceptTerm(term="alpha-helix")],
    )
    evidence = _evidence(system=SystemEvidence(studied_systems=["beta-helix"]))

    result = match_evidence(profile, evidence)

    assert result.dimensions["studied_systems"].match == DimensionMatch.PARTIAL
    assert result.dimensions["studied_systems"].matched_terms


def test_method_class_matches_against_methodology_terms():
    profile = _profile(methodology_terms=[ConceptTerm(term="coarse-grained", synonyms=["cg"])])
    evidence = _evidence(method=MethodEvidence(model_class=["coarse-grained"]))

    result = match_evidence(profile, evidence)

    assert result.dimensions["method_class"].match == DimensionMatch.MATCHED


def test_method_class_mismatch_when_paper_describes_a_different_method():
    """The mass-spec-review case from the extraction benchmark: a paper can
    describe biological processes in its own text (e.g. from TOC-style
    section headings) with zero modeling/method evidence at all -- method_class
    must independently catch that, since task_process alone wouldn't."""
    profile = _profile(methodology_terms=[ConceptTerm(term="coarse-grained", synonyms=["cg"])])
    evidence = _evidence(method=MethodEvidence(model_class=["mass spectrometry"]))

    result = match_evidence(profile, evidence)

    assert result.dimensions["method_class"].match == DimensionMatch.MISMATCH


def test_method_class_no_evidence_when_paper_names_no_method_at_all():
    profile = _profile(methodology_terms=[ConceptTerm(term="coarse-grained")])
    evidence = _evidence()  # no model_class at all -- e.g. a pure review abstract

    result = match_evidence(profile, evidence)

    assert result.dimensions["method_class"].match == DimensionMatch.NO_EVIDENCE


def test_family_fold_matches_against_representative_systems_too():
    profile = _profile(representative_systems=[ConceptTerm(term="pertactin")])
    evidence = _evidence(system=SystemEvidence(protein_or_system_names=["pertactin"]))

    result = match_evidence(profile, evidence)

    assert result.dimensions["family_fold"].match == DimensionMatch.MATCHED


def test_family_fold_matches_against_family_or_fold_field():
    profile = _profile(related_families=[ConceptTerm(term="autotransporter passenger domains")])
    evidence = _evidence(system=SystemEvidence(family_or_fold=["autotransporter passenger domain family"]))

    result = match_evidence(profile, evidence)

    assert result.dimensions["family_fold"].match == DimensionMatch.MATCHED


def test_missing_evidence_never_fails_a_required_gate():
    """NO_EVIDENCE must never be conflated with MISMATCH -- a paper the
    extractor never ran on (or that just didn't state a process) shouldn't
    be filtered out by a required dimension the way an actual mismatch would."""
    profile = _profile(scientific_tasks=[ConceptTerm(term="protein folding")])
    criteria = FilterCriteria(task_process=DimensionRequirement.REQUIRED)

    result = match_evidence(profile, evidence=None, criteria=criteria)

    assert result.dimensions["task_process"].match == DimensionMatch.NO_EVIDENCE
    assert result.passes_required is True


def test_explicit_mismatch_fails_a_required_gate():
    profile = _profile(scientific_tasks=[ConceptTerm(term="protein folding")])
    evidence = _evidence(processes=[ProcessEvidenceItem(raw_text="aggregation studied", normalized_label="aggregation")])
    criteria = FilterCriteria(task_process=DimensionRequirement.REQUIRED)

    result = match_evidence(profile, evidence, criteria)

    assert result.passes_required is False
    assert "task_process" in result.failed_required_dimensions


def test_score_is_1_when_no_dimension_is_preferred():
    profile = _profile()
    criteria = FilterCriteria(
        task_process=DimensionRequirement.OPTIONAL,
        studied_systems=DimensionRequirement.OPTIONAL,
        family_fold=DimensionRequirement.OPTIONAL,
    )

    result = match_evidence(profile, evidence=None, criteria=criteria)

    assert result.score == 1.0


def test_score_reflects_fraction_of_preferred_dimensions_matched():
    profile = _profile(
        scientific_tasks=[ConceptTerm(term="protein folding")],
        structural_terms=[ConceptTerm(term="beta-helix")],
    )
    evidence = _evidence(
        processes=[ProcessEvidenceItem(raw_text="folding demonstrated", normalized_label="folding")],
        # studied_systems left empty -> NO_EVIDENCE, not MATCHED
    )
    criteria = FilterCriteria(
        task_process=DimensionRequirement.PREFERRED,
        studied_systems=DimensionRequirement.PREFERRED,
    )

    result = match_evidence(profile, evidence, criteria)

    assert result.preferred_total == 2
    assert result.preferred_match_count == 1
    assert result.score == 0.5


# --- rank_papers --------------------------------------------------------------


def test_rank_papers_sorts_passing_papers_before_failing_ones():
    profile = _profile(scientific_tasks=[ConceptTerm(term="protein folding")])
    criteria = FilterCriteria(task_process=DimensionRequirement.REQUIRED)

    matching_paper = Paper(
        title="Matches",
        evidence=PaperEvidence(processes=[ProcessEvidenceItem(raw_text="folding shown", normalized_label="folding")]),
    )
    mismatching_paper = Paper(
        title="Mismatches",
        evidence=PaperEvidence(processes=[ProcessEvidenceItem(raw_text="binding shown", normalized_label="binding")]),
    )

    ranked = rank_papers(profile, [mismatching_paper, matching_paper], criteria)

    assert [p.title for p, _ in ranked] == ["Matches", "Mismatches"]
    assert ranked[0][1].passes_required is True
    assert ranked[1][1].passes_required is False


def test_rank_papers_never_drops_a_paper_even_when_it_fails_required():
    profile = _profile(scientific_tasks=[ConceptTerm(term="protein folding")])
    criteria = FilterCriteria(task_process=DimensionRequirement.REQUIRED)
    paper = Paper(
        title="Off target",
        evidence=PaperEvidence(processes=[ProcessEvidenceItem(raw_text="binding shown", normalized_label="binding")]),
    )

    ranked = rank_papers(profile, [paper], criteria)

    assert len(ranked) == 1


def test_rank_papers_treats_unextracted_papers_as_no_evidence_not_failure():
    profile = _profile(scientific_tasks=[ConceptTerm(term="protein folding")])
    criteria = FilterCriteria(task_process=DimensionRequirement.REQUIRED)
    paper = Paper(title="Never extracted", evidence=None)

    ranked = rank_papers(profile, [paper], criteria)

    assert ranked[0][1].passes_required is True
