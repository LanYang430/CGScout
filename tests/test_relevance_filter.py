"""Unit tests for rule-based relevance filtering. No network calls."""

from __future__ import annotations

from beam_agent.models import Paper
from beam_agent.relevance_filter import RuleBasedRelevanceFilter


def _paper(title: str, abstract: str | None = None) -> Paper:
    return Paper(title=title, abstract=abstract, sources=["test"])


def test_retains_paper_with_title_overlap():
    f = RuleBasedRelevanceFilter()
    papers = [_paper("Coarse-grained model for protein folding of beta-helix structures")]

    retained, rejected = f.filter("coarse-grained protein folding beta-helix", papers)

    assert len(retained) == 1
    assert len(rejected) == 0
    assert retained[0].relevance_score is not None
    assert retained[0].relevance_score > 0
    assert "folding" in retained[0].matched_keywords


def test_rejects_paper_with_zero_keyword_overlap():
    f = RuleBasedRelevanceFilter()
    papers = [_paper("Quantum computing error correction codes")]

    retained, rejected = f.filter("coarse-grained protein folding beta-helix", papers)

    assert len(retained) == 0
    assert len(rejected) == 1
    assert rejected[0].relevance_score == 0.0


def test_abstract_only_match_is_still_retained():
    f = RuleBasedRelevanceFilter()
    papers = [
        _paper(
            "A generic simulation study",
            abstract="We model protein folding using a coarse-grained potential.",
        )
    ]

    retained, _ = f.filter("coarse-grained protein folding", papers)

    assert len(retained) == 1


def test_missing_abstract_does_not_crash_and_falls_back_to_title():
    f = RuleBasedRelevanceFilter()
    papers = [_paper("Coarse-grained protein folding review", abstract=None)]

    retained, rejected = f.filter("coarse-grained protein folding", papers)

    assert len(retained) == 1


def test_retained_and_rejected_sorted_by_score_descending():
    f = RuleBasedRelevanceFilter()
    weak = _paper("A folding paper")
    strong = _paper(
        "Coarse-grained protein folding model",
        abstract="folding coarse-grained protein",
    )

    retained, _ = f.filter("coarse-grained protein folding", [weak, strong])

    assert len(retained) == 2
    assert retained[0].title == strong.title
    assert retained[0].relevance_score >= retained[1].relevance_score


def test_stemming_catches_irregular_derivation_beta_helix_vs_helical():
    # Real regression: "Sequence-dependent Stability Of The Beta-helical Fold"
    # was a false negative under exact-string matching against a query
    # containing "beta-helix" -- these share no suffix relationship, only a
    # prefix, which is exactly what text_utils.stem() is designed to catch.
    f = RuleBasedRelevanceFilter()
    papers = [_paper("Sequence-dependent Stability Of The Beta-helical Fold")]

    retained, rejected = f.filter(
        "200-aa beta-helix protein folding coarse-grained model", papers
    )

    assert len(retained) == 1
    assert len(rejected) == 0
    assert "beta-helix" in retained[0].matched_keywords


def test_stemming_catches_plural_protein_proteins():
    f = RuleBasedRelevanceFilter()
    papers = [_paper("A generic study", abstract="This model describes proteins in solution.")]

    retained, _ = f.filter("protein model", papers)

    assert len(retained) == 1
    assert "protein" in retained[0].matched_keywords


def test_title_match_weighted_higher_than_abstract_match():
    f = RuleBasedRelevanceFilter()
    title_match = _paper("Protein folding simulation")
    abstract_match = _paper("A generic study", abstract="This paper discusses protein folding.")

    retained, _ = f.filter("protein folding", [title_match, abstract_match])

    scores = {p.title: p.relevance_score for p in retained}
    assert scores[title_match.title] > scores[abstract_match.title]
