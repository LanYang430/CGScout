"""Unit tests for the Stage 3 cheap rule-based relevance filter. No network calls."""

from __future__ import annotations

from beam_agent.concept_resolver import ConceptTerm, ScientificConceptProfile
from beam_agent.models import Paper
from beam_agent.relevance_filter import RuleBasedRelevanceFilter


def _paper(title: str, abstract: str | None = None) -> Paper:
    return Paper(title=title, abstract=abstract, sources=["test"])


def _profile(raw_query: str, **kwargs) -> ScientificConceptProfile:
    return ScientificConceptProfile(raw_query=raw_query, **kwargs)


def test_retains_paper_with_title_overlap():
    f = RuleBasedRelevanceFilter()
    papers = [_paper("Coarse-grained model for protein folding of beta-helix structures")]

    retained, rejected = f.filter(_profile("coarse-grained protein folding beta-helix"), papers)

    assert len(retained) == 1
    assert len(rejected) == 0
    assert retained[0].relevance_score is not None
    assert retained[0].relevance_score > 0
    assert "folding" in retained[0].matched_keywords


def test_rejects_paper_with_zero_keyword_overlap():
    f = RuleBasedRelevanceFilter()
    papers = [_paper("Quantum computing error correction codes")]

    retained, rejected = f.filter(_profile("coarse-grained protein folding beta-helix"), papers)

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

    retained, _ = f.filter(_profile("coarse-grained protein folding"), papers)

    assert len(retained) == 1


def test_missing_abstract_does_not_crash_and_falls_back_to_title():
    f = RuleBasedRelevanceFilter()
    papers = [_paper("Coarse-grained protein folding review", abstract=None)]

    retained, rejected = f.filter(_profile("coarse-grained protein folding"), papers)

    assert len(retained) == 1


def test_retained_and_rejected_sorted_by_score_descending():
    f = RuleBasedRelevanceFilter()
    weak = _paper("A folding paper")
    strong = _paper(
        "Coarse-grained protein folding model",
        abstract="folding coarse-grained protein",
    )

    retained, _ = f.filter(_profile("coarse-grained protein folding"), [weak, strong])

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
        _profile("200-aa beta-helix protein folding coarse-grained model"), papers
    )

    assert len(retained) == 1
    assert len(rejected) == 0
    assert "beta-helix" in retained[0].matched_keywords


def test_stemming_catches_plural_protein_proteins():
    f = RuleBasedRelevanceFilter()
    papers = [_paper("A generic study", abstract="This model describes proteins in solution.")]

    retained, _ = f.filter(_profile("protein model"), papers)

    assert len(retained) == 1
    assert "protein" in retained[0].matched_keywords


def test_title_match_weighted_higher_than_abstract_match():
    f = RuleBasedRelevanceFilter()
    title_match = _paper("Protein folding simulation")
    abstract_match = _paper("A generic study", abstract="This paper discusses protein folding.")

    retained, _ = f.filter(_profile("protein folding"), [title_match, abstract_match])

    scores = {p.title: p.relevance_score for p in retained}
    assert scores[title_match.title] > scores[abstract_match.title]


def test_methodology_synonym_closes_the_cg_coarse_grained_gap():
    # Real gap flagged during the Phase 1.5 review: a paper that only ever
    # says "CG model" (never "coarse-grained") used to score zero against a
    # query that said "coarse-grained", because the filter only matched
    # literal query text. Now it matches via profile.methodology_terms'
    # synonym list, without "cg" ever appearing in raw_query.
    f = RuleBasedRelevanceFilter()
    papers = [_paper("A CG model of protein folding")]
    profile = _profile(
        "coarse-grained protein folding",
        methodology_terms=[ConceptTerm(term="coarse-grained", synonyms=["cg", "coarse grained"])],
    )

    retained, rejected = f.filter(profile, papers)

    assert len(retained) == 1
    assert len(rejected) == 0
    assert "cg" in retained[0].matched_keywords


def test_related_family_term_contributes_to_score_even_when_absent_from_raw_query():
    f = RuleBasedRelevanceFilter()
    papers = [_paper("Structural studies of autotransporter passenger domains")]
    profile = _profile(
        "beta-helix protein folding",
        related_families=[ConceptTerm(term="autotransporter passenger domains")],
    )

    retained, rejected = f.filter(profile, papers)

    assert len(retained) == 1
    assert len(rejected) == 0
