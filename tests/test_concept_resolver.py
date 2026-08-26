"""Unit tests for the Stage 0 baseline concept resolver. No network calls."""

from __future__ import annotations

from beam_agent.concept_resolver import BaselineConceptResolver


def test_extracts_structural_term_and_its_synonyms():
    profile = BaselineConceptResolver().resolve("200-aa beta-helix protein folding")

    terms = {t.term for t in profile.structural_terms}
    assert "beta-helix" in terms
    matched = next(t for t in profile.structural_terms if t.term == "beta-helix")
    assert "beta-solenoid" in matched.synonyms
    assert "parallel beta-helix" in matched.synonyms


def test_matches_structural_term_via_synonym_form_in_query():
    # Query says "beta-solenoid", not "beta-helix" -- should still resolve to
    # the canonical "beta-helix" cluster.
    profile = BaselineConceptResolver().resolve("beta-solenoid protein folding")

    terms = {t.term for t in profile.structural_terms}
    assert "beta-helix" in terms


def test_extracts_methodology_term():
    profile = BaselineConceptResolver().resolve("coarse-grained model of protein folding")

    terms = {t.term for t in profile.methodology_terms}
    assert "coarse-grained" in terms


def test_extracts_scientific_task():
    profile = BaselineConceptResolver().resolve("folding kinetics of a small protein")

    terms = {t.term for t in profile.scientific_tasks}
    assert "folding kinetics" in terms


def test_extracts_size_constraint():
    profile = BaselineConceptResolver().resolve("200-aa beta-helix protein folding")

    assert len(profile.size_constraints) == 1
    c = profile.size_constraints[0]
    assert c.value == 200
    assert c.unit == "aa"
    assert c.raw_text.lower() == "200-aa"


def test_beta_helix_triggers_curated_related_family_associations():
    profile = BaselineConceptResolver().resolve("200-aa beta-helix protein folding")

    family_terms = {t.term for t in profile.related_families}
    assert "autotransporter passenger domains" in family_terms
    assert "phage tailspike proteins" in family_terms


def test_beta_helix_triggers_curated_representative_system_associations():
    profile = BaselineConceptResolver().resolve("200-aa beta-helix protein folding")

    system_terms = {t.term for t in profile.representative_systems}
    assert "pertactin" in system_terms
    assert "P22 tailspike" in system_terms


def test_curated_associations_carry_distinct_provenance_and_lower_confidence():
    # These are inferred from the structural term, not stated in the user's
    # own query text -- source and confidence should make that legible so a
    # future LLM/external-verification step knows these are candidates to
    # corroborate or overwrite, not ground truth.
    profile = BaselineConceptResolver().resolve("200-aa beta-helix protein folding")

    for term in profile.related_families + profile.representative_systems:
        assert term.source == "baseline-curated"
        assert term.confidence < 1.0

    # Direct lexical matches remain source="baseline", confidence=1.0, so the
    # two kinds of knowledge stay visibly distinct in the profile.
    for term in profile.structural_terms:
        assert term.source == "baseline"
        assert term.confidence == 1.0


def test_no_structural_term_means_no_curated_associations():
    profile = BaselineConceptResolver().resolve("xyz completely unrelated query")

    assert profile.related_families == []
    assert profile.representative_systems == []


def test_curated_associations_do_not_duplicate_across_multiple_structural_terms():
    # Both alpha-helix and beta-helix are present; only beta-helix has curated
    # associations in the table, and each association appears once even if
    # matched via more than one structural term (not the case here, but the
    # dedup-by-canonical-name logic should hold regardless).
    profile = BaselineConceptResolver().resolve("alpha-helix beta-helix protein folding")

    family_terms = [t.term for t in profile.related_families]
    assert len(family_terms) == len(set(family_terms))


def test_residual_words_land_in_other_constraints():
    profile = BaselineConceptResolver().resolve("200-aa beta-helix protein folding coarse-grained model")

    # "model" isn't part of any matched cluster or the size constraint, so it
    # should survive as an uncategorized residual term rather than vanish.
    assert "model" in profile.other_constraints


def test_matched_cluster_words_do_not_leak_into_other_constraints():
    profile = BaselineConceptResolver().resolve("200-aa beta-helix protein folding coarse-grained model")

    assert "coarse-grained" not in profile.other_constraints
    assert "folding" not in profile.other_constraints
    assert "protein" not in profile.other_constraints


def test_resolver_name_recorded_for_provenance():
    profile = BaselineConceptResolver().resolve("folding")
    assert profile.resolver == "BaselineConceptResolver"


def test_raw_query_preserved_verbatim_stripped():
    profile = BaselineConceptResolver().resolve("  folding kinetics  ")
    assert profile.raw_query == "folding kinetics"


def test_query_with_no_recognizable_concepts_does_not_crash():
    profile = BaselineConceptResolver().resolve("xyz completely unrecognized query terms")

    assert profile.structural_terms == []
    assert profile.methodology_terms == []
    assert profile.scientific_tasks == []
    assert profile.size_constraints == []


def test_all_terms_flattens_every_category():
    profile = BaselineConceptResolver().resolve("200-aa beta-helix protein folding coarse-grained model")

    flattened = profile.all_terms()
    assert len(flattened) == len(profile.structural_terms) + len(profile.related_families) + len(
        profile.representative_systems
    ) + len(profile.scientific_tasks) + len(profile.methodology_terms)
