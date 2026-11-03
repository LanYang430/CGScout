"""Unit tests for related_system_identity.py -- the entity_type/
identity_context-driven query-anchor policy (this session's fourth-pass
fix, replacing the earlier Schwartz-Hearst/regex-based approach entirely).

No LLM, no network: builds DiscoveredSystem/SupportingPaper objects by
hand, exactly as transfer_entity_extraction.py's real extraction ->
related_system_discovery._ground_mention()'s pass-through would produce
them.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks" / "concept_resolver_reasoning"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks" / "concept_resolver_reasoning" / "mellea_integration"))

from related_system_discovery import DiscoveredSystem, SupportingPaper  # noqa: E402
from related_system_identity import build_related_system_identity  # noqa: E402


def _ds(name, entity_type=None, identity_context=None, n_support=1, rank=0,
        name_as_extracted=None, title="Some Paper", quote="quote"):
    sp = SupportingPaper(
        paper_ref="10.x/1", title=title, found_by_query="q", retrieval_rank=rank,
        evidence_quote=quote, structural_term_matched="autotransporter",
        system_name=name, name_as_extracted=name_as_extracted or name,
        entity_type=entity_type, identity_context=identity_context,
    )
    return DiscoveredSystem(name=name, norm_name=name.lower(), supporting_papers=[sp],
                             n_support=n_support, first_seen_rank=rank)


def test_specific_protein_with_grounded_context_is_grounded_full():
    ds = _ds("Pet", entity_type="specific_protein_or_system", identity_context="autotransporter")
    ident = build_related_system_identity(ds, scientific_class="autotransporter passenger domain")
    assert ident.grounding == "grounded_full"
    assert ident.query_anchor == "Pet autotransporter"
    assert ident.label == "Pet"  # invariant: original label always preserved


def test_specific_protein_without_context_is_grounded_contextual_bare_label():
    ds = _ds("EspP", entity_type="specific_protein_or_system", identity_context=None)
    ident = build_related_system_identity(ds, scientific_class="autotransporter passenger domain")
    assert ident.grounding == "grounded_contextual"
    assert ident.query_anchor == "EspP"


def test_construct_or_variant_with_context_also_grounded_full():
    ds = _ds("EspPDelta1-851", entity_type="construct_or_variant", identity_context="truncated EspP construct")
    ident = build_related_system_identity(ds, scientific_class="autotransporter passenger domain")
    assert ident.grounding == "grounded_full"
    assert ident.query_anchor == "EspPDelta1-851 truncated EspP construct"


def test_organism_or_species_is_insufficient_not_a_named_system_anchor():
    ds = _ds("Bordetella pertussis", entity_type="organism_or_species", identity_context=None)
    ident = build_related_system_identity(ds, scientific_class="autotransporter passenger domain")
    assert ident.grounding == "insufficient"
    assert ident.query_anchor is None
    assert ident.label == "Bordetella pertussis"  # still preserved for audit, just not searched


def test_generic_descriptor_is_insufficient():
    ds = _ds("trimeric", entity_type="generic_descriptor", identity_context=None)
    ident = build_related_system_identity(ds, scientific_class="autotransporter passenger domain")
    assert ident.grounding == "insufficient"
    assert ident.query_anchor is None


def test_protein_family_or_class_is_insufficient_not_a_named_system_anchor():
    ds = _ds("SPATEs", entity_type="protein_family_or_class", identity_context=None)
    ident = build_related_system_identity(ds, scientific_class="autotransporter passenger domain")
    assert ident.grounding == "insufficient"
    assert ident.query_anchor is None


def test_missing_entity_type_is_conservatively_insufficient_not_guessed():
    ds = _ds("SomeName", entity_type=None, identity_context=None)
    ident = build_related_system_identity(ds, scientific_class="autotransporter passenger domain")
    assert ident.grounding == "insufficient"
    assert ident.query_anchor is None


def test_first_mention_wins_when_multiple_supporting_papers_present():
    sp1 = SupportingPaper(paper_ref="10.x/1", title="T1", found_by_query="q", retrieval_rank=0,
                          evidence_quote="q1", structural_term_matched="autotransporter",
                          system_name="BrkA", name_as_extracted="BrkA",
                          entity_type="specific_protein_or_system", identity_context="autotransporter")
    sp2 = SupportingPaper(paper_ref="10.x/2", title="T2", found_by_query="q", retrieval_rank=1,
                          evidence_quote="q2", structural_term_matched="autotransporter",
                          system_name="BrkA", name_as_extracted="BrkA",
                          entity_type="organism_or_species", identity_context=None)
    ds = DiscoveredSystem(name="BrkA", norm_name="brka", supporting_papers=[sp1, sp2], n_support=2, first_seen_rank=0)
    ident = build_related_system_identity(ds, scientific_class="autotransporter passenger domain")
    assert ident.grounding == "grounded_full"
    assert ident.query_anchor == "BrkA autotransporter"


def test_aliases_dedup_against_label():
    ds = _ds("Pet", entity_type="specific_protein_or_system", identity_context="autotransporter",
             name_as_extracted="the autotransporter Pet")
    ident = build_related_system_identity(ds, scientific_class="autotransporter passenger domain")
    assert "the autotransporter Pet" in ident.aliases
    assert ident.label == "Pet"


def test_n_support_carried_through_as_deterministic_signal_not_llm_confidence():
    ds = _ds("Pet", entity_type="specific_protein_or_system", identity_context="autotransporter", n_support=3)
    ident = build_related_system_identity(ds, scientific_class="autotransporter passenger domain")
    assert ident.n_support == 3


# --- canonical-class projection (basis == "distinctive_structure") -------
# Hop-2 recall-variance fix: identity_context's own free wording is ignored
# for query_anchor construction on this basis -- query_anchor is always
# "{label} {scientific_class}", regardless of how verbose/differently-
# worded identity_context is. identity_context still decides the grounding
# TIER (audit only), just not the anchor STRING.

def test_distinctive_structure_projects_onto_scientific_class_regardless_of_identity_context_wording():
    short = _ds("EspP", entity_type="specific_protein_or_system", identity_context="autotransporter")
    verbose = _ds("EspP", entity_type="specific_protein_or_system",
                   identity_context="passenger domain of the Escherichia coli O157:H7 autotransporter")
    ident_short = build_related_system_identity(
        short, scientific_class="autotransporter passenger domain", basis="distinctive_structure")
    ident_verbose = build_related_system_identity(
        verbose, scientific_class="autotransporter passenger domain", basis="distinctive_structure")
    assert ident_short.query_anchor == ident_verbose.query_anchor == "EspP autotransporter passenger domain"


def test_distinctive_structure_still_qualifies_short_names_not_bare_label():
    ds = _ds("Tsh", entity_type="specific_protein_or_system", identity_context=None)
    ident = build_related_system_identity(
        ds, scientific_class="autotransporter passenger domain", basis="distinctive_structure")
    assert ident.query_anchor == "Tsh autotransporter passenger domain"
    assert ident.grounding == "grounded_contextual"  # tier still reflects missing identity_context, for audit


def test_distinctive_structure_grounding_tier_still_tracks_identity_context_presence():
    with_ctx = _ds("EspP", entity_type="specific_protein_or_system", identity_context="autotransporter")
    without_ctx = _ds("EspP", entity_type="specific_protein_or_system", identity_context=None)
    ident_with = build_related_system_identity(
        with_ctx, scientific_class="autotransporter passenger domain", basis="distinctive_structure")
    ident_without = build_related_system_identity(
        without_ctx, scientific_class="autotransporter passenger domain", basis="distinctive_structure")
    assert ident_with.grounding == "grounded_full"
    assert ident_without.grounding == "grounded_contextual"
    # but the anchor STRING is identical either way -- wording never leaks in
    assert ident_with.query_anchor == ident_without.query_anchor == "EspP autotransporter passenger domain"


def test_non_distinctive_structure_basis_keeps_pre_existing_identity_context_behavior():
    ds = _ds("SomeIDP", entity_type="specific_protein_or_system",
             identity_context="phosphorylated disordered region")
    ident = build_related_system_identity(
        ds, scientific_class="phosphorylated intrinsically disordered proteins", basis="chemistry_representation")
    assert ident.query_anchor == "SomeIDP phosphorylated disordered region"  # verbatim, not projected


def test_non_distinctive_structure_basis_without_identity_context_stays_bare_label_not_scientific_class():
    ds = _ds("SomeIDP", entity_type="specific_protein_or_system", identity_context=None)
    ident = build_related_system_identity(
        ds, scientific_class="phosphorylated intrinsically disordered proteins", basis="chemistry_representation")
    assert ident.query_anchor == "SomeIDP"  # unchanged pre-existing behavior -- no unproven class injected


def test_basis_none_default_preserves_pre_existing_behavior_for_older_callers():
    ds = _ds("EspP", entity_type="specific_protein_or_system",
             identity_context="passenger domain of the Escherichia coli O157:H7 autotransporter")
    ident = build_related_system_identity(ds, scientific_class="autotransporter passenger domain")  # basis omitted
    assert ident.query_anchor == "EspP passenger domain of the Escherichia coli O157:H7 autotransporter"
