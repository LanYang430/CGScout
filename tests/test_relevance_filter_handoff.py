"""Unit tests for relevance_filter_handoff.py -- specifically the
methodology surface-form normalization fix (coarse-grained / coarse
grained / coarse graining / coarse-graining) and confirmation that
structural terms like beta-helix are unaffected.

No network, no Mellea dependency (relevance_filter_handoff.py only needs
beam_agent.models/text_utils/concept_resolver's read-only cluster table).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks" / "concept_resolver_reasoning" / "mellea_integration"))

from relevance_filter_handoff import filter_papers  # noqa: E402

from beam_agent.models import Paper  # noqa: E402


def _paper(title, found_by_queries, categories=("system_with_methodology",), abstract=None):
    return Paper(title=title, abstract=abstract, sources=["openalex"], found_by_queries=list(found_by_queries), query_categories=list(categories))


def _kept_titles(papers):
    retained, _, _ = filter_papers(papers)
    return {p.title for p in retained}


def test_coarse_grained_query_matches_coarse_graining_title():
    p = _paper("The power of coarse graining in biomolecular simulations", ["study folding using a coarse-grained model"])
    assert p.title in _kept_titles([p])


def test_coarse_grained_query_matches_coarse_grained_space_title():
    p = _paper("Coarse grained models of protein folding", ["study folding using a coarse-grained model"])
    assert p.title in _kept_titles([p])


def test_coarse_graining_query_matches_coarse_grained_title():
    p = _paper("A review of coarse grained protein models", ["coarse-graining approaches to folding"])
    assert p.title in _kept_titles([p])


def test_coarse_grained_hyphenated_title_matches_space_query():
    p = _paper("Coarse-grained simulation of membrane proteins", ["study folding using coarse grained models"])
    assert p.title in _kept_titles([p])


def test_cg_abbreviation_still_matches_via_existing_synonym():
    # "cg" is already a listed synonym in _METHODOLOGY_TERM_CLUSTERS --
    # confirms the equivalence map doesn't break the pre-existing mechanism.
    p = _paper("CG models for protein folding", ["study folding using a coarse-grained model"])
    assert p.title in _kept_titles([p])


def test_unrelated_paper_still_dropped():
    p = _paper("Coronavirus spike protein structural biology", ["study folding using a coarse-grained model"])
    assert p.title not in _kept_titles([p])


def test_beta_helix_bare_beta_still_not_matched():
    """The exact real regression case: "Amyloid beta" must NOT match a
    beta-helix query -- confirms the methodology equivalence map has no
    effect on structural terms."""
    p = _paper("Amyloid beta: structure, biology and structure-based therapeutic development", ["200-aa beta-helix protein folding"])
    assert p.title not in _kept_titles([p])


def test_beta_helix_exact_match_still_works():
    # Title avoids "protein" (also in the query) so this isolates the
    # beta-helix exact-hyphenated-match axis specifically, not an
    # incidental match on a shared generic word.
    p = _paper("Structural basis of beta-helix architectural stability", ["200-aa beta-helix protein folding"])
    assert p.title in _kept_titles([p])


def test_beta_helix_space_variant_deliberately_NOT_normalized():
    # "beta helix" (space) is ALREADY a listed synonym of beta-helix in
    # _STRUCTURAL_TERM_CLUSTERS -- but relevance_filter_handoff.py doesn't
    # read that table at all (only _METHODOLOGY_TERM_CLUSTERS), so this
    # exercises the UNCHANGED strict compound-stem path, not the new
    # equivalence map. Title deliberately avoids any other word the query
    # also contains (e.g. "protein") so the only possible match axis is
    # beta/helix itself, isolating what this test actually checks. Must
    # NOT match -- confirming no accidental structural-term leniency was
    # introduced by this pass, at the cost of a symmetrical (but
    # out-of-scope, per instruction) gap: structural terms could suffer
    # the same hyphen/space mismatch this fix solved for methodology
    # wording, but that's deliberately not addressed in this pass.
    p = _paper("Beta helix architecture in extracellular matrix", ["200-aa beta-helix protein folding"])
    assert p.title not in _kept_titles([p])


def test_methodology_equivalence_map_has_no_beta_helix_entries():
    """Direct structural check: the equivalence map built from
    _METHODOLOGY_TERM_CLUSTERS must never contain a key/value derived from
    beta-helix (which lives only in the separate _STRUCTURAL_TERM_CLUSTERS,
    never imported by this module)."""
    from relevance_filter_handoff import _METHODOLOGY_EQUIVALENCE

    assert "beta-helix" not in _METHODOLOGY_EQUIVALENCE.values()
    assert "beta-heli" not in _METHODOLOGY_EQUIVALENCE  # beta-helix's own compound stem


# --- structural-route (Hop-2) strict gate: "Hop-1 broad, Hop-2 strict" ----
# Real scenario this was designed against: Pertactin's Hop-2 query
# "pertactin folding coarse-grained model" -- Martini 3 (CG evidence, no
# Pertactin) and the 2006 Junker experimental folding paper (Pertactin
# evidence, no CG) both cleared the OLD >=1-keyword-hit rule; only the
# target 2026 paper (both) should survive the new gate.

def _structural_paper(title, discovered_system, abstract="", query="pertactin folding coarse-grained model"):
    return Paper(
        title=title, abstract=abstract, sources=["openalex"],
        found_by_queries=[query], query_categories=["structural"],
        structural_discovered_system=discovered_system,
    )


def test_structural_gate_drops_cg_evidence_without_system_evidence():
    # Martini 3 -- real CG evidence, never mentions Pertactin at all
    p = _structural_paper(
        "Improving Martini 3 for Disordered and Multidomain Proteins", "Pertactin",
        abstract="We present improvements to the Martini 3 coarse-grained force field for disordered proteins.",
    )
    assert p.title not in _kept_titles([p])


def test_structural_gate_drops_system_evidence_without_cg_evidence():
    # the real 2006 Junker experimental folding paper -- real Pertactin
    # evidence, no coarse-grained/CG simulation evidence at all
    p = _structural_paper(
        "Pertactin beta-helix folding mechanism suggests common themes for the secretion "
        "and folding of autotransporter proteins", "Pertactin",
        abstract="We characterize the folding mechanism of the Pertactin passenger domain by "
                 "experimental biophysical methods.",
    )
    assert p.title not in _kept_titles([p])


def test_structural_gate_keeps_paper_with_both():
    # the actual target paper's real title/abstract shape -- both present
    p = _structural_paper(
        "Autotransporter folding avoids a kinetic trap during vectorial translocation "
        "across the bacterial outer membrane", "Pertactin",
        abstract="Using coarse-grained and all-atom simulations, we study folding of the "
                 "C-terminal segment of the Pertactin passenger domain.",
    )
    assert p.title in _kept_titles([p])


def test_structural_gate_accepts_cg_simulation_and_cg_model_phrasing():
    """The two phrasings explicitly named in the design decision, not
    covered by the hyphen/space "coarse-grain(ed)" forms alone."""
    p1 = _structural_paper("EspP folding", "EspP", abstract="EspP was studied using a CG simulation of its passenger domain.")
    p2 = _structural_paper("EspP folding", "EspP", abstract="A CG model of EspP reveals its folding pathway.")
    assert p1.title in _kept_titles([p1])
    assert p2.title in _kept_titles([p2])


def test_structural_gate_requires_all_words_of_a_multiword_system_name():
    # "P22 tailspike" -- only "P22" present, "tailspike" (the word itself)
    # never appears anywhere -> still drop
    p = _structural_paper(
        "P22 phage capsid assembly", "P22 tailspike",
        abstract="A coarse-grained model of P22 bacteriophage capsid assembly and maturation.",
    )
    assert p.title not in _kept_titles([p])


def test_structural_gate_does_not_apply_to_non_structural_routes():
    """The exact same title/abstract that WOULD fail the gate must still
    pass the ordinary >=1-keyword-hit rule when it's NOT a structural-route
    paper -- confirms the gate is scoped to Hop-2 only, per instruction."""
    p = Paper(
        title="Improving Martini 3 for Disordered and Multidomain Proteins",
        abstract="We present improvements to the Martini 3 coarse-grained force field.",
        sources=["openalex"],
        found_by_queries=["coarse-grained model"],
        query_categories=["modeling_requirement"],  # NOT structural
        structural_discovered_system=None,
    )
    assert p.title in _kept_titles([p])


def test_structural_gate_does_not_apply_when_discovered_system_missing():
    """A structural-route paper with no structural_discovered_system set
    (e.g. old provenance, or the field never got wired at some call site)
    fails open to the general rule -- never silently drops everything."""
    p = Paper(
        title="Improving Martini 3 for Disordered and Multidomain Proteins",
        abstract="We present improvements to the Martini 3 coarse-grained force field.",
        sources=["openalex"],
        found_by_queries=["Pertactin folding coarse-grained model"],
        query_categories=["structural"],
        structural_discovered_system=None,
    )
    assert p.title in _kept_titles([p])  # matches "coarse-grained" against its own query vocabulary, general rule only


# --- structural-route gate: EXISTENTIAL over all retrieving systems --------
# This session's fix for an order-dependent false drop: a paper found by
# BOTH 'EspP' and 'pertactin' Hop-2 queries used to be gated only against
# whichever system happened to be the first claimant (an accumulator/
# processing-order artifact, not a relevance judgment). Now the gate passes
# if ANY of the systems that actually, independently retrieved the paper has
# grounded lexical evidence -- restricted to structural_discovered_systems
# (the real retrieving set), never the full universe of Hop-1-discovered
# systems.

_TARGET_TITLE = ("Autotransporter folding avoids a kinetic trap during vectorial translocation "
                  "across the bacterial outer membrane")
_TARGET_ABSTRACT = (
    "A longstanding puzzle is why pertactin folds and is secreted in vivo within minutes but "
    "refolds in vitro over hours to days. We introduce BEAM, a multiscale coarse-grained "
    "framework applied to the pertactin passenger domain."
)


def _multi_structural_paper(discovered_systems, query="EspP folding coarse-grained model"):
    return Paper(
        title=_TARGET_TITLE, abstract=_TARGET_ABSTRACT, sources=["openalex"],
        found_by_queries=[query], query_categories=["structural"],
        structural_discovered_system=discovered_systems[0] if discovered_systems else None,  # first-claimant, as the real reattachment loop sets it
        structural_discovered_systems=discovered_systems,
    )


def test_espp_alone_fails_the_gate_pertactin_alone_passes():
    """Isolating each retrieving system individually, exactly as the user's
    regression scenario specifies."""
    espp_only = _multi_structural_paper(["EspP"])
    pertactin_only = _multi_structural_paper(["Pertactin"])
    assert espp_only.title not in _kept_titles([espp_only])       # "EspP" never appears in the text
    assert pertactin_only.title in _kept_titles([pertactin_only])  # "pertactin" appears repeatedly


def test_target_paper_survives_when_retrieved_by_both_espp_and_pertactin():
    """The real diagnosed case: EspP claimed it first (Hop-2 processing
    order, after EspP's n_support was boosted by this session's own
    duplicate/interaction merge), but Pertactin's query ALSO independently
    retrieved it -- the paper must now survive."""
    p = _multi_structural_paper(["EspP", "Pertactin"])
    kept, dropped, verdicts = filter_papers([p])
    assert p.title in {k.title for k in kept}
    v = next(v for v in verdicts if v.title == p.title)
    assert v.keep is True
    assert "Pertactin" in v.reason or "structural gate" in v.reason


def test_existential_check_is_restricted_to_this_papers_own_retrieving_systems():
    """A noise paper found ONLY by a non-matching system ('SystemX', whose
    name never appears in its text) must NOT be let through just because
    some OTHER, unrelated Hop-1-discovered system (e.g. 'Pertactin') exists
    elsewhere in the run -- structural_discovered_systems must list ONLY
    what actually, independently retrieved THIS paper, never the full
    Hop-1-discovered-systems universe."""
    noise = Paper(
        title="A totally unrelated coarse-grained simulation study",
        abstract="This coarse-grained simulation study examines an unrelated membrane protein.",
        sources=["openalex"], found_by_queries=["SystemX folding coarse-grained model"],
        query_categories=["structural"],
        structural_discovered_system="SystemX",
        structural_discovered_systems=["SystemX"],  # ONLY SystemX actually retrieved this paper
    )
    assert noise.title not in _kept_titles([noise])


def test_falls_back_to_singular_field_when_plural_list_is_empty():
    """Backward compatibility: a Paper built the OLD way (only the singular
    field set, e.g. an older diagnostic/fixture that predates this list)
    must behave exactly as before -- single-system AND-gate, not silently
    fail open."""
    p = Paper(
        title="Improving Martini 3 for Disordered and Multidomain Proteins",
        abstract="We present improvements to the Martini 3 coarse-grained force field.",
        sources=["openalex"], found_by_queries=["Pertactin folding coarse-grained model"],
        query_categories=["structural"],
        structural_discovered_system="Pertactin",
        # structural_discovered_systems deliberately left unset (empty)
    )
    assert p.title not in _kept_titles([p])  # never mentions Pertactin -- same as the pre-existing single-field test above
