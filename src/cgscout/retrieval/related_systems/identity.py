"""related_system_identity (scientific-planner-v2 branch) -- deterministic
identity preservation for Hop-1-discovered related systems. NEW, additive
module. No LLM call of its own -- consumes entity_type/identity_context
already produced by transfer_entity_extraction.py's extraction call and
carried, grounding-verified, on related_system_discovery.SupportingPaper.

History (this session, two earlier fixes superseded by this one): the
first version tried to DETECT an abbreviation-expansion relationship
itself, via a "<phrase> (LABEL)" regex plus a Schwartz-Hearst-style
character-alignment validator. That correctly rejected "another SPATE
(EspP)" but also produced real false positives no purely syntactic pattern
can fix -- e.g. it validated "EspPDelta1-851" as an "expansion" of "EspP"
purely because it starts with those same letters, when it is actually a
truncated CONSTRUCT of EspP, not an abbreviation of it. Distinguishing
those requires the kind of judgment only the extractor (reading the
sentence) can make -- so that judgment moved upstream, into the SAME
per-paper extraction call, as the entity_type/identity_context fields.
This module no longer guesses at anything: it only reads what the
extractor already said and deterministically builds a query anchor from
it. The regex/character-alignment machinery is deleted, not kept as a
fallback -- per this session's explicit decision, no parallel legacy path.

Governing invariant (unchanged from the prior design, restated for this
simpler mechanism): the ORIGINAL GROUNDED LABEL a discovered system was
found under is never lost -- `query_anchor` is always built FROM `label`
(bare, or `label` + a verbatim-grounded `identity_context`), never
replaced by something else entirely.

Entity-type policy (this session's explicit decision):
  - "specific_protein_or_system" / "construct_or_variant" -> a genuine
    named-system anchor. With a grounded identity_context: `query_anchor =
    "{label} {identity_context}"` (grounded_full). Without one: `query_
    anchor = label` alone (grounded_contextual) -- still a real named
    system, just without extra disambiguating context this time.
  - "organism_or_species" / "protein_family_or_class" / "generic_
    descriptor" / unset -> insufficient. An organism or a family name is
    not a specific transferable system (e.g. "Bordetella pertussis" is not
    a protein), and a bare descriptor ("trimeric") is not a named entity at
    all -- Hop-2 is not run for these; kept only as an audit record.

Canonical-class projection for `basis == "distinctive_structure"` (this
session's later fix -- Hop-2 recall variance): `identity_context` is
Luna's own free-worded restatement of the SAME axis-level `scientific_
class` it was already given, and its wording/length varies call to call
even for the exact same entity (observed: the same EspP mention producing
"autotransporter passenger domain" one run and "passenger domain of the
Escherichia coli O157:H7 autotransporter" -- organism/strain included --
the next), making `query_anchor` non-deterministic across runs and, when
verbose, a weaker OpenAlex query. For a `distinctive_structure` axis this
free wording is now IGNORED for anchor construction: `query_anchor =
"{label} {scientific_class}"`, unconditionally, for any entity_type in
_NAMED_SYSTEM_ENTITY_TYPES -- `scientific_class` is the axis's own single
already-canonical, already-deterministic phrase (the same string that
seeded Hop-1 and was checked by the relationship-fidelity gate in
transfer_entity_extraction.py, so asserting it of this entity is not
speculative -- for this basis every accepted mention already had that
relationship verified by evidence at extraction time). `identity_context`
is still carried on `RelatedSystemIdentity` and still decides the
`grounded_full` vs `grounded_contextual` grounding TIER (for audit), it
just no longer contributes to the query_anchor STRING. Deliberately no
subtype-modifier extraction (e.g. "trimeric" for a trimeric autotransporter)
-- simplicity and reproducibility take priority for now; add it later only
if a benchmark shows subtype specificity is actually needed. For every
OTHER basis, this projection does not apply at all -- there is no
equivalent relationship-fidelity gate for those bases (see transfer_
entity_extraction.py's basis-scoped gate), so asserting `scientific_class`
of an entity there would be an unproven claim. Those bases get exactly the
pre-existing behavior above (identity_context verbatim, or bare label).
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

_NAMED_SYSTEM_ENTITY_TYPES = {"specific_protein_or_system", "construct_or_variant"}


def _norm(s: str) -> str:
    return " ".join((s or "").strip().split()).lower()


class RelatedSystemIdentity(BaseModel):
    label: str                                    # Hop-1's own grounded name -- NEVER dropped/overwritten
    entity_type: Optional[str] = None             # as classified by transfer_entity_extraction.py, if any
    identity_context: Optional[str] = None        # verbatim-grounded broader-context phrase, if any
    grounding: Literal["grounded_full", "grounded_contextual", "insufficient"]
    query_anchor: Optional[str] = None            # what Hop-2 actually searches; None only for insufficient
    aliases: list[str] = Field(default_factory=list)
    source_grounding: list[str] = Field(default_factory=list)   # verbatim evidence_quote(s)
    source_paper_titles: list[str] = Field(default_factory=list)
    n_support: int = 0                            # deterministic support signal (NOT an LLM confidence score)
    first_seen_rank: int = 10**9


def build_related_system_identity(ds, *, scientific_class: str, basis: Optional[str] = None) -> RelatedSystemIdentity:
    """`ds` is a related_system_discovery.DiscoveredSystem -- typed loosely
    here (duck-typed on .name/.supporting_papers/.n_support/.first_seen_rank)
    so this module has no import-time dependency on that class's exact
    location, only on the shape Hop-1 already guarantees. No paper_cache
    needed -- entity_type/identity_context already rode along on each
    SupportingPaper, already grounding-verified by _ground_mention().

    `basis` is the axis's own axis.basis (transfer_retrieval_bridge.py
    threads it in -- not a new field, execution_mode dispatch is untouched).
    It exists ONLY to gate the canonical-class-projection query_anchor rule
    described in this module's own docstring to "distinctive_structure"
    axes -- see there for why. Default None (an older/other caller that
    never threads it) falls to the pre-existing identity_context-based
    behavior, unchanged."""
    label = ds.name
    source_grounding = [sp.evidence_quote for sp in ds.supporting_papers]
    source_paper_titles = [sp.title for sp in ds.supporting_papers]
    aliases = sorted({sp.name_as_extracted for sp in ds.supporting_papers
                       if sp.name_as_extracted and _norm(sp.name_as_extracted) != _norm(label)})

    # First mention that carries an entity_type wins -- same "first
    # claimant" convention already used elsewhere in this codebase (e.g.
    # RetrievedPaper.hop1_provenance). A DiscoveredSystem can merge mentions
    # from several papers; they are not expected to disagree in practice,
    # and this module makes no attempt to arbitrate if they do.
    entity_type: Optional[str] = None
    identity_context: Optional[str] = None
    for sp in ds.supporting_papers:
        if sp.entity_type:
            entity_type = sp.entity_type
            identity_context = sp.identity_context
            break

    if entity_type in _NAMED_SYSTEM_ENTITY_TYPES:
        grounding: Literal["grounded_full", "grounded_contextual", "insufficient"] = (
            "grounded_full" if identity_context else "grounded_contextual")
        if basis == "distinctive_structure":
            # Canonical-class projection (see module docstring): ignore
            # identity_context's own free wording for the query STRING --
            # the grounding tier above still reflects whether it was
            # present, for audit -- and always anchor on the axis's single
            # already-canonical, already-verified scientific_class instead.
            query_anchor = f"{label} {scientific_class}".strip()
        elif identity_context:
            query_anchor = f"{label} {identity_context}".strip()
        else:
            query_anchor = label
    else:
        # organism_or_species / protein_family_or_class / generic_descriptor
        # / entity_type never set (e.g. this axis's extraction failed or
        # used an extract_fn that doesn't produce it) -- abstain rather
        # than guess.
        grounding = "insufficient"
        query_anchor = None

    return RelatedSystemIdentity(
        label=label, entity_type=entity_type, identity_context=identity_context,
        grounding=grounding, query_anchor=query_anchor, aliases=aliases,
        source_grounding=source_grounding, source_paper_titles=source_paper_titles,
        n_support=ds.n_support, first_seen_rank=ds.first_seen_rank,
    )


def build_related_system_identities(
    discovered_systems: list, *, scientific_class: str, basis: Optional[str] = None,
) -> list[RelatedSystemIdentity]:
    return [build_related_system_identity(ds, scientific_class=scientific_class, basis=basis)
            for ds in discovered_systems]
