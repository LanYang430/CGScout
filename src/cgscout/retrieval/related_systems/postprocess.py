"""postprocess -- what runs on Hop-1's result between discovery and Hop-2, in this order: the conservative cleanup of the discovered systems, then the
identity preservation that builds the Hop-2 query anchors.  Merged (consolidation A4) from two modules whose code is moved here unchanged; each section keeps
the module docstring it had.  One mechanical rename: identity's private `_norm` is `_identity_norm` here, because it is NOT discovery._norm (the other
one, imported below, is what cleanup uses): they normalize differently, so they must not be unified silently.

CLEANUP (formerly cleanup.py)
Hop-1 conservative cleanup: high-confidence duplicate/fragment handling
ONLY. Called from orchestrator.py, strictly between
discover_related_systems() (Hop-1 discovery + grounding, UNTOUCHED) and the
Hop-2 query building (UNTOUCHED) -- so
Hop-1's own recall behavior is provably unaffected by anything in this
module; this is a post-hoc cleanup layer over its RESULT, not a change to
how that result is produced.

Deliberately does NOT judge whether a candidate is a "concrete system" --
that boundary is unstable (diagnosed this session: Chlamydia/pylori/CRD/TPS/
short+long repeats(SRs/LRs)/Polymers are all left untouched on purpose, even
though some look borderline). Only three narrow, high-confidence
transformations:

  1. suffix duplicate merge -- a candidate whose normalized name ends, at a
     whole-word boundary, with an already-surviving system's normalized
     name (e.g. "of the Escherichia coli Autotransporter EspP" ends with
     "EspP") merges into it. Pure string containment, no semantic judgment
     about what the extra words mean.
  2. exact fragment blocklist -- a candidate whose (normalized) name is
     EXACTLY one of a small, hand-curated, closed set of pure relational/
     connective words that carry no naming content by themselves. Seeded
     ONLY with real observed cases ("related", "double-stranded") -- expand
     this set only by adding another CONFIRMED observed case, never by
     guessing plausible-sounding analogues ("single-stranded" is a
     predictable sibling but has NOT been observed, so it deliberately
     stays out).
  3. objective-aware interaction-duplicate merge -- a "{known system}...
     {interaction/complex/binding word}"-shaped candidate (e.g. "EspP-BamA
     interaction") folds back into the known system it names, but ONLY when
     the objective is classified (by a small closed keyword list, no LLM)
     as a monomer/structural-process objective (folding, unfolding,
     conformational stability, ...). For an interaction/binding-type
     objective -- or an objective this module cannot confidently classify
     -- the candidate is left standalone: interaction may BE the thing
     being studied, and this module must never guess wrong toward deleting
     evidence for it.

Every merge/drop is recorded in the returned log entry list -- never
silent; provenance (supporting_papers / evidence_quote / paper_ref) is
never discarded by a merge, only consolidated onto the surviving name.

IDENTITY (formerly identity.py)
identity (scientific-planner-v2 branch) -- deterministic
identity preservation for Hop-1-discovered related systems. NEW, additive
module. No LLM call of its own -- consumes entity_type/identity_context
already produced by extraction.py's extraction call and
carried, grounding-verified, on discovery.SupportingPaper.

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
extraction.py, so asserting it of this entity is not
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

import re
from typing import Literal, Optional

from pydantic import BaseModel, Field

from cgscout.retrieval.related_systems.discovery import DiscoveredSystem, _dnorm, _norm  # noqa: E402 -- reused, not reimplemented


# ===== cleanup: merged verbatim from `cleanup.py` ============================================================================================
__all__ = ["cleanup_discovered_systems"]


# --- 2. exact fragment blocklist -------------------------------------------
# Closed set. Every entry is a REAL case observed this session surviving
# Hop-1 grounding despite carrying zero naming content on its own (a bare
# connective/relational adjective torn out of the sentence immediately
# around the structural-class mention). See module docstring: expand only
# by adding another CONFIRMED case.
_FRAGMENT_BLOCKLIST = {_dnorm(x) for x in ("related", "double-stranded")}


# --- 1. suffix duplicate merge ---------------------------------------------
_MIN_MERGE_TARGET_ALNUM_LEN = 4  # a candidate shorter than this can never be a merge TARGET


def _alnum_len(s: str) -> int:
    return len(re.sub(r"[^a-z0-9]", "", s.lower()))


def _ends_with_word_boundary(a: str, b: str) -> bool:
    """True if normalized `a` ends with normalized `b` at a whole-word
    boundary (not merely as a raw substring) -- "...autotransporter espp"
    ends with "espp" this way; a hypothetical "prespp" would not."""
    if not a or not b or not a.endswith(b):
        return False
    if len(a) == len(b):
        return True
    return not a[-(len(b) + 1)].isalnum()


def _contains_word_boundary(haystack: str, needle: str) -> bool:
    if not needle:
        return False
    pattern = rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])"
    return re.search(pattern, haystack) is not None


def _merge_supporting_papers(target: DiscoveredSystem, source: DiscoveredSystem) -> None:
    """Appends `source`'s supporting papers onto `target`'s -- never
    prepended, so target.supporting_papers[0] (used elsewhere as "the"
    primary evidence for this system) always stays the target's own
    original evidence, never the absorbed duplicate's. Dedupes by
    paper_ref so the same paper is never listed twice."""
    existing_refs = {sp.paper_ref for sp in target.supporting_papers}
    for sp in source.supporting_papers:
        if sp.paper_ref not in existing_refs:
            target.supporting_papers.append(sp)
            existing_refs.add(sp.paper_ref)
    target.first_seen_rank = min(target.first_seen_rank, source.first_seen_rank)
    target.n_support = len(target.supporting_papers)


def _merge_suffix_duplicates(systems: list[DiscoveredSystem], log: list[str]) -> list[DiscoveredSystem]:
    # Process SHORTEST names first, so a short/simple name is already an
    # established candidate TARGET by the time a longer candidate is
    # checked against it (a longer candidate needs the short one to
    # already be present in `survivors` to find it).
    ordered = sorted(systems, key=lambda d: _alnum_len(d.norm_name))
    survivors: list[DiscoveredSystem] = []
    for d in ordered:
        target = None
        for s in survivors:
            if s.norm_name == d.norm_name:
                continue  # identical normalized names are already merged upstream (Hop-1's own dict-by-key merge)
            if _alnum_len(s.norm_name) < _MIN_MERGE_TARGET_ALNUM_LEN:
                continue
            if _ends_with_word_boundary(_dnorm(d.norm_name), _dnorm(s.norm_name)):
                target = s
                break
        if target is not None:
            _merge_supporting_papers(target, d)
            log.append(
                f"duplicate_merge: {d.name!r} -> {target.name!r} "
                f"(normalized name ends with existing system name at a word boundary)"
            )
        else:
            survivors.append(d)
    return survivors


# --- 3. objective-aware interaction-duplicate merge -------------------------
_INTERACTION_WORDS = ("interaction", "interactions", "complex", "complexes", "binding", "interface")

_MONOMER_STRUCTURAL_OBJECTIVE_KEYWORDS = (
    "folding", "unfolding", "refolding", "misfolding", "fold",
    "conformational stability", "conformation", "stability", "denaturation",
    "structural stability",
)
_INTERACTION_OBJECTIVE_KEYWORDS = (
    "binding", "interaction", "protein-protein interaction", "complex formation",
    "translocation", "docking", "recognition", "assembly",
)


def _objective_allows_interaction_merge(clean_objective: str) -> bool:
    """Conservative, closed-keyword-only classification -- no LLM, no new
    classifier. If the objective mentions ANY interaction-type keyword, an
    interaction candidate is always left standalone (interaction may BE the
    thing being studied). Merge is allowed ONLY when the objective clearly
    names a monomer/structural-process keyword AND mentions none of the
    interaction-type keywords. An empty, unrecognized, or ambiguous
    objective defaults to NOT merging -- the safer direction when uncertain."""
    obj = _norm(clean_objective)
    if not obj:
        return False
    if any(kw in obj for kw in _INTERACTION_OBJECTIVE_KEYWORDS):
        return False
    return any(kw in obj for kw in _MONOMER_STRUCTURAL_OBJECTIVE_KEYWORDS)


def _merge_interaction_duplicates(
    systems: list[DiscoveredSystem], clean_objective: str, log: list[str],
) -> list[DiscoveredSystem]:
    if not _objective_allows_interaction_merge(clean_objective):
        return systems
    known = [d for d in systems if _alnum_len(d.norm_name) >= _MIN_MERGE_TARGET_ALNUM_LEN]
    survivors: list[DiscoveredSystem] = []
    for d in systems:
        dn = _dnorm(d.norm_name)
        has_interaction_word = any(_contains_word_boundary(dn, w) for w in _INTERACTION_WORDS)
        target = None
        if has_interaction_word:
            # Existing rank order (systems is already ranked coming in)
            # decides which known name wins if a candidate happens to name
            # more than one -- first match in that order, always logged so
            # it stays inspectable rather than silently arbitrary.
            for k in known:
                if k.norm_name == d.norm_name:
                    continue
                if _contains_word_boundary(dn, _dnorm(k.norm_name)):
                    target = k
                    break
        if target is not None:
            _merge_supporting_papers(target, d)
            log.append(
                f"interaction_merge: {d.name!r} -> {target.name!r} "
                f"(contains known system name + interaction-type word; "
                f"objective classified as monomer/structural-process)"
            )
        else:
            survivors.append(d)
    return survivors


# --- orchestrator -----------------------------------------------------------

def cleanup_discovered_systems(
    discovered_systems: list[DiscoveredSystem], *, clean_objective: str,
) -> tuple[list[DiscoveredSystem], list[str]]:
    """Applies, in order: (1) exact fragment blocklist, (2) suffix duplicate
    merge, (3) objective-aware interaction-duplicate merge -- then re-ranks
    survivors by the SAME (-n_support, first_seen_rank) key
    discover_related_systems() already uses (reused, not a new ranking
    rule: merging can raise a survivor's n_support, e.g. EspP 1->2, so
    re-applying the existing tie-break after merging is necessary for the
    result to stay correctly ordered, not an added scientific judgment).

    Order matters: the fragment filter runs FIRST so a blocklisted word
    (e.g. "related") can never itself become a duplicate-merge TARGET
    before being dropped.

    Returns (cleaned_systems, audit_log) -- audit_log has one line per
    merge/drop, never silent."""
    log: list[str] = []
    systems = list(discovered_systems)

    kept = []
    for d in systems:
        if _dnorm(d.name) in _FRAGMENT_BLOCKLIST or _dnorm(d.norm_name) in _FRAGMENT_BLOCKLIST:
            log.append(f"fragment_drop: {d.name!r} (exact match in closed fragment blocklist)")
            continue
        kept.append(d)
    systems = kept

    systems = _merge_suffix_duplicates(systems, log)
    systems = _merge_interaction_duplicates(systems, clean_objective, log)

    systems = sorted(systems, key=lambda s: (-s.n_support, s.first_seen_rank))
    return systems, log


# ===== identity: merged from `identity.py` (code unchanged; `_norm` -> `_identity_norm`, see the module docstring) =============================
_NAMED_SYSTEM_ENTITY_TYPES = {"specific_protein_or_system", "construct_or_variant"}


def _identity_norm(s: str) -> str:
    return " ".join((s or "").strip().split()).lower()


class RelatedSystemIdentity(BaseModel):
    label: str                                    # Hop-1's own grounded name -- NEVER dropped/overwritten
    entity_type: Optional[str] = None             # as classified by extraction.py, if any
    identity_context: Optional[str] = None        # verbatim-grounded broader-context phrase, if any
    grounding: Literal["grounded_full", "grounded_contextual", "insufficient"]
    query_anchor: Optional[str] = None            # what Hop-2 actually searches; None only for insufficient
    aliases: list[str] = Field(default_factory=list)
    source_grounding: list[str] = Field(default_factory=list)   # verbatim evidence_quote(s)
    source_paper_titles: list[str] = Field(default_factory=list)
    n_support: int = 0                            # deterministic support signal (NOT an LLM confidence score)
    first_seen_rank: int = 10**9


def build_related_system_identity(ds, *, scientific_class: str, basis: Optional[str] = None) -> RelatedSystemIdentity:
    """`ds` is a discovery.DiscoveredSystem -- typed loosely
    here (duck-typed on .name/.supporting_papers/.n_support/.first_seen_rank)
    so this module has no import-time dependency on that class's exact
    location, only on the shape Hop-1 already guarantees. No paper_cache
    needed -- entity_type/identity_context already rode along on each
    SupportingPaper, already grounding-verified by _ground_mention().

    `basis` is the axis's own axis.basis (orchestrator.py
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
                       if sp.name_as_extracted and _identity_norm(sp.name_as_extracted) != _identity_norm(label)})

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
