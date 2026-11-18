"""extraction -- richer Hop-1 per-paper extraction for the
transfer-aware related-system branch (P5).

Same ONE-call-per-paper extraction slot discovery.py's
discover_related_systems() uses (via its injectable extract_fn) -- NOT a
new LLM call. It asks that per-paper call for two fields the model must
genuinely judge from the text: entity_type and identity_context (plus the
class_membership_quote described below).

Backend-agnostic ("one run, one LLM backend" invariant): the SAME
`llm.LLMBackend` every other LLM-calling module takes is an explicit,
REQUIRED argument here (no default -- see llm.get_backend()'s own
docstring: no module may resolve or fall back to a backend on its own), and
this module calls its `.structured()` method.

Defined below: system_extraction_user_message() (pure input formatting,
schema-agnostic), the SystemMention / SystemExtraction shape the discovery layer
consumes, the deterministic structural_term_matched filler
_find_structural_term_matched(), and the _MAX_ABSTRACT_CHARS_FOR_EXTRACTION /
_EXTRACTION_TIMEOUT_SECONDS constants; the timeout is passed to the injected
backend via `LLMBackend.structured()`'s own `timeout` parameter.

Relationship validation is aligned with the transfer axis (revision after the
Q15-Q20 Hop-1 audit): the old gate -- a fixed list of fold/shape words -- was
written for the beta-helix/autotransporter case and deleted correct answers for
family-, process- and function-type classes (dynamin, MscS, ribosomes). Now the
user message carries a RELATIONSHIP CRITERION chosen by the axis basis (family /
architecture / physical process / chemistry / environment), and every listed
entity must come with a `class_membership_quote` -- a verbatim span of the
title/abstract showing that THIS entity meets that criterion. The only
deterministic check left is the same grounding discipline every other field gets:
the quote must be a verbatim substring of the paper text. No word list.

Design (per this session's explicit decisions):
  - LLM-facing schema stays as narrow as possible -- only name,
    evidence_quote, entity_type, identity_context.
  - NO confidence field: n_support (already computed by related_system_
    discovery.py from how many independent mentions/papers support a
    DiscoveredSystem) is the deterministic, auditable support signal --
    never an LLM-assigned number.
  - NO transfer_basis field: that's an axis-level property (axis.basis,
    already available), never needs re-deriving per mention.
  - NO LLM-generated retrieval_anchor: built deterministically in
    postprocess.py (identity) from entity_type + identity_context, once
    both have flowed through discovery.py's existing
    verbatim-grounding checks.

No side-channel dict anywhere in this design: entity_type/identity_context
ride on discovery.SupportingPaper itself (two new, additive,
Optional fields on that existing object -- see its own docstring comment),
via _ground_mention()'s existing getattr()-based pass-through, which also
verbatim-verifies identity_context against the paper's own text before
ever accepting it -- the same discipline evidence_quote/name already get.
"""
from __future__ import annotations

import json
from typing import Literal, Optional

from pydantic import BaseModel, Field


from cgscout import cache  # noqa: E402 -- dev-speed cache, own namespace below
from cgscout.llm import LLMBackend  # noqa: E402 -- the single-backend-per-run abstraction, injected by callers
from cgscout.retrieval.related_systems.discovery import _dnorm  # noqa: E402 -- normalization primitive used by _find_structural_term_matched (below)
from cgscout.retrieval.related_systems.terminology import _STRUCTURAL_MORPHEMES  # noqa: E402 -- reused, not duplicated

# --- Hop-1 per-paper extraction: input shape + user message (moved here from intake_contract.py, verbatim) -----------------------------------------------
# RelatedSystemDiscovery -- Hop-1 per-paper extraction. Given ONE paper's TITLE + ABSTRACT and a structural class, name the SPECIFIC system(s) THAT
# TEXT says have that structure. The name is later validated as a verbatim substring of the paper (deterministic grounding, in
# discovery.py). The model must NEVER add a name from its own knowledge -- only what the given text states.

class SystemMention(BaseModel):
    name: str                     # verbatim from the title/abstract
    evidence_quote: str           # shortest verbatim span naming the system AND the structure
    structural_term_matched: str  # the exact structural-class word/phrase in the text
    # ^ NO LONGER LLM-generated (schema-slimming pass, audited first -- see
    # run_hop1_stage4_schema_audit): the LLM only ever names a term that must
    # already be a verbatim substring of evidence_quote, checked against a
    # KNOWN, already-available list (structural_class + also_known_as) --
    # nothing here required the model's own judgment. mellea_adapter.py's
    # extract_related_systems_with_mellea() now asks the LLM for only `name`
    # + `evidence_quote` (see SystemMentionLLM below) and fills this field
    # deterministically afterward, in Python, before handing a full
    # SystemMention to discovery.py -- which is UNCHANGED,
    # still receives the exact same shape it always has and still runs the
    # exact same verbatim-grounding checks against this field.
    #
    # related_category / category_evidence_quote (the paused GroundedRelation
    # next-hop design's phase-1 calibration fields) REMOVED entirely, same
    # pass: grep across the whole codebase found zero production consumers
    # (only the paused design's own standalone scripts --
    # related_category_grounding.py / run_related_category_extraction_
    # calibration.py -- referenced them, and those two scripts will need
    # their own SystemMention(...) construction calls updated if that paused
    # track is ever resumed).


class SystemExtraction(BaseModel):
    systems: list[SystemMention] = Field(default_factory=list)


def system_extraction_user_message(title: str, abstract: str, structural_class: str,
                                   also_known_as: "list[str] | None" = None) -> str:
    aka = ""
    if also_known_as:
        aka = "\nRelated terms and constructed search phrases for this class: " + \
              ", ".join(f'"{t}"' for t in also_known_as) + "."
    return (
        f'CLASS: "{structural_class}"{aka}\n\n'
        f'TITLE: {title}\n\n'
        f'ABSTRACT: {abstract or "(no abstract available)"}'
    )


# --- Hop-1 helpers: structural-term filler, abstract-length cap, call timeout ----------------------------------------------------------------------

def _find_structural_term_matched(evidence_quote: str, structural_class: str, also_known_as) -> str:
    """Deterministic replacement for what the LLM used to report: which
    known structural-class term (structural_class itself, or one of
    also_known_as) is tied to this mention. The LLM was only ever allowed to
    name a term that must ALREADY be a verbatim (normalized) substring of
    evidence_quote, checked against this exact known list -- nothing about
    picking it required the model's own judgment. Longest term checked
    first so a more specific alias (e.g. "beta-helix adhesin") is preferred
    over a shorter one it contains (e.g. "beta-helix"). Reports the KNOWN
    term itself, not an extracted quote substring -- evidence_quote's own
    exact wording is untouched/unaffected elsewhere; this field's only real
    job in discovery.py's _ground_mention() is to be a
    normalized substring of evidence_quote, checked case/dash-insensitively
    regardless. Empty string if no known term matches verbatim (rare) --
    grounding's own independent _quote_ties_to_class() canon-normalized
    (Greek-letter-aware) fallback check still runs either way, so this can
    never silently break grounding, it only means this one field goes
    unpopulated for that one mention."""
    qn = _dnorm(evidence_quote)
    for term in sorted({structural_class, *(also_known_as or [])}, key=len, reverse=True):
        if term and _dnorm(term) in qn:
            return term
    return ""


# --- Hop-1 extraction robustness: abstract length cap + call timeout ------
#
# Diagnosed against a real case (cold-profile investigation):
# one retrieved paper's abstract, reconstructed by cgscout.sources.
# openalex._reconstruct_abstract() from OpenAlex's own abstract_inverted_
# index, came out 23,046 characters long and visibly garbled/repetitive --
# ~10-40x the observed normal range (500-2,500 chars). The
# reconstruction code itself is correct (a standard inverted-index
# rebuild); the SOURCE data for this one paper is malformed upstream, not
# something we can fix on our end. Feeding that text to the (earlier,
# local-model) backend caused a single Hop-1 extraction call to hang for
# ~300s before eventually failing anyway -- and because a failed call
# is never cached (cache.py's own design: a failure must never be mistaken
# for "this input genuinely has no result"), this exact ~300s cost recurred
# on EVERY run, cold or warm, silently (already-existing extraction_errors
# bookkeeping just recorded "1 failure," never that it cost 5 minutes).
#
# Two independent, narrowly-scoped safeguards -- Hop-1 extraction ONLY:
#
#   1. _MAX_ABSTRACT_CHARS_FOR_EXTRACTION -- truncates ONLY what THIS
#      function sends to the LLM. discovery.py's own
#      grounding (_ground_mention, via _paper_text) always reads the
#      paper's REAL, untruncated abstract from the Paper object directly --
#      completely unaffected by this cap, so grounding integrity for
#      whatever a truncated call DOES return is unchanged.
#   2. _EXTRACTION_TIMEOUT_SECONDS -- passed as `timeout` to
#      LLMBackend.structured(), bounds the worst-case wall time of ONE
#      call. discover_related_systems() already treats a failed extraction
#      gracefully (this paper contributes 0 systems, logged to
#      extraction_errors, the rest of the batch unaffected) -- this only
#      changes HOW LONG that failure takes to surface (~300s -> a few
#      seconds past the bound), never what happens after it. A dedicated,
#      single-use ThreadPoolExecutor per call (not a shared pool) with
#      shutdown(wait=False) on timeout, so an abandoned/hung call is never
#      waited on and can never starve or block any other call.
_MAX_ABSTRACT_CHARS_FOR_EXTRACTION = 4000
_EXTRACTION_TIMEOUT_SECONDS = 20

# --- end of the Hop-1 helper block ---------------------------------------------------------------------------------------------------------------

# A handful of entries in the shared morpheme list (terminology.
# _STRUCTURAL_MORPHEMES -- used elsewhere as discovery.py's
# _quote_ties_to_class() loose, backward-compatible fallback for the OLD
# pipeline) are too common in ordinary protein-biology prose to indicate
# genuine structural-CLASS identity by themselves: "domain", "membrane",
# "class", "family", "superfamily", "motif", "topology", "architecture",
# "turn", "loop" all appear in virtually any protein abstract regardless of
# fold (e.g. CyaA's OWN "receptor-binding domain" -- a domain, yes, but not
# evidence of a beta-helix/autotransporter relationship). Excluded ONLY for
# this additive, rich-extraction-only relationship-fidelity gate below --
# the shared list, and the old pipeline's looser use of it, are untouched.
_TOO_GENERIC_ALONE = {
    "domain", "membrane", "class", "family", "superfamily", "motif",
    "topology", "architecture", "turn", "loop",
}
_SPECIFIC_STRUCTURAL_MORPHEMES = tuple(
    m for m in _STRUCTURAL_MORPHEMES if m not in _TOO_GENERIC_ALONE
)


def _quote_supports_structural_relationship(evidence_quote: str, structural_term_matched: str) -> bool:
    """Deterministic relationship-fidelity gate (no LLM, no new field): the
    evidence_quote must contain either a real, known structural-class term
    (structural_term_matched -- structural_class itself or an also_known_as
    alias, already verbatim-checked by _find_structural_term_matched) OR a
    genuinely shape-specific morpheme (the trimmed list above). A quote that
    names an entity using only generic biology vocabulary unrelated to this
    specific structural class -- no known term, no shape-specific morpheme,
    just something like a bare "domain" -- is not evidence that entity
    shares (or belongs to) this transfer class; such a mention is dropped
    rather than accepted on a weak family resemblance."""
    if structural_term_matched:
        return True
    qc = evidence_quote.lower()
    return any(m in qc for m in _SPECIFIC_STRUCTURAL_MORPHEMES)

# Own cache namespace -- distinct prompt/schema, must never collide with
# the entries of any other extraction.
_CACHE_NAMESPACE = "hop1_extract_rich"


EntityType = Literal[
    "specific_protein_or_system",
    "protein_family_or_class",
    "organism_or_species",
    "generic_descriptor",
    "construct_or_variant",
]


class RichSystemMentionLLM(BaseModel):
    name: str
    evidence_quote: str
    entity_type: EntityType
    identity_context: Optional[str] = None
    class_membership_quote: str = ""


class RichSystemExtractionLLM(BaseModel):
    systems: list[RichSystemMentionLLM] = Field(default_factory=list)


class RichSystemMention(BaseModel):
    """Shape-compatible with intake_contract.SystemMention -- has the same
    .name / .evidence_quote / .structural_term_matched attributes related_
    system_discovery.py's _ground_mention() reads -- plus the two new
    fields it additionally passes through via getattr()."""
    name: str
    evidence_quote: str
    structural_term_matched: str
    entity_type: str
    identity_context: Optional[str] = None
    class_membership_quote: Optional[str] = None     # verbatim span showing the axis criterion is met (validated, see extract_related_systems_rich)


class RichSystemExtraction(BaseModel):
    systems: list[RichSystemMention] = Field(default_factory=list)


RICH_SYSTEM_EXTRACTION_SYSTEM_PROMPT = """\
You are given the TITLE and ABSTRACT of ONE scientific paper, a CLASS \
(a structural class such as "ankyrin-repeat domain", a molecular family, a physical \
process, a chemistry or an environment), and a RELATIONSHIP CRITERION that \
says what kind of relationship to that class counts for this search.

List every SPECIFIC protein, protein complex, protein domain, or named \
biological system that THIS TEXT explicitly describes as HAVING -- or being \
an example of -- that class (or a term the text uses for the same class), \
in the sense given by the RELATIONSHIP CRITERION.

For each, produce:
- name: the specific name, copied VERBATIM from the title or abstract \
(character for character -- do not expand abbreviations, do not add a \
species, do not normalise capitalisation or hyphens)
- evidence_quote: the SHORTEST verbatim span from the title or abstract \
that BOTH names the system AND ties it to the class
- entity_type: classify what KIND of thing `name` refers to, using ONLY \
what this text states:
  - "specific_protein_or_system": a specific, individually-named protein, \
protein complex, or protein domain (e.g. a named enzyme, receptor, or domain \
construct).
  - "protein_family_or_class": a family or class name covering multiple \
members, not one specific individuated system.
  - "organism_or_species": an organism, species, or genus name -- NOT \
itself a protein or biological system.
  - "generic_descriptor": a non-specific structural/relational word with no \
independent identity (e.g. a bare adjective like "trimeric", or "the \
core") -- this should rarely be extracted as `name` at all, but classify it \
this way if it is.
  - "construct_or_variant": a specific experimental construct, mutant, \
deletion, truncation, or fusion explicitly built from a named protein.
- class_membership_quote: the SHORTEST verbatim span from the title or \
abstract showing that THIS entity satisfies the RELATIONSHIP CRITERION (it \
may be the same span as evidence_quote). If the text contains no such span \
for this entity, do NOT list the entity at all.
- identity_context: ONLY when entity_type is "specific_protein_or_system" \
or "construct_or_variant", AND this SAME text states what broader \
class/context this entity belongs to (e.g. calling it "the ankyrin-repeat protein \
X" states "ankyrin-repeat protein" as X's context) -- copy that context phrase \
VERBATIM from the text. Leave this null if entity_type doesn't call for it, \
or if the text simply does not state a broader context for this entity. \
Never infer or supply a context from your own outside knowledge -- only \
what this text itself says.

Rules -- these are hard:
- Use ONLY what is written in the TITLE and ABSTRACT below. Do NOT use any \
knowledge from outside this text. If you are relying on what you already \
know about a protein rather than on what this text says, do NOT list it.
- Do NOT list a protein the text mentions for a DIFFERENT reason -- a \
comparison, a control, a method, a downstream application, a homolog named \
only in passing, a fusion/chimera CARGO grafted onto a system that has this \
structure, or a protein secreted/transported/studied THROUGH a system with \
this structure -- unless the text says THAT protein itself has this \
structure. Being attached to, carried by, or processed alongside a system \
with this structure does not make an entity an example of it.
- A generic phrase with no specific name ("the protein", "this enzyme", "a \
bacterial adhesin", "many virulence factors", "several proteins") is NOT a \
specific system -- skip it.
- The class itself ("ankyrin-repeat proteins", "an alpha-solenoid fold", "a \
beta-sandwich architecture", "serpin-family protease inhibitors") is NOT a system name -- skip \
it. A specific, individually-named member of the class (e.g. one named \
protein) IS a system name, even when its name also appears inside the \
class label.
- Apply the RELATIONSHIP CRITERION exactly as written: it is the test for \
whether an entity belongs on the list, and the text itself must show it.
- If the text names no specific system with this structure, return an \
EMPTY list. An empty list is a correct, expected answer. Padding the list \
with guesses is a failure.

Respond with a single JSON object matching the given schema, and nothing else."""


# What counts as "related to the class" depends on WHY the class was proposed (the transfer axis). Natural-language
# criteria for the extraction prompt -- not match lists -- so evidence is judged by what the axis claims.
AXIS_RELATIONSHIP_CRITERIA = {
    "family_homology": (
        "The entity must belong to the same molecular family as the class, or be a homolog, ortholog or evolutionarily "
        "related member of it. The text itself must state or clearly describe that family membership or homology."),
    "distinctive_structure": (
        "The entity must share the architecture or structural organization that defines the class (for example its "
        "fold, domain composition or assembly geometry). The text itself must describe that structure for this entity."),
    "physical_process": (
        "The entity must undergo or carry out the same physical process or mechanism that defines the class (for "
        "example the same kind of mechanical, thermodynamic, assembly or transport process). The text itself must "
        "describe that process for this entity."),
    "chemistry_representation": (
        "The entity must share the chemical composition, chemistry or molecular-representation issue that defines "
        "the class. The text itself must state that chemistry for this entity."),
    "environment_context": (
        "The entity must be studied in, or defined by, the same environment or context that defines the class (for "
        "example the same membrane, crowding or cellular setting). The text itself must describe that context for "
        "this entity."),
}
_DEFAULT_RELATIONSHIP_CRITERION = (
    "The text itself must explicitly describe the entity as belonging to, or being an example of, the class.")


def relationship_criterion(basis: Optional[str]) -> str:
    return AXIS_RELATIONSHIP_CRITERIA.get(basis or "", _DEFAULT_RELATIONSHIP_CRITERION)


def _norm_text(s: str) -> str:
    return " ".join((s or "").lower().split())


def _membership_is_grounded(mention: "RichSystemMention", title: str, abstract: str) -> bool:
    """The one deterministic check on the axis relationship: the model must supply a membership quote and it must be a
    verbatim substring of the paper's own title/abstract. (No word list; the axis criterion itself is judged by the
    model, the quote proves it did not invent the evidence.)"""
    q = _norm_text(mention.class_membership_quote or "")
    return bool(q) and q in _norm_text(f"{title} {abstract}")


_RICH_PROMPT_HASH = cache.content_hash(RICH_SYSTEM_EXTRACTION_SYSTEM_PROMPT)
_RICH_SCHEMA_HASH = cache.content_hash(
    json.dumps(RichSystemExtractionLLM.model_json_schema(), sort_keys=True))


def extract_related_systems_rich(
    title: str, abstract: str, structural_class: str, also_known_as=None,
    *, backend: LLMBackend, basis: Optional[str] = None, case_id: Optional[str] = None,
) -> RichSystemExtraction:
    """Drop-in extract_fn for discovery.discover_related_
    systems() -- same one-call-per-paper slot, with abstract truncation, a
    call timeout and a dev-cache (own cache namespace). Raises on failure --
    the caller (discover_related_systems) treats a failed paper as
    contributing nothing.

    `backend` is REQUIRED, no default -- the single LLMBackend this whole
    CGScout run/session already resolved once (llm.get_backend(),
    called by the top-level entry point, threaded down through every
    caller in between). This function never resolves or falls back to a
    backend on its own -- see llm.get_backend()'s own docstring on
    the "one run, one backend" invariant. `case_id` just rides through to
    the usage log, same as every other .structured() call already does.

    `basis` is the axis's own axis.basis (agent.py threads
    it in through a closure; execution_mode dispatch still reads only
    axis.execution_mode). It selects the RELATIONSHIP CRITERION appended to
    the user message (AXIS_RELATIONSHIP_CRITERIA: family / architecture /
    physical process / chemistry / environment; None or an unknown basis gets
    a generic "belongs to the class" criterion), so it is part of the cache
    key through the user message -- a result produced under one criterion is
    never served for another. Every returned mention carries a validated
    `class_membership_quote` (verbatim in the title/abstract); a mention
    without one is dropped, for every basis. That post-filter runs AFTER the
    cache get/set below, over the same cached raw extraction."""
    abstract = (abstract or "")[:_MAX_ABSTRACT_CHARS_FOR_EXTRACTION]
    user_message = (system_extraction_user_message(title, abstract, structural_class, also_known_as)
                    + f"\n\nRELATIONSHIP CRITERION: {relationship_criterion(basis)}")
    # Cache key includes backend.name/backend.model -- extractions of the SAME
    # paper by different backends/models must never collide (they are
    # genuinely different model outputs), so switching backend or model is
    # itself a natural cache-cold-start, never a stale cross-backend hit.
    cache_key = cache.content_hash(
        _RICH_PROMPT_HASH, _RICH_SCHEMA_HASH, user_message, backend.name, backend.model)
    cached = cache.get(_CACHE_NAMESPACE, cache_key)
    raw_result: Optional[RichSystemExtraction] = None
    if cached is not None:
        try:
            raw_result = RichSystemExtraction.model_validate(cached)
        except Exception:  # noqa: BLE001 -- a cached shape that no longer validates is a miss, not a crash
            raw_result = None

    if raw_result is None:
        raw = backend.structured(
            system=RICH_SYSTEM_EXTRACTION_SYSTEM_PROMPT, user=user_message,
            schema=RichSystemExtractionLLM, label="hop1_extract_rich",
            purpose="hop1_related_system_extraction", case_id=case_id,
            timeout=_EXTRACTION_TIMEOUT_SECONDS,
        )  # already a validated RichSystemExtractionLLM instance -- .structured()'s own contract
        raw_result = RichSystemExtraction(systems=[
            RichSystemMention(
                name=m.name, evidence_quote=m.evidence_quote,
                structural_term_matched=_find_structural_term_matched(m.evidence_quote, structural_class, also_known_as),
                entity_type=m.entity_type, identity_context=m.identity_context,
                class_membership_quote=m.class_membership_quote,
            )
            for m in raw.systems
        ])
        cache.set(_CACHE_NAMESPACE, cache_key, raw_result.model_dump(mode="json"))

    # Axis-aligned relationship validation (replaces the old distinctive_structure-only shape-word gate): every mention,
    # whatever the axis, must carry a class_membership_quote that is verbatim in the paper. Applied AFTER the cache
    # get/set as a pure post-filter over the same cached raw extraction.
    kept = [m for m in raw_result.systems if _membership_is_grounded(m, title, abstract)]
    return RichSystemExtraction(systems=kept)
