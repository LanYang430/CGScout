"""Evidence Route Planner: Multi-route evidence discovery planning.

Given grounded system + objective from intake, identify which evidence dimensions
would make prior CG work informative. Routes are independent:
  1. direct_system: what the user explicitly named
  2. process: the scientific process/property being studied
  3. context: grounded physical context (membrane, environment, etc.)
  4. modeling_requirement: what simulation capability would be needed
  5. structural: meaningful fold classes (can be empty!)

Structural terms: USER_GROUNDED or LITERATURE_GROUNDED only. No LLM inference
about what fold a named protein belongs to.

Modeling requirements: Always PLANNER_DERIVED + explicit justification. Never
represented as extracted literature evidence.

Grounding principle:
  no grounded meaningful structural feature → no structural route
"""
from __future__ import annotations

import re
from typing import Optional, Literal, Dict, Any, Callable
from pydantic import BaseModel, Field

# Reused, NOT duplicated: the same closed, morphology-based structural
# vocabulary and generic-head/connective word sets Hop-1's class-modifier
# stripping already uses (related_system_discovery._strip_class_modifier is
# the inverse operation of the lexical extraction below -- that peels a
# structural modifier OFF a name; this recovers the modifier ITSELF from
# grounded system text). Importing these rather than re-declaring a second,
# possibly-drifting copy is the point: one structural vocabulary for the
# whole codebase, not a growing biology ontology invented here.
from cgscout.retrieval.related_systems.terminology import _STRUCTURAL_MORPHEMES  # noqa: E402
from cgscout.retrieval.related_systems.discovery import _CLASS_HEAD_WORDS, _CONNECTIVES  # noqa: E402

_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", (s or "").strip().lower())


class RouteTerm(BaseModel):
    """A single evidence route term with full provenance."""
    term: str                           # the actual search term
    route_type: Literal[
        "direct_system",
        "process",
        "context",
        "modeling_requirement",
        "structural"
    ]
    source: Literal[
        "USER_GROUNDED",
        "LITERATURE_GROUNDED",
        "PLANNER_DERIVED"
    ]
    supporting_span: Optional[str] = None      # if grounded: verbatim source text
    provenance: Optional[Dict[str, Any]] = None  # source details: paper DOI, user turn, fact_id
    confidence: Optional[Literal["high", "exploratory"]] = None  # required for PLANNER_DERIVED
    justification: Optional[str] = None        # required for PLANNER_DERIVED (why this term?)


class EvidenceRoutePlan(BaseModel):
    """Complete multi-route evidence plan."""
    scientific_objective: str           # normalized user objective
    system_description: str             # normalized system

    # Five independent routes
    direct_system_terms: list[RouteTerm] = Field(default_factory=list)
    process_terms: list[RouteTerm] = Field(default_factory=list)
    context_terms: list[RouteTerm] = Field(default_factory=list)
    modeling_requirement_terms: list[RouteTerm] = Field(default_factory=list)
    structural_terms: list[RouteTerm] = Field(default_factory=list)

    # Metadata
    route_justification: Dict[str, Optional[str]] = Field(default_factory=dict)
    # e.g., {"structural": None} if no structural route was justified
    empty_routes: list[str] = Field(default_factory=list)
    # e.g., ["structural", "context"] if those routes found no grounding

    # ScientificPlanner.search_axes (complete scientific concepts the planner already produced); read only by Query Construction v2
    search_axes: list[str] = Field(default_factory=list)


# --- Deterministic extraction: direct_system and process ------------------

def _extract_direct_system_terms(system_description: str) -> list[RouteTerm]:
    """Direct system: what the user explicitly said.

    In full pipeline, this comes from A's FactType.SYSTEM_IDENTITY.
    Here (standalone), we take the user input as-is.
    """
    if not system_description or not system_description.strip():
        return []
    return [
        RouteTerm(
            term=system_description.strip(),
            route_type="direct_system",
            source="USER_GROUNDED",
            supporting_span=system_description.strip(),
            provenance={"source": "user_input"}
        )
    ]


def _extract_process_terms_primary(objective: str) -> list[RouteTerm]:
    """Primary process term: the normalized scientific objective (deterministic).

    In full pipeline, this comes from B's grounded scientific_objective.
    Here (standalone), we take the objective as provided.
    """
    if not objective or not objective.strip():
        return []
    return [
        RouteTerm(
            term=objective.strip(),
            route_type="process",
            source="USER_GROUNDED",
            supporting_span=objective.strip(),
            provenance={"source": "user_objective"}
        )
    ]


# --- Structural: USER_GROUNDED or LITERATURE_GROUNDED only ----------------

_MAX_MODIFIER_WORDS = 3  # bounds a runaway leftward walk; "left-handed beta-helix"-scale phrases fit

# Case B's edge-trimming set (see extract_lexical_structural_feature below):
# a deliberately NARROWER subset of _CONNECTIVES, not the whole set reused
# mechanically. _CONNECTIVES mixes two different jobs today: a pure empty
# wrapper (an article/determiner/example-marker -- stripping it changes
# nothing about what's left) and a word that, at the EDGE of a span with
# nothing on the other side, would silently change the claim if stripped
# rather than merely trim it. Case A only ever uses _CONNECTIVES as a STOP
# boundary during its backward walk (never crosses one, never removes one),
# so this distinction never mattered there -- Case B is different: it
# actively deletes edge tokens, so it needs the narrower list.
#
# Excluded, and why (never edge-trimmed by Case B):
#   "and", "or"      -- coordinating conjunctions; a dangling one at an edge
#                        signals an incomplete/conjoined fragment ("beta
#                        helix and ...", "... and beta helix"), not a clean
#                        wrapper -- silently dropping it could as easily
#                        paper over a genuinely truncated span as trim a
#                        harmless one.
#   "of", "with", "in" -- relational prepositions connecting two distinct
#                        things ("domain OF X", "protein WITH X"); stripping
#                        one at an edge risks discarding the very relation
#                        the text was expressing, not just filler.
#   "like", "based"   -- similarity/derivation markers ("beta-helix-LIKE",
#                        "beta-helix-BASED fold") -- these change the CLAIM
#                        itself (resembles X / built on X) rather than just
#                        decorating it; stripping one would turn "resembles
#                        a beta-helix" into "IS a beta-helix", a real
#                        semantic change, not a trim.
#   "containing"      -- same relational-gerund concern as "of"/"with".
#   "type", "types", "class" -- classifier suffixes ("beta-helix TYPE",
#                        "beta-helix CLASS") that mean "a kind of X", not X
#                        itself -- same derivation concern as "based".
# Kept (pure empty wrapper, safe to delete): "the"/"a"/"an" (articles),
# "other"/"others" (quantifier, doesn't relate two things), "such"/"as"/"eg"
# (example-introducing markers with no relational content of their own).
_EDGE_WRAPPER_TOKENS = _CONNECTIVES - {
    "and", "or", "of", "with", "in", "like", "based", "containing",
    "type", "types", "class",
}


def extract_lexical_structural_feature(text: str) -> Optional[str]:
    """Narrow deterministic EXTRACTION (never inference) of an explicit
    structural-class phrase already present verbatim in grounded text.

    Hard invariant: explicit structural wording present in the grounded text
    -> may extract. A structural feature known only from biology/domain
    knowledge (e.g. "Pertactin" IS a beta-helix protein, "GPCR"s ARE membrane
    receptors) -> must NOT extract, because that knowledge is not lexically
    present in the text itself.

    Algorithm: find a generic class-head word (_CLASS_HEAD_WORDS -- "protein",
    "enzyme", "domain", "fold", ...) and walk backward from it collecting
    words, stopping at the first connective/article (_CONNECTIVES -- "a",
    "the", "and", "of", ...) or after _MAX_MODIFIER_WORDS. If at least one
    collected word contains a structural morpheme (_STRUCTURAL_MORPHEMES --
    "helix", "barrel", "coil", "sheet", ...), the collected run IS the
    structural feature -- returned as a VERBATIM substring of `text`
    (original casing/hyphenation preserved), not reconstructed.

    "beta-helix protein" -> "beta-helix" (helix is a morpheme)
    "TIM barrel enzyme" -> "TIM barrel" (barrel is a morpheme; TIM rides
      along because it's contiguous with barrel and not a connective)
    "coiled-coil protein" -> "coiled-coil" (coil is a morpheme)
    "I have a beta-helix protein and want to study folding" -> "beta-helix"
      (the walk stops at "a", so filler words never enter the run)
    "Pertactin" / "Wnt-Wntless" / "GPCR" / "lysozyme" -> None (no
      _CLASS_HEAD_WORDS token present at all -- nothing to anchor a walk on)

    Case B (fallback, only tried when Case A above finds nothing at all):
    the grounded text has no class-head word to anchor on, but IS ITSELF,
    once a purely-empty leading/trailing wrapper token (_EDGE_WRAPPER_TOKENS
    -- a narrower, explicitly curated subset of _CONNECTIVES; see that
    constant's own comment for exactly which words were excluded and why)
    is trimmed off both ends, a short (<= _MAX_MODIFIER_WORDS) span
    containing a structural morpheme. Handles the case where the grounded
    system span (e.g. session.system_span from a message like "I want to
    study folding of beta helix") IS the bare structural-class name itself,
    with no "protein"/"fold"/... noun for Case A to anchor on.

    "beta helix" -> "beta helix" (Case B: no class-head word at all, but
      the whole (already short) span contains the morpheme "helix")
    "a beta helix" -> "beta helix" (leading article trimmed first)
    "coiled coil" / "TIM barrel" -> extracted the same way
    "beta helix and coiled coil" -> None (Case B does NOT split on a
      mid-span conjunction; "and" is never at an edge here, so nothing is
      trimmed, the core stays 4 tokens > _MAX_MODIFIER_WORDS, too long to
      trust)
    "Pertactin" / "EspP" -> still None (no morpheme anywhere, Case B can't
      manufacture one either)

    This function is NEVER handed a raw, un-grounded user sentence -- only
    an already-grounded span (session.system_span, an A-extracted
    SYSTEM_IDENTITY/STRUCTURAL_DESCRIPTION source_span, or similar). Case B
    does not attempt to parse a full sentence for a structural mention
    buried inside it; a long, un-anchored span still correctly returns None.

    This function has NO knowledge of what any named protein's fold is. It
    only ever returns a substring that was already, lexically, sitting next
    to a generic class-head word (Case A) or that the ENTIRE (wrapper-
    trimmed) grounded text already collapses to (Case B) in the given text.
    """
    if not text:
        return None
    # tokens: (lowercased_word, start_offset, end_offset) preserving original
    # text for the final verbatim substring
    tokens = [(m.group(0).lower(), m.start(), m.end())
              for m in re.finditer(r"[A-Za-z][A-Za-z0-9-]*", text)]
    for i, (word, _, _) in enumerate(tokens):
        if word not in _CLASS_HEAD_WORDS:
            continue
        run_start_idx = i
        j = i - 1
        found_morpheme = False
        collected = 0
        while j >= 0 and collected < _MAX_MODIFIER_WORDS:
            w = tokens[j][0]
            if w in _CONNECTIVES:
                break
            if any(m in w for m in _STRUCTURAL_MORPHEMES):
                found_morpheme = True
            run_start_idx = j
            collected += 1
            j -= 1
        if found_morpheme and run_start_idx < i:
            start = tokens[run_start_idx][1]
            end = tokens[i - 1][2]
            return text[start:end]

    # Case B: no class-head word anywhere to anchor Case A on -- try the
    # whole (edge-trimmed) grounded span as a bare structural-class mention.
    lo, hi = 0, len(tokens)
    while lo < hi and tokens[lo][0] in _EDGE_WRAPPER_TOKENS:
        lo += 1
    while hi > lo and tokens[hi - 1][0] in _EDGE_WRAPPER_TOKENS:
        hi -= 1
    core = tokens[lo:hi]
    # A _CONNECTIVES member (including one of the relational/derivational
    # words deliberately excluded from _EDGE_WRAPPER_TOKENS above -- "of",
    # "with", "like", "based", ...) surviving INSIDE the trimmed core --
    # not just at an edge -- means edge-trimming alone couldn't clean this
    # span: something in it is still relationally incomplete ("beta helix
    # of", "beta helix like", "beta helix and"). Reject rather than return
    # a span carrying an unresolved relation as if it were a clean name.
    core_has_stray_connective = any(w in _CONNECTIVES for w, _, _ in core)
    if (core and len(core) <= _MAX_MODIFIER_WORDS and not core_has_stray_connective
            and any(any(m in w for m in _STRUCTURAL_MORPHEMES) for w, _, _ in core)):
        start, end = core[0][1], core[-1][2]
        return text[start:end]
    return None


def _extract_structural_terms(
    system_description: str,
    grounded_structural_feature: Optional[str] = None,
    literature_structural_mentions: Optional[list[str]] = None
) -> tuple[list[RouteTerm], Optional[str]]:
    """Extract structural route with strict grounding.

    RULE: Only USER_GROUNDED or LITERATURE_GROUNDED.
          No LLM inference about what fold a named protein belongs to.

    Args:
        system_description: the user's system (e.g., "Wnt-Wntless", "beta-helix protein")
        grounded_structural_feature: if A extracted a FactType.STRUCTURAL_CLASS
        literature_structural_mentions: if papers already retrieved mention structural features

    Returns:
        (structural_terms, justification_for_empty)
        If structural_terms is empty, justification_for_empty explains why.
    """
    terms: list[RouteTerm] = []
    justification: Optional[str] = None

    # Case 1: User explicitly provided a structural class (e.g., "beta-helix protein")
    if grounded_structural_feature:
        terms.append(RouteTerm(
            term=grounded_structural_feature,
            route_type="structural",
            source="USER_GROUNDED",
            supporting_span=grounded_structural_feature,
            provenance={"source": "user_grounded_structural_class"}
        ))
        return terms, None

    # Case 2: Literature (already retrieved, not mined by Hop-1) mentions a meaningful
    #         structural feature (e.g., papers on "beta-helix protein" explicitly note the fold)
    if literature_structural_mentions:
        for mention in literature_structural_mentions:
            if mention and mention.strip():
                terms.append(RouteTerm(
                    term=mention.strip(),
                    route_type="structural",
                    source="LITERATURE_GROUNDED",
                    supporting_span=mention.strip(),
                    provenance={"source": "literature_structural_mention"}
                ))
        if terms:
            return terms, None

    # Case 3: no explicit fact/literature grounding yet -- try narrow lexical
    # extraction directly from the grounded system_description text itself.
    # Still USER_GROUNDED (it's a verbatim substring of the user's own
    # grounded system text), never inferred.
    if system_description:
        lexical = extract_lexical_structural_feature(system_description)
        if lexical:
            terms.append(RouteTerm(
                term=lexical,
                route_type="structural",
                source="USER_GROUNDED",
                supporting_span=lexical,
                provenance={"source": "lexical_extraction_from_system_description"}
            ))
            return terms, None

    # Case 4: Named system (e.g., "Wnt-Wntless") without any grounded structural feature
    # → DO NOT invent a structural route
    if system_description:
        justification = (
            f"System '{system_description}' has no grounded structural feature. "
            "No structural route created."
        )
        return [], justification

    # Case 5: Empty system description
    justification = "System description is empty. No structural route created."
    return [], justification


# --- Context: grounded only ------------------------------------------------
#
# Finding C: grounded context may already be present INSIDE system_description
# ("membrane receptor") without the user stating it as a separate sentence.
# This is a narrow, deterministic LEXICAL extraction -- a context term is only
# produced when one of a small, explicit set of physical-context markers
# appears VERBATIM in system_description. It is extraction of grounded text
# already stated by the user, never inference from common/scientific
# knowledge: "GPCR" does NOT yield "membrane" (GPCRs are membrane proteins,
# but the word "membrane" is not present in the text -- nothing is
# extracted). Keep this list narrow -- it is NOT a general context-inference
# table, and it must never be asked to reach a verdict an LLM would have to
# reason about (that would make it PLANNER_DERIVED, which context_terms must
# never be).
_CONTEXT_LEXICAL_MARKERS = (
    "membrane-bound", "membrane-embedded", "membrane-associated",
    "cell-surface", "extracellular", "intracellular",
    "cytosolic", "cytoplasmic", "membrane",
)


def _extract_context_from_system_description(system_description: str) -> list[RouteTerm]:
    """Deterministic verbatim marker extraction from system_description.
    Longer/compound markers are matched before their shorter substrings
    ("membrane-bound" before "membrane") and a claimed span blocks any
    shorter marker from also firing on the same text, so "membrane-bound
    complex" yields ONE term ("membrane-bound"), not two."""
    text = system_description or ""
    tl = text.lower()
    claimed: list[tuple[int, int]] = []
    found: list[tuple[int, RouteTerm]] = []
    for marker in sorted(_CONTEXT_LEXICAL_MARKERS, key=len, reverse=True):
        pattern = r'\b' + re.escape(marker) + r'\b'
        for m in re.finditer(pattern, tl):
            span = (m.start(), m.end())
            if any(not (span[1] <= c[0] or span[0] >= c[1]) for c in claimed):
                continue  # overlaps an already-claimed (longer) marker
            claimed.append(span)
            verbatim = text[span[0]:span[1]]  # preserve original casing
            found.append((span[0], RouteTerm(
                term=verbatim,
                route_type="context",
                source="USER_GROUNDED",
                supporting_span=verbatim,
                provenance={"source": "system_description_lexical_marker"}
            )))
    found.sort(key=lambda x: x[0])
    return [t for _, t in found]


def _extract_context_terms(
    system_description: Optional[str] = None,
    user_stated_context: Optional[str] = None,
    literature_context_mentions: Optional[list[str]] = None
) -> tuple[list[RouteTerm], Optional[str]]:
    """Extract context route: only USER_GROUNDED or LITERATURE_GROUNDED.

    RULE: Do NOT invent from model knowledge.
          Only if user explicitly said it (directly, or as a lexical marker
          already present inside the grounded system_description), or papers
          already retrieved mention it.

    Args:
        system_description: the grounded system text itself (Finding C: may
            carry an explicit lexical context marker, e.g. "membrane receptor")
        user_stated_context: if user said "membrane protein" or "in crowded environment"
            as a separate statement
        literature_context_mentions: context clues from already-retrieved papers

    Returns:
        (context_terms, justification_for_empty)
    """
    terms: list[RouteTerm] = []
    justification: Optional[str] = None

    # Lexical markers already present in the grounded system text (Finding C)
    if system_description:
        for t in _extract_context_from_system_description(system_description):
            if not any(_norm(x.term) == _norm(t.term) for x in terms):
                terms.append(t)

    # User-stated context (separate statement)
    if user_stated_context and user_stated_context.strip():
        candidate = user_stated_context.strip()
        if not any(_norm(x.term) == _norm(candidate) for x in terms):
            terms.append(RouteTerm(
                term=candidate,
                route_type="context",
                source="USER_GROUNDED",
                supporting_span=candidate,
                provenance={"source": "user_stated_context"}
            ))

    # Literature-grounded context
    if literature_context_mentions:
        for mention in literature_context_mentions:
            if mention and mention.strip():
                candidate = mention.strip()
                if not any(_norm(t.term) == _norm(candidate) for t in terms):
                    terms.append(RouteTerm(
                        term=candidate,
                        route_type="context",
                        source="LITERATURE_GROUNDED",
                        supporting_span=candidate,
                        provenance={"source": "literature_context_mention"}
                    ))

    if not terms:
        justification = "No user-stated, system-description-lexical, or literature-grounded context. Context route remains empty."

    return terms, justification


# --- Modeling requirements: PLANNER_DERIVED only, with strict justification ---

def _infer_modeling_requirements(
    objective: str,
    system_description: str,
    confidence_threshold: Literal["high", "exploratory"] = "high"
) -> tuple[list[RouteTerm], Optional[str]]:
    """Infer modeling requirements as planning hypotheses.

    RULE: Always PLANNER_DERIVED.
          Must include explicit justification.
          Never represented as extracted literature evidence.

    Well-justified cases:
      - "binding" / "interface" / "interaction" objective
        → interface preservation (key CG modeling challenge for binding)
      - "folding" objective
        → folding landscape sampling (multi-state sampling is core to folding CG)
      - "conformational change" objective + recognized system patterns
        → conformational flexibility (dynamic rearrangement in CG)
      - otherwise: empty

    Args:
        objective: the normalized scientific objective
        system_description: the normalized system description
        confidence_threshold: only return terms meeting this confidence bar

    Returns:
        (modeling_requirement_terms, justification_for_empty)
    """
    terms: list[RouteTerm] = []
    justification: Optional[str] = None

    obj_lower = (objective or "").lower()
    sys_lower = (system_description or "").lower()

    # Rule 1: binding / interface / interaction → interface preservation
    # These are fundamental CG challenges for accurate binding representation
    if any(kw in obj_lower for kw in ["binding", "interface", "interaction", "dock"]):
        terms.append(RouteTerm(
            term="interface preservation",
            route_type="modeling_requirement",
            source="PLANNER_DERIVED",
            confidence="high",
            justification=(
                "Protein-protein binding and complexes require precise geometric preservation "
                "of interaction interfaces in CG models to maintain binding specificity and modes. "
                "Interface fidelity is a standard CG modeling concern. "
                "(CG model design principle)"
            ),
            provenance={"source": "planner", "reasoning": "binding/interface objective → interface preservation"}
        ))

    # Rule 2: folding objective → folding landscape sampling
    # Folding is fundamentally about multi-state sampling and pathway exploration
    if "folding" in obj_lower:
        terms.append(RouteTerm(
            term="folding landscape sampling",
            route_type="modeling_requirement",
            source="PLANNER_DERIVED",
            confidence="high",
            justification=(
                "Protein folding requires CG models to sample multiple conformational states and "
                "pathways. Capturing the folding landscape—including transition states and "
                "intermediates—is a core objective of CG folding studies. "
                "(standard CG methodology)"
            ),
            provenance={"source": "planner", "reasoning": "folding objective → landscape sampling"}
        ))

    # Rule 3: conformational change objective → conformational flexibility
    # Conformational dynamics inherently requires modeling flexibility
    if "conformational" in obj_lower and "change" in obj_lower:
        terms.append(RouteTerm(
            term="conformational flexibility",
            route_type="modeling_requirement",
            source="PLANNER_DERIVED",
            confidence="high",
            justification=(
                "Systems undergoing conformational changes require explicit modeling of domain and "
                "inter-domain flexibility in CG models. Multi-state dynamics and rearrangement "
                "pathways are central to conformational-change studies. "
                "(standard CG modeling practice)"
            ),
            provenance={"source": "planner", "reasoning": "conformational change objective → flexibility"}
        ))

    if not terms:
        justification = (
            f"Objective '{objective}' does not match any well-justified modeling requirement. "
            "Modeling requirements remain empty."
        )

    return terms, justification


# (The narrow LLM modeling-requirement planner that used to live here -- prompt, schema, deterministic gate -- is historical code,
# kept in the git tag `legacy-pre-cgscout`; only the rule-based _infer_modeling_requirements above is active.)

CallFn = Callable[[str, str, dict], str]   # (system_prompt, user_message, schema) -> raw JSON content


# --- Orchestrator -----------------------------------------------------------

def plan_evidence_routes(
    system_description: str,
    scientific_objective: str,
    grounded_structural_feature: Optional[str] = None,
    user_stated_context: Optional[str] = None,
    literature_structural_mentions: Optional[list[str]] = None,
    literature_context_mentions: Optional[list[str]] = None,
    *,
    call: Optional["CallFn"] = None,
    modeling_requirement_mode: Literal["rule"] = "rule",
) -> EvidenceRoutePlan:
    """Orchestrate multi-route evidence planning.

    Args:
        system_description: normalized system name/description from user
        scientific_objective: normalized scientific objective from user
        grounded_structural_feature: if A extracted FactType.STRUCTURAL_CLASS
        user_stated_context: if user mentioned physical context
        literature_structural_mentions: structural features from retrieved papers
        literature_context_mentions: context clues from retrieved papers
        call: accepted for signature compatibility; not used (there is no LLM
            modeling-requirement planner any more).
        modeling_requirement_mode: only "rule" is supported (the keyword
            prototype _infer_modeling_requirements); any other value raises
            ValueError. The LLM mode was removed -- historical code, git tag
            `legacy-pre-cgscout`.

    Returns:
        Complete EvidenceRoutePlan with all five routes
    """
    plan = EvidenceRoutePlan(
        system_description=system_description,
        scientific_objective=scientific_objective
    )

    # Direct system (deterministic)
    plan.direct_system_terms = _extract_direct_system_terms(system_description)

    # Process (deterministic primary)
    plan.process_terms = _extract_process_terms_primary(scientific_objective)

    # Structural (deterministic; strict grounding)
    struct_terms, struct_just = _extract_structural_terms(
        system_description,
        grounded_structural_feature=grounded_structural_feature,
        literature_structural_mentions=literature_structural_mentions
    )
    plan.structural_terms = struct_terms
    if struct_just:
        plan.route_justification["structural"] = struct_just

    # Context (deterministic; grounding-only; includes Finding C lexical extraction)
    ctx_terms, ctx_just = _extract_context_terms(
        system_description=system_description,
        user_stated_context=user_stated_context,
        literature_context_mentions=literature_context_mentions
    )
    plan.context_terms = ctx_terms
    if ctx_just:
        plan.route_justification["context"] = ctx_just

    # Modeling requirements: rule prototype only (the LLM mode was removed).
    if modeling_requirement_mode != "rule":
        raise ValueError(
            f"plan_evidence_routes() supports only modeling_requirement_mode='rule' (got {modeling_requirement_mode!r}); "
            "the LLM modeling-requirement mode was removed."
        )
    req_terms, req_just = _infer_modeling_requirements(
        scientific_objective,
        system_description
    )
    plan.modeling_requirement_terms = req_terms
    if req_just:
        plan.route_justification["modeling_requirement"] = req_just

    # Track empty routes
    if not plan.structural_terms:
        plan.empty_routes.append("structural")
    if not plan.context_terms:
        plan.empty_routes.append("context")
    if not plan.modeling_requirement_terms:
        plan.empty_routes.append("modeling_requirement")

    return plan


# --- Validation (hard stops) -----------------------------------------------

def validate_hard_stops(plan: EvidenceRoutePlan) -> Optional[str]:
    """Check for hard-stop failure conditions.

    Returns an error message if any hard-stop condition is met, else None.
    """

    # Hard stop 1: Named system (no grounded structural feature) gets an invented structural route
    if plan.system_description and "structural" not in plan.empty_routes:
        # Check: does it look like a named system (no natural fold descriptor)?
        if not any(fold in plan.system_description.lower()
                   for fold in ["helix", "barrel", "fold", "solenoid", "sandwich", "barrel", "bundle"]):
            # It's a named system. Check if structural route was grounded.
            if plan.structural_terms:
                for st in plan.structural_terms:
                    if st.source != "USER_GROUNDED" and st.source != "LITERATURE_GROUNDED":
                        return (
                            f"HARD STOP: Named system '{plan.system_description}' got a structural route "
                            f"that is not grounded: {st.term} (source: {st.source}). "
                            "This violates the structural grounding rule."
                        )

    # Hard stop 2: Generic alpha/beta content as structural route
    for st in plan.structural_terms:
        term_lower = st.term.lower()
        if any(generic in term_lower for generic in ["alpha helix", "beta strand", "alpha-helix",
                                                       "beta-strand", "globular", "mixed alpha"]):
            return (
                f"HARD STOP: Generic structural descriptor '{st.term}' was accepted as a structural route. "
                "Only named, meaningful folds are allowed."
            )

    # Hard stop 3: Context from model memory (no grounding)
    for ct in plan.context_terms:
        if ct.source == "PLANNER_DERIVED":
            return (
                f"HARD STOP: Context term '{ct.term}' has source='PLANNER_DERIVED'. "
                "Context must be USER_GROUNDED or LITERATURE_GROUNDED only."
            )

    # Hard stop 4: Modeling requirements presented as evidence
    # (This is about downstream representation, not the planner itself.
    #  The hard stop here is that they must ALL be PLANNER_DERIVED.)
    for mt in plan.modeling_requirement_terms:
        if mt.source != "PLANNER_DERIVED":
            return (
                f"HARD STOP: Modeling requirement '{mt.term}' has source='{mt.source}'. "
                "All modeling requirements must be PLANNER_DERIVED."
            )
        if not mt.justification:
            return (
                f"HARD STOP: Modeling requirement '{mt.term}' is missing justification. "
                "Every PLANNER_DERIVED term must have an explicit justification."
            )

    return None
