"""Evidence Route Planner: Multi-route evidence discovery planning.

Given grounded system + objective from intake, identify which evidence dimensions
would make prior CG work informative. Routes are independent:
  1. direct_system: (no longer filled here; the system-specific CG queries are built from system_description by route_plan.py)
  2. process: the scientific process/property being studied
  3. context: grounded physical context (membrane, environment, etc.)
  4. modeling_requirement: what simulation capability would be needed (route disabled: never filled)
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


# --- Deterministic extraction: process -------------------------------------

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


# (The modeling-requirement planners that used to live here -- the LLM planner and the rule-based keyword prototype -- are historical code,
# kept in the git tag `legacy-pre-cgscout`; the modeling_requirement route is disabled: no terms are derived and no queries are built for it.)

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
        modeling_requirement_mode: only "rule" is accepted (kept so existing
            calls keep working); any other value raises ValueError. Neither
            the LLM mode nor the keyword prototype exists any more -- historical
            code, git tag `legacy-pre-cgscout`; modeling_requirement_terms stays empty.

    Returns:
        Complete EvidenceRoutePlan with all five routes
    """
    plan = EvidenceRoutePlan(
        system_description=system_description,
        scientific_objective=scientific_objective
    )

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

    # Modeling requirements: the route is disabled (nothing is derived); only the historical argument value is accepted.
    if modeling_requirement_mode != "rule":
        raise ValueError(
            f"plan_evidence_routes() supports only modeling_requirement_mode='rule' (got {modeling_requirement_mode!r}); "
            "the LLM modeling-requirement mode was removed."
        )
    # Track empty routes
    if not plan.structural_terms:
        plan.empty_routes.append("structural")
    if not plan.context_terms:
        plan.empty_routes.append("context")
    if not plan.modeling_requirement_terms:
        plan.empty_routes.append("modeling_requirement")

    return plan
