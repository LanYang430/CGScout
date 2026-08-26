"""Structured schema for a search result (`Paper`) and its extracted evidence.

`Paper` is the normalized-across-sources search result schema (Phase 1). The
`PaperEvidence` types below it are the single-pass, query-independent
evidence-extraction schema (Stage 4a, see evidence_extraction.py) -- they live
here rather than in evidence_extraction.py specifically because `Paper.evidence`
needs to reference `PaperEvidence` as a field type, and evidence_extraction.py
already imports `Paper` from this module; putting the schema on the other side
of that edge would be a circular import.

Design note: earlier iterations of this stage had the LLM itself produce a
relevant/not_relevant-style verdict (`Stage4Assessment`/`CGEligibility`/
`TaskMatch`/etc., and a later two-call split of the same idea across
`PaperEvidenceExtraction`/`QueryMatchResult`). Both were replaced by the
`PaperEvidence` schema below: the LLM now only extracts objective,
query-independent facts (once per paper, cacheable), and *all* comparison
against a user's `ScientificConceptProfile` -- including any eventual
relevant/not_relevant-style call -- is deterministic Python in
evidence_matching.py. See `benchmarks/stage4_design_experiments/` for the
single-call vs two-call experiments that motivated this.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class ExternalIds(BaseModel):
    semantic_scholar_id: Optional[str] = None
    openalex_id: Optional[str] = None
    arxiv_id: Optional[str] = None
    pmid: Optional[str] = None


class MethodEvidence(BaseModel):
    """What this paper's own text says about the modeling/simulation method
    itself. Deliberately NOT an eligibility judgment ("is this paper
    eligible/not_eligible for X") -- just what class of model the paper
    discusses and what role that model played in it. Whether that's useful
    for a given query is a downstream, deterministic comparison
    (evidence_matching.py), not something decided here.
    """

    model_class: list[str] = Field(
        default_factory=list,
        description="Model/method class(es) explicitly named or clearly described in the paper's own text, e.g. "
        "'coarse-grained', 'all-atom', 'structure-based', 'elastic-network', 'Go model'. Open vocabulary -- not a "
        "fixed enum. Empty if the paper doesn't discuss a simulation model/method at all.",
    )
    study_role: list[str] = Field(
        default_factory=list,
        description="What role the model/method played in THIS paper, e.g. 'developed' (introduces a new "
        "model/method), 'applied' (uses an existing model to study a system), 'benchmarked', 'validated', "
        "'reviewed_only' (a review/survey that discusses models without running one). Multiple roles can apply. "
        "Only what is stated or clearly evident -- never inferred beyond that.",
    )
    grounding_text: Optional[str] = Field(
        default=None, description="Abstract text (quote or close paraphrase) supporting the above, for audit."
    )


class SystemEvidence(BaseModel):
    """What this paper's own text explicitly says about the physical system(s) it
    studied, simulated, benchmarked, developed against, or applied its method to.

    v3 design: one open-text `studied_systems` field for identity + structure/
    architecture + qualitative scale together, instead of splitting them across a
    `system_type`/`structural_class`/`approx_size_residues`/`environment` schema an
    LLM can misassign content between. Real failures under the split schema, all
    confirmed against source abstracts (see benchmarks/paper_evidence_extraction/):
    "a three-helix bundle" -> approx_size_residues=3 (a topology count misread as a
    residue count); "approximately 9 ms" -> approx_size_residues=9 (a simulation
    duration misread as a residue count); a genuine beta-helix compressed into the
    coarse bucket "globular"; and `environment` conflating biological context,
    simulation setup, and general discussion into one field with no way to tell
    which. Removing the split fields removes the *slots* those errors landed in,
    rather than just prompting more carefully against the same slots.
    """

    studied_systems: list[str] = Field(
        default_factory=list,
        description="What system(s) the paper actually studied, simulated, benchmarked, developed against, or "
        "applied its method to -- kept close to the abstract's own words, one entry per distinct system or "
        "system-group. Capture identity + structure/architecture + qualitative scale together as one faithful "
        "phrase, e.g. 'three-helix bundle', 'twelve proteins with multiple secondary-structure arrangements', "
        "'C-terminal segment of the pertactin passenger domain', 'intrinsically disordered proteins', "
        "'protein-ligand complexes'. Never infer from general biological/domain background what the abstract "
        "itself doesn't say. NEVER include a simulation parameter (duration, temperature, timestep, box size) "
        "here even if a number appears right next to a system description in the same sentence -- only include "
        "what the number/phrase says about the SYSTEM, never about how long/how it was simulated. A general "
        "capability claim about the METHOD (e.g. 'this method can handle proteins up to N residues') is not a "
        "studied system either, unless the paper's own studied examples are actually that size.",
    )
    family_or_fold: list[str] = Field(
        default_factory=list,
        description="Fold/family name(s) explicitly stated AS SUCH (e.g. 'TIM-barrel fold', 'immunoglobulin "
        "fold', 'autotransporter passenger domain family') -- only when the abstract names a fold/family "
        "classification. An inline topology phrase like 'three-helix bundle' or 'three-stranded antiparallel "
        "beta-sheet' belongs in studied_systems, not here, even though it describes architecture -- do not "
        "promote a studied_systems-style description into this field just to have something to put here.",
    )
    protein_or_system_names: list[str] = Field(
        default_factory=list, description="Bare specific protein/system name(s) explicitly named in the paper's own text."
    )
    grounding_text: Optional[str] = Field(
        default=None, description="Abstract text (quote or close paraphrase) supporting the above, for audit."
    )


class ProcessEvidenceItem(BaseModel):
    """One process/capability this paper's text actually demonstrates or
    studies. Paired free text + normalized label in a single object (rather
    than two parallel lists) so the two can never drift out of index
    alignment with each other.
    """

    raw_text: str = Field(description="Close to the abstract's own words. Only what is stated, never inferred.")
    normalized_label: str = Field(
        description="Short, generic, reusable snake_case label, e.g. 'folding', 'unfolding', "
        "'conformational_dynamics', 'pocket_gating', 'binding', 'aggregation', 'assembly', 'translocation', "
        "'kinetics', 'thermodynamics', 'stability'. Open vocabulary -- NOT a fixed, folding-specific enum; invent "
        "a new label if none of the common ones fit, but prefer reusing an existing one over a near-duplicate.",
    )


class PaperEvidence(BaseModel):
    """Single-pass, query-independent structured extraction from a paper's
    title/abstract/metadata alone. See evidence_extraction.py for the full
    design rationale.

    No `ScientificConceptProfile` is ever given to the extractor that
    produces this -- every field describes the paper itself, never a search
    target, so target-concept leakage is structurally impossible (not just
    prompted against), and the result is safe to cache and reuse across
    different queries against the same paper. Matching this against a
    particular query's profile -- and any relevant/not_relevant-style
    decision -- happens entirely afterward in evidence_matching.py, as plain
    deterministic Python. This schema itself carries no eligibility,
    evidence-strength, confidence, or relevance judgment of any kind; those
    can be reconsidered at the full-text stage if still needed.
    """

    method: MethodEvidence = Field(default_factory=MethodEvidence)
    system: SystemEvidence = Field(default_factory=SystemEvidence)
    processes: list[ProcessEvidenceItem] = Field(default_factory=list)

    missing_information: list[str] = Field(
        default_factory=list,
        description="Aspects the abstract doesn't state clearly -- candidates for a later full-text pass.",
    )
    reasoning_summary: Optional[str] = None
    extractor: str = "baseline"


class Paper(BaseModel):
    title: str
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    doi: Optional[str] = None
    abstract: Optional[str] = None
    venue: Optional[str] = None

    sources: list[str] = Field(default_factory=list)
    external_ids: ExternalIds = Field(default_factory=ExternalIds)

    is_open_access: Optional[bool] = None
    oa_pdf_url: Optional[str] = None
    url: Optional[str] = None

    retrieved_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Populated by a RelevanceFilter (see relevance_filter.py; Stage 3, cheap
    # keyword-overlap check). None means "never scored" -- distinct from 0.0,
    # which means "scored and found to have no lexical overlap with the query."
    relevance_score: Optional[float] = None
    matched_keywords: list[str] = Field(default_factory=list)

    # Populated by a PaperEvidenceExtractor (see evidence_extraction.py;
    # Stage 4a, single-pass query-independent extraction). None means "never
    # extracted" -- distinct from an extracted-but-empty PaperEvidence, which
    # means the extractor ran and the paper's own text just didn't state
    # much. Query-independent, so this is exactly what a per-paper cache
    # would store/reuse across different queries against the same paper.
    evidence: Optional[PaperEvidence] = None

    def dedupe_key_doi(self) -> Optional[str]:
        """Normalized DOI for exact-match dedup, or None if no DOI."""
        if not self.doi:
            return None
        return self.doi.strip().lower().removeprefix("https://doi.org/")

    def dedupe_key_title_year(self) -> str:
        """Fallback fuzzy-ish key: lowercased, whitespace-collapsed title + year.

        Not a substitute for real fuzzy matching, but catches the common case of
        the same paper returned by both sources with identical titles and no DOI
        from one of them.
        """
        normalized_title = " ".join(self.title.lower().split())
        return f"{normalized_title}::{self.year or ''}"
