"""Data structures of Hop-2 of Literature-grounded RelatedSystemDiscovery.

Hop-1 turns a grounded structural class into CONCRETE real systems, each with
an exact quote + paper provenance (Hop1Provenance). Hop-2 then asks, for each
of those, whether the system has SYSTEM-SPECIFIC literature relevant to the
user's objective and/or coarse-grained modelling.

This module holds only the result types of that chain -- Hop1Provenance,
Hop2Query, Hop2Paper, Hop2SystemResult, Hop2Result (they are part of the
route_retrieval_plan result schema) -- and the coarse-grained surface forms
_CG_MARKERS that Stage 3 (relevance_filter_handoff) reuses. The old Hop-2
implementation that lived here (query building, discover_hop2) was replaced
by hop2_converged.py and is historical code, kept in the git tag
`legacy-pre-cgscout`.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

# coarse-grained methodology surface forms (for the report-only CG flag)
_CG_MARKERS = ("coarse-grained", "coarse grained", "coarsegrained", "coarse-grain ",
               "coarse grain ")


# --------------------------------------------------------------------------
# data structures -- the provenance chain
# --------------------------------------------------------------------------
class Hop1Provenance(BaseModel):
    """Where this system came from in Hop-1 -- carried verbatim, never re-derived."""
    structural_class: str
    also_known_as: list[str] = Field(default_factory=list)   # Hop-1 terminology expansion
    discovered_system: str
    n_support: int = 0
    first_seen_rank: int = 0
    supporting_paper_ref: str = ""
    supporting_paper_title: str = ""
    evidence_quote: str = ""                                 # exact Hop-1 quote, unmodified
    structural_term_matched: str = ""
    hop1_found_by_query: str = ""


class Hop2Query(BaseModel):
    system_name: str
    role: str                       # one of the three role names: system_objective / system_methodology / system_objective_methodology
    query: str
    provenance: Hop1Provenance


class Hop2Paper(BaseModel):
    paper_ref: str
    title: str
    year: Optional[int] = None
    has_abstract: bool = False
    # which discovered system(s) / role(s) / query string(s) retrieved it
    found_by_system: str
    found_by_roles: list[str] = Field(default_factory=list)
    found_by_queries: list[str] = Field(default_factory=list)
    retrieval_rank: int = 10**9     # first rank across this system's queries
    # coarse deterministic keyword signals (report diagnostic only)
    mentions_system: bool = False
    mentions_objective: bool = False
    mentions_cg: bool = False
    also_in_generic_retrieval: bool = False


class Hop2SystemResult(BaseModel):
    system_name: str
    provenance: Hop1Provenance
    queries: list[Hop2Query] = Field(default_factory=list)
    papers: list[Hop2Paper] = Field(default_factory=list)
    n_papers: int = 0
    n_papers_only_hop2: int = 0            # not in the generic-retrieval set
    # system-specific evidence buckets (mentions_system AND ...)
    folding_evidence_refs: list[str] = Field(default_factory=list)
    cg_evidence_refs: list[str] = Field(default_factory=list)
    folding_and_cg_evidence_refs: list[str] = Field(default_factory=list)
    has_folding_evidence: bool = False     # explicit -- absence is a real answer
    has_cg_evidence: bool = False
    noisy_queries: list[str] = Field(default_factory=list)   # returned papers, none mention the system
    empty_queries: list[str] = Field(default_factory=list)   # returned nothing
    search_errors: list[str] = Field(default_factory=list)


class Hop2Result(BaseModel):
    structural_class: str
    scientific_objective: str
    methodology_phrase: str
    also_known_as: list[str] = Field(default_factory=list)
    systems_available: int = 0
    systems_carried: int = 0
    system_selection_note: str = ""
    per_system: list[Hop2SystemResult] = Field(default_factory=list)
    all_papers: list[Hop2Paper] = Field(default_factory=list)
    n_papers_total_rows: int = 0          # sum over systems (a paper can recur across systems)
    n_papers_unique: int = 0
    # comparison with the existing generic retrieval
    n_generic: int = 0
    n_overlap_with_generic: int = 0
    n_hop2_only: int = 0
    hop2_only_refs: list[str] = Field(default_factory=list)
    # rollups
    systems_with_folding_evidence: list[str] = Field(default_factory=list)
    systems_with_cg_evidence: list[str] = Field(default_factory=list)
    systems_with_no_downstream_evidence: list[str] = Field(default_factory=list)
    search_errors: list[str] = Field(default_factory=list)
    retrieval_seconds: float = 0.0
