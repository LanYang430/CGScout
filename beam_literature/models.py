"""Structured schema for a single search result, normalized across sources.

Kept deliberately small for Phase 1 (search only). This is NOT the full BEAM
evidence schema described in the project plan (folding evidence, structural
fidelity, dynamics, ...) -- that gets built on top of this once full-text /
LLM extraction phases exist. This model only needs to answer: "what paper is
this, where did it come from, and can we get more out of it later?"
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

    # Populated by a RelevanceFilter (see relevance_filter.py). None means
    # "never scored" -- distinct from 0.0, which means "scored and found to
    # have no lexical overlap with the query."
    relevance_score: Optional[float] = None
    matched_keywords: list[str] = Field(default_factory=list)

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
