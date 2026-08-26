"""Stage 1: turn a ScientificConceptProfile into search queries.

    ScientificConceptProfile -> QueryPlanner.plan() -> QueryPlan -> search

Replaces the earlier query_expansion.py, which generated query variants
directly from raw query text (stopword trimming, size-qualifier removal, a
synonym table). That logic has moved "up" into concept_resolver.py as
structured extraction; this module's job is now query *strategy* given an
already-resolved profile, not text extraction.

Two categories of query, both important:

- `system_specific`   tied to the user's literal system/constraints (their
                       exact size, their exact structural term combination).
                       Roughly what the old expander produced.
- `general_discovery` deliberately drops system specifics and searches for
                       "<methodology> <structural term / family / task>" on
                       its own. This exists because a useful CG model may
                       never have been tested on the user's exact system --
                       system-specific queries alone would never surface it.

`QueryPlan.unique_query_strings()` flattens+dedupes both categories into the
flat list actually used for searching; `QueryPlan.queries` (with categories
and rationale) is kept for reporting/reproducibility.

`QueryPlanner` is a Protocol specifically so an LLM-backed planner (e.g. one
that proposes queries no template here would think of) can be dropped in
later without changing anything that calls it.
"""

from __future__ import annotations

import re
from typing import Optional, Protocol

from pydantic import BaseModel, Field

from beam_agent.concept_resolver import ScientificConceptProfile
from beam_agent.text_utils import core_keyword_string


class PlannedQuery(BaseModel):
    query: str
    category: str  # "system_specific" | "general_discovery"
    rationale: Optional[str] = None


class QueryPlan(BaseModel):
    profile: ScientificConceptProfile
    queries: list[PlannedQuery] = Field(default_factory=list)
    planner: str = "baseline"

    def unique_query_strings(self) -> list[str]:
        """Flatten both categories into a deduped list of query strings, in order."""
        seen: set[str] = set()
        out: list[str] = []
        for pq in self.queries:
            key = " ".join(pq.query.lower().split())
            if key and key not in seen:
                seen.add(key)
                out.append(pq.query)
        return out


class QueryPlanner(Protocol):
    def plan(self, profile: ScientificConceptProfile) -> QueryPlan:
        ...


class BaselineQueryPlanner:
    def __init__(self, max_queries_per_category: int = 5) -> None:
        self.max_queries_per_category = max_queries_per_category

    def plan(self, profile: ScientificConceptProfile) -> QueryPlan:
        queries = self._system_specific(profile) + self._general_discovery(profile)
        return QueryPlan(profile=profile, queries=queries, planner="BaselineQueryPlanner")

    def _system_specific(self, profile: ScientificConceptProfile) -> list[PlannedQuery]:
        query = profile.raw_query
        candidates = [
            PlannedQuery(query=query, category="system_specific", rationale="original user query")
        ]

        core = core_keyword_string(query)
        if core and core.lower() != query.lower():
            candidates.append(
                PlannedQuery(query=core, category="system_specific", rationale="stopword-trimmed")
            )

        if profile.size_constraints:
            broadened = query
            for c in profile.size_constraints:
                broadened = broadened.replace(c.raw_text, "")
            broadened = " ".join(broadened.split())
            if broadened and broadened.lower() != query.lower():
                candidates.append(
                    PlannedQuery(
                        query=broadened,
                        category="system_specific",
                        rationale="size qualifier dropped",
                    )
                )

        for term in profile.structural_terms + profile.methodology_terms + profile.scientific_tasks:
            if not term.synonyms:
                continue
            synonym = term.synonyms[0]
            pattern = re.compile(re.escape(term.term), re.IGNORECASE)
            if pattern.search(query):
                variant = pattern.sub(synonym, query, count=1)
                candidates.append(
                    PlannedQuery(
                        query=variant,
                        category="system_specific",
                        rationale=f"synonym: {term.term} -> {synonym}",
                    )
                )

        return self._dedupe(candidates)[: self.max_queries_per_category]

    def _general_discovery(self, profile: ScientificConceptProfile) -> list[PlannedQuery]:
        methodology_phrase = " ".join(t.term for t in profile.methodology_terms)
        methodology_phrase = methodology_phrase or "coarse-grained model"

        candidates: list[PlannedQuery] = []

        for term in profile.structural_terms:
            candidates.append(
                PlannedQuery(
                    query=f"{methodology_phrase} {term.term}".strip(),
                    category="general_discovery",
                    rationale=f"structural_term: {term.term}, system-agnostic",
                )
            )
        for term in profile.related_families:
            candidates.append(
                PlannedQuery(
                    query=f"{methodology_phrase} {term.term}".strip(),
                    category="general_discovery",
                    rationale=f"related_family: {term.term}",
                )
            )
        for term in profile.representative_systems:
            candidates.append(
                PlannedQuery(
                    query=f"{methodology_phrase} {term.term}".strip(),
                    category="general_discovery",
                    rationale=f"representative_system: {term.term}",
                )
            )
        for task in profile.scientific_tasks:
            candidates.append(
                PlannedQuery(
                    query=f"{methodology_phrase} {task.term}".strip(),
                    category="general_discovery",
                    rationale=f"scientific_task: {task.term}, no system specifics",
                )
            )

        if not candidates:
            # Baseline resolver found no known clusters -- fall back to the
            # raw query so general discovery still returns something.
            candidates.append(
                PlannedQuery(
                    query=profile.raw_query,
                    category="general_discovery",
                    rationale="fallback: no concepts resolved",
                )
            )

        return self._dedupe(candidates)[: self.max_queries_per_category]

    @staticmethod
    def _dedupe(queries: list[PlannedQuery]) -> list[PlannedQuery]:
        seen: set[str] = set()
        out: list[PlannedQuery] = []
        for pq in queries:
            key = " ".join(pq.query.lower().split())
            if key and key not in seen:
                seen.add(key)
                out.append(pq)
        return out
