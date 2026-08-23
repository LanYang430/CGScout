"""Phase 1.5: query expansion -> multi-query search -> dedupe -> relevance filter.

Wires the modular pieces together:

    user_query
      -> QueryExpander.expand()          (several complementary queries)
      -> search each query x each source (raw candidate pool)
      -> dedupe_papers()                 (merge duplicates across queries+sources)
      -> RelevanceFilter.filter()        (drop obvious irrelevant candidates)
      -> retained papers, ranked by relevance score

Every stage is swappable via constructor args -- in particular `query_expander`
and `relevance_filter`, both typed as Protocols in their own modules, so an
LLM-backed implementation of either can be dropped in later without touching
this orchestration or the CLI that calls it. Still no LLM calls anywhere in
this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from beam_literature.dedupe import dedupe_papers
from beam_literature.models import Paper
from beam_literature.query_expansion import QueryExpander, RuleBasedQueryExpander
from beam_literature.relevance_filter import RelevanceFilter, RuleBasedRelevanceFilter
from beam_literature.sources.openalex import search_openalex
from beam_literature.sources.semantic_scholar import search_semantic_scholar


@dataclass
class PipelineResult:
    user_query: str
    expanded_queries: list[str]
    raw_candidate_count: int
    deduped_count: int
    filtered_count: int
    retained_papers: list[Paper]
    rejected_papers: list[Paper] = field(default_factory=list)
    source_errors: list[str] = field(default_factory=list)


def run_pipeline(
    user_query: str,
    *,
    limit_per_query: int = 15,
    sources: tuple[str, ...] = ("semantic_scholar", "openalex"),
    query_expander: QueryExpander | None = None,
    relevance_filter: RelevanceFilter | None = None,
) -> PipelineResult:
    expander = query_expander or RuleBasedQueryExpander()
    filterer = relevance_filter or RuleBasedRelevanceFilter()

    expanded_queries = expander.expand(user_query)

    raw: list[Paper] = []
    errors: list[str] = []

    for q in expanded_queries:
        if "semantic_scholar" in sources:
            try:
                raw.extend(search_semantic_scholar(q, limit=limit_per_query))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"semantic_scholar[{q!r}]: {exc}")
        if "openalex" in sources:
            try:
                raw.extend(search_openalex(q, limit=limit_per_query))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"openalex[{q!r}]: {exc}")

    deduped = dedupe_papers(raw)
    retained, rejected = filterer.filter(user_query, deduped)

    return PipelineResult(
        user_query=user_query,
        expanded_queries=expanded_queries,
        raw_candidate_count=len(raw),
        deduped_count=len(deduped),
        filtered_count=len(retained),
        retained_papers=retained,
        rejected_papers=rejected,
        source_errors=errors,
    )
