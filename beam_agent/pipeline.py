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

`run_pipeline()` does no file I/O -- it returns a `PipelineResult` in memory
and nothing more. Reproducibility snapshots (writing that result to disk) are
a separate, opt-in concern; see `snapshot.py`. Keeping this function pure
means it stays easy to unit test and to call from contexts (a notebook, a
future web UI) that don't want files appearing on disk as a side effect.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from beam_agent.dedupe import dedupe_papers
from beam_agent.models import Paper
from beam_agent.query_expansion import QueryExpander, RuleBasedQueryExpander
from beam_agent.relevance_filter import RelevanceFilter, RuleBasedRelevanceFilter
from beam_agent.sources.openalex import search_openalex
from beam_agent.sources.semantic_scholar import search_semantic_scholar

_JSON_SAFE_TYPES = (str, int, float, bool, type(None), list, dict)


def _config_snapshot(obj: object) -> dict:
    """Best-effort, generic snapshot of a component's config for reproducibility.

    Records the class name plus any public instance attribute that's a plain
    JSON-serializable value (str/int/float/bool/None/list/dict); anything else
    (e.g. an API client instance a future LLM-backed component might hold) is
    silently skipped rather than crashing on it. Generic on purpose so this
    keeps working for any future `QueryExpander`/`RelevanceFilter` without
    needing to know its attribute names in advance.
    """
    config = {"class": type(obj).__name__}
    for key, value in vars(obj).items():
        if not key.startswith("_") and isinstance(value, _JSON_SAFE_TYPES):
            config[key] = value
    return config


@dataclass
class PipelineResult:
    user_query: str
    expanded_queries: list[str]
    raw_papers: list[Paper]
    deduped_papers: list[Paper]
    retained_papers: list[Paper]
    rejected_papers: list[Paper] = field(default_factory=list)
    source_errors: list[str] = field(default_factory=list)
    sources: tuple[str, ...] = ("semantic_scholar", "openalex")
    limit_per_query: int = 15
    expander_config: dict = field(default_factory=dict)
    filter_config: dict = field(default_factory=dict)

    @property
    def raw_candidate_count(self) -> int:
        return len(self.raw_papers)

    @property
    def deduped_count(self) -> int:
        return len(self.deduped_papers)

    @property
    def filtered_count(self) -> int:
        return len(self.retained_papers)


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
        raw_papers=raw,
        deduped_papers=deduped,
        retained_papers=retained,
        rejected_papers=rejected,
        source_errors=errors,
        sources=sources,
        limit_per_query=limit_per_query,
        expander_config=_config_snapshot(expander),
        filter_config=_config_snapshot(filterer),
    )
