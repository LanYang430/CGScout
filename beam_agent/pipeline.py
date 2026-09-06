"""Orchestrates the full rule-based-plus-single-LLM-pass front half of BEAM's
literature pipeline:

    user_query
      -> ConceptResolver.resolve()          Stage 0: structured concept profile
      -> QueryPlanner.plan()                Stage 1: system-specific + general-
                                             discovery search queries
      -> search each query x each source    Stage 2: raw candidate pool,
                                             each Paper stamped with which
                                             PlannedQuery(s)/categor(y/ies)
                                             found it (found_by_queries/
                                             query_categories -- see
                                             stamp_provenance() below)
      -> dedupe_papers()                    merge duplicates across queries+sources,
                                             unioning provenance (dedupe.py)
      -> RelevanceFilter.filter()           Stage 3: cheap keyword-overlap filter
      -> PaperEvidenceExtractor.extract_many()  Stage 4a: single-pass, query-
                                             independent evidence extraction
                                             (baseline: honest stub, no LLM
                                             call -- see evidence_extraction.py)

Stage 4b/5 (deterministic matching against the profile, and user-configurable
filtering/ranking; see evidence_matching.py) are deliberately NOT part of
`run_pipeline()` -- extraction is the expensive, cacheable, query-independent
part; matching is cheap and rerunnable with different `FilterCriteria`
without re-running retrieval or the LLM. Call `evidence_matching.rank_papers()`
on `PipelineResult.evidence_papers` separately, as many times as needed.

Every stage is swappable via constructor args -- `concept_resolver`,
`query_planner`, `relevance_filter`, and `evidence_extractor` are all typed as
Protocols in their own modules, so an LLM-backed implementation of any one of
them can be dropped in later without touching this orchestration or the CLI
that calls it.

`run_pipeline()` does no file I/O -- it returns a `PipelineResult` in memory
and nothing more. Reproducibility snapshots (writing that result to disk) are
a separate, opt-in concern; see `snapshot.py`. Keeping this function pure
means it stays easy to unit test and to call from contexts (a notebook, a
future web UI) that don't want files appearing on disk as a side effect.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from beam_agent.concept_resolver import BaselineConceptResolver, ConceptResolver, ScientificConceptProfile
from beam_agent.dedupe import dedupe_papers
from beam_agent.evidence_extraction import BaselineEvidenceExtractor, PaperEvidenceExtractor
from beam_agent.models import Paper
from beam_agent.query_planner import BaselineQueryPlanner, PlannedQuery, QueryPlan, QueryPlanner
from beam_agent.relevance_filter import RelevanceFilter, RuleBasedRelevanceFilter
from beam_agent.sources.openalex import search_openalex
from beam_agent.sources.semantic_scholar import search_semantic_scholar

_JSON_SAFE_TYPES = (str, int, float, bool, type(None), list, dict)


def group_queries_by_normalized_key(queries: list[PlannedQuery]) -> dict[str, tuple[str, list[str]]]:
    """Maps a normalized query key -> (first-seen original query string, every
    category that produced it). Same normalization QueryPlan.unique_query_strings()
    and BaselineQueryPlanner._dedupe() already use for the same purpose (lowercased,
    whitespace-collapsed) -- kept as its own copy here rather than imported, matching
    how those two already duplicate the same one-liner rather than sharing it.

    A query string that happens to come from more than one category (rare, but not
    impossible -- system_specific and general_discovery build queries independently)
    is searched exactly once, carrying every category it belongs to, rather than
    searched twice for the same string.
    """
    grouped: dict[str, tuple[str, list[str]]] = {}
    for pq in queries:
        key = " ".join(pq.query.lower().split())
        if not key:
            continue
        if key not in grouped:
            grouped[key] = (pq.query, [pq.category])
        else:
            original, categories = grouped[key]
            if pq.category not in categories:
                categories.append(pq.category)
    return grouped


def stamp_provenance(papers: list[Paper], *, query: str, categories: list[str]) -> None:
    """Attach which planned query (and its category/categories) found each paper,
    right after the search call that produced it -- before anything gets flattened
    across queries/sources and this context becomes unrecoverable (see dedupe.py's
    _merge(), which unions this across duplicate papers the same way it already
    unions `sources`). Mutates in place: each Paper here was just constructed by
    this call's search_*() function and isn't shared/aliased anywhere else yet.
    """
    for paper in papers:
        if query not in paper.found_by_queries:
            paper.found_by_queries.append(query)
        for category in categories:
            if category not in paper.query_categories:
                paper.query_categories.append(category)


def _config_snapshot(obj: object) -> dict:
    """Best-effort, generic snapshot of a component's config for reproducibility.

    Records the class name plus any public instance attribute that's a plain
    JSON-serializable value (str/int/float/bool/None/list/dict); anything else
    (e.g. an API client instance a future LLM-backed component might hold) is
    silently skipped rather than crashing on it. Generic on purpose so this
    keeps working for any future component without needing to know its
    attribute names in advance.
    """
    config = {"class": type(obj).__name__}
    for key, value in vars(obj).items():
        if not key.startswith("_") and isinstance(value, _JSON_SAFE_TYPES):
            config[key] = value
    return config


@dataclass
class PipelineResult:
    user_query: str
    concept_profile: ScientificConceptProfile
    query_plan: QueryPlan
    raw_papers: list[Paper]
    deduped_papers: list[Paper]
    retained_papers: list[Paper]
    rejected_papers: list[Paper] = field(default_factory=list)
    evidence_papers: list[Paper] = field(default_factory=list)
    source_errors: list[str] = field(default_factory=list)
    sources: tuple[str, ...] = ("semantic_scholar", "openalex")
    limit_per_query: int = 15
    resolver_config: dict = field(default_factory=dict)
    planner_config: dict = field(default_factory=dict)
    filter_config: dict = field(default_factory=dict)
    extractor_config: dict = field(default_factory=dict)

    @property
    def expanded_queries(self) -> list[str]:
        """Flat, deduped query strings actually used for searching."""
        return self.query_plan.unique_query_strings()

    @property
    def raw_candidate_count(self) -> int:
        return len(self.raw_papers)

    @property
    def deduped_count(self) -> int:
        return len(self.deduped_papers)

    @property
    def filtered_count(self) -> int:
        return len(self.retained_papers)

    def evidence_summary(self) -> dict[str, int]:
        """Coarse counts over `evidence_papers` for reporting. Deliberately
        NOT a relevant/not_relevant/needs_review bucket -- Stage 4a produces
        no such verdict; see evidence_matching.py for the deterministic,
        user-criteria-driven comparison that happens afterward, separately.
        """
        extracted = [p for p in self.evidence_papers if p.evidence is not None]
        no_abstract = [
            p
            for p in extracted
            if any(m.startswith("no abstract available") for m in p.evidence.missing_information)
        ]
        with_process = [p for p in extracted if p.evidence.processes]
        with_method = [p for p in extracted if p.evidence.method.model_class or p.evidence.method.study_role]
        with_system = [
            p
            for p in extracted
            if p.evidence.system.studied_systems or p.evidence.system.protein_or_system_names
        ]
        return {
            "extracted": len(extracted),
            "no_abstract": len(no_abstract),
            "with_process_evidence": len(with_process),
            "with_method_evidence": len(with_method),
            "with_system_evidence": len(with_system),
        }


def run_pipeline(
    user_query: str,
    *,
    limit_per_query: int = 15,
    sources: tuple[str, ...] = ("semantic_scholar", "openalex"),
    concept_resolver: ConceptResolver | None = None,
    query_planner: QueryPlanner | None = None,
    relevance_filter: RelevanceFilter | None = None,
    evidence_extractor: PaperEvidenceExtractor | None = None,
) -> PipelineResult:
    resolver = concept_resolver or BaselineConceptResolver()
    planner = query_planner or BaselineQueryPlanner()
    filterer = relevance_filter or RuleBasedRelevanceFilter()
    extractor = evidence_extractor or BaselineEvidenceExtractor()

    profile = resolver.resolve(user_query)
    plan = planner.plan(profile)

    # Grouped (not plan.unique_query_strings()) so PlannedQuery.category
    # survives into the search calls below -- unique_query_strings() would
    # flatten it away before we ever get a chance to stamp provenance.
    grouped_queries = group_queries_by_normalized_key(plan.queries)

    raw: list[Paper] = []
    errors: list[str] = []

    for query, categories in grouped_queries.values():
        if "semantic_scholar" in sources:
            try:
                papers = search_semantic_scholar(query, limit=limit_per_query)
                stamp_provenance(papers, query=query, categories=categories)
                raw.extend(papers)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"semantic_scholar[{query!r}]: {exc}")
        if "openalex" in sources:
            try:
                papers = search_openalex(query, limit=limit_per_query)
                stamp_provenance(papers, query=query, categories=categories)
                raw.extend(papers)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"openalex[{query!r}]: {exc}")

    deduped = dedupe_papers(raw)
    retained, rejected = filterer.filter(profile, deduped)
    # Stage 4a is query-independent by design -- extract_many() never sees
    # `profile`. Matching that extraction against this (or any other) profile
    # happens separately, in evidence_matching.py.
    evidence_papers = extractor.extract_many(retained)

    return PipelineResult(
        user_query=user_query,
        concept_profile=profile,
        query_plan=plan,
        raw_papers=raw,
        deduped_papers=deduped,
        retained_papers=retained,
        rejected_papers=rejected,
        evidence_papers=evidence_papers,
        source_errors=errors,
        sources=sources,
        limit_per_query=limit_per_query,
        resolver_config=_config_snapshot(resolver),
        planner_config=_config_snapshot(planner),
        filter_config=_config_snapshot(filterer),
        extractor_config=_config_snapshot(extractor),
    )
