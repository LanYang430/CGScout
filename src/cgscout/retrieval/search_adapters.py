"""Search wrappers shared by the retrieval routes.

This module owns zero scientific logic and zero retrieval logic of its own.
It only:
  1. wraps beam_agent.search.search_and_dedupe() in capturing closures so
     every real Paper object retrieved (across the routes) is cached by
     paper_ref -- the sidecar RouteRetrievalResult.papers alone cannot carry
     (it only has paper_ref/title/year, not abstract/authors/etc.), and
     Stage 3 (filter.filter_papers) needs full Paper
     objects;
  2. defines the Hop-1 / Hop-2 retrieval depths and sources
     (STRUCTURAL_*), and the two small adapters papers_for_stage3() /
     provenance_index() from a RouteRetrievalResult to the Stage 3 contract.

The old ConceptResolverSession -> EvidenceRoutePlan wiring that used to live
here (build_evidence_route_plan_from_session / run_live_route_retrieval) is
historical code, kept in the git tag `legacy-pre-cgscout`.
"""
from __future__ import annotations


from cgscout.retrieval.direct.route_plan import (  # noqa: E402
    RouteRetrievalBudgets,
    RouteRetrievalResult,
)
from cgscout.retrieval.related_systems.discovery import _paper_ref  # noqa: E402 -- same paper_ref convention used throughout

from cgscout.models import Paper  # noqa: E402
from cgscout.retrieval.search import search_and_dedupe  # noqa: E402

# Hop-1 per-query retrieval depth: fixed at 15, per the cross-case retrieval-
# depth sensitivity experiment (beta-helix/coiled-coil/TIM-barrel, this
# session) -- was 8.
STRUCTURAL_HOP1_SEARCH_LIMIT = 15
# Hop-2 per-query retrieval depth: was 8, then 15 (diagnosed against the
# live end-to-end run: the query "pertactin coarse-grained model" genuinely
# returns the target paper at rank 14/24 -- top-8 cut it off; not a ranking
# problem, a pure depth cutoff). Now 20, per the Hop-2 simplification
# (one query per system instead of three) -- this is now the ONLY retrieval
# depth per system, so it carries what three queries' worth of depth used
# to. Hop-2 ranking ((-n_support, first_seen_rank) for system selection)
# is unchanged by this.
STRUCTURAL_HOP2_SEARCH_LIMIT = 20
STRUCTURAL_SEARCH_SOURCES = ("openalex",)


def _make_capturing_search_fns(paper_cache: dict[str, Paper], budgets: RouteRetrievalBudgets):
    """One shared paper_cache across all 5 routes -- the sidecar paper_ref ->
    full Paper mapping Stage 3 needs, since RouteRetrievalResult.papers
    itself only carries paper_ref/title/year, not abstract/authors/etc."""

    def flat_search_fn(query: str, limit: int) -> list[Paper]:
        papers = search_and_dedupe(query, limit=limit)
        for p in papers:
            paper_cache.setdefault(_paper_ref(p), p)
        return papers

    def structural_hop1_search_fn(query: str) -> list[Paper]:
        papers = search_and_dedupe(query, limit=STRUCTURAL_HOP1_SEARCH_LIMIT, sources=STRUCTURAL_SEARCH_SOURCES)
        for p in papers:
            paper_cache.setdefault(_paper_ref(p), p)
        return papers

    def structural_hop2_search_fn(query: str) -> list[Paper]:
        papers = search_and_dedupe(query, limit=STRUCTURAL_HOP2_SEARCH_LIMIT, sources=STRUCTURAL_SEARCH_SOURCES)
        for p in papers:
            paper_cache.setdefault(_paper_ref(p), p)
        return papers

    return flat_search_fn, structural_hop1_search_fn, structural_hop2_search_fn


def make_need_semantic_search_fn(paper_cache: dict[str, Paper]):
    """P3: the SEMANTIC counterpart of flat_search_fn, for the `need_semantic` route (OpenAlex `search.semantic=`, capped at 50
    by the engine).  Same shared paper_cache as the other routes, so Stage 3 sees the full Paper."""
    def need_semantic_search_fn(query: str, limit: int) -> list[Paper]:
        papers = search_and_dedupe(query, limit=limit, mode="semantic")
        for p in papers:
            paper_cache.setdefault(_paper_ref(p), p.model_copy(update={"retrieval_modes": ["semantic"]}))
        return papers

    return need_semantic_search_fn


def papers_for_stage3(result: RouteRetrievalResult, paper_cache: dict[str, Paper]) -> list[Paper]:
    """Smallest adapter to the existing Stage 3 contract (filter_papers(list[Paper])).
    Provenance is NOT flattened away: result.papers (indexable by paper_ref)
    remains the sidecar; the caller reattaches it after Stage 3/4 by looking
    up each retained/extracted Paper's paper_ref in result.papers."""
    out = []
    missing = []
    for rp in result.papers:
        p = paper_cache.get(rp.paper_ref)
        if p is None:
            missing.append(rp.paper_ref)
            continue
        out.append(p)
    if missing:
        # Should not happen -- every RouteRetrievalResult.papers entry was
        # produced by a search_fn call this module itself wrapped. Surfaced,
        # not silently dropped, if it ever does.
        import sys as _sys
        print(f"route_retrieval_live_adapter: WARNING -- {len(missing)} paper_ref(s) in "
              f"RouteRetrievalResult had no cached Paper object: {missing}", file=_sys.stderr)
    return out


def provenance_index(result: RouteRetrievalResult) -> dict[str, "object"]:
    """paper_ref -> RetrievedPaper (route provenance + hop1_provenance),
    for reattaching provenance to a Paper after Stage 3/4 by paper_ref."""
    return {rp.paper_ref: rp for rp in result.papers}
