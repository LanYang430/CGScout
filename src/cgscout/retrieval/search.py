"""Orchestrator: query both sources, dedupe, return one ranked list of Papers.

Ranking for Phase 1 is intentionally simple and deterministic (no LLM, no
score guessing): a paper found by both sources is ranked above one found by
only one, and ties keep each source's own relevance order. Anything smarter
than that belongs in a later phase once there's a defined evidence schema to
rank against.
"""

from __future__ import annotations

from cgscout import cache
from cgscout.models import Paper
from cgscout.sources.openalex import search_openalex
from cgscout.sources.semantic_scholar import search_semantic_scholar

# ===== moved verbatim from `retrieval/search_adapters.py` (consolidation A3): the Hop-1 / Hop-2 retrieval depths and sources, used by the 3B route ========
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

# ===== merged from `retrieval/dedupe.py` (moved verbatim; its docstring is this comment) ===================================
# Cross-source deduplication.
#
# This is the piece the plan specifically calls out as easy to skip and important
# not to: without it, a paper returned by both Semantic Scholar and OpenAlex looks
# like two separate pieces of evidence to anything built downstream (LLM extraction,
# comparison tables). Merge first, always.
#
# Strategy, in order:
# 1. Exact DOI match (normalized, case-insensitive) -- authoritative when present.
# 2. Fallback: normalized title + year match, for papers missing a DOI on one side.
#
# When two Paper records merge, we keep the richer one as the base (prefer whichever
# already has an abstract / DOI / more fields filled in) and union their `sources`,
# `found_by_queries`, `query_categories`, and `external_ids` so provenance from both
# is preserved -- a paper found by two different planned queries, or by both
# sources, keeps every query/source that found it after merging into one record.
# ===========================================================================================================================

def _richness(p: Paper) -> tuple[int, int, int]:
    """Rough completeness score used to pick which duplicate becomes the base."""
    return (
        1 if p.abstract else 0,
        1 if p.doi else 0,
        1 if p.oa_pdf_url else 0,
    )


def _merge(a: Paper, b: Paper) -> Paper:
    base, other = (a, b) if _richness(a) >= _richness(b) else (b, a)
    merged = base.model_copy(deep=True)

    merged.sources = sorted(set(base.sources) | set(other.sources))
    merged.found_by_queries = sorted(set(base.found_by_queries) | set(other.found_by_queries))
    merged.query_categories = sorted(set(base.query_categories) | set(other.query_categories))

    if not merged.doi and other.doi:
        merged.doi = other.doi
    if not merged.abstract and other.abstract:
        merged.abstract = other.abstract
    if not merged.venue and other.venue:
        merged.venue = other.venue
    if not merged.oa_pdf_url and other.oa_pdf_url:
        merged.oa_pdf_url = other.oa_pdf_url
    if merged.is_open_access is None and other.is_open_access is not None:
        merged.is_open_access = other.is_open_access
    if not merged.url and other.url:
        merged.url = other.url

    for field in ("semantic_scholar_id", "openalex_id", "arxiv_id", "pmid"):
        if not getattr(merged.external_ids, field):
            setattr(merged.external_ids, field, getattr(other.external_ids, field))

    return merged


def dedupe_papers(papers: list[Paper]) -> list[Paper]:
    """Merge duplicate Papers across sources, preserving input order otherwise.

    `doi_to_merge_key`/`ty_to_merge_key` map a paper's *natural* key (its DOI,
    or its title+year) to the merge-group key it currently belongs to, and
    `merged_by_key` maps that merge-group key to the actual merged Paper.
    Every lookup goes through this same indirection -- doi_to_merge_key and
    ty_to_merge_key are never allowed to point at a merge-group key that
    doesn't exist in merged_by_key. (An earlier version tracked Paper objects
    directly in a by-title-year cache instead of merge keys, which could fall
    out of sync with merged_by_key whenever a paper merged under its DOI key
    while a *different* paper later matched it only by title+year -- that
    second lookup landed on a `ty::...` key that had never actually been
    registered as a merge group, and crashed. Keeping every index as
    "natural key -> merge key" rather than "natural key -> Paper" closes that
    gap: whichever merge key a paper actually merged into is the only thing
    ever recorded against its natural keys.)

    If a paper's DOI matches one existing group while its title+year
    coincidentally matches a different group, DOI wins (matching the "DOI is
    authoritative" precedence documented in the module docstring), and that
    paper's title+year key gets pointed at the DOI-matched group going
    forward. That's a deliberate simplification, not an oversight: two
    genuinely different papers sharing both a title and a publication year is
    the kind of rare coincidence not worth adding a third fallback layer for.
    """
    doi_to_merge_key: dict[str, str] = {}
    ty_to_merge_key: dict[str, str] = {}
    merged_by_key: dict[str, Paper] = {}
    order: list[str] = []  # merge keys in first-seen order, for stable output

    for p in papers:
        doi_key = p.dedupe_key_doi()
        ty_key = p.dedupe_key_title_year()

        existing_merge_key = None
        if doi_key and doi_key in doi_to_merge_key:
            existing_merge_key = doi_to_merge_key[doi_key]
        elif ty_key in ty_to_merge_key:
            existing_merge_key = ty_to_merge_key[ty_key]

        if existing_merge_key is not None:
            merged_by_key[existing_merge_key] = _merge(merged_by_key[existing_merge_key], p)
            if doi_key:
                doi_to_merge_key[doi_key] = existing_merge_key
            ty_to_merge_key[ty_key] = existing_merge_key
        else:
            merge_key = f"doi::{doi_key}" if doi_key else f"ty::{ty_key}"
            merged_by_key[merge_key] = p
            order.append(merge_key)
            if doi_key:
                doi_to_merge_key[doi_key] = merge_key
            ty_to_merge_key[ty_key] = merge_key

    return [merged_by_key[k] for k in order]


# ===== end of the code merged from `retrieval/dedupe.py` ===================================================================


_SEARCH_CACHE_NAMESPACE = "search_and_dedupe"
# v1: 24h, simple and fixed -- OpenAlex/Semantic Scholar indexing and ranking
# both change over time, unlike the Hop-1/Stage-4 extraction caches (whose
# key already fully determines the answer: paper content + exact prompt/
# schema/model). Past this age a cached entry is treated as a miss and
# overwritten with a fresh result, exactly like any other miss.
_SEARCH_CACHE_TTL_SECONDS = 24 * 60 * 60


def search_and_dedupe(
    query: str,
    limit: int = 20,
    *,
    sources: tuple[str, ...] = ("openalex",),
    mode: str = "lexical",
) -> list[Paper]:
    """Search the given sources, merge duplicates, and return a ranked list.

    `mode` ("lexical" default, or "semantic") is threaded straight to
    search_openalex()'s own `mode` -- see that function's docstring.
    Semantic Scholar has no equivalent mode; `mode="semantic"` with
    "semantic_scholar" in `sources` is not a supported combination (no
    caller does this today).

    v1: defaults to OpenAlex only -- Semantic Scholar removed from the
    default, not just deprioritized. Diagnosed live against this session's
    canonical case (no S2_API_KEY set in this environment): all 5 real
    flat-route queries failed after exhausting retries against S2's
    unauthenticated rate limit, contributing exactly 0 papers, while still
    costing ~10.5s/query (~40s total) in retry/backoff before giving up --
    pure waste, not a recall/speed tradeoff. Callers that want Semantic
    Scholar back (e.g. with an S2_API_KEY) can still pass
    sources=("semantic_scholar", "openalex") explicitly; nothing about
    search_semantic_scholar() itself changed.

    A failure in one source is not fatal to the other -- this is a search
    agent, not a database transaction; partial results with a note beats no
    results. Failures are printed to stderr-equivalent via a raised warning
    collected in `errors` accessible through `search_and_dedupe.last_errors`.

    Dev-speed cache (cgscout.cache): keyed on (query, limit, sources, mode) --
    the exact inputs that determine what this function asks the sources for.
    Only a call with ZERO source errors is ever cached, so a rate-limit/
    timeout/malformed-response is never mistaken for "this query genuinely
    has no results" and baked in permanently. A cache hit reproduces the
    same Paper list (same fields, same order) a fresh call would have
    returned at cache-write time -- the ranking/dedupe logic below is
    identical on both paths, only the network round-trip is skipped.
    Entries older than _SEARCH_CACHE_TTL_SECONDS are treated as a miss and
    re-fetched + overwritten -- OpenAlex/S2 indexing and ranking both drift
    over time, unlike the paper-content-keyed extraction caches.
    """
    cache_key = cache.content_hash(query, str(limit), ",".join(sorted(sources)), mode)
    cached = cache.get(_SEARCH_CACHE_NAMESPACE, cache_key, ttl_seconds=_SEARCH_CACHE_TTL_SECONDS)
    if cached is not None:
        search_and_dedupe.last_errors = []  # type: ignore[attr-defined] -- only error-free results are ever cached
        return [Paper.model_validate(d) for d in cached]

    raw: list[Paper] = []
    errors: list[str] = []

    if "semantic_scholar" in sources:
        try:
            raw.extend(search_semantic_scholar(query, limit=limit))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"semantic_scholar: {exc}")

    if "openalex" in sources:
        try:
            raw.extend(search_openalex(query, limit=limit, mode=mode))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"openalex: {exc}")

    search_and_dedupe.last_errors = errors  # type: ignore[attr-defined]

    deduped = dedupe_papers(raw)
    deduped.sort(key=lambda p: len(p.sources), reverse=True)
    result = deduped[:limit] if limit else deduped

    if not errors:
        cache.set(_SEARCH_CACHE_NAMESPACE, cache_key, [p.model_dump(mode="json") for p in result])

    return result


search_and_dedupe.last_errors = []  # type: ignore[attr-defined]
