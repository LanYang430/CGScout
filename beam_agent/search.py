"""Orchestrator: query both sources, dedupe, return one ranked list of Papers.

Ranking for Phase 1 is intentionally simple and deterministic (no LLM, no
score guessing): a paper found by both sources is ranked above one found by
only one, and ties keep each source's own relevance order. Anything smarter
than that belongs in a later phase once there's a defined evidence schema to
rank against.
"""

from __future__ import annotations

from beam_agent import cache
from beam_agent.dedupe import dedupe_papers
from beam_agent.models import Paper
from beam_agent.sources.openalex import search_openalex
from beam_agent.sources.semantic_scholar import search_semantic_scholar

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

    Dev-speed cache (beam_agent.cache): keyed on (query, limit, sources, mode) --
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
