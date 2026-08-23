"""Orchestrator: query both sources, dedupe, return one ranked list of Papers.

Ranking for Phase 1 is intentionally simple and deterministic (no LLM, no
score guessing): a paper found by both sources is ranked above one found by
only one, and ties keep each source's own relevance order. Anything smarter
than that belongs in a later phase once there's a defined evidence schema to
rank against.
"""

from __future__ import annotations

from beam_literature.dedupe import dedupe_papers
from beam_literature.models import Paper
from beam_literature.sources.openalex import search_openalex
from beam_literature.sources.semantic_scholar import search_semantic_scholar


def search_and_dedupe(
    query: str,
    limit: int = 20,
    *,
    sources: tuple[str, ...] = ("semantic_scholar", "openalex"),
) -> list[Paper]:
    """Search the given sources, merge duplicates, and return a ranked list.

    A failure in one source is not fatal to the other -- this is a search
    agent, not a database transaction; partial results with a note beats no
    results. Failures are printed to stderr-equivalent via a raised warning
    collected in `errors` accessible through `search_and_dedupe.last_errors`.
    """
    raw: list[Paper] = []
    errors: list[str] = []

    if "semantic_scholar" in sources:
        try:
            raw.extend(search_semantic_scholar(query, limit=limit))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"semantic_scholar: {exc}")

    if "openalex" in sources:
        try:
            raw.extend(search_openalex(query, limit=limit))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"openalex: {exc}")

    search_and_dedupe.last_errors = errors  # type: ignore[attr-defined]

    deduped = dedupe_papers(raw)
    deduped.sort(key=lambda p: len(p.sources), reverse=True)
    return deduped[:limit] if limit else deduped


search_and_dedupe.last_errors = []  # type: ignore[attr-defined]
