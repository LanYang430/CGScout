"""Thin client for the OpenAlex Works API (search only).

Docs: https://docs.openalex.org/api-entities/works/search-works

No API key required and no registration needed. OpenAlex asks callers to send
a contact email via the `mailto` param to get into their faster "polite pool" --
that's opt-in here via the OPENALEX_MAILTO env var (unset by default) rather than
hardcoded, since it's your contact info going to a third-party service.
"""

from __future__ import annotations

import os
import time
from typing import Any

import requests

from beam_agent.models import ExternalIds, Paper

_BASE_URL = "https://api.openalex.org/works"
_SELECT = ",".join(
    [
        "id",
        "doi",
        "title",
        "publication_year",
        "authorships",
        "primary_location",
        "open_access",
        "abstract_inverted_index",
    ]
)


def search_openalex(
    query: str,
    limit: int = 20,
    *,
    max_retries: int = 3,
    timeout: float = 15.0,
) -> list[Paper]:
    """Query OpenAlex's search-works endpoint and return normalized Papers."""
    params: dict[str, Any] = {
        "search": query,
        "per_page": min(limit, 200),  # API cap per request
        "select": _SELECT,
    }
    mailto = os.environ.get("OPENALEX_MAILTO")
    if mailto:
        params["mailto"] = mailto
    # OpenAlex's current (2026) usage-based pricing: keyless/anonymous callers
    # get a $0.10/day budget (~10 search requests/day at 10 credits each); a
    # free API key raises that 10x to $1/day. Opt-in via env var, same
    # reasoning as OPENALEX_MAILTO above -- never hardcoded, since it's the
    # user's own key.
    api_key = os.environ.get("OPENALEX_API_KEY")
    if api_key:
        params["api_key"] = api_key

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(_BASE_URL, params=params, timeout=timeout)
            if resp.status_code == 429:
                time.sleep(2**attempt)
                continue
            resp.raise_for_status()
            data = resp.json()
            return [_to_paper(item) for item in data.get("results", [])]
        except requests.RequestException as exc:  # noqa: PERF203
            last_error = exc
            time.sleep(min(2**attempt, 8))

    raise RuntimeError(f"OpenAlex search failed after {max_retries} attempts") from last_error


def _to_paper(item: dict[str, Any]) -> Paper:
    authorships = item.get("authorships") or []
    authors = [
        a.get("author", {}).get("display_name", "")
        for a in authorships
        if a.get("author", {}).get("display_name")
    ]

    primary_location = item.get("primary_location") or {}
    source = primary_location.get("source") or {}
    venue = source.get("display_name")

    open_access = item.get("open_access") or {}
    doi = item.get("doi")
    if doi:
        doi = doi.removeprefix("https://doi.org/")

    openalex_id = item.get("id")
    if openalex_id:
        openalex_id = openalex_id.removeprefix("https://openalex.org/")

    return Paper(
        title=item.get("title") or "(untitled)",
        authors=authors,
        year=item.get("publication_year"),
        doi=doi,
        abstract=_reconstruct_abstract(item.get("abstract_inverted_index")),
        venue=venue,
        sources=["openalex"],
        external_ids=ExternalIds(openalex_id=openalex_id),
        is_open_access=open_access.get("is_oa"),
        oa_pdf_url=open_access.get("oa_url"),
        url=primary_location.get("landing_page_url"),
    )


def _reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str | None:
    """OpenAlex stores abstracts as {word: [positions]} to dodge copyright issues.

    Reassemble into plain text; return None if there's nothing to reconstruct.
    """
    if not inverted_index:
        return None
    positions: dict[int, str] = {}
    for word, idxs in inverted_index.items():
        for idx in idxs:
            positions[idx] = word
    if not positions:
        return None
    return " ".join(positions[i] for i in sorted(positions))
