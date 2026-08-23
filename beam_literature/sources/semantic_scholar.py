"""Thin client for the Semantic Scholar Graph API (relevance search only).

Docs: https://api.semanticscholar.org/api-docs/graph

No API key required. If S2_API_KEY is set in the environment, it's sent as the
x-api-key header, which raises your rate limit -- everything works without it,
just more slowly under load.
"""

from __future__ import annotations

import os
import time
from typing import Any

import requests

from beam_literature.models import ExternalIds, Paper

_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_FIELDS = ",".join(
    [
        "title",
        "authors",
        "year",
        "venue",
        "abstract",
        "externalIds",
        "openAccessPdf",
        "url",
    ]
)


def search_semantic_scholar(
    query: str,
    limit: int = 20,
    *,
    max_retries: int = 3,
    timeout: float = 15.0,
) -> list[Paper]:
    """Query Semantic Scholar's relevance search and return normalized Papers.

    Retries with backoff on 429 (rate limit); raises on other HTTP errors after
    exhausting retries so a broken source fails loudly instead of silently
    returning an empty result set.
    """
    headers = {}
    api_key = os.environ.get("S2_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    params = {
        "query": query,
        "limit": min(limit, 100),  # API cap per request
        "fields": _FIELDS,
    }

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(
                _BASE_URL, params=params, headers=headers, timeout=timeout
            )
            if resp.status_code == 429:
                time.sleep(2**attempt)
                continue
            resp.raise_for_status()
            data = resp.json()
            return [_to_paper(item) for item in data.get("data", [])]
        except requests.RequestException as exc:  # noqa: PERF203
            last_error = exc
            time.sleep(min(2**attempt, 8))

    raise RuntimeError(
        f"Semantic Scholar search failed after {max_retries} attempts"
    ) from last_error


def _to_paper(item: dict[str, Any]) -> Paper:
    external = item.get("externalIds") or {}
    oa_pdf = item.get("openAccessPdf") or {}
    return Paper(
        title=item.get("title") or "(untitled)",
        authors=[a.get("name", "") for a in (item.get("authors") or []) if a.get("name")],
        year=item.get("year"),
        doi=external.get("DOI"),
        abstract=item.get("abstract"),
        venue=item.get("venue") or None,
        sources=["semantic_scholar"],
        external_ids=ExternalIds(
            semantic_scholar_id=item.get("paperId"),
            arxiv_id=external.get("ArXiv"),
            pmid=external.get("PubMed"),
        ),
        is_open_access=bool(oa_pdf.get("url")) if oa_pdf else None,
        oa_pdf_url=oa_pdf.get("url"),
        url=item.get("url"),
    )
