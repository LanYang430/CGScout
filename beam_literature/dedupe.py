"""Cross-source deduplication.

This is the piece the plan specifically calls out as easy to skip and important
not to: without it, a paper returned by both Semantic Scholar and OpenAlex looks
like two separate pieces of evidence to anything built downstream (LLM extraction,
comparison tables). Merge first, always.

Strategy, in order:
1. Exact DOI match (normalized, case-insensitive) -- authoritative when present.
2. Fallback: normalized title + year match, for papers missing a DOI on one side.

When two Paper records merge, we keep the richer one as the base (prefer whichever
already has an abstract / DOI / more fields filled in) and union their `sources`
and `external_ids` so provenance from both is preserved.
"""

from __future__ import annotations

from beam_literature.models import Paper


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
    """Merge duplicate Papers across sources, preserving input order otherwise."""
    by_doi: dict[str, Paper] = {}
    by_title_year: dict[str, Paper] = {}
    order: list[str] = []  # dedupe keys in first-seen order, for stable output

    def _key_for(p: Paper) -> str:
        doi_key = p.dedupe_key_doi()
        return f"doi::{doi_key}" if doi_key else f"ty::{p.dedupe_key_title_year()}"

    merged_by_key: dict[str, Paper] = {}

    for p in papers:
        doi_key = p.dedupe_key_doi()
        ty_key = p.dedupe_key_title_year()

        existing_key = None
        if doi_key and doi_key in by_doi:
            existing_key = f"doi::{doi_key}"
        elif ty_key in by_title_year:
            existing_key = f"ty::{ty_key}"

        if existing_key is not None:
            merged = _merge(merged_by_key[existing_key], p)
            merged_by_key[existing_key] = merged
            if doi_key:
                by_doi[doi_key] = merged
            by_title_year[ty_key] = merged
        else:
            key = _key_for(p)
            merged_by_key[key] = p
            order.append(key)
            if doi_key:
                by_doi[doi_key] = p
            by_title_year[ty_key] = p

    return [merged_by_key[k] for k in order]
