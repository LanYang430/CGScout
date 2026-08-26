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

from beam_agent.models import Paper


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
