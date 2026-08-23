"""Unit tests for cross-source dedup -- the part the plan flagged as easy to get wrong.

No network calls: builds Paper objects by hand to simulate "same paper, returned by
both sources, with slightly different metadata completeness."
"""

from beam_literature.dedupe import dedupe_papers
from beam_literature.models import ExternalIds, Paper


def _paper(**kwargs) -> Paper:
    defaults = dict(title="Some Paper", year=2020, sources=["semantic_scholar"])
    defaults.update(kwargs)
    return Paper(**defaults)


def test_same_doi_merges_into_one():
    a = _paper(
        title="Coarse-grained folding of beta-helix proteins",
        doi="10.1000/xyz123",
        sources=["semantic_scholar"],
        external_ids=ExternalIds(semantic_scholar_id="s2-1"),
    )
    b = _paper(
        title="Coarse-grained folding of beta-helix proteins",
        doi="https://doi.org/10.1000/XYZ123",  # different casing/prefix
        abstract="An abstract only OpenAlex had.",
        sources=["openalex"],
        external_ids=ExternalIds(openalex_id="W123"),
    )

    result = dedupe_papers([a, b])

    assert len(result) == 1
    merged = result[0]
    assert set(merged.sources) == {"semantic_scholar", "openalex"}
    assert merged.abstract == "An abstract only OpenAlex had."
    assert merged.external_ids.semantic_scholar_id == "s2-1"
    assert merged.external_ids.openalex_id == "W123"


def test_same_title_year_no_doi_merges():
    a = _paper(title="Upside model benchmarking on 200 residue proteins", year=2019, doi=None)
    b = _paper(
        title="  UPSIDE model  benchmarking on 200 residue proteins ",  # whitespace/case differ
        year=2019,
        doi=None,
        sources=["openalex"],
    )

    result = dedupe_papers([a, b])
    assert len(result) == 1
    assert set(result[0].sources) == {"semantic_scholar", "openalex"}


def test_different_papers_stay_separate():
    a = _paper(title="Paper A", year=2020, doi="10.1/a")
    b = _paper(title="Paper B", year=2021, doi="10.1/b")

    result = dedupe_papers([a, b])
    assert len(result) == 2


def test_richer_record_wins_as_base():
    sparse = _paper(title="Same Paper", year=2020, doi="10.1/x", abstract=None, url=None)
    rich = _paper(
        title="Same Paper",
        year=2020,
        doi="10.1/x",
        abstract="Full abstract text.",
        url="https://example.com/paper",
        oa_pdf_url="https://example.com/paper.pdf",
        is_open_access=True,
        sources=["openalex"],
    )

    result = dedupe_papers([sparse, rich])
    assert len(result) == 1
    assert result[0].abstract == "Full abstract text."
    assert result[0].oa_pdf_url == "https://example.com/paper.pdf"
    assert result[0].is_open_access is True
