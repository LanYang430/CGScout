"""Unit tests for Stage 4a evidence extraction. No real network calls --
OllamaPaperEvidenceExtractor is exercised against a fake `requests.Session`.
"""

from __future__ import annotations

import json

import pytest

from beam_agent.evidence_extraction import BaselineEvidenceExtractor, OllamaPaperEvidenceExtractor
from beam_agent.models import Paper


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"message": {"content": json.dumps(self._payload)}}


class _FailingResponse:
    def raise_for_status(self) -> None:
        raise RuntimeError("connection refused")


class _FakeSession:
    """Records every request made and replays a queue of canned responses,
    one per call, in order -- enough to test multiple papers in one run
    without needing a real Ollama server."""

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.requests: list[dict] = []

    def post(self, url, *, json, timeout):  # noqa: A002 -- matches requests.Session.post's signature
        self.requests.append({"url": url, "json": json, "timeout": timeout})
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _paper(**overrides) -> Paper:
    defaults = dict(
        title="A coarse-grained model for protein folding",
        abstract="We present a coarse-grained model and demonstrate folding of a 100-residue protein.",
        year=2020,
    )
    defaults.update(overrides)
    return Paper(**defaults)


# --- BaselineEvidenceExtractor ------------------------------------------------


def test_baseline_extractor_makes_no_network_calls_and_fills_empty_evidence():
    extractor = BaselineEvidenceExtractor()
    papers = extractor.extract_many([_paper()])

    assert papers[0].evidence is not None
    assert papers[0].evidence.method.model_class == []
    assert papers[0].evidence.processes == []
    assert papers[0].evidence.extractor == "BaselineEvidenceExtractor"


def test_baseline_extractor_flags_missing_abstract():
    extractor = BaselineEvidenceExtractor()
    papers = extractor.extract_many([_paper(abstract=None)])

    assert papers[0].evidence.missing_information == [
        "no abstract available; cannot extract without full text or an abstract"
    ]


def test_baseline_extractor_never_mutates_the_input_paper_in_place():
    extractor = BaselineEvidenceExtractor()
    original = _paper()

    extractor.extract_many([original])

    assert original.evidence is None


# --- OllamaPaperEvidenceExtractor --------------------------------------------


def test_ollama_extractor_never_sends_a_profile_or_query_in_the_request():
    """The whole point of Stage 4a: no ScientificConceptProfile anywhere in
    the request payload, only the paper's own title/abstract/metadata."""
    raw_response = {
        "method": {"model_class": ["coarse-grained"], "study_role": ["applied"]},
        "system": {
            "studied_systems": ["three-helix bundle"],
            "protein_or_system_names": [],
        },
        "processes": [{"raw_text": "folding demonstrated", "normalized_label": "folding"}],
        "missing_information": [],
        "reasoning_summary": "clear CG folding paper",
    }
    session = _FakeSession([_FakeResponse(raw_response)])
    extractor = OllamaPaperEvidenceExtractor(session=session)

    papers = extractor.extract_many([_paper()])

    sent_payload = json.dumps(session.requests[0]["json"])
    assert "ScientificConceptProfile" not in sent_payload
    assert "structural_terms" not in sent_payload
    assert papers[0].evidence.method.model_class == ["coarse-grained"]
    assert papers[0].evidence.system.studied_systems == ["three-helix bundle"]
    assert papers[0].evidence.processes[0].normalized_label == "folding"


def test_ollama_extractor_passes_through_system_evidence_fields_unmodified():
    """SystemEvidence v3 fields are all open-text lists -- no numeric
    coercion needed (the old approx_size_residues int-coercion logic was
    removed along with the field itself). This just confirms the new fields
    round-trip through validation untouched."""
    raw_response = {
        "method": {},
        "system": {
            "studied_systems": ["twelve proteins with multiple secondary structure arrangements"],
            "family_or_fold": ["TIM-barrel fold"],
            "protein_or_system_names": ["ubiquitin"],
        },
        "processes": [],
    }
    session = _FakeSession([_FakeResponse(raw_response)])
    extractor = OllamaPaperEvidenceExtractor(session=session)

    papers = extractor.extract_many([_paper()])

    ev = papers[0].evidence.system
    assert ev.studied_systems == ["twelve proteins with multiple secondary structure arrangements"]
    assert ev.family_or_fold == ["TIM-barrel fold"]
    assert ev.protein_or_system_names == ["ubiquitin"]


def test_ollama_extractor_degrades_on_request_failure_without_crashing():
    session = _FakeSession([_FailingResponse()])
    extractor = OllamaPaperEvidenceExtractor(session=session)

    papers = extractor.extract_many([_paper()])

    assert papers[0].evidence is not None
    assert "extraction failed" in papers[0].evidence.missing_information[0]


def test_ollama_extractor_degrades_on_schema_validation_failure():
    # "processes" entries are missing the required normalized_label field.
    raw_response = {"method": {}, "system": {}, "processes": [{"raw_text": "folding"}]}
    session = _FakeSession([_FakeResponse(raw_response)])
    extractor = OllamaPaperEvidenceExtractor(session=session)

    papers = extractor.extract_many([_paper()])

    assert "schema validation" in papers[0].evidence.missing_information[0]


def test_ollama_extractor_skips_papers_without_an_abstract():
    session = _FakeSession([])  # no calls expected
    extractor = OllamaPaperEvidenceExtractor(session=session)

    papers = extractor.extract_many([_paper(abstract=None)])

    assert session.requests == []
    assert "no abstract available" in papers[0].evidence.missing_information[0]


def test_ollama_extractor_handles_multiple_papers_independently():
    responses = [
        _FakeResponse({"method": {"model_class": ["coarse-grained"]}, "system": {}, "processes": []}),
        _FakeResponse({"method": {"model_class": ["all-atom"]}, "system": {}, "processes": []}),
    ]
    session = _FakeSession(responses)
    extractor = OllamaPaperEvidenceExtractor(session=session)

    papers = extractor.extract_many([_paper(title="Paper 1"), _paper(title="Paper 2")])

    assert papers[0].evidence.method.model_class == ["coarse-grained"]
    assert papers[1].evidence.method.model_class == ["all-atom"]
    assert len(session.requests) == 2


# --- Stage 4 extraction cache (tests/conftest.py isolates BEAM_CACHE_DIR per test) --


def test_ollama_extractor_second_identical_call_is_a_cache_hit_no_second_request():
    raw = {"method": {"model_class": ["coarse-grained"]}, "system": {}, "processes": []}
    # only ONE response queued -- a second real request would raise IndexError
    # (list.pop(0) on empty), so this only passes if the cache actually skips it
    session = _FakeSession([_FakeResponse(raw)])
    extractor = OllamaPaperEvidenceExtractor(session=session)

    first = extractor.extract_many([_paper()])
    second = extractor.extract_many([_paper()])  # identical paper -- must be a cache hit

    assert len(session.requests) == 1
    assert first[0].evidence.method.model_class == ["coarse-grained"]
    assert second[0].evidence.method.model_class == ["coarse-grained"]
    assert second[0].evidence.extractor == extractor.name  # re-stamped fresh on the hit path too


def test_ollama_extractor_different_paper_content_is_a_cache_miss():
    responses = [
        _FakeResponse({"method": {"model_class": ["coarse-grained"]}, "system": {}, "processes": []}),
        _FakeResponse({"method": {"model_class": ["all-atom"]}, "system": {}, "processes": []}),
    ]
    session = _FakeSession(responses)
    extractor = OllamaPaperEvidenceExtractor(session=session)

    a = extractor.extract_many([_paper(abstract="Abstract A")])
    b = extractor.extract_many([_paper(abstract="Abstract B")])

    assert len(session.requests) == 2
    assert a[0].evidence.method.model_class == ["coarse-grained"]
    assert b[0].evidence.method.model_class == ["all-atom"]


def test_ollama_extractor_does_not_cache_a_request_failure():
    session = _FakeSession([_FailingResponse(), _FakeResponse(
        {"method": {"model_class": ["coarse-grained"]}, "system": {}, "processes": []})])
    extractor = OllamaPaperEvidenceExtractor(session=session)

    first = extractor.extract_many([_paper()])
    assert "extraction failed" in first[0].evidence.missing_information[0]

    second = extractor.extract_many([_paper()])  # same paper -- must NOT reuse the failure
    assert len(session.requests) == 2
    assert second[0].evidence.method.model_class == ["coarse-grained"]


def test_ollama_extractor_does_not_cache_a_schema_validation_failure():
    bad = _FakeResponse({"method": {}, "system": {}, "processes": [{"raw_text": "folding"}]})  # missing normalized_label
    good = _FakeResponse({"method": {"model_class": ["coarse-grained"]}, "system": {}, "processes": []})
    session = _FakeSession([bad, good])
    extractor = OllamaPaperEvidenceExtractor(session=session)

    first = extractor.extract_many([_paper()])
    assert "schema validation" in first[0].evidence.missing_information[0]

    second = extractor.extract_many([_paper()])  # same paper -- must NOT reuse the failure
    assert len(session.requests) == 2
    assert second[0].evidence.method.model_class == ["coarse-grained"]


def test_ollama_extractor_different_model_name_is_a_cache_miss():
    """Same paper, same prompt/schema -- but a different declared model must
    still be treated as a different cache entry (switching models must never
    silently reuse another model's cached output)."""
    responses = [
        _FakeResponse({"method": {"model_class": ["coarse-grained"]}, "system": {}, "processes": []}),
        _FakeResponse({"method": {"model_class": ["all-atom"]}, "system": {}, "processes": []}),
    ]
    session = _FakeSession(responses)

    a = OllamaPaperEvidenceExtractor(model="qwen3:8b", session=session).extract_many([_paper()])
    b = OllamaPaperEvidenceExtractor(model="qwen3:14b", session=session).extract_many([_paper()])

    assert len(session.requests) == 2
    assert a[0].evidence.method.model_class == ["coarse-grained"]
    assert b[0].evidence.method.model_class == ["all-atom"]
