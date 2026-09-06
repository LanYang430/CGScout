"""Unit tests for the Stage 0 concept resolvers. No real network calls --
LLMConceptResolver is exercised against a fake `requests.Session`."""

from __future__ import annotations

import json

from beam_agent.concept_resolver import BaselineConceptResolver, LLMConceptResolver


def test_extracts_structural_term_and_its_synonyms():
    profile = BaselineConceptResolver().resolve("200-aa beta-helix protein folding")

    terms = {t.term for t in profile.structural_terms}
    assert "beta-helix" in terms
    matched = next(t for t in profile.structural_terms if t.term == "beta-helix")
    assert "beta-solenoid" in matched.synonyms
    assert "parallel beta-helix" in matched.synonyms


def test_matches_structural_term_via_synonym_form_in_query():
    # Query says "beta-solenoid", not "beta-helix" -- should still resolve to
    # the canonical "beta-helix" cluster.
    profile = BaselineConceptResolver().resolve("beta-solenoid protein folding")

    terms = {t.term for t in profile.structural_terms}
    assert "beta-helix" in terms


def test_extracts_methodology_term():
    profile = BaselineConceptResolver().resolve("coarse-grained model of protein folding")

    terms = {t.term for t in profile.methodology_terms}
    assert "coarse-grained" in terms


def test_extracts_scientific_task():
    profile = BaselineConceptResolver().resolve("folding kinetics of a small protein")

    terms = {t.term for t in profile.scientific_tasks}
    assert "folding kinetics" in terms


def test_extracts_size_constraint():
    profile = BaselineConceptResolver().resolve("200-aa beta-helix protein folding")

    assert len(profile.size_constraints) == 1
    c = profile.size_constraints[0]
    assert c.value == 200
    assert c.unit == "aa"
    assert c.raw_text.lower() == "200-aa"


def test_baseline_resolver_never_populates_related_families_or_representative_systems():
    # The old hardcoded beta-helix -> autotransporter/pertactin/P22 tailspike
    # association table was removed entirely -- BaselineConceptResolver is
    # Layer 0 only (lexical vocabulary matching), never LLM-backed concept
    # expansion. These fields stay on the schema for existing consumers but
    # are never populated by this resolver, for any query.
    profile = BaselineConceptResolver().resolve("200-aa beta-helix protein folding")

    assert profile.related_families == []
    assert profile.representative_systems == []
    assert profile.expanded_concepts == []


def test_baseline_resolver_direct_lexical_matches_keep_source_and_confidence():
    profile = BaselineConceptResolver().resolve("200-aa beta-helix protein folding")

    for term in profile.structural_terms:
        assert term.source == "baseline"
        assert term.confidence == 1.0


def test_residual_words_land_in_other_constraints():
    profile = BaselineConceptResolver().resolve("200-aa beta-helix protein folding coarse-grained model")

    # "model" isn't part of any matched cluster or the size constraint, so it
    # should survive as an uncategorized residual term rather than vanish.
    assert "model" in profile.other_constraints


def test_matched_cluster_words_do_not_leak_into_other_constraints():
    profile = BaselineConceptResolver().resolve("200-aa beta-helix protein folding coarse-grained model")

    assert "coarse-grained" not in profile.other_constraints
    assert "folding" not in profile.other_constraints
    assert "protein" not in profile.other_constraints


def test_resolver_name_recorded_for_provenance():
    profile = BaselineConceptResolver().resolve("folding")
    assert profile.resolver == "BaselineConceptResolver"


def test_raw_query_preserved_verbatim_stripped():
    profile = BaselineConceptResolver().resolve("  folding kinetics  ")
    assert profile.raw_query == "folding kinetics"


def test_query_with_no_recognizable_concepts_does_not_crash():
    profile = BaselineConceptResolver().resolve("xyz completely unrecognized query terms")

    assert profile.structural_terms == []
    assert profile.methodology_terms == []
    assert profile.scientific_tasks == []
    assert profile.size_constraints == []


def test_all_terms_flattens_every_category():
    profile = BaselineConceptResolver().resolve("200-aa beta-helix protein folding coarse-grained model")

    flattened = profile.all_terms()
    assert len(flattened) == len(profile.structural_terms) + len(profile.related_families) + len(
        profile.representative_systems
    ) + len(profile.scientific_tasks) + len(profile.methodology_terms)


# --- LLMConceptResolver -------------------------------------------------------


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
    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.requests: list[dict] = []

    def post(self, url, *, json, timeout):  # noqa: A002 -- matches requests.Session.post's signature
        self.requests.append({"url": url, "json": json, "timeout": timeout})
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_llm_resolver_runs_layer_0_deterministically_underneath():
    session = _FakeSession([_FakeResponse({"expanded_concepts": []})])
    resolver = LLMConceptResolver(session=session)

    profile = resolver.resolve("200-aa beta-helix protein folding coarse-grained model")

    terms = {t.term for t in profile.structural_terms}
    assert "beta-helix" in terms
    assert len(profile.size_constraints) == 1


def test_llm_resolver_populates_expanded_concepts_with_no_confidence_field():
    raw_response = {
        "expanded_concepts": [
            {
                "term": "autotransporter passenger domains",
                "relationship": "biological system class known to adopt this fold",
                "basis": "many autotransporter passenger domains fold into right-handed beta-helices",
            }
        ]
    }
    session = _FakeSession([_FakeResponse(raw_response)])
    resolver = LLMConceptResolver(session=session)

    profile = resolver.resolve("200-aa beta-helix protein folding coarse-grained model")

    assert len(profile.expanded_concepts) == 1
    concept = profile.expanded_concepts[0]
    assert concept.term == "autotransporter passenger domains"
    assert concept.relationship == "biological system class known to adopt this fold"
    assert concept.basis
    assert concept.source == "llm-proposed"
    assert not hasattr(concept, "confidence")


def test_llm_resolver_never_populates_related_families_or_representative_systems():
    session = _FakeSession([_FakeResponse({"expanded_concepts": []})])
    resolver = LLMConceptResolver(session=session)

    profile = resolver.resolve("200-aa beta-helix protein folding")

    assert profile.related_families == []
    assert profile.representative_systems == []


def test_llm_resolver_degrades_to_empty_expansion_on_request_failure():
    session = _FakeSession([_FailingResponse()])
    resolver = LLMConceptResolver(session=session)

    profile = resolver.resolve("200-aa beta-helix protein folding")

    assert profile.expanded_concepts == []
    # Layer 0 still ran -- a failed LLM call degrades only the expansion, not
    # the whole resolve() call.
    assert profile.structural_terms


def test_llm_resolver_records_its_own_name_as_resolver_provenance():
    session = _FakeSession([_FakeResponse({"expanded_concepts": []})])
    resolver = LLMConceptResolver(model="qwen3:8b", session=session)

    profile = resolver.resolve("folding")

    assert profile.resolver == "LLMConceptResolver:qwen3:8b"
