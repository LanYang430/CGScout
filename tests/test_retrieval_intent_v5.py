"""build_retrieval_intent (v5): FactType.SYSTEM_IDENTITY -> grounded_system_
description, and the SystemIdentityExtractor's span is used when A produced
no system fact ("chignolin folding" fused case). No LLM.

Transitively imports RetrievalIntent -> concept_resolver_session -> mellea,
so this needs beam_agent/.venv-mellea.
"""
import sys
from pathlib import Path

import pytest

pytest.importorskip("mellea")

BENCH = Path(__file__).resolve().parent.parent / "benchmarks" / "concept_resolver_reasoning"
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(BENCH / "mellea_integration"))

import run_abcd_test_v5 as v5  # noqa: E402
from concept_resolver_session import ConceptResolverSession  # noqa: E402
from query_planner_handoff import build_retrieval_intent  # noqa: E402


def _fact(fid, ftype, span, turn=1):
    return v5.GroundedFact(fact_id=fid, fact_type=ftype, value=span, source_span=span, source_turn=turn)


def test_system_identity_fact_feeds_grounded_system_description():
    s = ConceptResolverSession(authoritative_facts=[
        _fact("f1", v5.FactType.SYSTEM_IDENTITY, "Wnt-Wls"),
        _fact("f2", v5.FactType.SCIENTIFIC_OBJECTIVE, "the binding interface"),
    ])
    intent = build_retrieval_intent(s)
    assert "Wnt-Wls" in intent.grounded_system_description
    assert "the binding interface" in intent.scientific_objective


def test_extractor_span_used_when_A_made_no_system_fact():
    # "chignolin folding" -> A tagged it all as one scientific_objective fact
    s = ConceptResolverSession(
        authoritative_facts=[_fact("f1", v5.FactType.SCIENTIFIC_OBJECTIVE, "chignolin folding")],
        system_identified=True,
        system_span="chignolin",
    )
    intent = build_retrieval_intent(s)
    assert "chignolin" in intent.grounded_system_description


def test_extractor_span_not_duplicated_when_A_already_captured_it():
    s = ConceptResolverSession(
        authoritative_facts=[_fact("f1", v5.FactType.SYSTEM_IDENTITY, "a 200-aa beta-helix protein")],
        system_identified=True,
        system_span="200-aa beta-helix protein",
    )
    intent = build_retrieval_intent(s)
    assert len(intent.grounded_system_description) == 1


def test_system_identity_and_structural_description_accumulate():
    s = ConceptResolverSession(authoritative_facts=[
        _fact("f1", v5.FactType.SYSTEM_IDENTITY, "a beta-helix protein"),
        _fact("f2", v5.FactType.STRUCTURAL_DESCRIPTION, "right-handed parallel beta-helix"),
        _fact("f3", v5.FactType.SYSTEM_SIZE, "200-aa"),
    ])
    intent = build_retrieval_intent(s)
    assert "a beta-helix protein" in intent.grounded_system_description
    assert "right-handed parallel beta-helix" in intent.grounded_system_description
    assert "200-aa" in intent.constraints  # size stays separate, no overwrite
