"""Enforces the "one run, one LLM backend, no mixed execution" invariant
(llm_backend.py::get_backend()'s own docstring) at the real call site that
matters most: Hop-1 related-system extraction must call exactly the backend
it was given and never fall back to any other transport.

Also covers get_backend(): `openai` (default) is the only supported value --
anything else is an explicit error, never a silent fallback.

No live network: OpenAILunaBackend is a real class here (construction only
builds a lazy client), and its `.structured()` is monkeypatched per-test to a
counting stub. This proves the ISOLATION property itself (which backend's
transport got touched), not any particular model's output.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks" / "concept_resolver_reasoning"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks" / "concept_resolver_reasoning" / "mellea_integration"))

import pytest  # noqa: E402

from llm_backend import get_backend, OpenAILunaBackend  # noqa: E402
from transfer_entity_extraction import extract_related_systems_rich, RichSystemExtractionLLM  # noqa: E402


class _CountingBackend:
    """A minimal LLMBackend stand-in used only to prove `extract_related_
    systems_rich` calls exactly the backend it was given -- not to test
    either real backend's own transport."""
    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.calls = 0

    def structured(self, *, system, user, schema, label, effort=None,
                    purpose=None, case_id=None, timeout=None):
        self.calls += 1
        return schema(systems=[])


def test_get_backend_openai_env_returns_openai_backend(monkeypatch):
    monkeypatch.setenv("BEAM_LLM_BACKEND", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")  # __init__ only builds a lazy client
    backend = get_backend()
    assert isinstance(backend, OpenAILunaBackend)
    assert backend.name == "openai"


def test_get_backend_unset_env_defaults_to_openai(monkeypatch):
    monkeypatch.delenv("BEAM_LLM_BACKEND", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    backend = get_backend()
    assert isinstance(backend, OpenAILunaBackend)
    assert backend.name == "openai"


@pytest.mark.parametrize("name", ["qwen", "claude", "gemini"])
def test_get_backend_rejects_any_non_openai_backend(monkeypatch, name):
    """Only 'openai' is supported: another value is a hard ValueError, never a silent fallback to openai."""
    monkeypatch.setenv("BEAM_LLM_BACKEND", name)
    with pytest.raises(ValueError, match="openai"):
        get_backend()


def test_hop1_extraction_calls_only_the_injected_backend(monkeypatch):
    """The injected backend is the ONLY one whose .structured() runs: the real OpenAILunaBackend transport is never touched."""
    injected = _CountingBackend("openai", "gpt-5.6-luna")
    openai_spy_calls = []
    monkeypatch.setattr(
        OpenAILunaBackend, "structured",
        lambda self, **kw: openai_spy_calls.append(kw) or RichSystemExtractionLLM(systems=[]),
    )

    result = extract_related_systems_rich(
        title="A paper about an autotransporter passenger domain",
        abstract="This autotransporter has a beta-helix passenger domain.",
        structural_class="autotransporter passenger domain",
        backend=injected, basis="distinctive_structure",
    )

    assert injected.calls == 1
    assert openai_spy_calls == []  # OpenAILunaBackend.structured() was NEVER invoked
    assert result.systems == []


def test_backend_is_a_required_argument_no_hidden_default():
    """No hidden mixed-backend fallback: omitting `backend` is a hard
    TypeError, never a silent default to some other backend."""
    with pytest.raises(TypeError):
        extract_related_systems_rich(  # noqa: this call is expected to raise
            title="t", abstract="a", structural_class="c",
        )


def test_cache_key_is_backend_specific(monkeypatch, tmp_path):
    """Extractions of the exact same paper text by two different backends
    (name + model) must never collide in the dev cache -- they are genuinely
    different model outputs. Verified by observing two DIFFERENT backends
    both get a real .structured() call for the identical (title, abstract,
    structural_class), i.e. neither one's result was served from a cache
    entry the other backend wrote."""
    monkeypatch.setenv("BEAM_CACHE_DIR", str(tmp_path / "cache"))

    gpt = _CountingBackend("openai", "gpt-5.6-luna")
    other = _CountingBackend("other-backend", "other-model")
    kwargs = dict(
        title="A paper about an autotransporter passenger domain",
        abstract="This autotransporter has a beta-helix passenger domain.",
        structural_class="autotransporter passenger domain",
        basis="distinctive_structure",
    )
    extract_related_systems_rich(backend=gpt, **kwargs)
    extract_related_systems_rich(backend=other, **kwargs)
    assert gpt.calls == 1
    assert other.calls == 1  # NOT served from gpt's cache entry
