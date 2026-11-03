"""Enforces this session's "one BEAM run, one LLM backend, no mixed
execution" invariant (llm_backend.py::get_backend()'s own docstring) at the
two real call sites that used to violate it: Hop-1 related-system
extraction and its terminology-expansion step -- both used to hardcode
Ollama/Qwen (mellea_adapter._instruct*) regardless of the configured
backend.

No live network / no live Ollama: OpenAILunaBackend and MelleaQwenBackend
are both real classes here (construction has no network/Ollama side
effect -- see their own __init__), but their `.structured()` methods are
monkeypatched per-test to counting stubs that return a validated schema
instance. This proves the ISOLATION property itself (which backend's
transport got touched), not any particular model's output.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks" / "concept_resolver_reasoning"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks" / "concept_resolver_reasoning" / "mellea_integration"))

import pytest  # noqa: E402

from llm_backend import get_backend, OpenAILunaBackend, MelleaQwenBackend  # noqa: E402
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


def test_get_backend_qwen_env_returns_qwen_backend(monkeypatch):
    monkeypatch.setenv("BEAM_LLM_BACKEND", "qwen")
    backend = get_backend()
    assert isinstance(backend, MelleaQwenBackend)
    assert backend.name == "qwen"


def test_qwen_backend_constructs_with_zero_mellea_ollama_dependency():
    """MelleaQwenBackend() must be constructible (and its .model readable)
    without importing mellea/ollama -- the import is lazy, inside
    .structured() only. If mellea_adapter (which does `import mellea` at
    module load) were already imported by something else in this process,
    this assertion is trivially true but harmless; the real guarantee this
    locks in is that MelleaQwenBackend.__init__ itself never imports it."""
    backend = MelleaQwenBackend()
    assert backend.model == "qwen3:8b"
    assert backend.name == "qwen"


def test_gpt_backend_hop1_extraction_never_touches_qwen_backend(monkeypatch):
    gpt = _CountingBackend("openai", "gpt-5.6-luna")
    qwen_spy_calls = []
    monkeypatch.setattr(
        MelleaQwenBackend, "structured",
        lambda self, **kw: qwen_spy_calls.append(kw) or RichSystemExtractionLLM(systems=[]),
    )

    result = extract_related_systems_rich(
        title="A paper about an autotransporter passenger domain",
        abstract="This autotransporter has a beta-helix passenger domain.",
        structural_class="autotransporter passenger domain",
        backend=gpt, basis="distinctive_structure",
    )

    assert gpt.calls == 1
    assert qwen_spy_calls == []  # the Qwen/Ollama backend's .structured() was NEVER invoked
    assert result.systems == []


def test_qwen_backend_hop1_extraction_never_touches_openai_backend(monkeypatch):
    qwen = _CountingBackend("qwen", "qwen3:8b")
    openai_spy_calls = []
    monkeypatch.setattr(
        OpenAILunaBackend, "structured",
        lambda self, **kw: openai_spy_calls.append(kw) or RichSystemExtractionLLM(systems=[]),
    )

    result = extract_related_systems_rich(
        title="A paper about an autotransporter passenger domain",
        abstract="This autotransporter has a beta-helix passenger domain.",
        structural_class="autotransporter passenger domain",
        backend=qwen, basis="distinctive_structure",
    )

    assert qwen.calls == 1
    assert openai_spy_calls == []  # the OpenAI backend's .structured() was NEVER invoked
    assert result.systems == []


def test_backend_is_a_required_argument_no_hidden_default():
    """No hidden mixed-backend fallback: omitting `backend` is a hard
    TypeError, never a silent default to some other backend."""
    with pytest.raises(TypeError):
        extract_related_systems_rich(  # noqa: this call is expected to raise
            title="t", abstract="a", structural_class="c",
        )


def test_cache_key_is_backend_specific(monkeypatch, tmp_path):
    """A GPT-mode extraction and a qwen-mode extraction of the exact same
    paper text must never collide in the dev cache -- they are genuinely
    different model outputs. Verified by observing two DIFFERENT backends
    both get a real .structured() call for the identical (title, abstract,
    structural_class), i.e. neither one's result was served from a cache
    entry the other backend wrote."""
    monkeypatch.setenv("BEAM_CACHE_DIR", str(tmp_path / "cache"))

    gpt = _CountingBackend("openai", "gpt-5.6-luna")
    qwen = _CountingBackend("qwen", "qwen3:8b")
    kwargs = dict(
        title="A paper about an autotransporter passenger domain",
        abstract="This autotransporter has a beta-helix passenger domain.",
        structural_class="autotransporter passenger domain",
        basis="distinctive_structure",
    )
    extract_related_systems_rich(backend=gpt, **kwargs)
    extract_related_systems_rich(backend=qwen, **kwargs)
    assert gpt.calls == 1
    assert qwen.calls == 1  # NOT served from gpt's cache entry
