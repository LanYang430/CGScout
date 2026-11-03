"""Which LLM writes the benchmark.  The point is to NOT use the model family BEAM itself runs on (GPT-5.6 Luna in the
planner / expert prior / Stage 4), so queries do not carry that model's wording.  No Claude / Gemini key exists in this
environment yet, so the pilot falls back to the existing OpenAI backend and records the model name on every row
(`generator_model`).  To switch, add a backend with the same `structured(...)` signature (see LLMBackend in
benchmarks/concept_resolver_reasoning/mellea_integration/llm_backend.py) and select it with BENCH_GEN_BACKEND."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MI = ROOT / "benchmarks" / "concept_resolver_reasoning" / "mellea_integration"
for p in (str(ROOT), str(MI), str(MI.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)


def get_generator_backend():
    name = os.environ.get("BENCH_GEN_BACKEND", "openai").strip().lower()
    if name == "openai":
        from llm_backend import OpenAILunaBackend
        model = os.environ.get("BENCH_GEN_MODEL")
        return OpenAILunaBackend(model=model) if model else OpenAILunaBackend()
    raise SystemExit(f"BENCH_GEN_BACKEND={name!r} is not implemented yet (only 'openai'); see this file's docstring.")


def label(backend) -> str:
    return f"{backend.name}:{backend.model}"
