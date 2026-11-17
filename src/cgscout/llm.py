"""LLM backend abstraction.

The LLMBackend protocol is what every LLM-calling module takes as its
`backend` argument. This is the ONLY file that imports the `openai` package
or constructs an OpenAI client; no other file should reach for it directly.

One backend implements the protocol:
  - OpenAILunaBackend: GPT-5.6 Luna via the Responses API + structured
    outputs (`client.responses.parse(text_format=<pydantic model>, ...)`).
    The backend of the live path (cli.py) and of every
    evaluated run.
The protocol is kept as the single extension point for another backend; no
other backend is part of this project (the earlier optional local Qwen/Mellea
backend is historical code, kept only in the git tag `legacy-pre-cgscout`).

Usage logging: every .structured() call appends ONE line to
CGSCOUT_USAGE_LOG_PATH (default .cache/llm_usage.jsonl -- .cache/ is already
gitignored) before returning. Fields: ts, backend, model, effort, label,
purpose, case_id, input_tokens, cached_input_tokens, output_tokens,
latency_s, estimated_cost_usd. This is the ONLY source of truth for real
spend anywhere in this codebase -- nothing here or downstream estimates
cost from prompt length; every number in the log came from the API
response's own `usage` block.

OPENAI_API_KEY is read by the `openai` SDK from the environment
(`OpenAI()` with no arguments) -- it is never read, echoed, or logged by
code in this module, and never appears in the usage log.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional, Protocol, TypeVar

from pydantic import BaseModel

from cgscout import config as _env

T = TypeVar("T", bound=BaseModel)


class LLMBackend(Protocol):
    name: str  # "openai"
    model: str  # e.g. "gpt-5.6-luna" -- exposed for cache keys, labels, logging

    def structured(
        self, *, system: str, user: str, schema: type[T],
        label: str, effort: Optional[str] = None,
        purpose: Optional[str] = None, case_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> T:
        """One structured-output call. Returns a validated instance of
        `schema`. Raises on failure (backend error, refusal, or a response
        with no parseable output) -- callers decide how to handle that,
        this layer never silently returns a guessed/default value.

        `timeout` (seconds) bounds this ONE call's worst-case wall time when
        given; omit it to keep a backend's own default (unbounded for
        OpenAI's SDK default). Every caller that batches many calls concurrently (e.g. Hop-1
        related-system extraction, one call per paper) should pass one, so a
        single hung call can never stall the whole batch -- see
        extraction.py."""
        ...


# ---------------------------------------------------------------------------
# CallFn adapter -- for the OLDER (str, str, dict) -> str call convention
# ---------------------------------------------------------------------------
#
# intake_classifiers.py's generic classifiers (expand_terminology(), ...)
# take a `call: CallFn` of shape `(system_prompt: str, user_message: str,
# schema: dict) -> str` -- predates LLMBackend. make_call_fn() bridges an
# injected LLMBackend into that shape, so every call site (e.g.
# agent.py's terminology expansion step, part of the
# Hop-1 chain) goes through the one configured backend -- the "one run, one
# backend" invariant get_backend() documents.

def make_call_fn(
    backend: "LLMBackend", *, response_schema: type[T], label: str,
    case_id: Optional[str] = None, timeout: Optional[float] = None,
) -> Callable[[str, str, dict], str]:
    """Returns a CallFn-shaped closure over `backend` for ONE specific
    pydantic response schema (a CallFn's own third argument -- a raw JSON-
    schema dict -- is a legacy artifact; this bridge ignores it and validates
    against `response_schema` instead, which is the REAL schema every
    existing caller already builds its prompt/cache-key around)."""
    def call(system_prompt: str, user_message: str, _schema_dict: dict) -> str:
        result = backend.structured(
            system=system_prompt, user=user_message, schema=response_schema,
            label=label, case_id=case_id, timeout=timeout,
        )
        return result.model_dump_json()
    return call


# ---------------------------------------------------------------------------
# usage logging -- shared by every backend, one JSONL row per call
# ---------------------------------------------------------------------------

_USAGE_LOG_PATH = Path(_env.get("USAGE_LOG_PATH") or ".cache/llm_usage.jsonl")


@dataclass
class UsageRecord:
    ts: float
    backend: str
    model: str
    effort: Optional[str]
    label: str
    purpose: Optional[str]
    case_id: Optional[str]
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    latency_s: float
    estimated_cost_usd: float


def _log_usage(rec: UsageRecord) -> None:
    _USAGE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_USAGE_LOG_PATH, "a") as f:
        f.write(json.dumps(asdict(rec)) + "\n")


def read_usage_log(path: Optional[Path] = None) -> list[dict]:
    """Convenience reader for benchmark reporting -- parses every row
    written by _log_usage above. Returns [] if the log doesn't exist yet."""
    p = path or _USAGE_LOG_PATH
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# OpenAI GPT-5.6 Luna backend
# ---------------------------------------------------------------------------

# Confirmed against the user's own OpenAI account pricing at setup time
# (2026-09), GPT-5.6 Luna standard tier. A price change or a different
# model is a one-line edit here, not a rewrite -- every cost computation
# below reads from this dict, nothing is hardcoded inline.
LUNA_PRICE_PER_1M_USD = {"input": 0.20, "cached_input": 0.02, "output": 1.20}

DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_EFFORT = "medium"  # configurable -- see get_backend() / CGSCOUT_LLM_EFFORT


class OpenAILunaBackend:
    name = "openai"

    def __init__(
        self, model: str = DEFAULT_MODEL, default_effort: str = DEFAULT_EFFORT,
        price_per_1m: dict[str, float] = LUNA_PRICE_PER_1M_USD,
    ):
        from openai import OpenAI  # local import -- see module docstring
        self._client = OpenAI()  # reads OPENAI_API_KEY from env; never pass a literal key
        self.model = model
        self.default_effort = default_effort
        self._price = price_per_1m

    def structured(self, *, system, user, schema, label, effort=None,
                    purpose=None, case_id=None, timeout=None):
        effort = effort or self.default_effort
        client = self._client.with_options(timeout=timeout) if timeout is not None else self._client
        t0 = time.time()
        resp = client.responses.parse(
            model=self.model,
            instructions=system,
            input=user,
            text_format=schema,
            reasoning={"effort": effort},
        )
        latency = time.time() - t0

        u = resp.usage
        cached = (u.input_tokens_details.cached_tokens if u and u.input_tokens_details else 0) or 0
        input_tokens = u.input_tokens if u else 0
        output_tokens = u.output_tokens if u else 0
        billable_input = max(input_tokens - cached, 0)
        cost = (
            billable_input / 1e6 * self._price["input"]
            + cached / 1e6 * self._price["cached_input"]
            + output_tokens / 1e6 * self._price["output"]
        )
        _log_usage(UsageRecord(
            ts=t0, backend=self.name, model=self.model, effort=effort, label=label,
            purpose=purpose, case_id=case_id, input_tokens=input_tokens,
            cached_input_tokens=cached, output_tokens=output_tokens,
            latency_s=round(latency, 3), estimated_cost_usd=round(cost, 6),
        ))

        if resp.output_parsed is None:
            raise RuntimeError(
                f"OpenAILunaBackend {label!r}: no parsed output (refusal or schema "
                f"mismatch) -- output_text={resp.output_text!r}"
            )
        return resp.output_parsed


def get_backend() -> LLMBackend:
    """CGSCOUT_LLM_BACKEND=openai (default; the only supported value -- anything
    else raises ValueError). CGSCOUT_LLM_MODEL / CGSCOUT_LLM_EFFORT override the
    openai backend's model / default reasoning effort.  (Every CGSCOUT_* name
    also reads the legacy BEAM_* spelling -- see cgscout.config.)

    Global invariant (one BEAM run uses exactly one LLM backend, no mixed
    execution): call this ONCE per run/session and thread the SAME instance
    into every module that needs an LLM -- ScientificPlanner,
    ExpertPriorPlanner, and Hop-1 related-system extraction
    (extraction.extract_related_systems_rich) alike. None of
    those modules call get_backend() themselves or fall back to a different
    backend on their own; they take `backend` as an explicit, required
    argument. Any future module that needs an LLM call should follow the
    same pattern -- take `backend`, don't resolve it internally."""
    name = _env.get("LLM_BACKEND", "openai").strip().lower()
    if name == "openai":
        model = _env.get("LLM_MODEL", DEFAULT_MODEL)
        effort = _env.get("LLM_EFFORT", DEFAULT_EFFORT)
        return OpenAILunaBackend(model=model, default_effort=effort)
    raise ValueError(f"Unsupported CGSCOUT_LLM_BACKEND={name!r} (only 'openai' is supported)")


def backend_label(backend: LLMBackend) -> str:
    """Human-readable one-line description of the active backend, for a
    CLI startup banner (`print(f"LLM backend: {backend_label(backend)}")`)
    -- makes the single-backend-per-run invariant visible to the user, not
    just enforced in code."""
    if backend.name == "openai":
        return "GPT-5.6 Luna"
    return f"{backend.name} ({getattr(backend, 'model', 'unknown model')})"
