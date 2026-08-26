"""Stage 4a: single-pass, query-independent evidence extraction (title +
abstract + metadata only).

Sits between the cheap keyword-overlap filter and deterministic matching:

    search -> dedupe -> cheap rule filter -> HERE -> deterministic matching -> full-text retrieval -> ...

Design: one LLM call per paper, no profile, no decision
--------------------------------------------------------
An extractor (`OllamaPaperEvidenceExtractor` or `BaselineEvidenceExtractor`)
is given ONLY a paper's title/abstract/metadata -- never a
`ScientificConceptProfile` -- and produces every field of a `PaperEvidence`
(see models.py): what modeling method the paper used and what role it played
(`MethodEvidence`), what physical system it studied (`SystemEvidence`), and
what process(es)/capabilities it demonstrated (`ProcessEvidenceItem` list).

This is deliberately the ONLY LLM call in the whole pipeline, and it never
sees a search target:

- Query-independent means cacheable. The same paper surfaced by two
  different user queries (or the same query run twice) is extracted once;
  see evidence_matching.py for how a cached PaperEvidence is compared
  against a *particular* query's profile afterward, entirely in Python.
- No profile in the prompt means target-concept leakage into
  `system.studied_systems` / `system.family_or_fold` / etc. is structurally
  impossible, not just prompted against -- there is nothing to copy from.
- No decision. Earlier iterations of this stage asked the LLM to also judge
  eligibility/task_match/evidence_strength and derive a relevant/not_relevant
  verdict (see `benchmarks/stage4_design_experiments/` for the single-call vs
  two-call experiments that explored this). Both approaches showed the same
  problem: a small model doing several *judgments* in one schema-constrained
  generation conflates them (e.g. task_match absorbing system/topology-
  mismatch reasoning it shouldn't). The fix that actually worked was removing
  judgment from the LLM's job entirely, not further prompt engineering.
  Extraction-only also means every field here is objective and checkable
  against the abstract, which two-call's own diagnostics (~2x slower per
  paper, two of seventeen inputs producing runaway generation at the 300s
  timeout) never fully escaped either -- a smaller, single, judgment-free
  schema per paper is both simpler AND cheaper than either previous design,
  and cacheability makes repeated queries cheaper still.

`PaperEvidenceExtractor` is a Protocol specifically so a different LLM
backend (Claude, OpenAI, a bigger local model) can be dropped in later
without touching the pipeline that calls it.
"""

from __future__ import annotations

import json
from typing import Optional, Protocol

import requests

from beam_agent.models import (
    MethodEvidence,
    Paper,
    PaperEvidence,
    ProcessEvidenceItem,
    SystemEvidence,
)

__all__ = [
    "MethodEvidence",
    "SystemEvidence",
    "ProcessEvidenceItem",
    "PaperEvidence",
    "PaperEvidenceExtractor",
    "BaselineEvidenceExtractor",
    "OllamaPaperEvidenceExtractor",
]


class PaperEvidenceExtractor(Protocol):
    def extract_many(self, papers: list[Paper]) -> list[Paper]:
        """Return copies of `papers` with `.evidence` set. Never takes a
        `ScientificConceptProfile` -- see module docstring."""
        ...


def _no_abstract_evidence(extractor_name: str) -> PaperEvidence:
    """The one case with no real extraction to do: no abstract at all means
    no extractor (LLM or baseline) can say anything about the paper's own
    text. Distinct from an extracted-but-empty PaperEvidence."""
    return PaperEvidence(
        missing_information=["no abstract available; cannot extract without full text or an abstract"],
        extractor=extractor_name,
    )


def _annotate_paper(paper: Paper, evidence: PaperEvidence) -> Paper:
    return paper.model_copy(update={"evidence": evidence})


class BaselineEvidenceExtractor:
    """Makes no LLM calls; every paper with an abstract still gets an empty
    (all fields blank) PaperEvidence rather than one populated with real
    judgment. Useful for tests and for running the rest of the pipeline
    without a local model available.
    """

    NAME = "BaselineEvidenceExtractor"

    def extract_many(self, papers: list[Paper]) -> list[Paper]:
        out = []
        for paper in papers:
            if not paper.abstract:
                evidence = _no_abstract_evidence(self.NAME)
            else:
                evidence = PaperEvidence(
                    reasoning_summary="baseline extractor makes no LLM calls; every field left blank",
                    extractor=self.NAME,
                )
            out.append(_annotate_paper(paper, evidence))
        return out


# ---------------------------------------------------------------------------
# Ollama-backed extractor
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a scientific literature extraction assistant for BEAM, a tool that helps computational \
biophysicists organize coarse-grained (CG) molecular simulation literature. You are given ONLY a \
paper's title, abstract, and basic metadata -- no search query, no research target, and nothing to \
judge relevance against. Your ONLY job is to extract objective facts this paper's own text states \
about (1) the modeling/simulation method it used, (2) the physical system it studied, and (3) the \
process(es)/capabilities it demonstrated. Do not invent anything not stated -- if the abstract does \
not say something, leave that field empty ([]) or null, never guess.

This extraction is reused across many different future research questions, so it must describe the \
paper itself, not any particular search or target.

METHOD EVIDENCE: model_class is the modeling/method class(es) explicitly named or clearly described \
(e.g. "coarse-grained", "all-atom", "structure-based", "elastic-network", "Go model") -- open \
vocabulary, leave empty if no simulation model/method is actually discussed. study_role is what role \
the model played in THIS paper -- "developed" (introduces a new model/method), "applied" (uses an \
existing model to study a system), "benchmarked"/"validated" (tests or compares a model's \
performance/accuracy against something), "reviewed_only" (a review/survey that discusses models \
without running one). Multiple roles can apply. Do NOT judge whether this makes the paper "eligible" \
for anything -- just report what the text says.

SYSTEM EVIDENCE: studied_systems is what system(s) the paper actually studied, simulated, \
benchmarked, developed against, or applied its method to -- kept close to the abstract's own words, \
one entry per distinct system or system-group. Capture identity + structure/architecture + \
qualitative scale together as ONE faithful phrase rather than splitting them apart or normalizing \
into a coarser category -- e.g. "three-helix bundle", "twelve proteins with multiple \
secondary-structure arrangements", "C-terminal segment of the pertactin passenger domain", \
"intrinsically disordered proteins", "protein-ligand complexes". Never infer from general \
biological/domain background what the abstract itself doesn't say -- a protein you happen to know is \
typically membrane-associated is NOT evidence this paper studied it in a membrane, unless the \
abstract itself says so. NEVER include a simulation parameter (duration, temperature, timestep, box \
size) in studied_systems even if a number sits right next to a system description in the same \
sentence -- only include what the text says about the SYSTEM, never about how long or how it was \
simulated ("approximately 9 ms" describes simulation duration, not system size; never let it become \
"9" of anything). A general capability claim about the METHOD (e.g. "this force field can handle \
proteins containing about 1000 residues") is NOT a studied system either, unless the paper's own \
actually-studied examples are that size -- read for what was really used as a benchmark/test/example \
system, not what the method is generically capable of.

family_or_fold is a fold/family name ONLY when the abstract explicitly names one AS a fold/family \
classification (e.g. "TIM-barrel fold", "immunoglobulin fold", "autotransporter passenger domain \
family") -- leave empty otherwise. An inline topology phrase like "three-helix bundle" or \
"three-stranded antiparallel beta-sheet" belongs in studied_systems, not here, even though it \
describes architecture -- do not promote a studied_systems-style description into family_or_fold \
just to have something to put there.

protein_or_system_names is any specific protein/system name explicitly named in the paper's own \
text -- bare names, not full descriptive phrases (those belong in studied_systems). It is completely \
normal for studied_systems and protein_or_system_names to overlap or say the same thing when a name \
is all the abstract gives you.

Leave any system-evidence field empty/null if the paper doesn't specify -- that is the correct, \
honest answer, not a strike against anything.

PROCESS EVIDENCE: processes is a list of {raw_text, normalized_label} pairs, one per process or \
capability the paper's abstract states the model/method was actually shown to study or demonstrate. \
raw_text should stay close to the abstract's own words -- only what is stated, never inferred. \
normalized_label restates the same capability as a short, generic, reusable snake_case label (e.g. \
folding, unfolding, conformational_dynamics, pocket_gating, binding, aggregation, assembly, \
translocation, kinetics, thermodynamics, stability) so the same underlying capability gets the same \
label across different papers -- prefer an existing common term over inventing a near-duplicate, but \
you are not restricted to a fixed list; invent a new concise label if none fit. Do NOT force \
everything into a folding-specific vocabulary -- report whatever process the paper actually studied, \
even if it's something else entirely (e.g. pocket gating, membrane insertion, phase separation).

MISSING INFORMATION: list anything a full-text read would likely clarify that the abstract leaves \
unstated (e.g. "abstract doesn't state residue count", "abstract doesn't name a specific protein \
system").

Do not assess eligibility, evidence strength, confidence, or relevance to any target -- none of that \
is part of your job here.

Respond with a single JSON object matching the given schema, and nothing else."""


def _build_user_message(paper: Paper) -> str:
    return (
        f"Paper to extract evidence from:\n"
        f"Title: {paper.title}\n"
        f"Year: {paper.year or 'unknown'}\n"
        f"Venue: {paper.venue or 'unknown'}\n"
        f"Abstract: {paper.abstract}"
    )


_LLM_FACING_FIELDS = {"method", "system", "processes", "missing_information", "reasoning_summary"}


def _llm_output_schema() -> dict:
    schema = PaperEvidence.model_json_schema()
    schema["properties"] = {k: v for k, v in schema["properties"].items() if k in _LLM_FACING_FIELDS}
    schema["required"] = [k for k in schema.get("required", []) if k in _LLM_FACING_FIELDS]
    # Drop class-level docstrings (PaperEvidence's own, and each nested
    # model's in $defs) from the schema sent to the model -- those describe
    # this codebase's internal design (e.g. mention `ScientificConceptProfile`
    # for context) and add nothing the per-field `description`s below don't
    # already say more usefully. Field-level descriptions are untouched;
    # they carry the actual per-field prompting instructions.
    schema.pop("description", None)
    for defn in schema.get("$defs", {}).values():
        defn.pop("description", None)
    return schema


def _post_ollama_chat(
    *,
    session: requests.Session,
    host: str,
    model: str,
    timeout: float,
    system_prompt: str,
    user_message: str,
    schema: dict,
    think: bool = False,
) -> dict:
    """One Ollama /api/chat request with JSON-schema-constrained output
    (constrained decoding -- malformed JSON is mechanically impossible), NOT
    tool-calling, since tool-call reliability has documented regressions on
    some current small local models.

    `think=False` by default: qwen3:8b is a hybrid-reasoning model that emits
    a `<think>...</think>` preamble before the actual JSON unless told not
    to. Measured on this extraction schema, disabling it cut wall-clock time
    from ~49s to ~18s per paper with no loss in extraction quality -- pure
    latency the thinking pass wasn't buying anything for on a
    schema-constrained, single-fact-extraction task like this one. Exposed
    as a parameter (not hardcoded) so a harder future extraction task, or a
    different model, can turn it back on.
    """
    response = session.post(
        f"{host}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "format": schema,
            "stream": False,
            "think": think,
            "options": {"temperature": 0},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    content = response.json()["message"]["content"]
    return json.loads(content)


class OllamaPaperEvidenceExtractor:
    """Stage 4a extractor backed by a local model served through Ollama.

    Default model `qwen3:8b` was chosen for structured-extraction quality at
    the 7-8B tier suitable for a 16GB unified-memory machine; `model`/`host`
    are constructor args specifically so this stays swappable -- to a bigger
    local model, or to a different `PaperEvidenceExtractor` implementation
    entirely (Claude, OpenAI, ...) behind the same Protocol. `think=False` by
    default -- see `_post_ollama_chat`'s docstring; pass `think=True` to
    re-enable qwen3's reasoning preamble if a harder future extraction task
    seems to need it.
    """

    def __init__(
        self,
        model: str = "qwen3:8b",
        host: str = "http://localhost:11434",
        timeout: float = 120.0,
        *,
        think: bool = False,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.think = think
        self._session = session or requests.Session()

    @property
    def name(self) -> str:
        return f"OllamaPaperEvidenceExtractor:{self.model}"

    def extract_many(self, papers: list[Paper]) -> list[Paper]:
        out = []
        for paper in papers:
            if not paper.abstract:
                evidence = _no_abstract_evidence(self.name)
            else:
                evidence = self._extract_one(paper)
            out.append(_annotate_paper(paper, evidence))
        return out

    def _extract_one(self, paper: Paper) -> PaperEvidence:
        try:
            raw = _post_ollama_chat(
                session=self._session,
                host=self.host,
                model=self.model,
                timeout=self.timeout,
                system_prompt=_SYSTEM_PROMPT,
                user_message=_build_user_message(paper),
                schema=_llm_output_schema(),
                think=self.think,
            )
        except Exception as exc:  # noqa: BLE001 -- any failure degrades, never crashes the run
            return PaperEvidence(
                missing_information=[f"extraction failed: {exc}"],
                extractor=self.name,
            )

        try:
            return PaperEvidence.model_validate({**raw, "extractor": self.name})
        except Exception as exc:  # noqa: BLE001 -- malformed model output degrades, doesn't crash
            return PaperEvidence(
                missing_information=[f"extraction output failed schema validation: {exc}"],
                extractor=self.name,
            )
