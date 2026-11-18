# CGScout

CGScout was developed under the working name BEAM.

CGScout is a literature-retrieval agent for computational biophysicists. Given a modeling question about a protein system, it finds and ranks papers
that show how coarse-grained (CG) models were built and used for that system — or for structurally and functionally related systems.

## Pipeline

```
user request
  → planning        P1 scientific planner (system, objective, search context)  →  P2 expert prior (related-system hypotheses)
  → retrieval       Direct route (flat queries) + 3A (concept queries) + 3B (related-system discovery, Hop-1 → Hop-2; P3–P6)
  → Stage 3         cheap rule filter, review filter, embedding classifier ("did the authors simulate?"), Europe PMC abstract recovery
  → Stage 4         P7 tier-1 screen → P8 per-paper evidence extraction → P9 judge → ranked, explained recommendations
```

The order of these steps is fixed and lives in one place, `cgscout/agent.py` (`run` -> `execute` -> `retrieve` -> `assess`): a workflow-driven agent, where the LLM does the scientific
planning and the evidence judgement and results are not guaranteed to repeat run to run. Every LLM step goes through one injected backend (`cgscout.llm`, OpenAI `gpt-5.6-luna`);
retrieval uses OpenAlex (and Europe PMC for abstracts).
Stage -> function -> prompt: [docs/architecture.md](docs/architecture.md). The nine system prompts as text files: [prompts/](prompts/INDEX.md).

## Install

Python ≥ 3.12.

```bash
pip install .            # or: pip install ".[dev]" to run the tests
```

Keys (environment): `OPENAI_API_KEY` (planning, extraction, judging, and the Stage 3 embedding classifier); `OPENALEX_API_KEY` and `OPENALEX_MAILTO` are optional but
recommended — keyless OpenAlex access has a very small daily budget.

## Use

```bash
cgscout                       # interactive: describe your system and question; clarifying questions are asked when needed
cgscout --no-stage4           # stop after Stage 3 (list ordered by route tier and year, not a relevance ranking)
cgscout --case-id my-run      # tag every LLM call of the session in the usage log
```

### From Python

```python
from cgscout.agent import Clarification, run
from cgscout.llm import get_backend

backend = get_backend()                                   # one backend per run (needs OPENAI_API_KEY)
out = run("How were coarse-grained models built for the EspP autotransporter beta-helix?", backend=backend)
while isinstance(out, Clarification):                     # the planner asks only when the ambiguity changes what to search
    out = run(input(out.question + "\n> "), backend=backend, session=out.session)
if out.assessment is None:                                # Stage 4 runs only when something survived Stage 3
    print("no candidates found")
else:
    for a in out.assessment.recommended:
        print(f"[{a.rank}] {a.title} ({a.year})  grade {a.final_grade}/3")
```

`run` returns an `AgentResult` (`.plan`, `.prior`, `.retrieval` with the per-axis logs and the Stage-3 verdicts, `.assessment`). To start from a plan you already have, call
`execute(plan, prior, backend=backend)`; `AgentConfig(budgets=..., stage4=...)` changes the retrieval switches and the Stage-4 settings (the defaults are what the command line runs with).

Settings (the legacy `BEAM_*` spellings of these are still read, with a one-time notice):

| variable | meaning | default |
|---|---|---|
| `CGSCOUT_CACHE_DIR` | cache directory (search results, extractions, judgments) | `<repo>/.cache` in a checkout, `~/.cache/cgscout` when installed |
| `CGSCOUT_USAGE_LOG_PATH` | JSONL log with one row per LLM call (tokens, cost) | `.cache/llm_usage.jsonl` |
| `CGSCOUT_LLM_MODEL` / `CGSCOUT_LLM_EFFORT` | model and reasoning effort | `gpt-5.6-luna` / `medium` |
| `CGSCOUT_LLM_BACKEND` | only `openai` is supported; any other value is an error | `openai` |

## Repository layout

```
src/cgscout/          the package: agent (the pipeline), planning/ retrieval/ sources/ screening/ assessment/ + llm, cache, models, config, cli
prompts/              the nine system prompts as text files + SHA256SUMS + INDEX.md, generated from the code (tests/test_prompts_export.py keeps them in sync)
tests/                unit and structure tests (pytest); test_golden_replay.py is skipped unless a replay fixture is configured
benchmark/            not installed: evaluation/ (freeze_plans, run_benchmark, arms, provenance, harness, recorder, replay), baselines/, generation/ (benchmark construction), training/
docs/                 architecture.md, reproduce.md, phase3_cleanup_record.md (what the Phase 3 cleanup removed or moved, and how to undo it)
```

## Tests

```bash
pip install ".[dev]"
pytest
```

## Reproducing the paper evaluation

See [docs/reproduce.md](docs/reproduce.md). The paper configuration is `cgscout.config.PAPER`; the default entry point (`cgscout`) uses the library defaults.

## History

The working name survives only in the history and in the legacy `BEAM_*` environment variables, which are still read.
The earlier v1 pipeline, the old session path and the benchmark-construction and regression scripts of that time are not part of this repository's main line; they are kept in the git tag
`legacy-pre-cgscout` (Python files only: the data, results and raw outputs of those experiments are not published, so the tag shows how things were done but cannot reproduce them; see
[docs/reproduce.md](docs/reproduce.md)).
