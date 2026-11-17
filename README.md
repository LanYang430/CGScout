# CGScout

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

Every LLM step goes through one injected backend (`cgscout.llm`, OpenAI `gpt-5.6-luna`); retrieval uses OpenAlex (and Europe PMC for abstracts).
Module-level details: [docs/architecture.md](docs/architecture.md).

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

Settings (the legacy `BEAM_*` spellings of these are still read, with a one-time notice):

| variable | meaning | default |
|---|---|---|
| `CGSCOUT_CACHE_DIR` | cache directory (search results, extractions, judgments) | `<repo>/.cache` in a checkout, `~/.cache/cgscout` when installed |
| `CGSCOUT_USAGE_LOG_PATH` | JSONL log with one row per LLM call (tokens, cost) | `.cache/llm_usage.jsonl` |
| `CGSCOUT_LLM_MODEL` / `CGSCOUT_LLM_EFFORT` | model and reasoning effort | `gpt-5.6-luna` / `medium` |
| `CGSCOUT_LLM_BACKEND` | only `openai` is supported; any other value is an error | `openai` |

## Repository layout

```
src/cgscout/          the package: planning/ retrieval/ sources/ screening/ assessment/ + llm, cache, models, text, config, cli
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

The earlier v1 pipeline, the old session path and the one-off experiment scripts are not part of this repository's main line; they are kept in the git tag
`legacy-pre-cgscout` (and the branch `legacy`).
