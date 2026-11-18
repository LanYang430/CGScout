# Reproducing the evaluation

## Environment

- Python ≥ 3.12 (the verified dependency set needs it); `pip install ".[dev]"` and `pip install -r benchmark/requirements.txt`.
- Record the exact environment of a run: `pip freeze > requirements-lock.txt` next to the results. The package pins only what must be pinned
  (`scikit-learn>=1.9.1,<1.10`, because the shipped Stage 3 classifier is a scikit-learn pickle saved with 1.9.1); everything else is a lower bound.
- Keys: `OPENAI_API_KEY`; `OPENALEX_API_KEY` (a free key; keyless access allows only a handful of searches per day).

## Run directory (not in the repository)

`CGSCOUT_BENCH_DIR=<run dir>` holds `queries.csv` (columns `Paper ID`, `Latest query (v2)`, `DOI`, `OpenAlex ID`) and everything a run writes.
Use a **new, empty directory** for the final run. Every evaluation setting comes from one place, `cgscout.config.PAPER`; `benchmark/evaluation/arms.py` and the runner read it.

```bash
export CGSCOUT_BENCH_DIR=/path/to/run
python benchmark/evaluation/freeze_plans.py --expect-rows 200      # P1 + P2 once per case -> plans_frozen/ (LLM only, no OpenAlex); add --dry-run to see the state first
export CGSCOUT_STOP_USD=40                                         # spend ceiling for the arm; a full run refuses to start without it
python benchmark/evaluation/run_benchmark.py CUR --expect-rows 200 # retrieval -> Stage 3 -> Stage 4 per case -> results/CUR/<id>.json (resumable)
python benchmark/evaluation/run_benchmark.py CUR --pids B001,B002  # a subset (smoke test); the ceiling then defaults to $5
```

**Selection.** Without `--pids` the run is *full*: it needs `--expect-rows N` (queries.csv must have exactly N rows) and an explicit `CGSCOUT_STOP_USD`. `--pids` must name at least one
case and every name must exist; an empty value or a typo is an error, never "run everything" and never silently dropped. `freeze_plans.py` follows the same rules.

**Exit codes of `run_benchmark.py`.** `0` every selected case has a final result (`OK`, `REFUSED`, `NOT_RUN_CLARIFICATION`) · `2` not started (bad command line, input or plans; nothing spent) ·
`3` OpenAlex budget · `4` started but **incomplete**: the spend ceiling stopped it, or cases are still `ERROR`/`DEGRADED` after 3 passes. The last lines of the log (and stderr) list the unfinished cases.
Raise the ceiling (or fix the cause) and run the same command again: finished cases are never re-run; only unfinished ones are.

**Frozen plans and versions.** Each `plans_frozen/<id>.json` carries a `provenance` block: sha256 of the three modules that define P1 and P2
(`cgscout/planning/{scientific_planner,expert_prior_planner,session}.py`), the model and effort, `cgscout` version, git commit. `run_benchmark.py` refuses to start when a selected case has no plan,
a plan without provenance (older `freeze_plans.py`, or copied from another run), a plan made with different P1/P2 source, another model/effort, or for another query text;
`freeze_plans.py` refuses to add plans next to plans of a different version. Consequences: (1) freeze **after** the last edit of those three modules (prompt wording included), in a fresh directory;
(2) any later edit of them, even a comment, invalidates the plans on purpose; (3) resuming under the same code works normally. P3-P9 are not frozen; they run live with the code of the current checkout.

**What a run records.** `run_config_CUR.json` (rewritten at every start): the profile, the **full effective configuration** (all retrieval budgets, all Stage-4 settings, model and effort,
Stage-3 classifier file hash and threshold, the signature defaults of the retrieval/Stage-4 entry points), which of those differ from the library defaults, input / script / package hashes, git commit and dirty flag,
the selection, the spend ceiling and a summary of the plans' provenance. `run_starts_CUR.jsonl` appends one line per start, and a restart whose configuration or code differs from the previous start logs a warning.
`tests/test_run_benchmark.py` pins the effective values of the evaluated CUR configuration, so a changed library default (e.g. `max_papers_per_query`) or PAPER value fails a test instead of silently changing the run.

## Regression check without LLM or network

`tests/test_golden_replay.py` replays the pipeline offline against a fixed fixture (cached search and LLM results, a stub Stage 3 classifier built from stored flags) and
compares the outputs byte for byte with stored golden outputs:

```bash
python benchmark/evaluation/build_replay_fixture.py --run-dir <earlier run dir> --golden-cases <expected outputs> --out <fixture>
CGSCOUT_REPLAY_FIXTURE=<fixture> pytest tests/test_golden_replay.py
```

## Baselines

`benchmark/baselines/`: `gpt_only.py`, `gpt_web_search.py`, `openalex_semantic.py` (same run-directory convention; query workbook via `CGSCOUT_QUERIES_FILE`).
A Claude or Gemini web-search baseline would be a further independent script here, with its SDK listed only in `benchmark/requirements.txt`.

## Data that is not in the repository

The 200-query workbook, per-run result JSONs, LLM usage logs, `plans_frozen/`, `terminology_memo.json`, and the Stage 3 classifier's training CSV
(`benchmark/training/train_sim_classifier.py` needs it; the shipped `sim_classifier_model.joblib` is used otherwise) belong to the data release.

The scripts that need the author's labelled Stage-3 exports have no built-in path: `benchmark/training/train_sim_classifier.py` takes `--train-csv PATH` (or `CGSCOUT_STAGE3_TRAIN_CSV`;
the older name `BEAM_STAGE3_TRAIN_CSV` is still read), and `benchmark/generation/{l3_verify,overlap_audit,propose14}.py` take `--stage3-data-dir DIR` (or `CGSCOUT_STAGE3_DATA_DIR`) naming the directory
that holds the four `beam_stage3_*.csv` files. Without them each script stops with a message that says what to set; `--help` works without any data.

## Publisher abstracts are not distributed

The construction data of the benchmark under `benchmark/generation/data/` keeps the DOI, title, authors, year and venue of every seed and candidate paper, but not the abstract text, which belongs to the publishers. Where a record once carried `abstract` (`seed_abstract` in the `pilot_v*` files) it now carries `abstract_sha256` (the SHA-256 of the UTF-8 text) and `abstract_chars` (its length; `seed_abstract_sha256` / `seed_abstract_chars` in the pilot files). To check a paper, fetch its abstract by DOI from OpenAlex and compare `hashlib.sha256(text.encode("utf-8")).hexdigest()` with the stored value (a publisher may have corrected an abstract since, in which case they differ). 248 abstracts were replaced this way; the scripts that read them are the archived ones listed below.

## Benchmark-generation scripts are a historical record

`benchmark/generation/` records how the 200-query benchmark was built. It was written against the code before the CGScout restructure and has not been maintained for the
current version: **nine of its scripts are archived / historical workflows that cannot run on CGScout** -- importing them raises `ModuleNotFoundError`. Their outputs under
`benchmark/generation/data/` are kept as they were produced; nothing was re-run, and no script was repaired or migrated.

| script | import that no longer exists | output kept |
|---|---|---|
| `routing_smoke.py` | `cgscout.retrieval.orchestrator` (renamed `cgscout.retrieval.transfer`); `regression_lib` | `data/routing_smoke.json` |
| `routing_smoke_v2.py` | the same two | `data/routing_smoke_v2*.json` (seven files) |
| `generate.py` | `regression_lib` | `data/pilot_v0.*`, `data/pilot_v1.*` |
| `seeds.py` | `regression_lib` | `data/seed_pool.json`, `data/pilot_seeds.json` |
| `propose14.py` | `regression_lib` | `data/seed_proposal_14.json` |
| `overlap_audit.py` | `regression_lib`; `sim_classifier_train` (now `benchmark/training/train_sim_classifier.py`) | `data/curated_pool_audit.json` |
| `l3_find_more.py` | `regression_lib` | `data/l3_more_candidates*.json` |
| `l3_verify.py` | `regression_lib` | `data/l3_verification.json` |
| `probe.py` | `regression_lib` | none kept (it wrote `--out`, by default `data/pilot_v0_probed.*`) |

`regression_lib` was replaced by `benchmark/evaluation/harness.py`, and `run_expert_prior_retrieval` became the body of `cgscout.agent.retrieve`. For the scripts that take
`--stage3-data-dir`, `--help` and the data-directory check still run. The other files of the directory (`finalize_pilot_seeds.py`, `hop1_audit.py`, `make_review.py`, `curated_pool.py`,
`leakage.py`, `prompts.py`, `schemas.py`, `gen_backend.py`, `stage3_data.py`) are helpers or post-processing steps of the same workflow; treat the whole directory as an archive and its
files under `data/` as the construction record of the benchmark.

To trace how the benchmark was built, and to read the scripts next to the modules they were written against, check out the git tag `legacy-pre-cgscout` (`git worktree add ../cgscout-legacy legacy-pre-cgscout`). That tag is the code as it was before the CGScout restructure: the v1 `beam_agent/` package and, under `benchmarks/`, **30 reviewed Python files** -- the benchmark-construction scripts (`benchmarks/beam_bench_v1/`, the originals of the scripts in `benchmark/generation/`), the 13-case regression scripts (`benchmarks/regression_13/`, including `regression_lib.py`) and `benchmarks/concept_resolver_reasoning/mellea_integration/sim_classifier_train.py`. **Only those 30 Python files are kept from `benchmarks/` in the published history.** The data, results, reports and raw LLM outputs that those experiments produced or read are not part of the published history (what belongs to the data release is listed above). Publisher abstracts were removed from every version of the published history before publication. So the tag shows how things were done; it cannot reproduce the historical experiments or their outputs, which needs the inputs, an API key and live OpenAlex / OpenAI access, and no experiment was re-run for this repository. The evaluation itself (`benchmark/evaluation/`, above) does not depend on any of these scripts.
