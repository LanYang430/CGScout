# Reproducing the evaluation

## Environment

- Python ≥ 3.12 (the verified dependency set needs it); `pip install ".[dev]"` and `pip install -r benchmark/requirements.txt`.
- Record the exact environment of a run: `pip freeze > requirements-lock.txt` next to the results. The package pins only what must be pinned
  (`scikit-learn>=1.9.1,<1.10`, because the shipped Stage 3 classifier is a scikit-learn pickle saved with 1.9.1); everything else is a lower bound.
- Keys: `OPENAI_API_KEY`; `OPENALEX_API_KEY` (a free key; keyless access allows only a handful of searches per day).

## Run directory (not in the repository)

`CGSCOUT_BENCH_DIR=<run dir>` holds `queries.csv` (columns `Paper ID`, `Latest query (v2)`, `DOI`, `OpenAlex ID`) and everything a run writes.
The paper configuration is `cgscout.profiles.PAPER`; `benchmark/evaluation/arms.py` builds the evaluation arm from it.

```bash
export CGSCOUT_BENCH_DIR=/path/to/run
python benchmark/evaluation/freeze_plans.py            # P1 + P2 once per case -> plans_frozen/ (LLM only, no OpenAlex)
python benchmark/evaluation/run_benchmark.py CUR       # retrieval -> Stage 3 -> Stage 4 per case -> results/CUR/<id>.json (resumable)
```

`run_benchmark.py` records, in `run_config_CUR.json`, the model, efforts, Stage 4 settings, input hash, script hash, `cgscout` version and git commit.
Rules (checkpointing, retries, OpenAlex budget handling, spend ceiling `CGSCOUT_STOP_USD`) are listed at the top of the script.

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
