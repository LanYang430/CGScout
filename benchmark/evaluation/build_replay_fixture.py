"""Build the (large, not versioned) fixture for the offline golden replay.

usage: python build_replay_fixture.py --run-dir <dir of an earlier paper-profile run> --golden-cases <dir with the expected <pid>.json outputs> --out <fixture_dir>

<run dir> must contain queries.csv (or queries37.csv), plans_frozen/, terminology_memo.json, cache_work/ and results/CUR/ -- i.e. what run_benchmark.py leaves behind
(see arms.py).  <golden cases> are the replay outputs of a reference code version (python replay.py <fixture> <dir> <pids> run on that version).
Copies only; the run directory is not modified.
"""
import argparse, shutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True); ap.add_argument("--golden-cases", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
run, out = Path(a.run_dir), Path(a.out)
out.mkdir(parents=True, exist_ok=True)
q = next(p for p in (run / "queries.csv", run / "queries37.csv") if p.exists())
shutil.copy(q, out / "queries.csv")
for d in ("plans_frozen", "results/CUR"):
    shutil.copytree(run / d, out / d, dirs_exist_ok=True)
shutil.copy(run / "terminology_memo.json", out / "terminology_memo.json")
shutil.copytree(run / "cache_work", out / "cache_work", dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("*.lock", "openalex_usage.jsonl", "openalex_semantic.ts", "openalex_budget_state.json"))
shutil.copytree(a.golden_cases, out / "golden" / "cases", dirs_exist_ok=True)
print("fixture written to", out, "| cases:", sorted(p.stem for p in (out / "golden" / "cases").glob("*.json")))
