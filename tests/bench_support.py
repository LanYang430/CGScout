"""Helpers shared by tests/test_run_benchmark.py and tests/test_plan_provenance.py: make benchmark/evaluation importable, build a run directory, launch a script like a user would.
Nothing here touches the network or an LLM."""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "benchmark" / "evaluation"
if str(EVAL) not in sys.path:
    sys.path.insert(0, str(EVAL))              # arms, provenance, run_benchmark, freeze_plans, harness, recorder

import provenance as PV  # noqa: E402
from cgscout.config import PAPER  # noqa: E402

COLUMNS = ["Paper ID", "Latest query (v2)", "DOI", "OpenAlex ID"]


def pid_of(i):
    return f"B{i:03d}"


def write_queries(run_dir, n, *, query=lambda i: f"query text {i}"):
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "queries.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(COLUMNS)
        for i in range(1, n + 1):
            w.writerow([pid_of(i), query(i), f"10.1/{i}", f"W{i}"])
    return [pid_of(i) for i in range(1, n + 1)]


def read_queries(run_dir):
    with open(Path(run_dir) / "queries.csv", encoding="utf-8-sig") as f:
        return {r["Paper ID"]: r for r in csv.DictReader(f)}


def plan_for(pid, query, prov, *, status="READY"):
    plan = {"pid": pid, "query": query, "attempts": [{"n": 1, "status": "ready", "clarifications": []}], "created": "2026-11-10 10:00:00", "model": prov["model"], "status": status,
            "llm_cost_usd": 0.0, "llm_calls": 0, "provenance": prov}
    if status == "READY":
        plan.update(planner={"system": "s", "target_component": None, "scientific_objective": "o", "search_context": [], "search_axes": []}, expert_prior={})
    return plan


def write_plans(run_dir, pids=None, *, prov=None, drop_provenance=False, query_override=None):
    """Frozen plans for the given cases, stamped with the CURRENT P1/P2 code (or with `prov`); `drop_provenance` imitates a plan from an older freeze_plans.py."""
    run_dir = Path(run_dir); rows = read_queries(run_dir)
    prov = prov or PV.stamp(model=PAPER["llm"]["model"], effort=PAPER["llm"]["effort"])        # one stamp for all files (it runs git)
    (run_dir / "plans_frozen").mkdir(exist_ok=True)
    for pid in (pids or list(rows)):
        plan = plan_for(pid, query_override or rows[pid]["Latest query (v2)"], prov)
        if drop_provenance:
            del plan["provenance"]
        (run_dir / "plans_frozen" / f"{pid}.json").write_text(json.dumps(plan))
    return prov


def write_result(run_dir, pid, status="OK", cost=0.0, arm="CUR"):
    d = Path(run_dir) / "results" / arm; d.mkdir(parents=True, exist_ok=True)
    (d / f"{pid}.json").write_text(json.dumps({"pid": pid, "status": status, "llm": {"cost_usd": cost}, "seed": {}}))


def run_script(script, args, run_dir, *, env_extra=None, drop_env=("CGSCOUT_STOP_USD", "NEW37_STOP_USD", "CGSCOUT_PLANS_DIR", "NEW37_PLANS_DIR")):
    """Run benchmark/evaluation/<script> in a fresh interpreter exactly like the docs do, from a neutral working directory."""
    env = {k: v for k, v in os.environ.items() if k not in drop_env}
    env.update({"CGSCOUT_BENCH_DIR": str(run_dir), "PYTHONDONTWRITEBYTECODE": "1", "OPENAI_API_KEY": "sk-test-not-a-real-key",
                "PYTHONPATH": os.pathsep.join([str(ROOT / "src")] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))})
    env.update(env_extra or {})
    return subprocess.run([sys.executable, str(EVAL / script), *args], capture_output=True, text=True, env=env, cwd=str(Path(run_dir).parent), timeout=300)
