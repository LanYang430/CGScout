"""Run configuration of the paper evaluation: one arm, "CUR" = the live CGScout code run with the named profile `cgscout.profiles.PAPER`.

PAPER is the single source of every evaluation setting: the five retrieval switches (P1 flat_anchor_short_context, P2 flat_query_hygiene, P3 need_semantic, Stage-3 abstract
recovery, query_construction_v2 off), the Stage-4 settings (efforts, deep_cap, the 4000 abstract characters the tier-1 screen sees), the LLM (model, effort, lifting explicit
"low" requests to the profile effort) and the shipped Stage-3 classifier.  Everything the profile does not name (e.g. max_papers_per_query, Stage-4 workers) is the library default;
run_benchmark.py records the full effective values of all of them in run_config_<arm>.json, and tests/test_run_benchmark.py pins them.
`ARMS["CUR"]` keeps label / budgets / screen_chars / classifier as copies taken from the profile at import (replay.py and external verification scripts read them);
the runner itself reads the profile through profile_for().  Pure data + small readers; no network here.

The run's data directory (queries.csv, plans_frozen/, cache_work/, terminology_memo.json, results/) is NOT part of the repository: point CGSCOUT_BENCH_DIR
at it (default: the current directory).  CGSCOUT_PLANS_DIR (legacy name NEW37_PLANS_DIR also read) overrides where the frozen plans are read from; whichever directory
is used, run_benchmark.py only accepts plans whose provenance matches the code that runs (provenance.py).
"""
from pathlib import Path
import csv, json, os

from cgscout.profiles import PAPER, PROFILES

ROOT = Path(os.environ.get("CGSCOUT_BENCH_DIR", ".")).resolve()
SHARED_CACHE = ROOT / "cache_work"          # fresh cache dir: no reuse of any earlier dev-run cache
INPUT = ROOT / "queries.csv"                # columns: REQUIRED_COLUMNS

REQUIRED_COLUMNS = ("Paper ID", "Latest query (v2)", "DOI", "OpenAlex ID")

# single source of truth: cgscout.profiles.PAPER (pinned value-by-value in tests/test_profiles.py)
ARMS = {
    "CUR": dict(profile="paper", label=PAPER["label"], classifier=PAPER["stage3_classifier"], screen_chars=PAPER["stage4"]["screen_abstract_chars"], budgets=dict(PAPER["budgets"])),
}


class PreflightError(Exception):
    """A run (or a freeze) must not start: bad command line, bad input, unusable plans.  The message says what to fix; nothing has been spent."""


def profile_for(arm):
    return PROFILES[ARMS[arm]["profile"]]


def read_rows():
    """queries.csv as a list of dict rows; PreflightError when the file is missing, lacks a required column or repeats a Paper ID."""
    if not INPUT.exists():
        raise PreflightError(f"{INPUT} not found (CGSCOUT_BENCH_DIR={ROOT})")
    with open(INPUT, encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (rd.fieldnames or [])]
        if missing:
            raise PreflightError(f"{INPUT.name} lacks column(s) {missing}; expected {list(REQUIRED_COLUMNS)}")
        rows = list(rd)
    ids = [r["Paper ID"] for r in rows]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise PreflightError(f"{INPUT.name}: duplicate Paper ID(s) {dup[:10]}")
    return rows


def select_pids(ids, pids_arg, expect_rows):
    """Which cases a command selects.  Returns (selected ids in queries.csv order, is_full_run).
    * `--expect-rows N`, when given, must equal the number of rows of queries.csv;
    * no `--pids` = a FULL run (every row) and then `--expect-rows` is mandatory, so a stale or truncated queries.csv cannot be run by accident;
    * `--pids` must name at least one case and every name must exist: an empty value or a typo is an error, never 'run everything' and never silently dropped."""
    if expect_rows is not None and len(ids) != expect_rows:
        raise PreflightError(f"queries.csv has {len(ids)} rows but --expect-rows is {expect_rows}")
    if pids_arg is None:
        if expect_rows is None:
            raise PreflightError("a full run (no --pids) needs --expect-rows N, the number of cases queries.csv must contain (e.g. --expect-rows 200)")
        return list(ids), True
    want = [p.strip() for p in pids_arg.split(",") if p.strip()]
    if not want:
        raise PreflightError("--pids is empty; give a comma-separated list of Paper IDs, or omit --pids for a full run")
    unknown = [p for p in dict.fromkeys(want) if p not in set(ids)]
    if unknown:
        raise PreflightError(f"--pids names case(s) that are not in queries.csv: {unknown}")
    return [i for i in ids if i in set(want)], False


def case_ids():
    return [r["Paper ID"] for r in csv.DictReader(open(INPUT, encoding="utf-8-sig"))], []


PLANS = Path(os.environ.get("CGSCOUT_PLANS_DIR", os.environ.get("NEW37_PLANS_DIR", str(ROOT / "plans_frozen"))))


def frozen_plan(pid):
    """freeze_plans.py output: ScientificPlanner + ExpertPrior generated ONCE and replayed by the run."""
    return json.load(open(PLANS / f"{pid}.json"))


def fixed_expert_prior(pid):
    from cgscout.planning.expert_prior_planner import ExpertPriorOutput
    return ExpertPriorOutput.model_validate(frozen_plan(pid)["expert_prior"])


def budgets_for(arm, RP):
    spec = profile_for(arm)["budgets"]
    return None if spec is None else RP.RouteRetrievalBudgets(**spec)


def stage4_config(arm="CUR"):
    from cgscout.assessment.stage4 import Stage4Config
    return Stage4Config(**profile_for(arm)["stage4"])
