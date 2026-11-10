"""Run configuration of the paper evaluation: one arm, "CUR" = the live CGScout code with every switch passed EXPLICITLY (nothing is left to defaults).

P1 (flat_anchor_short_context) ON, P2 (flat_query_hygiene) ON, P3 (need_semantic) ON, Stage-3 abstract recovery ON, query_construction_v2 OFF,
Stage-4 tier-1 sees up to 4000 abstract characters, the shipped Stage-3 classifier.  Pure data + three small readers; no network here.

The run's data directory (queries.csv, plans_frozen/, cache_work/, terminology_memo.json, results/) is NOT part of the repository: point CGSCOUT_BENCH_DIR
at it (default: the current directory).  CGSCOUT_PLANS_DIR (legacy name NEW37_PLANS_DIR also read) overrides where the frozen plans are read from.
"""
from pathlib import Path
import csv, json, os

from cgscout.profiles import PAPER

ROOT = Path(os.environ.get("CGSCOUT_BENCH_DIR", ".")).resolve()
SHARED_CACHE = ROOT / "cache_work"          # fresh cache dir: no reuse of any earlier dev-run cache
INPUT = ROOT / "queries.csv"                # columns: Paper ID, Latest query (v2), DOI, OpenAlex ID

# single source of truth: cgscout.profiles.PAPER (pinned value-by-value in tests/test_profiles.py)
ARMS = {
    "CUR": dict(label=PAPER["label"], classifier="old", screen_chars=PAPER["stage4"]["screen_abstract_chars"], budgets=dict(PAPER["budgets"])),
}


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
    spec = ARMS[arm]["budgets"]
    return None if spec is None else RP.RouteRetrievalBudgets(**spec)
