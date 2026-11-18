"""The named run configurations (cgscout.config) are pure data and are pinned here value by value:
  * PAPER must equal the evaluation arm that produced the paper's runs (values copied from the original arm definition);
  * INTERACTIVE is the library defaults, i.e. no overrides.
Changing a default (or a PAPER value) therefore fails a test instead of silently changing what a run means."""
import sys
from pathlib import Path

from cgscout.config import INTERACTIVE, PAPER, PROFILES
from cgscout.retrieval.direct.route_plan import RouteRetrievalBudgets

EVAL = Path(__file__).resolve().parents[1] / "benchmark" / "evaluation"

# the original "CUR" arm (an archived benchmark arm file), spelled out
ORIGINAL_CUR_BUDGETS = dict(flat_anchor_short_context=True, flat_query_hygiene=True, need_semantic_enabled=True, stage3_recover_abstracts=True, query_construction_v2=False)
ORIGINAL_CUR_SCREEN_CHARS = 4000


def test_paper_profile_equals_the_original_evaluation_arm():
    assert PAPER["budgets"] == ORIGINAL_CUR_BUDGETS
    assert PAPER["stage4"] == dict(screen_effort="medium", extract_effort="medium", judge_effort="medium", deep_cap=500, screen_abstract_chars=ORIGINAL_CUR_SCREEN_CHARS)
    assert PAPER["llm"] == dict(model="gpt-5.6-luna", effort="medium", lift_low_effort_to_medium=True)


def test_evaluation_arm_is_built_from_the_paper_profile():
    sys.path.insert(0, str(EVAL))
    try:
        import arms
    finally:
        sys.path.remove(str(EVAL))
    cur = arms.ARMS["CUR"]
    assert cur["budgets"] == ORIGINAL_CUR_BUDGETS and cur["screen_chars"] == ORIGINAL_CUR_SCREEN_CHARS and cur["classifier"] == PAPER["stage3_classifier"] == "shipped"


def test_profile_budgets_are_valid_budget_fields():
    for p in PROFILES.values():
        RouteRetrievalBudgets(**p["budgets"])          # raises on an unknown field


def test_interactive_profile_is_the_library_defaults():
    assert INTERACTIVE["budgets"] == {} and INTERACTIVE["stage4"] == {} and INTERACTIVE["llm"] == {}
    assert RouteRetrievalBudgets(**INTERACTIVE["budgets"]) == RouteRetrievalBudgets()


def test_paper_differs_from_the_defaults_only_where_it_says_so():
    d, p = RouteRetrievalBudgets(), RouteRetrievalBudgets(**PAPER["budgets"])
    changed = {k for k in RouteRetrievalBudgets.model_fields if getattr(d, k) != getattr(p, k)}
    assert changed <= set(PAPER["budgets"])            # nothing outside the explicit overrides differs
