"""cgscout configuration: the environment variables and the named run configurations.

ENVIRONMENT  `get(name, default)`: `CGSCOUT_<NAME>` is the name; the legacy `BEAM_<NAME>` is still read (a one-time notice on stderr when only the legacy name
             is set), so existing scripts and caches keep working.  Semantics are exactly those of os.environ.get: a variable that is present (even empty) wins,
             the new name before the legacy one.  (Imported as `_env` by cache.py and llm.py.)

PROFILES     pure data, no imports of the pipeline.

PAPER        the configuration the paper evaluation runs with ("CUR" arm of benchmark/evaluation).  It names the five retrieval switches below, the Stage-4 efforts / deep_cap /
             screen abstract length, the LLM model and effort, and the Stage-3 classifier.  Every setting it does NOT name (e.g. max_papers_per_query, the Stage-4 worker counts)
             is the library default; benchmark/evaluation/run_benchmark.py records ALL effective values in run_config_<arm>.json and tests/test_run_benchmark.py pins them.
INTERACTIVE  the library defaults, unchanged (empty overrides).  The `cgscout` command line does not read profiles: it runs with the library defaults, which is what this profile describes.

A profile is a dict of overrides for the pieces that take configuration: `budgets` -> retrieval.direct.route_plan.RouteRetrievalBudgets(**budgets),
`stage4` -> assessment.stage4.Stage4Config (+ the abstract length shown to the tier-1 screen), `llm` -> model / reasoning effort (the evaluation harness
additionally lifts any explicit effort="low" request to "medium" for this profile).  PAPER is selected only by the evaluation runner (benchmark/evaluation/arms.py).
"""
import os
import sys

_WARNED: set = set()


def get(name: str, default=None):
    new, old = f"CGSCOUT_{name}", f"BEAM_{name}"
    if new in os.environ:
        return os.environ[new]
    if old in os.environ:
        if old not in _WARNED:
            _WARNED.add(old)
            print(f"cgscout: environment variable {old} is deprecated; use {new}", file=sys.stderr)
        return os.environ[old]
    return default


# ---------------------------------------------------------------- named run configurations (formerly profiles.py)
PAPER = {
    "label": "paper: P1+P2+P3 on, Stage-3 abstract recovery on, query_construction_v2 off, tier-1 sees 4000 abstract characters",
    "budgets": dict(flat_anchor_short_context=True, flat_query_hygiene=True, need_semantic_enabled=True,
                    stage3_recover_abstracts=True, query_construction_v2=False),
    "stage3_classifier": "shipped",     # screening/sim_classifier_model.joblib
    "stage4": dict(screen_effort="medium", extract_effort="medium", judge_effort="medium", deep_cap=500, screen_abstract_chars=4000),
    "llm": dict(model="gpt-5.6-luna", effort="medium", lift_low_effort_to_medium=True),
}

INTERACTIVE = {
    "label": "interactive: library defaults, unchanged",
    "budgets": {},
    "stage3_classifier": "shipped",
    "stage4": {},
    "llm": {},
}

PROFILES = {"paper": PAPER, "interactive": INTERACTIVE}
