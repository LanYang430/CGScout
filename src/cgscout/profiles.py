"""Named run configurations -- pure data, no code and no imports of the pipeline.

PAPER        the configuration the paper evaluation runs with ("CUR" arm of benchmark/evaluation): every switch passed explicitly, nothing left to defaults.
INTERACTIVE  the library / CLI defaults, unchanged (empty overrides): what `cgscout` does when no configuration is given.

A profile is a dict of overrides for the pieces that take configuration: `budgets` -> retrieval.direct.route_plan.RouteRetrievalBudgets(**budgets),
`stage4` -> assessment.stage4.Stage4Config (+ the abstract length shown to the tier-1 screen), `llm` -> model / reasoning effort (the evaluation harness
additionally lifts any explicit effort="low" request to "medium" for this profile).  The default entry point stays INTERACTIVE; selecting PAPER is an explicit choice.
"""

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
