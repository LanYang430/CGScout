"""Consolidation safety net, Phase 0 -- static guards and the effective-configuration record.  Offline, no LLM, no network, no production edit.

1. Patch points (rules R-1 / R-2): every name the evaluation harness, the CLI tests or the recorder assigns to must exist on its module today, and the
   flow functions must look the patched names up as module globals at call time (so an assignment takes effect).  A refactor that moves a name
   must move the patch point in the same commit; this test then fails on the old spelling instead of the patch silently doing nothing.
2. Recorder coverage: every module that binds `search_and_dedupe` at module level must be in recorder.MODULES, and the kinds the recorder derives from
   the caller's qualified name must still be function names that exist.  The one known blind spot (KNOWN-1: a function-local import in route_plan) is
   pinned, not fixed.
3. Effective configuration (E-1..E-4 of the safeguards addendum): `config=None` == `Stage4Config()`, `budgets=None` == `RouteRetrievalBudgets()`, the CLI runs
   with the library defaults, the PAPER profile differs from them by design -- all recorded in tests/fixtures/consolidation/effective_config_baseline.json.
Regenerate the config fixture ONLY on the baseline:  python tests/test_consolidation_guards.py --write"""
import ast
import dataclasses
import importlib
import inspect
import json
import sys
from pathlib import Path

import pytest

import consolidation_layout as LY

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "cgscout"
EVAL = ROOT / "benchmark" / "evaluation"
CONFIG_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "consolidation" / "effective_config_baseline.json"


def _eval_module(name):
    sys.path.insert(0, str(EVAL))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(EVAL))


# ---------------------------------------------------------------------------------------------------------------- 1. patch points

# (module, attribute): assigned somewhere in tests/ or benchmark/evaluation/ today
PATCH_POINTS = [
    ("cgscout.retrieval.orchestrator", "expand_terminology"), ("cgscout.retrieval.orchestrator", "filter_papers"),
    ("cgscout.retrieval.orchestrator", "run_expert_prior_retrieval"), ("cgscout.retrieval.orchestrator", "rank_for_display"),
    ("cgscout.retrieval.orchestrator", "search_and_dedupe"),
    ("cgscout.cli", "run_expert_prior_retrieval"), ("cgscout.cli", "run_stage4_v2"), ("cgscout.cli", "rank_for_display"),
    ("harness", "run_expert_prior_retrieval"), ("harness", "run_stage4_v2"), ("harness", "run_plan"), ("harness", "run_plan_checked"),
    ("harness", "rank_for_display"),
    ("cgscout.retrieval.search_adapters", "search_and_dedupe"),
]

# names the 3B chain looks up as globals of the flow module at call time (a patch on the module takes effect only because of this).
# Function-specific and therefore BASELINE-ONLY (`_run_related_system_discovery` disappears in Phase 3, the loop moves in Phase 2b): re-pointed in the
# commit that moves them.  The layout-driven test below (every patch target is looked up by some function of its module) is what survives a move unedited.
LOOKED_UP_AT_CALL_TIME = {
    "_run_related_system_discovery": ["expand_terminology", "rephrase_hop1_semantic_intent", "discover_related_systems", "cleanup_discovered_systems",
                                      "build_related_system_identities", "normalize_search_anchors", "merge_and_select", "run_converged_hop2",
                                      "extract_related_systems_rich", "_make_search_fn"],
    "run_expert_prior_retrieval": ["filter_papers", "run_retrieval_from_plan", "papers_for_stage3", "provenance_index", "_run_direct_concept_search",
                                   "_run_related_system_discovery", "_merge_papers", "render_direct_concept_queries"],
    "_make_search_fn": ["search_and_dedupe"],
    "_run_direct_concept_search": ["search_and_dedupe"],
}


@pytest.mark.parametrize("mod,attr", PATCH_POINTS)
def test_patch_point_exists(mod, attr):
    m = _eval_module(mod) if "." not in mod else importlib.import_module(mod)
    assert hasattr(m, attr), f"R-2: {mod}.{attr} is assigned by tests/benchmark code; it must exist (an assignment to a missing name silently does nothing)"
    assert callable(getattr(m, attr))


def _names_loaded_inside_functions(module):
    """Names read (not imported, not assigned) inside any function or closure of the module's source."""
    tree = ast.parse(inspect.getsource(module))
    out = set()
    for fn in (n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))):
        out |= {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return out


@pytest.mark.parametrize("logical", LY.COLLABORATORS)
def test_every_patch_target_is_looked_up_by_a_function_of_its_module(logical):
    """R-1, layout-driven: assigning to `module.attr` takes effect only if some function of that module reads `attr` as a global at call time.  If the
    migrated flow binds the collaborator elsewhere, the layout's table must list THAT module; this test fails for a target nobody looks up."""
    for module, attr in LY.active_layout().patch_targets[logical]:
        assert attr in _names_loaded_inside_functions(importlib.import_module(module)), (
            f"R-1: no function of {module} reads {attr!r}; a patch on it would silently do nothing")


def test_flow_functions_look_patched_names_up_at_call_time():
    flow = importlib.import_module("cgscout.retrieval.orchestrator")
    for fn, names in LOOKED_UP_AT_CALL_TIME.items():
        code = getattr(flow, fn).__code__
        # a nested closure's names live in the code objects of the enclosing function's constants
        allnames = set(code.co_names)
        for c in code.co_consts:
            if inspect.iscode(c):
                allnames |= set(c.co_names)
        missing = [n for n in names if n not in allnames]
        assert not missing, f"R-1: {fn} no longer references {missing} as globals; patches on the module would not reach it"


# ---------------------------------------------------------------------------------------------------------------- 2. recorder coverage

def _binders():
    """({module: lineno} binding `search_and_dedupe` at MODULE level, {module: lineno} binding it inside a function)."""
    top, lazy = {}, {}
    for p in sorted(SRC.rglob("*.py")):
        mod = ".".join(("cgscout",) + p.relative_to(SRC).with_suffix("").parts)
        tree = ast.parse(p.read_text())
        top_nodes = {id(n) for n in ast.walk(tree)}
        for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            for n in ast.walk(fn):
                if isinstance(n, ast.ImportFrom) and any(a.name == "search_and_dedupe" for a in n.names):
                    lazy[mod] = n.lineno
                top_nodes.discard(id(n))
        for n in ast.walk(tree):
            if id(n) in top_nodes and isinstance(n, ast.ImportFrom) and any(a.name == "search_and_dedupe" for a in n.names):
                top[mod] = n.lineno
    return top, lazy


def test_recorder_modules_cover_every_module_level_binder():
    rec = _eval_module("recorder")
    top, lazy = _binders()
    layout = LY.active_layout()
    assert set(top) == set(layout.recorder_binders), f"binders of search_and_dedupe changed: {sorted(top)} (the layout lists {sorted(layout.recorder_binders)})"
    assert set(top) <= set(rec.MODULES), "a module that binds search_and_dedupe is not wrapped by the recorder: its searches would go unrecorded"


def test_known_blind_spot_is_pinned_not_fixed():
    """KNOWN-1: route_plan._default_flat_search imports search_and_dedupe inside the function, which module-attribute wrapping cannot see.  It is only reachable
    when run_route_retrieval gets flat_search_fn=None; the only production caller (from_planner.run_retrieval_from_plan) always injects one."""
    _, lazy = _binders()
    assert set(lazy) == {"cgscout.retrieval.direct.route_plan"}
    src = (SRC / "retrieval" / "direct" / "from_planner.py").read_text()
    assert "run_route_retrieval(evidence_plan, flat_search_fn=flat_search_fn" in src


def test_recorder_kinds_still_name_existing_functions():
    rec = _eval_module("recorder")
    defined = set()
    for p in SRC.rglob("*.py"):
        defined |= {n.name for n in ast.walk(ast.parse(p.read_text())) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    needles = {needle for needle, _ in rec._KINDS} | {"_make_search_fn"}
    assert needles <= defined, f"recorder classifies by function name; no longer defined: {sorted(needles - defined)}"


# ---------------------------------------------------------------------------------------------------------------- 3. effective configuration

def current_effective():
    from cgscout.assessment.stage4 import Stage4Config
    from cgscout.profiles import INTERACTIVE, PAPER
    from cgscout.retrieval.direct.route_plan import RouteRetrievalBudgets

    def d4(c):
        return {k: v for k, v in dataclasses.asdict(c).items() if k != "abstract_lookup"}

    lib = {"stage4": d4(Stage4Config()), "budgets": json.loads(RouteRetrievalBudgets().model_dump_json())}
    paper = {"stage4": d4(Stage4Config(**PAPER["stage4"])), "budgets": json.loads(RouteRetrievalBudgets(**PAPER["budgets"]).model_dump_json())}
    interactive = {"stage4": d4(Stage4Config(**INTERACTIVE["stage4"])), "budgets": json.loads(RouteRetrievalBudgets(**INTERACTIVE["budgets"]).model_dump_json())}

    def diff(a, b):
        return {sec: {k: {"library_default": a[sec][k], "paper": b[sec][k]} for k in a[sec] if a[sec][k] != b[sec][k]} for sec in a}

    return {"library_default": lib, "paper": paper, "interactive": interactive, "paper_minus_default": diff(lib, paper)}


def test_effective_config_equals_baseline_record():
    assert current_effective() == json.loads(CONFIG_FIXTURE.read_text())


def test_none_equals_default_construction():
    from cgscout.assessment.stage4 import Stage4Config, run_stage4_v2
    from cgscout.retrieval.direct.route_plan import RouteRetrievalBudgets
    from cgscout.retrieval.orchestrator import run_expert_prior_retrieval
    # `config or Stage4Config()` / `budgets or RouteRetrievalBudgets()`: a default-constructed instance is truthy, so None and "the default instance" behave alike
    assert bool(Stage4Config()) and bool(RouteRetrievalBudgets())
    assert inspect.signature(run_stage4_v2).parameters["config"].default is None
    assert inspect.signature(run_expert_prior_retrieval).parameters["budgets"].default is None
    assert "cfg = config or Stage4Config()" in inspect.getsource(run_stage4_v2)
    assert "budgets = budgets or RouteRetrievalBudgets()" in inspect.getsource(run_expert_prior_retrieval)


def test_cli_uses_library_defaults_and_paper_differs_by_design():
    rec = json.loads(CONFIG_FIXTURE.read_text())
    assert rec["interactive"] == rec["library_default"]                      # INTERACTIVE = the library defaults (the CLI does not read profiles)
    assert rec["paper"] != rec["library_default"]                            # PAPER is NOT the CLI configuration -- by design (profiles.py docstring)
    assert set(rec["paper_minus_default"]["stage4"]) == {"screen_effort", "extract_effort", "deep_cap"}


def test_default_budgets_reach_the_direct_route_unchanged():
    """E-1b / E-2b, executed: budgets=None reaches run_retrieval_from_plan as RouteRetrievalBudgets(); explicit PAPER budgets reach it as given."""
    import hashlib
    from cgscout.profiles import PAPER
    from cgscout.retrieval.direct.route_plan import RouteRetrievalBudgets
    import consolidation_world as W

    def h(b):
        return hashlib.sha1(json.dumps(b.model_dump(), sort_keys=True, default=str).encode()).hexdigest()[:8]
    base = json.loads(W.FIXTURE.read_text())
    assert [e["budgets"] for e in base["S01_3b_happy"]["events"] if e["kind"] == "direct_route"] == [h(RouteRetrievalBudgets())]
    assert [e["budgets"] for e in base["S12_multi_axis_loop"]["events"] if e["kind"] == "direct_route"] == [h(RouteRetrievalBudgets(**PAPER["budgets"]))]


if __name__ == "__main__":
    if "--write" not in sys.argv:
        sys.exit("usage: python tests/test_consolidation_guards.py --write   (baseline only)")
    sys.path.insert(0, str(ROOT / "src"))
    CONFIG_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FIXTURE.write_text(json.dumps(current_effective(), indent=1, sort_keys=True) + "\n")
    print("wrote", CONFIG_FIXTURE)
