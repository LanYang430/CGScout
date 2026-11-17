"""Consolidation safety net, Phase 0 -- proof that the trace world SURVIVES a move of the flow (offline, no production change).

Phases 2a/2b/3 move the retrieval flow to another module and another entry; the world must then reproduce the stored baselines with only the APPROVED
renames applied.  Here the move is SIMULATED: the source of cgscout.retrieval.orchestrator is executed once more under another module path
(`cgscout.retrieval._sim_flow`, which exists only inside these tests).  A layout that points at it must reproduce every baseline trace of both scenario
families (positive control), and each way a real migration could silently go wrong must be reported (negative controls):
  * a collaborator patched in the OLD module while the new flow looks it up in the new one  -> patch never called / real collaborator reached
  * a search binding the layout forgot                                                        -> the guarded HTTP boundary is reached
  * a module rename that was not approved                                                     -> the recorder's module list differs
  * the recorder's MODULES list not updated in the same commit                                -> unrecorded external calls (U)
  * an entry adapter that returns a wrapper instead of the retrieval result                   -> contract violation
  * a patch target that does not exist (R-2) / a collaborator the layout does not declare     -> assertion
The baseline fixtures are read-only here.  Approved aliases can rename a module path; they cannot hide a missing, extra or unmapped module."""
import dataclasses
import importlib
import importlib.util
import json
import sys
from types import SimpleNamespace

import pytest

import consolidation_layout as LY
import consolidation_world as W

SIM = "cgscout.retrieval._sim_flow"
BASE_A = json.loads(W.FIXTURE.read_text())
BASE_P5 = json.loads(W.FIXTURE_P5.read_text())
BASELINE_OF = {**BASE_A, **BASE_P5}


@pytest.fixture
def sim(monkeypatch):
    """The orchestrator's code, executed under another module path, plus the 'same commit' update of the recorder's module list (the old module path
    leaves it, the new one joins it: after a real move the old module no longer binds search_and_dedupe)."""
    orch = importlib.import_module(LY.BASE_FLOW)
    spec = importlib.util.spec_from_file_location(SIM, orch.__file__)
    m = importlib.util.module_from_spec(spec)
    sys.modules[SIM] = m
    spec.loader.exec_module(m)
    rec = W._recorder()
    monkeypatch.setattr(rec, "MODULES", [x for x in rec.MODULES if x != LY.BASE_FLOW] + [SIM])
    try:
        yield m
    finally:
        sys.modules.pop(SIM, None)


def sim_layout(*, entry=SIM, patch=SIM, aliases=True, name="simulated_move"):
    """The ACTIVE layout with the orchestrator's role played by SIM: same collaborators, the binder `orchestrator` replaced by SIM, and the approved alias SIM -> orchestrator
    added on top of whatever aliases the active layout already carries."""
    cur = LY.active_layout()
    return LY.Layout(name=name, entry=(entry, "run_expert_prior_retrieval"), adapter="legacy_retrieval", patch_targets=LY.baseline_targets(patch),
                     recorder_binders=tuple(SIM if b == LY.BASE_FLOW else b for b in cur.recorder_binders),
                     module_aliases={**cur.module_aliases, SIM: LY.BASE_FLOW} if aliases else {}, approval="self-test only")


_runs = {"n": 0}


def run(name, layout, tmp_path, monkeypatch):
    _runs["n"] += 1
    monkeypatch.setenv("CGSCOUT_CACHE_DIR", str(tmp_path / f"cache-{_runs['n']}-{name}"))          # a cold cache for EVERY run, like the baseline writer
    return W.run_scenario(name, layout)


# ---------------------------------------------------------------------------------------------------------------- positive control
@pytest.mark.parametrize("name", sorted(W.ALL_SCENARIOS))
def test_simulated_move_reproduces_the_baseline(name, sim, tmp_path, monkeypatch):
    got = run(name, sim_layout(), tmp_path, monkeypatch)
    why = LY.diff_against_baseline(got, BASELINE_OF[name], sim_layout())
    assert not why, f"{name}: {why}"
    if name == "R01_real_search_stack_recorder":
        assert got["recorder_installed_in"] == sorted(sim_layout().recorder_binders)                       # really wrapped in the NEW module (not the old) ...
        assert LY.normalize(got, sim_layout())["recorder_installed_in"] == BASE_A[name]["recorder_installed_in"]   # ... and equal to the old list after the approved rename


# ---------------------------------------------------------------------------------------------------------------- negative controls
def test_patch_left_on_the_old_module_is_detected(sim, tmp_path, monkeypatch):
    t = dict(LY.baseline_targets(SIM))
    t["expand_terminology"] = ((LY.BASE_FLOW, "expand_terminology"),)          # moved flow, patch table not updated for ONE collaborator
    layout = dataclasses.replace(sim_layout(), patch_targets=t)
    why = LY.diff_against_baseline(run("S01_3b_happy", layout, tmp_path, monkeypatch), BASE_A["S01_3b_happy"], layout)
    assert "patch_never_called" in why and "expand_terminology" in why["patch_never_called"]
    assert "events" in why or "runs" in why                                       # and the unpatched collaborator was really reached (P3 got the scripted LLM)


def test_unpatched_search_binding_reaches_the_guarded_boundary_not_the_network(sim, tmp_path, monkeypatch):
    t = dict(LY.baseline_targets(SIM))
    t["search_and_dedupe"] = ((LY.BASE_FLOW, "search_and_dedupe"),)            # the new flow's own binding is not patched
    layout = dataclasses.replace(sim_layout(), patch_targets=t)
    got = run("S01_3b_happy", layout, tmp_path, monkeypatch)
    assert any(e["kind"] == "UNPATCHED_HTTP_BOUNDARY_REACHED" for e in got["events"])
    assert LY.diff_against_baseline(got, BASE_A["S01_3b_happy"], layout)


def test_unapproved_module_rename_is_not_hidden(sim, tmp_path, monkeypatch):
    for label, aliases in (("no alias", {}), ("alias to the wrong module", {SIM: "cgscout.retrieval.search_adapters"}),
                           ("alias for another module", {"cgscout.retrieval.other": LY.BASE_FLOW})):
        layout = dataclasses.replace(sim_layout(), module_aliases=aliases)
        why = LY.diff_against_baseline(run("R01_real_search_stack_recorder", layout, tmp_path, monkeypatch), BASE_A["R01_real_search_stack_recorder"], layout)
        assert set(why) == {"recorder_installed_in"}, f"{label}: {why}"       # only the module list differs; the records themselves are equal


def test_recorder_modules_not_updated_in_the_same_commit_is_detected(sim, tmp_path, monkeypatch):
    rec = W._recorder()
    monkeypatch.setattr(rec, "MODULES", [m for m in rec.MODULES if m != SIM])  # the commit moved the flow but forgot recorder.MODULES
    layout = sim_layout()
    why = LY.diff_against_baseline(run("R01_real_search_stack_recorder", layout, tmp_path, monkeypatch), BASE_A["R01_real_search_stack_recorder"], layout)
    assert "recorder" in why and "recorder_installed_in" in why               # external searches outside any record window (U) AND the wrapped-module list


def test_adapter_returning_a_wrapper_violates_the_contract(sim, monkeypatch):
    monkeypatch.setitem(LY.ADAPTERS, "wrapper", lambda layout, plan, prior, **kw: SimpleNamespace(retrieval=LY.legacy_retrieval(layout, plan, prior, **kw)))
    layout = dataclasses.replace(sim_layout(), adapter="wrapper")
    with pytest.raises(LY.ContractViolation, match="wrapper"):               # raised out of the run, not recorded as the flow's own exception
        W.run_scenario("S01_3b_happy", layout)


def test_missing_patch_target_and_undeclared_collaborator_are_refused(sim):
    t = dict(LY.baseline_targets(SIM))
    t["expand_terminology"] = ((SIM, "expand_terminology_renamed"),)
    with pytest.raises(AssertionError, match="R-2"):
        W.run_scenario("S01_3b_happy", dataclasses.replace(sim_layout(), patch_targets=t))
    t = dict(LY.baseline_targets(SIM))
    del t["filter_papers"]
    with pytest.raises(AssertionError, match="declares no patch target"):
        W.run_scenario("S01_3b_happy", dataclasses.replace(sim_layout(), patch_targets=t))


# ---------------------------------------------------------------------------------------------------------------- the registry itself
def test_normalization_touches_only_the_module_fields():
    layout = dataclasses.replace(sim_layout(), module_aliases={SIM: LY.BASE_FLOW})
    tr = {"recorder_installed_in": [SIM, "x.y"], "events": [{"kind": "k", "query": SIM}], "patch_calls": {SIM: 1}}
    out = LY.normalize(tr, layout)
    assert out["recorder_installed_in"] == sorted([LY.BASE_FLOW, "x.y"])        # an unmapped module stays as it is
    assert out["events"] == tr["events"] and out["patch_calls"] == tr["patch_calls"]
    assert tr["recorder_installed_in"] == [SIM, "x.y"]                            # the input is not modified


@pytest.mark.parametrize("layout", sorted(LY.LAYOUTS.values(), key=lambda l: l.name), ids=lambda l: l.name)
def test_every_registered_layout_is_consistent(layout):
    assert layout.approval.strip()
    assert set(layout.patch_targets) == set(LY.COLLABORATORS)
    assert layout.adapter in LY.ADAPTERS
    m = importlib.import_module(layout.entry[0])
    assert callable(getattr(m, layout.entry[1]))
    for logical, targets in layout.patch_targets.items():
        assert targets, logical
        for module, attr in targets:
            assert hasattr(importlib.import_module(module), attr), f"R-2: {module}.{attr}"
    baseline_binders = set(LY.BASELINE.recorder_binders)
    assert {layout.module_aliases.get(b, b) for b in layout.recorder_binders} == baseline_binders      # the approved renames map the new binders onto the old ones
    assert set(layout.module_aliases.values()) <= baseline_binders | {LY.BASE_FLOW}
    rec = W._recorder()
    assert set(layout.recorder_binders) <= set(rec.MODULES), "recorder.MODULES must list every module the layout says binds search_and_dedupe"


def test_the_baseline_fixtures_are_the_tagged_ones():
    """Read-only: the stored baselines carry the structural facts of tag consol-0-baseline; a cheap tripwire against an accidental regeneration."""
    assert sorted(BASE_A) == sorted(W.SCENARIOS) and sorted(BASE_P5) == sorted(W.SCENARIOS_P5)
    assert BASE_A["R01_real_search_stack_recorder"]["recorder_installed_in"] == ["cgscout.retrieval.orchestrator", "cgscout.retrieval.search_adapters"]
