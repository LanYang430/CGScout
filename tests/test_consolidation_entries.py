"""Consolidation safety net -- what each ENTRY hands to the flow (the three-entry equivalence, T7-T9).  Offline.

Phase 0 recorded how the CLI (cli.run_retrieval) and the evaluation harness (harness.run_plan) each drove  retrieval -> Stage 4.  They are not identical, and a unified
entry must PRESERVE the differences, not erase them:
  * CLI      retrieval gets no budgets (library defaults); Stage 4 runs only when it was asked for and at least one paper was retained; no Stage-4 config;
  * harness  budgets are forwarded only when given; Stage 4 runs when `stage4` is set and the run is not degraded (an axis error or a failed HTTP request) --
             also when nothing was retained; the Stage-4 config is forwarded when given.
Since 2a all entries go through cgscout.agent.execute (the CLI via agent.run); they enter the flow through agent.retrieve and agent.assess, which are replaced by recorders
here, so no network and no LLM is involved.  What the entries hand over is the AgentConfig: `budgets` / `stage4` are None unless the caller gave them (None means the
library default: `budgets or RouteRetrievalBudgets()` in retrieve, `config or Stage4Config()` in run_stage4_v2).  (Until 2a the same facts were asserted one level
deeper, at run_expert_prior_retrieval / run_stage4_v2; since 2b that function is the body of agent.retrieve, whose behaviour the trace world pins.)
Test-side coverage of the CLI itself (printing, loop, `--no-stage4`, no candidates, case_id) lives in tests/test_cli.py."""
from types import SimpleNamespace

import pytest

import bench_support  # noqa: F401  (puts benchmark/evaluation on sys.path)
import harness as H
from cgscout import agent
from cgscout.assessment.stage4 import Stage4Config
from cgscout.models import Paper
from cgscout.planning.expert_prior_planner import ExpertPriorOutput
from cgscout.planning.scientific_planner import PlannerOutput
from cgscout.retrieval.direct.route_plan import RouteRetrievalBudgets

FIELDS = {"system": "Aquila", "scientific_objective": "folding mechanism", "search_context": [], "search_axes": []}


def _res(retained=(), axis_errors=()):
    return SimpleNamespace(
        axis_logs=[SimpleNamespace(error=e, axis=SimpleNamespace(basis="b", execution_mode="m"), executed=True, skip_reason=None, direct_queries=[],
                                   hop1_queries=[], hop2_queries=[], related_system_identities=[], n_papers_found=0) for e in axis_errors],
        direct_papers=[], transfer_papers=[], retained=list(retained), rejected=[], stage3_verdicts=[])


def _paper():
    return Paper(title="p", year=2020, query_categories=["system_specific_cg"])


class ScriptedBackend:
    """The planner answers with one ready plan, the expert prior with an empty prior: what the interactive path needs."""
    name, model, default_effort = "scripted", "scripted-model", "medium"

    def structured(self, *, system, user, schema, label, effort=None, purpose=None, case_id=None, timeout=None):
        if schema is PlannerOutput:
            return PlannerOutput(system=FIELDS["system"], scientific_objective=FIELDS["scientific_objective"], clarification_needed=False,
                                 search_context=[], search_axes=[])
        if schema is ExpertPriorOutput:
            return ExpertPriorOutput(transfer_needed=False, transfer_axes=[])
        raise AssertionError(schema)


@pytest.fixture
def rig(monkeypatch):
    calls = SimpleNamespace(ret=[], s4=[])
    box = SimpleNamespace(res=_res())

    def fake_ret(plan, ep, *, backend, config=None, case_id=None):
        calls.ret.append(dict(plan=plan, prior=ep, backend=backend, config=config, case_id=case_id))
        return box.res

    def fake_s4(plan, ep, retained, *, backend, config=None, case_id=None, user_request=None):
        calls.s4.append(dict(plan=plan, prior=ep, retained=list(retained), backend=backend, config=config, case_id=case_id, user_request=user_request))
        return SimpleNamespace(workload={}, assessments=[], by_key=lambda: {})
    monkeypatch.setattr(agent, "retrieve", fake_ret)
    monkeypatch.setattr(agent, "assess", fake_s4)
    monkeypatch.setattr(H, "read_usage_log", lambda: [])
    monkeypatch.setattr(H.U, "exhausted", lambda: None)
    monkeypatch.setattr(H.U, "reset_run", lambda *_: None)
    monkeypatch.setattr(H.U, "set_run_label", lambda *_: None)
    monkeypatch.setattr(H.U, "run_usage", lambda *_: {"failed_requests": 0, "http_requests": 0, "cache_hits": 0, "credits_charged": 0})
    ep = ExpertPriorOutput(transfer_needed=False, transfer_axes=[])
    return SimpleNamespace(calls=calls, box=box, ep=ep, run=lambda **kw: H.run_plan("BACKEND", dict(FIELDS), "CASE-1", {}, expert_prior=ep, **kw))


# ------------------------------------------------------------------------------------------------------------------ the harness entry
def test_harness_forwards_budgets_only_when_given(rig):
    rig.run()
    assert rig.calls.ret[0]["config"].budgets is None and rig.calls.ret[0]["case_id"] == "CASE-1" and rig.calls.ret[0]["backend"] == "BACKEND"
    b = RouteRetrievalBudgets(need_semantic_enabled=True)
    rig.run(budgets=b)
    assert rig.calls.ret[1]["config"].budgets is b


def test_harness_stage4_runs_when_not_degraded_even_with_nothing_retained(rig):
    cfg = Stage4Config(deep_cap=7)
    rig.run(stage4=True, stage4_config=cfg, user_request="req")
    assert len(rig.calls.s4) == 1
    s4 = rig.calls.s4[0]
    assert s4["retained"] == [] and s4["config"].stage4 is cfg and s4["user_request"] == "req" and s4["case_id"] == "CASE-1"      # empty retained still reaches Stage 4


def test_harness_stage4_config_is_forwarded_only_when_given(rig):
    rig.run(stage4=True)
    assert rig.calls.s4[0]["config"].stage4 is None             # None = `config or Stage4Config()` inside run_stage4_v2


def test_harness_stage4_skipped_when_an_axis_errored(rig):
    rig.box.res = _res(axis_errors=["boom"])
    rec, ar = rig.run(stage4=True)
    assert rig.calls.s4 == [] and rec["degraded"] is True and rec["axis_errors"] == ["boom"]
    assert ar.assessment is None and ar.assess_started_at is None


def test_harness_stage4_skipped_when_a_http_request_failed(rig, monkeypatch):
    monkeypatch.setattr(H.U, "run_usage", lambda *_: {"failed_requests": 1, "http_requests": 1, "cache_hits": 0, "credits_charged": 0})
    rec, ar = rig.run(stage4=True)
    assert rig.calls.s4 == [] and rec["degraded"] is True and ar.assessment is None


def test_harness_budget_exhaustion_leaves_the_gate_as_an_exception_before_stage4(rig, monkeypatch):
    monkeypatch.setattr(H.U, "exhausted", lambda: {"headers": {}, "reset_s": 5})
    monkeypatch.setattr(H.U, "exhausted_message", lambda ex: "exhausted")
    with pytest.raises(H.U.OpenAlexBudgetExhausted):
        rig.run(stage4=True)
    assert rig.calls.s4 == []


def test_harness_without_stage4_never_calls_it(rig):
    rig.run(stage4=False)
    assert rig.calls.s4 == []


def test_harness_returns_the_record_and_the_agent_result_and_times_stage4_from_the_agent(rig, monkeypatch):
    rows = [{"ts": 10.0, "label": "stage4_screen", "input_tokens": 1, "output_tokens": 2, "estimated_cost_usd": 0.5},
            {"ts": 30.0, "label": "stage4_screen", "input_tokens": 10, "output_tokens": 20, "estimated_cost_usd": 1.0},
            {"ts": 31.0, "label": "hop2_search_anchor", "input_tokens": 99, "output_tokens": 99, "estimated_cost_usd": 9.0}]
    monkeypatch.setattr(H, "read_usage_log", lambda: rows)
    monkeypatch.setattr(agent.time, "time", lambda: 20.0)          # assess_started_at: only the rows from then on, and only stage4 labels, count
    rec, ar = rig.run(stage4=True)
    assert ar.assess_started_at == 20.0 and ar.retrieval is rig.box.res and ar.assessment is not None
    assert rec["stage4"]["llm_usage"] == {"llm_calls": 1, "input_tokens": 10, "output_tokens": 20, "cost_usd": 1.0}


# ---- partial results: what had completed when an exception ended the run (benchmark/evaluation/replay.py records it) ----

def _stage4_fails(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("injected Stage-4 failure")
    monkeypatch.setattr(agent, "assess", boom)


def test_partial_changes_nothing_on_the_normal_path(rig):
    rig.box.res = _res(retained=[_paper()])
    rec1, ar1 = rig.run(stage4=True)
    partial = {}
    rec2, ar2 = rig.run(stage4=True, partial=partial)
    assert rec1 == rec2 and ar1.retrieval is ar2.retrieval
    assert partial["retrieval"] is rig.box.res and partial["agent_result"] is ar2
    assert H.results_of(ar2, partial) == (ar2.retrieval, ar2.assessment)


def test_a_stage4_exception_keeps_the_retrieval_result_and_no_stage4_result(rig, monkeypatch):
    rig.box.res = _res(retained=[_paper()])
    _stage4_fails(monkeypatch)
    partial = {}
    with pytest.raises(RuntimeError, match="injected Stage-4 failure"):
        rig.run(stage4=True, partial=partial)
    assert partial == {"retrieval": rig.box.res}
    res, s4 = H.results_of(None, partial)
    assert res is rig.box.res and len(res.retained) == 1 and s4 is None


def test_an_exception_after_execute_keeps_the_agent_result_with_its_stage4_result(rig, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("post-processing failure")
    monkeypatch.setattr(H, "rank_for_display", boom)
    partial = {}
    with pytest.raises(RuntimeError, match="post-processing"):
        rig.run(stage4=True, partial=partial)
    res, s4 = H.results_of(None, partial)
    assert res is rig.box.res and s4 is not None and s4 is partial["agent_result"].assessment


def test_budget_exhaustion_in_the_gate_keeps_the_retrieval_result(rig, monkeypatch):
    monkeypatch.setattr(H.U, "exhausted", lambda: {"headers": {}, "reset_s": 5})
    monkeypatch.setattr(H.U, "exhausted_message", lambda ex: "exhausted")
    partial = {}
    with pytest.raises(H.U.OpenAlexBudgetExhausted):
        rig.run(stage4=True, partial=partial)
    assert H.results_of(None, partial) == (rig.box.res, None)


def test_a_failure_before_retrieval_completes_leaves_partial_empty_even_if_it_held_an_earlier_attempt(rig, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("retrieval failed")
    monkeypatch.setattr(agent, "retrieve", boom)
    partial = {"retrieval": "stale", "agent_result": "stale"}
    with pytest.raises(RuntimeError, match="retrieval failed"):
        rig.run(stage4=True, partial=partial)
    assert partial == {} and H.results_of(None, partial) == (None, None)


def test_run_plan_checked_passes_partial_through(rig, monkeypatch):
    _stage4_fails(monkeypatch)
    partial = {}
    with pytest.raises(RuntimeError):
        H.run_plan_checked("BACKEND", dict(FIELDS), "CASE-1", {}, max_tries=1, wait_s=0, expert_prior=rig.ep, stage4=True, partial=partial)
    assert partial["retrieval"] is rig.box.res


def test_results_of_prefers_the_agent_result_that_run_plan_returned(rig):
    ar = SimpleNamespace(retrieval="R", assessment="S")
    assert H.results_of(ar, {"retrieval": "other", "agent_result": SimpleNamespace(retrieval="X", assessment="Y")}) == ("R", "S")


def test_harness_run_plan_checked_retries_only_degraded_runs_and_reuses_the_prior(rig, monkeypatch):
    monkeypatch.setattr(H.time, "sleep", lambda *_: None)
    rig.box.res = _res(axis_errors=["boom"])
    rec, ar = H.run_plan_checked("BACKEND", dict(FIELDS), "CASE-1", {}, max_tries=3, wait_s=0, expert_prior=rig.ep, stage4=True)
    assert rec["attempts"] == 3 and rec["degraded"] is True and len(rig.calls.ret) == 3 and rig.calls.s4 == []
    assert ar.retrieval is rig.box.res                           # the agent result of the LAST attempt travels with the record
    rig.calls.ret.clear()
    rig.box.res = _res()
    rec, ar = H.run_plan_checked("BACKEND", dict(FIELDS), "CASE-1", {}, max_tries=3, wait_s=0, expert_prior=rig.ep)
    assert rec["attempts"] == 1 and len(rig.calls.ret) == 1


# ------------------------------------------------------------------------------------------------------------------ the three entries, side by side (T8)
def _calls(rig):
    """The calls the flow was entered with, without the objects whose identity differs run to run."""
    return ([{k: v for k, v in c.items() if k not in ("plan", "prior")} for c in rig.calls.ret],
            [{k: v for k, v in c.items() if k not in ("plan", "prior")} for c in rig.calls.s4])


def test_three_entries_enter_the_flow_with_the_same_calls(rig):
    """The interactive path (agent.run, what the CLI calls), the direct path (agent.execute) and the evaluation harness (H.run_plan) must hand retrieve and assess the same
    arguments for the same input.  The only differences allowed are the two gates: nothing retained -> the CLI/default gate skips Stage 4, the harness (healthy run) does
    not; and `case_id` / `user_request` are whatever each caller was given (here: the same)."""
    be = ScriptedBackend()
    plan = H.make_plan(FIELDS)
    ep = ExpertPriorOutput(transfer_needed=False, transfer_axes=[])
    paper = _paper()
    rig.box.res = _res(retained=[paper])

    out = agent.run("my request", backend=be, case_id="CASE-1")                                   # P1 + P2 + execute, default gate
    assert isinstance(out, agent.AgentResult) and out.assessment is not None
    via_run = _calls(rig)
    rig.calls.ret.clear(); rig.calls.s4.clear()

    agent.execute(plan, ep, backend=be, case_id="CASE-1", user_request="my request")           # the shared path, default gate
    via_execute = _calls(rig)
    rig.calls.ret.clear(); rig.calls.s4.clear()

    H.run_plan(be, dict(FIELDS), "CASE-1", {}, expert_prior=ep, user_request="my request", stage4=True)          # the evaluation entry
    via_harness = _calls(rig)

    assert via_run == via_execute == via_harness
    (ret,), (s4,) = via_run
    assert ret == {"backend": be, "config": agent.AgentConfig(), "case_id": "CASE-1"}           # library defaults: no budgets, no Stage-4 config, default threshold
    assert s4 == {"retained": [paper], "backend": be, "config": agent.AgentConfig(), "user_request": "my request", "case_id": "CASE-1"}


def test_the_gates_are_the_only_difference_between_the_entries(rig):
    be = ScriptedBackend()
    plan, ep = H.make_plan(FIELDS), ExpertPriorOutput(transfer_needed=False, transfer_axes=[])
    rig.box.res = _res()                                                                       # nothing retained
    agent.run("req", backend=be, case_id="CASE-1")
    agent.execute(plan, ep, backend=be, case_id="CASE-1")
    assert rig.calls.s4 == []                                                                  # CLI / default gate: no Stage 4 on an empty result ...
    H.run_plan(be, dict(FIELDS), "CASE-1", {}, expert_prior=ep, stage4=True)
    assert len(rig.calls.s4) == 1 and rig.calls.s4[0]["retained"] == []                       # ... the harness gate (healthy run) does run it


def test_the_agent_configuration_reaches_retrieve_and_assess_as_given(rig):
    be = ScriptedBackend()
    plan, ep = H.make_plan(FIELDS), ExpertPriorOutput(transfer_needed=False, transfer_axes=[])
    b, c = RouteRetrievalBudgets(need_semantic_enabled=True), Stage4Config(deep_cap=7)
    rig.box.res = _res(retained=[_paper()])
    cfg = agent.AgentConfig(budgets=b, stage4=c, stage3_threshold=0.9)
    agent.execute(plan, ep, backend=be, config=cfg, case_id="X", user_request="r")
    assert rig.calls.ret[0]["config"] is cfg and rig.calls.s4[0]["config"] is cfg


# ------------------------------------------------------------------------------------------------------------------ what retrieve does with the configuration
def test_retrieve_applies_the_budgets_default_and_passes_the_stage3_threshold(monkeypatch):
    """`budgets=None` -> RouteRetrievalBudgets(); the threshold and `recover_abstracts` reach Stage 3 (filter_papers), the budgets reach the direct stream."""
    seen = {}

    def fake_direct(plan_, budgets):
        seen["budgets"] = budgets
        return None, []

    def fake_filter(papers, **kw):
        seen["stage3"] = kw
        return [], [], []
    monkeypatch.setattr(agent, "retrieve_direct", fake_direct)
    monkeypatch.setattr(agent, "filter_papers", fake_filter)
    plan, ep = H.make_plan(FIELDS), ExpertPriorOutput(transfer_needed=False, transfer_axes=[])
    agent.retrieve(plan, ep, backend=ScriptedBackend())
    assert seen["budgets"] == RouteRetrievalBudgets() and seen["stage3"] == {"classifier_threshold": agent.SIM_CLASSIFIER_THRESHOLD, "recover_abstracts": RouteRetrievalBudgets().stage3_recover_abstracts}
    b = RouteRetrievalBudgets(stage3_recover_abstracts=True)
    agent.retrieve(plan, ep, backend=ScriptedBackend(), config=agent.AgentConfig(budgets=b, stage3_threshold=0.9))
    assert seen["budgets"] is b and seen["stage3"] == {"classifier_threshold": 0.9, "recover_abstracts": True}
