"""Consolidation safety net, Phase 0 -- what each ENTRY hands to the flow today (baseline of the three-entry equivalence, T7-T9).  Offline.

The CLI (cli.run_retrieval) and the evaluation harness (harness.run_plan) both run  retrieval -> Stage 4  but are not identical, and a unified entry
must PRESERVE these differences, not erase them:
  * CLI      retrieval is called with no budgets (library defaults); Stage 4 runs only when `use_stage4` and at least one paper was retained; no `config`;
             Stage 4 is not gated by any degraded-run check.
  * harness  budgets are forwarded only when given; Stage 4 runs when `stage4` is set and the run is not degraded (an axis error or a failed HTTP request),
             also when nothing was retained; `config=stage4_config` is always passed.
The two entry points of the flow (run_expert_prior_retrieval, run_stage4_v2) are replaced by recorders, so no network and no LLM is involved.
The CLI side of this record lives in tests/test_cli.py (case_id, plan/prior pass-through, --no-stage4, no-candidates)."""
import types
from types import SimpleNamespace

import pytest

import bench_support  # noqa: F401  (puts benchmark/evaluation on sys.path)
import harness as H
from cgscout.assessment.stage4 import Stage4Config
from cgscout.planning.expert_prior_planner import ExpertPriorOutput
from cgscout.retrieval.direct.route_plan import RouteRetrievalBudgets

FIELDS = {"system": "Aquila", "scientific_objective": "folding mechanism", "search_context": [], "search_axes": []}


def _res(retained=(), axis_errors=()):
    return SimpleNamespace(
        axis_logs=[SimpleNamespace(error=e, axis=SimpleNamespace(basis="b", execution_mode="m"), executed=True, skip_reason=None, direct_queries=[],
                                   hop1_queries=[], hop2_queries=[], related_system_identities=[], n_papers_found=0) for e in axis_errors],
        direct_papers=[], transfer_papers=[], retained=list(retained), rejected=[], stage3_verdicts=[])


@pytest.fixture
def rig(monkeypatch):
    calls = SimpleNamespace(ret=[], s4=[])
    box = SimpleNamespace(res=_res())

    def fake_ret(plan, ep, **kw):
        calls.ret.append(kw)
        return box.res

    def fake_s4(plan, ep, retained, backend, **kw):
        calls.s4.append(dict(retained=list(retained), **kw))
        return SimpleNamespace(workload={}, assessments=[], by_key=lambda: {})
    monkeypatch.setattr(H, "run_expert_prior_retrieval", fake_ret)
    monkeypatch.setattr(H, "run_stage4_v2", fake_s4)
    monkeypatch.setattr(H, "read_usage_log", lambda: [])
    monkeypatch.setattr(H.U, "exhausted", lambda: None)
    monkeypatch.setattr(H.U, "reset_run", lambda *_: None)
    monkeypatch.setattr(H.U, "set_run_label", lambda *_: None)
    monkeypatch.setattr(H.U, "run_usage", lambda *_: {"failed_requests": 0, "http_requests": 0, "cache_hits": 0, "credits_charged": 0})
    ep = ExpertPriorOutput(transfer_needed=False, transfer_axes=[])
    return SimpleNamespace(calls=calls, box=box, ep=ep, run=lambda **kw: H.run_plan("BACKEND", dict(FIELDS), "CASE-1", {}, expert_prior=ep, **kw))


def test_harness_forwards_budgets_only_when_given(rig):
    rig.run()
    assert "budgets" not in rig.calls.ret[0] and rig.calls.ret[0]["case_id"] == "CASE-1" and rig.calls.ret[0]["backend"] == "BACKEND"
    b = RouteRetrievalBudgets(need_semantic_enabled=True)
    rig.run(budgets=b)
    assert rig.calls.ret[1]["budgets"] is b


def test_harness_stage4_runs_when_not_degraded_even_with_nothing_retained(rig):
    cfg = Stage4Config(deep_cap=7)
    rig.run(stage4=True, stage4_config=cfg, user_request="req")
    assert len(rig.calls.s4) == 1
    s4 = rig.calls.s4[0]
    assert s4["retained"] == [] and s4["config"] is cfg and s4["user_request"] == "req" and s4["case_id"] == "CASE-1"      # empty retained still reaches Stage 4


def test_harness_stage4_skipped_when_an_axis_errored(rig):
    rig.box.res = _res(axis_errors=["boom"])
    rec = rig.run(stage4=True)
    assert rig.calls.s4 == [] and rec["degraded"] is True and rec["axis_errors"] == ["boom"]


def test_harness_stage4_skipped_when_a_http_request_failed(rig, monkeypatch):
    monkeypatch.setattr(H.U, "run_usage", lambda *_: {"failed_requests": 1, "http_requests": 1, "cache_hits": 0, "credits_charged": 0})
    rec = rig.run(stage4=True)
    assert rig.calls.s4 == [] and rec["degraded"] is True


def test_harness_without_stage4_never_calls_it(rig):
    rig.run(stage4=False)
    assert rig.calls.s4 == []


def test_harness_run_plan_checked_retries_only_degraded_runs_and_reuses_the_prior(rig, monkeypatch):
    monkeypatch.setattr(H.time, "sleep", lambda *_: None)
    rig.box.res = _res(axis_errors=["boom"])
    rec = H.run_plan_checked("BACKEND", dict(FIELDS), "CASE-1", {}, max_tries=3, wait_s=0, expert_prior=rig.ep, stage4=True)
    assert rec["attempts"] == 3 and rec["degraded"] is True and len(rig.calls.ret) == 3 and rig.calls.s4 == []
    rig.calls.ret.clear()
    rig.box.res = _res()
    rec = H.run_plan_checked("BACKEND", dict(FIELDS), "CASE-1", {}, max_tries=3, wait_s=0, expert_prior=rig.ep)
    assert rec["attempts"] == 1 and len(rig.calls.ret) == 1
