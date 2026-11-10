"""Frozen plans (plans_frozen/) carry the identity of the P1/P2 code that made them; a run or a freeze with different code, another query text, an old or foreign file is refused,
while the same code resumes normally.  Offline: freeze_plans.py is run with a fake planner session / backend; the checks are plain functions."""
import json
import types
from pathlib import Path

import pytest

from bench_support import PAPER, PV, pid_of, plan_for, read_queries, run_script, write_plans, write_queries  # noqa: F401

import arms  # noqa: E402

MODEL, EFFORT = PAPER["llm"]["model"], PAPER["llm"]["effort"]


# ===================================================================== the check itself ==================================================================================

def test_p1p2_source_hash_is_the_hash_of_the_real_source_files():
    import importlib.util
    cur = PV.p1p2_sources()
    assert set(cur) == set(PV.P1P2_MODULES) == {"cgscout.planning.scientific_planner", "cgscout.planning.expert_prior_planner", "cgscout.planning.session"}
    for name, digest in cur.items():
        assert digest == PV.sha256_file(importlib.util.find_spec(name).origin) and len(digest) == 64


def _plan(prov=None, **kw):
    return plan_for("B001", "query 1", prov or PV.stamp(model=MODEL, effort=EFFORT), **kw)


def _check(plan, **kw):
    args = dict(pid="B001", query="query 1", model=MODEL, effort=EFFORT); args.update(kw)
    return PV.check_plan(plan, **args)


def test_a_plan_made_by_the_current_code_is_accepted():
    assert _check(_plan()) == []
    assert _check(_plan(status="NOT_READY")) == []                              # a case the planner never cleared is also frozen under the same rules


def test_a_plan_without_provenance_is_refused():
    plan = _plan(); del plan["provenance"]
    assert "no provenance" in _check(plan)[0]


@pytest.mark.parametrize("module", PV.P1P2_MODULES)
def test_a_plan_made_with_different_p1_p2_code_is_refused(module):
    prov = PV.stamp(model=MODEL, effort=EFFORT)
    prov["p1p2_source_sha256"][module] = "0" * 64                              # as if that module had been edited after the freeze
    problems = _check(_plan(prov))
    assert len(problems) == 1 and module in problems[0] and "plan frozen with source" in problems[0]


def test_editing_a_p1_p2_source_file_after_the_freeze_invalidates_the_plan(tmp_path, monkeypatch):
    """The real mechanism: change one byte of a P1/P2 module -> its hash changes -> the plan frozen before is refused."""
    prov = PV.stamp(model=MODEL, effort=EFFORT)
    edited = tmp_path / "scientific_planner.py"
    edited.write_text(PV._module_file("cgscout.planning.scientific_planner").read_text() + "\n# an edit made after the freeze\n")
    real = PV._module_file
    monkeypatch.setattr(PV, "_module_file", lambda name: edited if name == "cgscout.planning.scientific_planner" else real(name))
    problems = _check(_plan(prov))
    assert len(problems) == 1 and "scientific_planner" in problems[0]


@pytest.mark.parametrize("change, expected", [
    (dict(model="another-model"), "plan frozen with model"),
    (dict(effort="low"), "plan frozen with effort"),
    (dict(pid="B002"), "belongs to case"),
    (dict(query="a different query"), "different query text"),
])
def test_plan_for_another_model_effort_case_or_query_is_refused(change, expected):
    plan = _plan()
    if "pid" in change:
        problems = _check(plan, pid="B002")
    elif "query" in change:
        problems = _check(plan, query="a different query")
    else:
        problems = _check(plan, **change)
    assert any(expected in p for p in problems), problems


def test_check_plans_summarises_missing_stale_and_usable(tmp_path):
    run = tmp_path / "run"; pids = write_queries(run, 4)
    write_plans(run, pids[:2]); write_plans(run, pids[2:3], drop_provenance=True)       # B001, B002 good; B003 old format; B004 missing
    res = PV.check_plans(run / "plans_frozen", read_queries(run), pids, model=MODEL, effort=EFFORT)
    assert res["missing"] == ["B004"] and list(res["problems"]) == ["B003"] and res["usable"] == 2 and res["checked"] == 4
    assert res["git_commits"] and res["p1p2_source_sha256"] == PV.p1p2_sources()
    text = PV.format_plan_problems(res, run / "plans_frozen")
    assert "2 of 4 usable" in text and "B004" in text and "B003" in text and "FINAL P1/P2" in text


def test_package_hash_changes_with_the_code(tmp_path, monkeypatch):
    import shutil, sys
    import cgscout
    h1 = PV.package_sha256()
    assert h1 == PV.package_sha256()                                           # deterministic
    copy = tmp_path / "cgscout"
    shutil.copytree(Path(cgscout.__file__).parent, copy, ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setitem(sys.modules, "cgscout", types.SimpleNamespace(__file__=str(copy / "__init__.py")))
    assert PV.package_sha256() == h1                                           # an identical copy hashes identically
    (copy / "text.py").write_text((copy / "text.py").read_text() + "\n# edit\n")
    assert PV.package_sha256() != h1


# ===================================================================== freeze_plans.py, in-process with fakes ============================================================

class FakeSession:
    created = []

    def __init__(self, backend, case_id):
        FakeSession.created.append(case_id)

    def submit_user_message(self, text):
        plan = types.SimpleNamespace(system="sys", target_component=None, scientific_objective="obj", search_context=[],
                                     model_dump=lambda: {"system": "sys", "target_component": None, "scientific_objective": "obj", "search_context": [], "search_axes": []})
        return types.SimpleNamespace(status="ready", plan=plan, display_question_to_user=None)


@pytest.fixture
def freeze(tmp_path, monkeypatch):
    import freeze_plans, harness
    import cgscout.llm, cgscout.planning.session
    run = tmp_path / "run"; run.mkdir()
    for name, value in (("ROOT", run), ("INPUT", run / "queries.csv"), ("PLANS", run / "plans_frozen")):
        monkeypatch.setattr(arms, name, value)
    for k in ("CGSCOUT_LLM_MODEL", "CGSCOUT_LLM_EFFORT", "CGSCOUT_CACHE_DIR", "CGSCOUT_USAGE_LOG_PATH", "PYTHONDONTWRITEBYTECODE"):
        monkeypatch.setenv(k, "x")
    FakeSession.created = []
    monkeypatch.setattr(cgscout.llm, "get_backend", lambda: types.SimpleNamespace(model=MODEL, default_effort=EFFORT, structured=lambda **kw: None))
    monkeypatch.setattr(cgscout.llm, "backend_label", lambda b: "FAKE")
    monkeypatch.setattr(cgscout.llm, "read_usage_log", lambda path=None: [])
    monkeypatch.setattr(cgscout.planning.session, "PlannerSession", FakeSession)
    monkeypatch.setattr(harness, "propose", lambda bw, fields, case_id: types.SimpleNamespace(model_dump=lambda: {"transfer_needed": False, "axes": []}))
    return types.SimpleNamespace(main=lambda *a: freeze_plans.main(list(a)), run=run)


def test_freeze_stamps_every_plan_and_a_rerun_with_the_same_code_resumes(freeze, capsys):
    pids = write_queries(freeze.run, 3)
    assert freeze.main("--expect-rows", "3") == 0
    assert len(FakeSession.created) == 3
    for pid in pids:
        plan = json.loads((freeze.run / "plans_frozen" / f"{pid}.json").read_text())
        assert plan["status"] == "READY" and plan["provenance"]["p1p2_source_sha256"] == PV.p1p2_sources() and plan["provenance"]["model"] == MODEL
    FakeSession.created.clear()
    assert freeze.main("--expect-rows", "3") == 0                               # same code: nothing regenerated, nothing refused
    assert FakeSession.created == [] and "already frozen" in capsys.readouterr().out


def test_freeze_continues_an_interrupted_run_under_the_same_code(freeze):
    pids = write_queries(freeze.run, 4)
    write_plans(freeze.run, pids[:2])                                           # two plans exist from the first, interrupted, invocation
    before = {p: (freeze.run / "plans_frozen" / f"{p}.json").read_bytes() for p in pids[:2]}
    assert freeze.main("--expect-rows", "4") == 0
    assert sorted(c.rsplit("_", 1)[0] for c in FakeSession.created) == ["plan_B003", "plan_B004"]
    assert all((freeze.run / "plans_frozen" / f"{p}.json").read_bytes() == before[p] for p in pids[:2])      # existing files are never overwritten


def test_freeze_refuses_to_mix_p1_p2_versions_in_one_directory(freeze, monkeypatch, capsys):
    pids = write_queries(freeze.run, 3)
    write_plans(freeze.run, pids[:1])
    edited = dict(PV.p1p2_sources()); edited["cgscout.planning.expert_prior_planner"] = "f" * 64          # the prompt module was edited between the two invocations
    monkeypatch.setattr(PV, "p1p2_sources", lambda: edited)
    assert freeze.main("--expect-rows", "3") == 2
    assert FakeSession.created == []                                            # refused before any LLM work
    assert sorted(p.name for p in (freeze.run / "plans_frozen").glob("*.json")) == ["B001.json"]
    assert "expert_prior_planner" in capsys.readouterr().err


def test_freeze_refuses_plans_for_another_query_text(freeze, capsys):
    pids = write_queries(freeze.run, 2)
    write_plans(freeze.run, pids[:1], query_override="what queries.csv used to say")
    assert freeze.main("--expect-rows", "2") == 2 and FakeSession.created == []
    assert "different query text" in capsys.readouterr().err


@pytest.mark.parametrize("args, expected", [([], "--expect-rows"), (["--expect-rows", "3"], "has 2 rows"), (["--pids", ""], "empty"), (["--pids", "B999"], "not in queries.csv")])
def test_freeze_selection_rules_are_the_same_as_the_runner(freeze, capsys, args, expected):
    write_queries(freeze.run, 2)
    assert freeze.main(*args) == 2 and FakeSession.created == []
    assert expected in capsys.readouterr().err


def test_freeze_dry_run_reports_state_and_never_calls_the_llm(freeze, capsys):
    pids = write_queries(freeze.run, 3)
    write_plans(freeze.run, pids[:1]); write_plans(freeze.run, pids[1:2], drop_provenance=True)
    assert freeze.main("--expect-rows", "3", "--dry-run") == 2                  # B002 is stale
    out = capsys.readouterr().out
    assert "B001 frozen, provenance OK" in out and "B002 STALE: no provenance" in out and "B003 TO DO" in out
    assert FakeSession.created == []
    assert sorted(p.name for p in (freeze.run / "plans_frozen").glob("*.json")) == ["B001.json", "B002.json"]


# ===================================================================== freeze_plans.py, real command line ================================================================

def test_freeze_cli_dry_run_and_resume_without_llm(tmp_path):
    run = tmp_path / "run"; pids = write_queries(run, 3)
    r = run_script("freeze_plans.py", ["--expect-rows", "3", "--dry-run"], run)
    assert r.returncode == 0 and r.stdout.count("TO DO") == 3 and "Traceback" not in r.stderr, r.stderr[-1500:]
    write_plans(run)                                                            # all frozen with the current code: the real command only confirms (no LLM, no key needed)
    r = run_script("freeze_plans.py", ["--expect-rows", "3"], run)
    assert r.returncode == 0 and r.stdout.count("already frozen") == 3, r.stdout[-800:] + r.stderr[-1500:]
    write_plans(run, pids[:1], drop_provenance=True)
    r = run_script("freeze_plans.py", ["--expect-rows", "3"], run)
    assert r.returncode == 2 and "no provenance" in r.stderr
