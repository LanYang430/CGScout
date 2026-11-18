"""The `cgscout` command line (src/cgscout/cli.py): start-up, the interactive loop, the printed result.  Offline.

cli.py had no test at all.  These tests PIN ITS CURRENT BEHAVIOUR (including the wording it prints today); they do not change it.
The real PlannerSession, ScientificPlanner, ExpertPriorPlanner and main() loop run against a scripted LLM backend; only the two network-bound
entry points (agent.retrieve, agent.assess: what the CLI reaches through agent.run) are replaced by recorders, and every argument they receive is asserted.
No network, no API key, no LLM."""
import builtins
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cgscout import agent, cli
from cgscout.assessment.stage4 import Judgment, PaperAssessment, Stage4Result
from cgscout.models import Paper
from cgscout.planning.expert_prior_planner import ExpertPriorOutput
from cgscout.planning.scientific_planner import PlannerOutput

ROOT = Path(__file__).resolve().parents[1]


def plan_out(*, system="beta-helix protein", objective="folding mechanism", question=None, context=(), axes=(), target=None):
    return PlannerOutput(system=system, target_component=target, scientific_objective=objective, clarification_needed=question is not None,
                         clarification_question=question, search_context=list(context), search_axes=list(axes))


class ScriptedBackend:
    """An LLMBackend whose answers are a script: planner turns in order, an empty expert prior."""
    name = "scripted"
    model = "scripted-model"
    default_effort = "medium"

    def __init__(self, planner_turns):
        self.turns = list(planner_turns)
        self.calls = []

    def structured(self, *, system, user, schema, label, effort=None, purpose=None, case_id=None, timeout=None):
        self.calls.append(dict(system=system, user=user, schema=schema, label=label, effort=effort, purpose=purpose, case_id=case_id))
        if schema is PlannerOutput:
            return self.turns.pop(0)
        if schema is ExpertPriorOutput:
            return ExpertPriorOutput(transfer_needed=False)
        raise AssertionError(f"unexpected schema {schema}")


def paper(title, year, *, cats=("system_specific_cg",), keywords=("coarse-grained",), venue=None, doi=None):
    return Paper(title=title, year=year, venue=venue, doi=doi, query_categories=list(cats), matched_keywords=list(keywords))


def judgment(*, grade=3, etype="direct_application"):
    return Judgment(verdict="graded", computational="coarse_grained", methods_used=["Go model"], system_relation="same_system", process_relation="same_process",
                    evidence_type=etype, supports_applicability="yes", grade=grade, explanation="Builds a CG model of the same protein.", supporting_quote="we built a coarse-grained model")


def assessment(rank, title, year, grade, *, doi=None, grounded=True, with_judgment=True):
    return PaperAssessment(key=title, title=title, doi=doi, year=year, status="assessed", final_grade=grade, rank=rank, quote_grounded=grounded,
                           judgment=judgment(grade=grade) if with_judgment else None)


class Rig:
    """Everything a run of main() needs, with the two retrieval entry points recorded."""

    def __init__(self, monkeypatch, capsys, inputs, planner_turns, argv=("cgscout",)):
        self.capsys = capsys
        self.backend = ScriptedBackend(planner_turns)
        self.retrieval_calls, self.stage4_calls = [], []
        self.retained = [paper("Old CG model", 2005), paper("New CG model", 2020, venue="JCTC", doi="10.1/new", cats=("transfer_direct",), keywords=("martini", "elastic network"))]
        self.stage4_result = None
        it = iter(inputs)

        def fake_input(prompt=""):
            try:
                v = next(it)
            except StopIteration:
                raise EOFError
            if isinstance(v, BaseException):
                raise v
            return v

        def fake_retrieval(plan, expert_prior, *, backend, config=None, case_id=None):
            self.retrieval_calls.append(dict(plan=plan, expert_prior=expert_prior, backend=backend, config=config, case_id=case_id))
            if isinstance(self.retrieval_error, Exception):
                raise self.retrieval_error
            return SimpleNamespace(direct_papers=self.retained[:1], transfer_papers=self.retained[1:], retained=list(self.retained), rejected=[paper("Filtered", 2001)])

        def fake_stage4(plan, expert_prior, retained, *, backend, config=None, case_id=None, user_request=None):
            self.stage4_calls.append(dict(plan=plan, expert_prior=expert_prior, retained=retained, backend=backend, config=config, user_request=user_request, case_id=case_id))
            return self.stage4_result

        self.retrieval_error = None
        monkeypatch.setattr(builtins, "input", fake_input)
        monkeypatch.setattr(sys, "argv", list(argv))
        monkeypatch.setattr(cli, "get_backend", lambda: self.backend)
        monkeypatch.setattr(agent, "retrieve", fake_retrieval)
        monkeypatch.setattr(agent, "assess", fake_stage4)

    def run(self):
        assert cli.main() is None
        return self.capsys.readouterr().out


@pytest.fixture
def rig(monkeypatch, capsys):
    def make(inputs, planner_turns=(), **kw):
        return Rig(monkeypatch, capsys, inputs, planner_turns, **kw)
    return make


# ======================================================================= start-up / loop =====================================================================

def test_banner_backend_line_and_quit(rig):
    r = rig(["quit"])
    out = r.run()
    assert out.startswith(cli.BANNER) and "LLM backend: scripted (scripted-model)" in out
    assert r.backend.calls == [] and r.retrieval_calls == []


@pytest.mark.parametrize("inputs", [[], [KeyboardInterrupt()], ["", "   ", "exit"]])
def test_eof_ctrl_c_blank_lines_and_exit_all_end_the_session_quietly(rig, inputs):
    r = rig(inputs)
    r.run()
    assert r.backend.calls == []                      # blank input never reaches the planner


def test_a_clarifying_question_is_shown_and_the_answer_continues_the_same_conversation(rig):
    r = rig(["I study a helix", "the folding of it", "quit"], [plan_out(question="Which helix protein?"), plan_out(system="alpha-helical bundle", objective="folding")])
    r.stage4_result = Stage4Result(assessments=[], problem_context="", workload={"n_retained": 0, "n_recommended": 0, "n_background": 0, "n_rejected": 0})
    out = r.run()
    assert "CGScout: Which helix protein?" in out
    assert len(r.retrieval_calls) == 1                # retrieval only after the second turn
    second_user_message = r.backend.calls[1]["user"]
    assert "[1] user: I study a helix" in second_user_message and "[2] Assistant (asked): Which helix protein?" in second_user_message and "[3] user: the folding of it" in second_user_message


def test_new_starts_an_empty_conversation(rig):
    r = rig(["first message", "new", "second message", "quit"], [plan_out(question="q?"), plan_out(question="q2?")])
    out = r.run()
    assert "CGScout: (started a new session)" in out
    assert r.backend.calls[1]["user"] == "CONVERSATION (most recent entry is what you are responding to):\n[1] user: second message"


def test_an_error_in_a_turn_is_printed_and_the_loop_goes_on(rig):
    r = rig(["go", "next", "quit"], [plan_out(), plan_out(question="still there?")])
    r.retrieval_error = RuntimeError("boom")
    out = r.run()
    assert "CGScout: (error) boom" in out and "CGScout: still there?" in out


def test_case_id_reaches_every_call(rig):
    r = rig(["go", "quit"], [plan_out()], argv=("cgscout", "--case-id", "CASE-7"))
    r.stage4_result = Stage4Result(assessments=[], problem_context="", workload={"n_retained": 0, "n_recommended": 0, "n_background": 0, "n_rejected": 0})
    r.run()
    assert [c["case_id"] for c in r.backend.calls] == ["CASE-7", "CASE-7"]            # planner, expert prior
    assert r.retrieval_calls[0]["case_id"] == "CASE-7" and r.stage4_calls[0]["case_id"] == "CASE-7"


# ======================================================================= what is handed to retrieval / Stage 4 ===============================================

def test_the_plan_and_expert_prior_are_passed_on_unchanged(rig):
    p = plan_out(system="EspP passenger domain", target=None, objective="folding", context=["membrane"], axes=["a", "b"])
    r = rig(["my request", "quit"], [p])
    r.stage4_result = Stage4Result(assessments=[], problem_context="", workload={"n_retained": 2, "n_recommended": 0, "n_background": 0, "n_rejected": 0})
    r.run()
    (call,) = r.retrieval_calls
    assert call["plan"] == p and call["expert_prior"] == ExpertPriorOutput(transfer_needed=False) and call["backend"] is r.backend
    assert call["config"] == agent.AgentConfig()                 # the command line runs with the library defaults: no budgets, no Stage-4 config
    (s4,) = r.stage4_calls
    assert s4["plan"] == p and s4["retained"] == r.retained and s4["user_request"] == "my request" and s4["backend"] is r.backend
    assert s4["config"] == agent.AgentConfig()
    ep_call = r.backend.calls[1]
    assert ep_call["label"] == "expert_prior_planner" and 'SYSTEM (already resolved): "EspP passenger domain"' in ep_call["user"] and "STATED CONTEXT: ['membrane']" in ep_call["user"]


# ======================================================================= printed result =======================================================================

def test_stage4_result_is_printed_ranked_with_evidence(rig):
    r = rig(["go", "quit"], [plan_out(context=["membrane"], axes=["axis one"], target="N-terminal domain")])
    r.stage4_result = Stage4Result(
        assessments=[assessment(1, "Direct CG paper", 2021, 3, doi="10.1/direct"), assessment(2, "Unverified quote paper", 2019, 2, grounded=False),
                     assessment(3, "Weak paper", 2010, 1), assessment(4, "Irrelevant paper", 2000, 0)],
        problem_context="", workload={"n_retained": 4, "n_recommended": 2, "n_background": 1, "n_rejected": 1})
    out = r.run()
    assert "CGScout: (plan) system='beta-helix protein' target_component='N-terminal domain'" in out
    assert "objective='folding mechanism'" in out and "search_context=['membrane']" in out and "search_axes=['axis one']" in out
    assert "CGScout: Evidence for your specific problem (ranked by scientific relevance)" in out
    assert "2 candidates -> 4 kept by the cheap filter -> 2 recommended, 1 background, 1 judged not relevant" in out
    assert "[1] Direct CG paper  (2021)  grade 3/3  [direct application]" in out and "doi=10.1/direct" in out
    assert "why: Builds a CG model of the same protein." in out and 'evidence: "we built a coarse-grained model"' in out
    assert "[2] Unverified quote paper  (2019)  grade 2/3" in out and "(quote not verified in text)" in out
    assert out.index("[1] Direct CG paper") < out.index("[2] Unverified quote paper")
    assert "Background (weaker applicability):" in out and "    - Weak paper (2010)" in out and "Irrelevant paper" not in out


def test_background_list_shows_at_most_ten_papers_and_keeps_stage4_order(rig):
    r = rig(["go", "quit"], [plan_out()])
    r.stage4_result = Stage4Result(assessments=[assessment(i, f"Background {i:02d}", 2000 + i, 1) for i in range(1, 13)], problem_context="",
                                   workload={"n_retained": 12, "n_recommended": 0, "n_background": 12, "n_rejected": 0})
    out = r.run()
    shown = [f"Background {i:02d}" for i in range(1, 13) if f"- Background {i:02d} (" in out]
    assert shown == [f"Background {i:02d}" for i in range(1, 11)]                 # the first ten, in Stage-4 order; 11 and 12 are not printed


def test_stage4_errors_are_counted_in_the_summary_line(rig):
    bad = PaperAssessment(key="bad", title="Bad", status="error", error="LLM failed")
    r = rig(["go", "quit"], [plan_out()])
    r.stage4_result = Stage4Result(assessments=[bad], problem_context="", workload={"n_retained": 2, "n_recommended": 0, "n_background": 0, "n_rejected": 0})
    assert ", 1 could not be evaluated" in r.run()


def test_no_stage4_prints_the_stage3_list_in_route_tier_then_year_order(rig):
    r = rig(["go", "quit"], [plan_out()], argv=("cgscout", "--no-stage4"))
    r.retained = [paper("Transfer paper 2024", 2024, cats=("transfer_direct",)), paper("Direct 2010", 2010), paper("Direct 2022", 2022, venue="JCTC", doi="10.1/d22", keywords=("Go model",))]
    out = r.run()
    assert r.stage4_calls == []
    assert "CGScout: Evidence for your specific problem\n" in out and "(ranked by" not in out
    assert "3 candidates -> 3 worth reading (1 filtered for lacking explicit CG evidence)" in out       # candidates = direct + transfer papers (the rig splits retained 1 + 2)
    assert out.index("[1] Direct 2022") < out.index("[2] Direct 2010") < out.index("[3] Transfer paper 2024")      # tier 0 before tier 1, newer first inside a tier
    assert "2022  JCTC  doi=10.1/d22" in out and "found via: system_specific_cg" in out and "CG evidence: Go model" in out


def test_no_candidates_message_even_with_stage4_on(rig):
    r = rig(["go", "quit"], [plan_out()])
    r.retained = []
    out = r.run()
    assert r.stage4_calls == [] and "(no candidates found across direct or transfer evidence)" in out


# ======================================================================= the real command line ===============================================================

def _run_cli(args, *, stdin=""):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CGSCOUT_", "BEAM_"))}
    env.update(PYTHONPATH=os.pathsep.join([str(ROOT / "src")] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])),
               PYTHONDONTWRITEBYTECODE="1", OPENAI_API_KEY="sk-test-not-a-real-key")
    return subprocess.run([sys.executable, "-m", "cgscout.cli", *args], input=stdin, capture_output=True, text=True, env=env, cwd=str(ROOT.parent), timeout=120)


def test_python_m_cgscout_cli_help():
    r = _run_cli(["--help"])
    assert r.returncode == 0 and "--case-id" in r.stdout and "--no-stage4" in r.stdout and "Traceback" not in r.stderr


def test_python_m_cgscout_cli_starts_and_ends_on_empty_stdin_without_any_api_call():
    r = _run_cli([], stdin="")                       # EOF at the first prompt; the real get_backend() builds the OpenAI backend object, no request is made
    assert r.returncode == 0 and "LLM backend: GPT-5.6 Luna" in r.stdout and "Traceback" not in r.stderr, r.stdout[-500:] + r.stderr[-1500:]


def test_python_m_cgscout_cli_quit_command():
    r = _run_cli([], stdin="quit\n")
    assert r.returncode == 0 and "Type 'quit' to exit" in r.stdout


def test_console_script_entry_point_is_declared():
    assert 'cgscout = "cgscout.cli:main"' in (ROOT / "pyproject.toml").read_text() and callable(cli.main)
