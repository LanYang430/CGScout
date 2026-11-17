"""The README's "From Python" example runs as written (offline: a scripted backend, and the two network-bound steps agent.retrieve / agent.assess replaced)."""
import builtins
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from cgscout import agent, llm
from cgscout.assessment.stage4 import Judgment, PaperAssessment, Stage4Result
from cgscout.models import Paper
from cgscout.planning.expert_prior_planner import ExpertPriorOutput
from cgscout.planning.scientific_planner import PlannerOutput

ROOT = Path(__file__).resolve().parents[1]


def _readme_example() -> str:
    text = (ROOT / "README.md").read_text()
    section = text.split("### From Python", 1)[1]
    return re.search(r"```python\n(.*?)```", section, re.S).group(1)


class Scripted:
    name, model, default_effort = "scripted", "scripted-model", "medium"

    def __init__(self):
        self.turns = [PlannerOutput(system="EspP", scientific_objective="folding", clarification_needed=True, clarification_question="Which domain of EspP?"),
                      PlannerOutput(system="EspP", target_component="passenger domain", scientific_objective="folding", clarification_needed=False)]

    def structured(self, *, system, user, schema, label, effort=None, purpose=None, case_id=None, timeout=None):
        if schema is PlannerOutput:
            return self.turns.pop(0)
        if schema is ExpertPriorOutput:
            return ExpertPriorOutput(transfer_needed=False)
        raise AssertionError(schema)


def _assessment():
    j = Judgment(verdict="graded", computational="coarse_grained", methods_used=["Go model"], system_relation="same_system", process_relation="same_process",
                 evidence_type="direct_application", supports_applicability="yes", grade=3, explanation="x", supporting_quote="we built a coarse-grained model")
    return Stage4Result(assessments=[PaperAssessment(key="k", title="A CG model of EspP", year=2021, status="assessed", final_grade=3, rank=1, quote_grounded=True, judgment=j)],
                        problem_context="", workload={"n_retained": 1, "n_recommended": 1, "n_background": 0, "n_rejected": 0})


@pytest.mark.parametrize("retained", [True, False])
def test_the_readme_example_runs_as_written(monkeypatch, capsys, retained):
    be = Scripted()
    papers = [Paper(title="A CG model of EspP", year=2021, query_categories=["system_specific_cg"])] if retained else []
    monkeypatch.setattr(llm, "get_backend", lambda: be)
    monkeypatch.setattr(builtins, "input", lambda prompt="": "the passenger domain")
    monkeypatch.setattr(agent, "retrieve", lambda plan, prior, **kw: SimpleNamespace(retained=papers, direct_papers=papers, transfer_papers=[], rejected=[], axis_logs=[], stage3_verdicts=[]))
    monkeypatch.setattr(agent, "assess", lambda *a, **kw: _assessment())
    exec(compile(_readme_example(), "README.md", "exec"), {"__name__": "readme_example"})
    out = capsys.readouterr().out
    assert ("[1] A CG model of EspP (2021)  grade 3/3" in out) if retained else ("no candidates found" in out)
    assert be.turns == []                                              # both planner turns were consumed: the clarification loop ran once
