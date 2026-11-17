"""cgscout.agent -- the one place where the pipeline is run (a workflow-driven agent: the order of the steps is fixed, the LLM does the scientific planning
and the evidence judgement, and results are not guaranteed to repeat run to run).

    run(text)            P1 scientific planner -> P2 expert prior -> execute        (the interactive entry; returns a Clarification when P1 needs an answer)
    execute(plan, prior) retrieve -> gate -> assess                                  (the shared path: used by the CLI, by the evaluation harness, by tests)
    retrieve(...)        direct routes + 3A + 3B -> merge -> Stage 3                 -> ExpertPriorRetrievalResult
    assess(...)          Stage 4 (tier-1 screen -> evidence extraction -> judge -> rank)  -> Stage4Result
    propose_prior(plan)  P2, written once (the four planner fields are not repeated by every caller)

Consolidation step 2a: `retrieve` and `assess` still delegate to the existing entry points (`run_expert_prior_retrieval`, `run_stage4_v2`); the call shapes are
exactly what the CLI and the harness used to make themselves: `budgets` / the Stage-4 `config` are forwarded only when they are given (None and "omitted" are the same
call: `budgets or RouteRetrievalBudgets()`, `config or Stage4Config()`), and nothing else is passed.

The two callers differ on purpose and `gate` is how that is expressed.  The CLI's default gate runs Stage 4 only when something was retained.  The evaluation harness
passes a gate that first checks the run for budget exhaustion and degradation (failed HTTP requests, axis errors) and runs Stage 4 on every healthy run, also when nothing
was retained.  A gate may raise (e.g. budget exhausted); the exception leaves `execute` unchanged.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

from cgscout.assessment.stage4 import Stage4Config, Stage4Result, run_stage4_v2
from cgscout.llm import LLMBackend
from cgscout.planning.expert_prior_planner import ExpertPriorOutput, ExpertPriorPlanner
from cgscout.planning.scientific_planner import PlannerOutput
from cgscout.planning.session import PlannerSession
from cgscout.retrieval.direct.route_plan import RouteRetrievalBudgets
from cgscout.retrieval.orchestrator import ExpertPriorRetrievalResult, run_expert_prior_retrieval
from cgscout.screening.sim_classifier import SIM_CLASSIFIER_THRESHOLD


@dataclass(frozen=True)
class AgentConfig:
    """What the caller configures.  The defaults are the library defaults (what the `cgscout` command line runs with); the paper configuration is built from
    `cgscout.config.PAPER` by the evaluation code, never here."""
    budgets: Optional[RouteRetrievalBudgets] = None      # None = RouteRetrievalBudgets() (retrieval switches and caps)
    stage4: Optional[Stage4Config] = None                # None = Stage4Config()
    stage3_threshold: float = SIM_CLASSIFIER_THRESHOLD


@dataclass
class AgentResult:
    plan: PlannerOutput
    prior: ExpertPriorOutput
    retrieval: ExpertPriorRetrievalResult
    assessment: Optional[Stage4Result]                   # None when the gate said no
    assess_started_at: Optional[float]                   # time.time() right before the first Stage-4 call; the evaluation windows the Stage-4 LLM usage with it


@dataclass
class Clarification:
    """P1 needs an answer from the user.  Hand `session` back to `run` with the answer to continue the same conversation."""
    question: Optional[str]
    session: PlannerSession


Gate = Callable[[ExpertPriorRetrievalResult], bool]


def propose_prior(plan: PlannerOutput, *, backend: LLMBackend, case_id: Optional[str] = None) -> ExpertPriorOutput:
    return ExpertPriorPlanner(backend).propose(
        system=plan.system, target_component=plan.target_component,
        scientific_objective=plan.scientific_objective, search_context=plan.search_context,
        case_id=case_id,
    )


def retrieve(plan: PlannerOutput, prior: ExpertPriorOutput, *, backend: LLMBackend, config: Optional[AgentConfig] = None,
             case_id: Optional[str] = None) -> ExpertPriorRetrievalResult:
    cfg = config or AgentConfig()
    kw = {}
    if cfg.budgets is not None:
        kw["budgets"] = cfg.budgets
    if cfg.stage3_threshold != SIM_CLASSIFIER_THRESHOLD:
        kw["stage3_classifier_threshold"] = cfg.stage3_threshold
    return run_expert_prior_retrieval(plan, prior, backend=backend, case_id=case_id, **kw)


def assess(plan: PlannerOutput, prior: ExpertPriorOutput, retained: list, *, backend: LLMBackend, config: Optional[AgentConfig] = None,
           case_id: Optional[str] = None, user_request: Optional[str] = None) -> Stage4Result:
    cfg = config or AgentConfig()
    kw = {"config": cfg.stage4} if cfg.stage4 is not None else {}
    return run_stage4_v2(plan, prior, retained, backend, user_request=user_request, case_id=case_id, **kw)


def execute(plan: PlannerOutput, prior: ExpertPriorOutput, *, backend: LLMBackend, config: Optional[AgentConfig] = None,
            case_id: Optional[str] = None, user_request: Optional[str] = None, gate: Optional[Gate] = None) -> AgentResult:
    cfg = config or AgentConfig()
    res = retrieve(plan, prior, backend=backend, config=cfg, case_id=case_id)
    go = gate(res) if gate is not None else bool(res.retained)
    s4, started = None, None
    if go:
        started = time.time()
        s4 = assess(plan, prior, list(res.retained), backend=backend, config=cfg, case_id=case_id, user_request=user_request)
    return AgentResult(plan=plan, prior=prior, retrieval=res, assessment=s4, assess_started_at=started)


def run(text: str, *, backend: LLMBackend, session: Optional[PlannerSession] = None, config: Optional[AgentConfig] = None,
        case_id: Optional[str] = None, gate: Optional[Gate] = None,
        on_plan: Optional[Callable[[PlannerOutput], None]] = None) -> "Clarification | AgentResult":
    """One user turn.  `on_plan(plan)` is called as soon as P1 has produced a ready plan, before P2 and retrieval start (the command line prints the plan there,
    so that it appears before the long steps, exactly as it always did)."""
    session = session or PlannerSession(backend=backend, case_id=case_id)
    turn = session.submit_user_message(text)
    if turn.status == "awaiting_user":
        return Clarification(question=turn.display_question_to_user, session=session)
    if turn.status != "ready":
        raise ValueError(f"unknown planner session status {turn.status!r}")
    plan = session.last_plan
    if on_plan is not None:
        on_plan(plan)
    prior = propose_prior(plan, backend=backend, case_id=case_id)
    return execute(plan, prior, backend=backend, config=config, case_id=case_id, user_request=text, gate=gate)
