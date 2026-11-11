"""PlannerSession (scientific-planner-v2 branch) -- the Plan-B intake
orchestrator. NEW, additive module.

Replaces ConceptResolverSession's submit_user_message() (A -> SystemIdentity
-> B -> Guard -> IntakeReadinessGate -> C -> D, concept_resolver_session.py)
with: one ScientificPlanner call + a thin deterministic router. Does NOT
import concept_resolver_session.py -- the two session types are unrelated
and the old one is untouched.

    user query -> ScientificPlanner.plan() -> clarification_needed?
        true  -> PlannerTurnResult(status="awaiting_user", ...)
        false -> PlannerTurnResult(status="ready", plan=...)
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

from cgscout.llm import LLMBackend, get_backend
from cgscout.planning.scientific_planner import PlannerOutput, ScientificPlanner

SessionStatus = Literal["awaiting_user", "ready"]


class PlannerTurnResult(BaseModel):
    status: SessionStatus
    display_question_to_user: Optional[str] = None
    plan: Optional[PlannerOutput] = None


class PlannerSession:
    """One conversation. `case_id`, if given, is threaded through to every
    LLM call this session makes, for the usage log / benchmark reporting."""

    def __init__(
        self, *, backend: Optional[LLMBackend] = None,
        effort: Optional[str] = None, case_id: Optional[str] = None,
    ):
        self._planner = ScientificPlanner(backend or get_backend(), effort=effort)
        self._history: list[str] = []
        self._case_id = case_id
        self.last_plan: Optional[PlannerOutput] = None

    def submit_user_message(self, text: str) -> PlannerTurnResult:
        self._history.append(f"user: {text}")
        plan = self._planner.plan(self._history, case_id=self._case_id)
        self.last_plan = plan

        if plan.clarification_needed and plan.clarification_question:
            self._history.append(f"Assistant (asked): {plan.clarification_question}")
            return PlannerTurnResult(
                status="awaiting_user",
                display_question_to_user=plan.clarification_question,
                plan=plan,
            )
        return PlannerTurnResult(status="ready", plan=plan)
