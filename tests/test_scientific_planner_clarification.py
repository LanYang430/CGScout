"""Live regression checks for ScientificPlanner's clarification_needed
judgment -- specifically the "target_component over-clarification" fix
(scientific_planner.py's PLANNER_SYSTEM_PROMPT).

These are NOT offline/deterministic like the rest of this suite: clarify-
or-not is a live LLM judgment call (one real Luna API call per test), so
they're SKIPPED BY DEFAULT to keep `pytest` fast, free, and network-free
for everyone else -- run explicitly via:

    BEAM_RUN_LIVE_PLANNER_TESTS=1 pytest tests/test_scientific_planner_clarification.py -v

Real diagnosed case (this session): "I want to simulate the intrinsically
disordered regions of eIF4B and understand their self-association and
phase behavior. What coarse-grained models should I consider?" -- the
query already names the system (eIF4B), the objective (self-association /
phase behavior), and the methodology intent (coarse-grained), but the
planner asked which IDR(s)/construct to focus on anyway, even though CG
literature on this system+objective is sparse enough that a single
broadly-scoped search would already cover any of those sub-scopes -- an
over-clarification the target_component rule's missing contrastive
example (fixed this round) was meant to prevent.
"""
import os

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("BEAM_RUN_LIVE_PLANNER_TESTS") != "1",
    reason="live LLM call -- set BEAM_RUN_LIVE_PLANNER_TESTS=1 to run",
)


def test_eif4b_idr_query_does_not_trigger_clarification():
    from cgscout.llm import get_backend
    from cgscout.planning.session import PlannerSession

    session = PlannerSession(backend=get_backend(), case_id="test_eif4b_no_clarify")
    turn = session.submit_user_message(
        "I want to simulate the intrinsically disordered regions of eIF4B and "
        "understand their self-association and phase behavior. What coarse-grained "
        "models should I consider?"
    )
    assert turn.status == "ready", (
        f"expected no clarification, got: {turn.display_question_to_user!r} "
        f"(plan: {turn.plan.model_dump()})"
    )
    assert turn.plan.clarification_needed is False
