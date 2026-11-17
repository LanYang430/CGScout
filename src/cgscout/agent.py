"""cgscout.agent -- the one place where the pipeline is run (a workflow-driven agent: the order of the steps is fixed, the LLM does the scientific planning
and the evidence judgement, and results are not guaranteed to repeat run to run).

    run(text)            P1 scientific planner -> P2 expert prior -> execute        (the interactive entry; returns a Clarification when P1 needs an answer)
    execute(plan, prior) retrieve -> gate -> assess                                  (the shared path: used by the CLI, by the evaluation harness, by tests)
    retrieve(...)        direct routes + 3A + 3B -> merge -> Stage 3                 -> ExpertPriorRetrievalResult  (+ rank_for_display: the display order)
    assess(...)          Stage 4 (tier-1 screen -> evidence extraction -> judge -> rank)  -> Stage4Result
    propose_prior(plan)  P2, written once (the four planner fields are not repeated by every caller)

Consolidation step 2b: `retrieve` is the dispatch (it was `run_expert_prior_retrieval` in retrieval/orchestrator.py): direct routes, the per-axis loop over the transfer
axes (3A and 3B executors live in retrieval/transfer.py), the merge and Stage 3 (a private named step).  `assess` still delegates to `run_stage4_v2`.  `budgets` / the
Stage-4 `config` are forwarded only when they are given (None and "omitted" are the same: `budgets or RouteRetrievalBudgets()`, `config or Stage4Config()`).

The two callers differ on purpose and `gate` is how that is expressed.  The CLI's default gate runs Stage 4 only when something was retained.  The evaluation harness
passes a gate that first checks the run for budget exhaustion and degradation (failed HTTP requests, axis errors) and runs Stage 4 on every healthy run, also when nothing
was retained.  A gate may raise (e.g. budget exhausted); the exception leaves `execute` unchanged.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

from pydantic import BaseModel, Field

from cgscout.assessment.stage4 import Stage4Config, Stage4Result, run_stage4_v2
from cgscout.llm import LLMBackend
from cgscout.models import Paper
from cgscout.planning.expert_prior_planner import ExpertPriorOutput, ExpertPriorPlanner, TransferAxis
from cgscout.planning.scientific_planner import PlannerOutput
from cgscout.planning.session import PlannerSession
from cgscout.retrieval.concept_3a import render_direct_concept_queries
from cgscout.retrieval.direct.from_planner import retrieve_direct
from cgscout.retrieval.direct.route_plan import RouteRetrievalBudgets
from cgscout.retrieval.related_systems.postprocess import RelatedSystemIdentity
# The transfer-axis executors.  The underscore names are imported on purpose: they are the executors this dispatch calls and nothing else uses them; renaming them would
# also change the recorder's classification by function name (benchmark/evaluation/recorder.py), so that is a separate step.  This module must not bind
# `search_and_dedupe` itself: every search is made by the executors in retrieval/transfer.py, which is where the recorder looks.
from cgscout.retrieval.transfer import _merge_papers, _run_direct_concept_search, _run_related_system_discovery
from cgscout.screening.filter import Stage3Verdict, filter_papers
from cgscout.screening.sim_classifier import SIM_CLASSIFIER_THRESHOLD

# Hard cap on how many direct_concept_search axes get a semantic query in
# one case/run -- a defensive bound, not something normal cases are
# expected to hit (every case seen so far proposes at most 2). Axes beyond
# the cap are skipped for retrieval (recorded in their axis_log), not
# silently truncated into a combined query.
MAX_DIRECT_CONCEPT_SEMANTIC_QUERIES_PER_CASE = 3

class TransferAxisExecutionLog(BaseModel):
    axis: TransferAxis
    executed: bool
    skip_reason: Optional[str] = None
    direct_queries: list[str] = Field(default_factory=list)          # direct_concept_search only
    hop1_queries: list[str] = Field(default_factory=list)            # discovery only
    related_system_identities: list[RelatedSystemIdentity] = Field(default_factory=list)
    hop2_queries: list[str] = Field(default_factory=list)            # exact per-identity Hop-2 query text
    hop2_plan: list[dict] = Field(default_factory=list)              # per system: search_anchor / core_process / stable_descriptor, queries + n_results, fallback_used
    n_papers_found: int = 0
    error: Optional[str] = None


class ExpertPriorRetrievalResult(BaseModel):
    transfer_needed: bool = False
    axis_logs: list[TransferAxisExecutionLog] = Field(default_factory=list)
    direct_papers: list[Paper] = Field(default_factory=list)
    transfer_papers: list[Paper] = Field(default_factory=list)
    retained: list[Paper] = Field(default_factory=list)
    rejected: list[Paper] = Field(default_factory=list)
    stage3_verdicts: list[Stage3Verdict] = Field(
        default_factory=list,
        description="One Stage3Verdict per paper in retained+rejected -- review_flagged/review_reason, "
        "classifier_score, keep, and reason, for per-paper audit (e.g. the Pertactin end-to-end check).",
    )


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


def _plan_axis_queries(plan: PlannerOutput, expert_prior: ExpertPriorOutput, evidence_plan, budgets: RouteRetrievalBudgets) -> dict[int, list[str]]:
    """The 3A query planning, moved verbatim from the dispatch body: which direct_concept_search axes get queries (at most
    MAX_DIRECT_CONCEPT_SEMANTIC_QUERIES_PER_CASE of them) and which queries.  Keys are indices into expert_prior.transfer_axes."""
    # direct_concept_search's queries: 3 fixed deterministic templates per
    # axis (no LLM call -- see concept_3a.py), capped
    # to at most MAX_DIRECT_CONCEPT_SEMANTIC_QUERIES_PER_CASE axes per
    # case (so at most 3x that many OpenAlex calls). Computed up front so
    # the per-axis loop below just looks its result up.
    direct_concept_axis_indices = [
        i for i, a in enumerate(expert_prior.transfer_axes)
        if a.confidence != "low" and a.execution_mode == "direct_concept_search"
    ]
    capped_indices = direct_concept_axis_indices[:MAX_DIRECT_CONCEPT_SEMANTIC_QUERIES_PER_CASE]
    if budgets.query_construction_v2:
        from cgscout.retrieval.direct.query_construction_v2 import analyze, render_direct_concept_queries_v2       # lazy: avoids a circular import
        _qc = analyze(evidence_plan, budgets)
        queries_by_axis_index: dict[int, list[str]] = {
            i: render_direct_concept_queries_v2(expert_prior.transfer_axes[i].scientific_class, _qc) for i in capped_indices
        }
    else:
        queries_by_axis_index = {
            i: render_direct_concept_queries(expert_prior.transfer_axes[i].scientific_class, plan.scientific_objective)
            for i in capped_indices
        }
    return queries_by_axis_index


def _stage3_screen(papers: list[Paper], *, classifier_threshold: float, recover_abstracts: bool):
    """Stage 3 (the keyword / CG-evidence gate, review filter, embedding classifier, abstract recovery), run ONCE over the merged candidates.  The result contract
    is filter_papers's: (retained, rejected, verdicts).  A named step, not a public stage; `filter_papers` is looked up in this module at call time."""
    return filter_papers(papers, classifier_threshold=classifier_threshold, recover_abstracts=recover_abstracts)


def retrieve(plan: PlannerOutput, prior: ExpertPriorOutput, *, backend: LLMBackend, config: Optional[AgentConfig] = None,
             case_id: Optional[str] = None) -> ExpertPriorRetrievalResult:
    """Direct routes + 3A + 3B -> merge -> Stage 3.  (The body of the former orchestrator.run_expert_prior_retrieval, moved here in consolidation 2b.)

    `backend` is REQUIRED (no default -- llm.get_backend()'s docstring: no module resolves or falls back to a backend on its own).  The caller resolves ONE backend for the
    whole run/session and passes the SAME instance here as it does to ScientificPlanner/ExpertPriorPlanner -- this function threads it into every Hop-1 LLM call a
    discovery-mode transfer axis makes.  Routes that make no LLM call at all (direct-route retrieval, direct_concept_search axes, Stage 3) don't need it and don't receive it.

    `config.stage3_threshold`: pass-through to filter_papers()'s own `classifier_threshold` -- the one knob to change the sim classifier's decision boundary for a whole
    run, without editing sim_classifier.py's module constant.  `config.budgets=None` means RouteRetrievalBudgets() (the library defaults)."""
    cfg = config or AgentConfig()
    budgets = cfg.budgets or RouteRetrievalBudgets()
    stage3_classifier_threshold = cfg.stage3_threshold
    expert_prior = prior
    result = ExpertPriorRetrievalResult(transfer_needed=expert_prior.transfer_needed)

    # -- direct evidence: UNCHANGED, reuse from_planner --
    # budgets are forwarded so the flat-query / need_semantic switches (RouteRetrievalBudgets) reach the direct-evidence routes too;
    # with budgets=None this is exactly the previous call (defaults are applied inside).
    evidence_plan, direct_papers = retrieve_direct(plan, budgets)
    result.direct_papers = direct_papers

    # -- transfer evidence: high/medium confidence axes only. execution_mode
    # is read verbatim from each axis, never mapped from a basis -> mode
    # table in this file. --

    queries_by_axis_index = _plan_axis_queries(plan, expert_prior, evidence_plan, budgets)

    transfer_paper_cache: dict[str, Paper] = {}
    merged_transfer: dict[str, Paper] = {}
    for axis_index, axis in enumerate(expert_prior.transfer_axes):
        if axis.confidence == "low":
            result.axis_logs.append(TransferAxisExecutionLog(
                axis=axis, executed=False,
                skip_reason="low confidence -- recorded internally, not retrieved, not shown",
            ))
            continue

        direct_queries: list[str] = []
        hop1_queries: list[str] = []
        identities: list[RelatedSystemIdentity] = []
        hop2_queries: list[str] = []
        hop2_plan: list[dict] = []
        try:
            if axis.execution_mode == "direct_concept_search":
                if axis_index not in queries_by_axis_index:
                    result.axis_logs.append(TransferAxisExecutionLog(
                        axis=axis, executed=False,
                        skip_reason=f"direct_concept_search semantic-query cap reached "
                                    f"({MAX_DIRECT_CONCEPT_SEMANTIC_QUERIES_PER_CASE} per case)",
                    ))
                    continue
                if not queries_by_axis_index[axis_index]:
                    result.axis_logs.append(TransferAxisExecutionLog(
                        axis=axis, executed=False,
                        skip_reason="direct_concept_search template rendering produced no query "
                                    "(empty scientific_class or objective)",
                    ))
                    continue
                papers, direct_queries = _run_direct_concept_search(
                    axis, queries=queries_by_axis_index[axis_index],
                    paper_cache=transfer_paper_cache, budgets=budgets)
                error: Optional[str] = None
            elif axis.execution_mode == "related_system_discovery":
                papers, hop1_queries, identities, hop2_queries, error, hop2_plan = _run_related_system_discovery(
                    axis, scientific_objective=plan.scientific_objective, original_system=plan.system,
                    paper_cache=transfer_paper_cache, budgets=budgets, backend=backend, case_id=case_id)
            else:  # unrecognized mode from a malformed/unexpected model output -- skip, don't guess
                papers = []
                error = f"unrecognized execution_mode {axis.execution_mode!r}"
        except Exception as exc:  # noqa: BLE001 -- one axis's failure never blocks the others
            papers, error = [], str(exc)

        _merge_papers(merged_transfer, papers)
        result.axis_logs.append(TransferAxisExecutionLog(
            axis=axis, executed=True, direct_queries=direct_queries, hop1_queries=hop1_queries,
            related_system_identities=identities, hop2_queries=hop2_queries, hop2_plan=hop2_plan,
            n_papers_found=len(papers), error=error,
        ))

    result.transfer_papers = list(merged_transfer.values())

    # -- merge direct + transfer, dedupe by paper_ref, run Stage 3 ONCE --
    merged: dict[str, Paper] = {}
    _merge_papers(merged, result.direct_papers)
    _merge_papers(merged, result.transfer_papers)

    retained, rejected, verdicts = _stage3_screen(list(merged.values()), classifier_threshold=stage3_classifier_threshold,
                                                  recover_abstracts=budgets.stage3_recover_abstracts)
    result.retained = retained
    result.rejected = rejected
    result.stage3_verdicts = verdicts
    return result


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


# --- display ranking (direct evidence first, then executed transfer axes) --

_TIER_LABELS = (
    ("system_specific_cg", 0), ("process", 0), ("context", 0), ("need_semantic", 0),
    ("structural", 1), ("transfer_direct", 1),
)


def provenance_tier(paper: Paper) -> int:
    """direct(0) < transfer high/medium(1). No separate low tier: a
    low-confidence axis never executes, so no paper can ever carry a
    low-confidence transfer label at this stage."""
    cats = set(paper.query_categories or [])
    tiers = [t for label, t in _TIER_LABELS if label in cats]
    return min(tiers) if tiers else 2


def rank_for_display(papers: list[Paper]) -> list[Paper]:
    return sorted(papers, key=lambda p: (provenance_tier(p), -(p.year or 0)))
