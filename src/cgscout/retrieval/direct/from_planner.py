"""from_planner (scientific-planner-v2 branch) -- turns a
PlannerOutput into a Stage-2 retrieval run. NEW, additive module.

Owns zero retrieval logic of its own: it only maps PlannerOutput fields
onto the EXISTING, UNMODIFIED plan_evidence_routes() / run_route_retrieval()
public functions (evidence_plan.py / route_plan.py --
shared, unedited infrastructure used by both the old and new intake path)
and reuses search_adapters.py's _make_capturing_search_fns /
papers_for_stage3 / provenance_index helpers UNCHANGED rather than
reimplementing them.

Deliberately does NOT touch the structural branch: grounded_structural_
feature is always None here, so EvidenceRoutePlan.structural_terms stays
empty and structural_branch_enabled is False -- the structural/related-
system branch stays fully paused, exactly as on the old path right now.
No hop1_search_fn/hop1_extract_fn/hop2_search_fn are passed to
run_route_retrieval() because they are never needed while structural_
terms is empty.

modeling_requirement_mode="rule" + call=None: the modeling_requirement
LLM call (dead -- that route is disabled, see route_plan.py) is
never made on this path, by construction, not by a special case.
"""
from __future__ import annotations

from typing import Optional


from cgscout.retrieval.direct.evidence_plan import EvidenceRoutePlan, RouteTerm, plan_evidence_routes  # noqa: E402
from cgscout.retrieval.direct.route_plan import (  # noqa: E402
    RouteRetrievalBudgets,
    RouteRetrievalResult,
    resolve_structural_and_safety_budgets,
    run_route_retrieval,
)
from cgscout.retrieval.search_adapters import (  # noqa: E402
    STRUCTURAL_HOP2_SEARCH_LIMIT,
    _make_capturing_search_fns,
    make_need_semantic_search_fn,
)
from cgscout.models import Paper  # noqa: E402

from cgscout.planning.scientific_planner import PlannerOutput  # noqa: E402


def _norm_ctx(s: str) -> str:
    return " ".join((s or "").lower().split())


def evidence_plan_from_planner_output(plan: PlannerOutput) -> EvidenceRoutePlan:
    system_description = plan.system
    if plan.target_component:
        system_description = f"{system_description} ({plan.target_component})"
    evidence_plan = plan_evidence_routes(
        system_description=system_description,
        scientific_objective=plan.scientific_objective,
        grounded_structural_feature=None,  # structural branch stays paused
        user_stated_context="; ".join(plan.search_context) or None,
        call=None,
        modeling_requirement_mode="rule",  # no modeling_requirement LLM call -- route is disabled
    )
    # Hard guarantee, independent of grounded_structural_feature above:
    # plan_evidence_routes() ALSO derives structural_terms lexically from
    # system_description itself (extract_lexical_structural_feature(),
    # evidence_plan.py -- unmodified, unrelated to this branch's
    # work). Diagnosed live: "EspP passenger domain" -> structural_terms=
    # ['EspP passenger'], because "passenger" is itself in
    # _STRUCTURAL_MORPHEMES (added for the earlier EspP/pertactin case) --
    # correct behavior of that OLD, untouched code, but this bridge must
    # not let it silently turn the structural branch on. Force-cleared
    # here rather than upstream, so the old lexical extractor's behavior
    # stays exactly as it is for whoever else relies on it.
    evidence_plan.structural_terms = []
    evidence_plan.search_axes = [a for a in (getattr(plan, "search_axes", None) or []) if a and a.strip()]     # used only when budgets.query_construction_v2 is set
    # Flat lexical retrieval needs SHORT queries. plan_evidence_routes() takes user_stated_context as ONE string, so the
    # joined search_context list above became a single context term (every item required in one query). Replace that
    # one joined term by one context term per search_context item (same USER_GROUNDED provenance, same order);
    # context terms that came from elsewhere (system-description markers) are left exactly as they were.
    items = [c.strip() for c in plan.search_context if c and c.strip()]
    joined = "; ".join(plan.search_context).strip()
    if len(items) > 1:
        kept = [ct for ct in evidence_plan.context_terms if ct.term.strip() != joined]
        have = {_norm_ctx(ct.term) for ct in kept}
        for it in items:
            if _norm_ctx(it) not in have:
                have.add(_norm_ctx(it))
                kept.append(RouteTerm(term=it, route_type="context", source="USER_GROUNDED", supporting_span=it,
                                      provenance={"source": "user_stated_context"}))
        evidence_plan.context_terms = kept
    return evidence_plan


def run_retrieval_from_plan(
    plan: PlannerOutput, budgets: Optional[RouteRetrievalBudgets] = None,
) -> tuple[EvidenceRoutePlan, RouteRetrievalResult, dict[str, Paper]]:
    evidence_plan = evidence_plan_from_planner_output(plan)
    budgets = budgets or RouteRetrievalBudgets()
    budgets = resolve_structural_and_safety_budgets(
        budgets, structural_hop2_search_limit=STRUCTURAL_HOP2_SEARCH_LIMIT)

    paper_cache: dict[str, Paper] = {}
    flat_search_fn, _hop1_fn, _hop2_fn = _make_capturing_search_fns(paper_cache, budgets)
    # P3: the semantic function exists only when the route is enabled -- with the default budgets nothing changes.
    need_fn = make_need_semantic_search_fn(paper_cache) if budgets.need_semantic_enabled else None

    result = run_route_retrieval(evidence_plan, flat_search_fn=flat_search_fn, budgets=budgets, need_semantic_search_fn=need_fn)
    return evidence_plan, result, paper_cache
