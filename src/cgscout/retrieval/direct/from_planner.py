"""from_planner (scientific-planner-v2 branch) -- turns a
PlannerOutput into a Stage-2 retrieval run. NEW, additive module.

Owns zero retrieval logic of its own: it only maps PlannerOutput fields
onto the EXISTING, UNMODIFIED plan_evidence_routes() / run_route_retrieval()
public functions (evidence_plan.py / route_plan.py --
shared, unedited infrastructure used by both the old and new intake path)
and owns the capturing search wrappers (_make_capturing_search_fns /
make_need_semantic_search_fn) and the papers_for_stage3 / provenance_index
adapters (formerly retrieval/search_adapters.py, merged here unchanged).

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
from cgscout.retrieval.related_systems.discovery import _paper_ref  # noqa: E402 -- same paper_ref convention used throughout
from cgscout.retrieval.search import (  # noqa: E402
    STRUCTURAL_HOP1_SEARCH_LIMIT,
    STRUCTURAL_HOP2_SEARCH_LIMIT,
    STRUCTURAL_SEARCH_SOURCES,
    search_and_dedupe,
)
from cgscout.models import Paper  # noqa: E402

from cgscout.planning.scientific_planner import PlannerOutput  # noqa: E402


# ===== moved verbatim from `retrieval/search_adapters.py` (consolidation A3) =================================================================
# The capturing search wrappers (every real Paper retrieved is cached by paper_ref for Stage 3) and the two small adapters from a
# RouteRetrievalResult to the Stage 3 contract.  `_make_capturing_search_fns` keeps its two unused Hop-1/Hop-2 closures (their removal is a
# separate, later cleanup).  The route code (3A / 3B in transfer.py) calls papers_for_stage3 / provenance_index from here.

def _make_capturing_search_fns(paper_cache: dict[str, Paper], budgets: RouteRetrievalBudgets):
    """One shared paper_cache across all 5 routes -- the sidecar paper_ref ->
    full Paper mapping Stage 3 needs, since RouteRetrievalResult.papers
    itself only carries paper_ref/title/year, not abstract/authors/etc."""

    def flat_search_fn(query: str, limit: int) -> list[Paper]:
        papers = search_and_dedupe(query, limit=limit)
        for p in papers:
            paper_cache.setdefault(_paper_ref(p), p)
        return papers

    def structural_hop1_search_fn(query: str) -> list[Paper]:
        papers = search_and_dedupe(query, limit=STRUCTURAL_HOP1_SEARCH_LIMIT, sources=STRUCTURAL_SEARCH_SOURCES)
        for p in papers:
            paper_cache.setdefault(_paper_ref(p), p)
        return papers

    def structural_hop2_search_fn(query: str) -> list[Paper]:
        papers = search_and_dedupe(query, limit=STRUCTURAL_HOP2_SEARCH_LIMIT, sources=STRUCTURAL_SEARCH_SOURCES)
        for p in papers:
            paper_cache.setdefault(_paper_ref(p), p)
        return papers

    return flat_search_fn, structural_hop1_search_fn, structural_hop2_search_fn


def make_need_semantic_search_fn(paper_cache: dict[str, Paper]):
    """P3: the SEMANTIC counterpart of flat_search_fn, for the `need_semantic` route (OpenAlex `search.semantic=`, capped at 50
    by the engine).  Same shared paper_cache as the other routes, so Stage 3 sees the full Paper."""
    def need_semantic_search_fn(query: str, limit: int) -> list[Paper]:
        papers = search_and_dedupe(query, limit=limit, mode="semantic")
        for p in papers:
            paper_cache.setdefault(_paper_ref(p), p.model_copy(update={"retrieval_modes": ["semantic"]}))
        return papers

    return need_semantic_search_fn


def papers_for_stage3(result: RouteRetrievalResult, paper_cache: dict[str, Paper]) -> list[Paper]:
    """Smallest adapter to the existing Stage 3 contract (filter_papers(list[Paper])).
    Provenance is NOT flattened away: result.papers (indexable by paper_ref)
    remains the sidecar; the caller reattaches it after Stage 3/4 by looking
    up each retained/extracted Paper's paper_ref in result.papers."""
    out = []
    missing = []
    for rp in result.papers:
        p = paper_cache.get(rp.paper_ref)
        if p is None:
            missing.append(rp.paper_ref)
            continue
        out.append(p)
    if missing:
        # Should not happen -- every RouteRetrievalResult.papers entry was
        # produced by a search_fn call this module itself wrapped. Surfaced,
        # not silently dropped, if it ever does.
        import sys as _sys
        print(f"route_retrieval_live_adapter: WARNING -- {len(missing)} paper_ref(s) in "
              f"RouteRetrievalResult had no cached Paper object: {missing}", file=_sys.stderr)
    return out


def provenance_index(result: RouteRetrievalResult) -> dict[str, "object"]:
    """paper_ref -> RetrievedPaper (route provenance + hop1_provenance),
    for reattaching provenance to a Paper after Stage 3/4 by paper_ref."""
    return {rp.paper_ref: rp for rp in result.papers}



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
    # Hard guarantee, independent of grounded_structural_feature above: this bridge never lets the structural branch turn on.
    # (Before Phase 3, plan_evidence_routes() also derived structural_terms lexically from system_description -- e.g.
    # "EspP passenger domain" -> ['EspP passenger'] -- and this line discarded them; that lexical extractor was removed because its
    # output was always discarded here. The line stays as the explicit safeguard.)
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


def retrieve_direct(plan: PlannerOutput, budgets: RouteRetrievalBudgets) -> tuple[EvidenceRoutePlan, list[Paper]]:
    """The direct-evidence stream (rule-based routes, no LLM): -> (evidence_plan, direct_papers).  Moved verbatim from the dispatch body of
    run_expert_prior_retrieval (consolidation 2b); each paper's `found_by_queries` / `query_categories` are written IN PLACE from the route provenance, as before."""
    # budgets are forwarded so the flat-query / need_semantic switches (RouteRetrievalBudgets) reach the direct-evidence routes too;
    # with budgets=None this is exactly the previous call (defaults are applied inside).
    evidence_plan, route_result, direct_paper_cache = run_retrieval_from_plan(plan, budgets=budgets)
    direct_papers = papers_for_stage3(route_result, direct_paper_cache)
    prov = provenance_index(route_result)
    for p in direct_papers:
        rp = prov.get(_paper_ref(p))
        if rp:
            p.found_by_queries = list(dict.fromkeys(e.query for e in rp.found_by))
            p.query_categories = list(dict.fromkeys(e.route for e in rp.found_by))
    return evidence_plan, direct_papers
