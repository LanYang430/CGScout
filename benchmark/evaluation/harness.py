"""Shared end-to-end runner + artifact saver for the 13-case regression suite
and the offline canonical-schema validation. Calls production code
UNMODIFIED (ScientificPlanner / ExpertPriorPlanner / cgscout.agent.execute)
and only records what it returns. NOT imported by beam_agent.

Route attribution is by SOURCE LIST, not by category label alone:
  in_flat      = target DOI present in result.direct_papers (flat/direct route)
  in_3a        = present in result.transfer_papers with 'transfer_direct' in its categories
  in_3b        = present in result.transfer_papers with 'structural' in its categories
                 (Hop-1/Hop-2 discovery)
A target counts as a 3B contribution only if in_3b is true; if it is ALSO
in_flat that is reported (`3b_and_flat`) rather than hidden.
"""
import os
import sys
import time


from cgscout.sources import openalex_usage as U  # noqa: E402  (budget / cache / ledger / latch)
from cgscout.sources import openalex as _OA  # noqa: E402
from cgscout import cache as _cache  # noqa: E402
from cgscout.planning.scientific_planner import PlannerOutput  # noqa: E402
from cgscout.agent import AgentConfig, execute as agent_execute, propose_prior, rank_for_display  # noqa: E402
from cgscout.assessment.stage4 import _paper_key as _s4_key  # noqa: E402
from cgscout.llm import read_usage_log  # noqa: E402


def harness_guard(estimated_searches: int, *, max_parallel: int = 1, check_budget: bool = True, what: str = "benchmark",
                  refresh: bool = False) -> dict:
    """Common start-up for every script that drives OpenAlex through the pipeline:
      * strict budget mode ON, so out-of-budget raises OpenAlexBudgetExhausted (which the
        pipeline's `except Exception` blocks cannot swallow) instead of becoming empty results
        that would be scored as retrieval misses;
      * at most `max_parallel` benchmark processes at once;
      * pre-flight: refuse to start if the remaining daily budget cannot cover the plan."""
    os.environ.setdefault("OPENALEX_STRICT_BUDGET", "1")
    if U.acquire_benchmark_slot(max_parallel) is None:
        sys.exit(f"[{what}] another benchmark process is already running (max_parallel={max_parallel}); "
                 f"refusing to start a second one against the same OpenAlex budget.")
    status = {"state": "skipped", "remaining_credits": None}
    if check_budget:
        if refresh:
            U.budget_status(force=True)   # live re-check, e.g. after adding funds; clears a stale exhaustion state
        status = U.require_budget(estimated_searches)
    rem = status.get("remaining_credits")
    print(f"[{what}] OpenAlex budget: {rem if rem is not None else 'remaining not stated by API (unverified)'} credits; "
          f"plan <= {estimated_searches} uncached searches (~{estimated_searches * U.CREDITS_PER_SEARCH_DEFAULT} credits); "
          f"already used today by this machine: {U.used_today_utc()}", flush=True)
    return status


def abort_message(e: BaseException) -> str:
    reset = getattr(e, "reset_s", None)
    return ("\n" + "*" * 100 + "\n*** BENCHMARK STOPPED: OpenAlex daily budget exhausted.\n"
            f"*** Resets at midnight UTC (~{reset}s from the response). Nothing more was attempted.\n"
            "*** The interrupted run is NOT recorded as a retrieval miss and is excluded from recall counts;\n"
            "*** completed runs before it are valid. Re-run after the reset (cached queries cost nothing).\n" + "*" * 100)


def run_guarded(fn, *args, **kw):
    """Run a benchmark entry point; budget problems become ONE clear message and exit code 3."""
    try:
        return fn(*args, **kw)
    except U.OpenAlexBudgetInsufficient as e:
        print(f"\n*** NOT STARTED: {e}", flush=True)
        sys.exit(3)
    except U.OpenAlexBudgetExhausted as e:
        print(abort_message(e), flush=True)
        sys.exit(3)


# (mode, limit) combinations the pipeline actually issues: 3A semantic at max_papers_per_query (20 or 50),
# flat/Hop-2 lexical, Hop-1 semantic at 15.
_QUERY_COMBOS = [("semantic", 20), ("semantic", 50), ("semantic", 15), ("lexical", 20), ("lexical", 15)]


def cached_target_ranks(query: str, dois: list) -> list:
    """Where the target ranks in the result list of ONE already-executed query. CACHE-ONLY: this never
    makes a network request. Looks in the request-level cache and in search_and_dedupe's cache (older
    entries). Returns one entry per (mode, limit) that is cached; an empty list means 'not cached', which
    is different from rank None (= cached, target absent from that query's results)."""
    out = []
    for mode, limit in _QUERY_COMBOS:
        per_page = min(limit, 200 if mode == "lexical" else 50)
        got = U.request_cache_get(U.request_cache_key(query, limit=per_page, mode=mode, select=_OA._SELECT))
        if got is None:
            got = _cache.get("search_and_dedupe", _cache.content_hash(query, str(limit), "openalex", mode), ttl_seconds=24 * 3600)
        if got is None:
            continue
        rank = next((i for i, d in enumerate(got, 1) if any(x.lower() in (d.get("doi") or "").lower() for x in dois)), None)
        out.append({"mode": mode, "limit": limit, "n_results": len(got), "rank": rank})
    return out


def _match(paper, dois):
    d = (getattr(paper, "doi", None) or "").lower()
    return any(x.lower() in d for x in dois)


def make_plan(plan_fields: dict) -> PlannerOutput:
    return PlannerOutput(system=plan_fields["system"], target_component=plan_fields.get("target_component"),
                         scientific_objective=plan_fields["scientific_objective"], clarification_needed=False,
                         search_context=plan_fields.get("search_context") or [], search_axes=plan_fields.get("search_axes") or [])


def propose(backend, plan_fields: dict, case_id: str):
    return propose_prior(make_plan(plan_fields), backend=backend, case_id=case_id)


def run_plan(backend, plan_fields: dict, case_id: str, targets: dict, top_n: int = 15,
             expert_prior=None, budgets=None, user_request=None, stage4: bool = False, stage4_config=None, partial: dict = None) -> tuple:
    """-> (rec, agent_result): the record dict, and the cgscout.agent.AgentResult of the run (its .retrieval / .assessment are the retrieval and Stage-4 results).
    targets: {label: [doi, ...]} (list = acceptable alternates).
    expert_prior: reuse a previously proposed ExpertPrior (isolates planner-input effects
    from downstream effects). budgets: optional RouteRetrievalBudgets override passed
    to the retrieval through the agent configuration (no production edit).
    partial: optional dict that the caller keeps; it is emptied at the start and then filled with what had been completed when an exception ends the
    run: "retrieval" (the retrieval result, as soon as the gate has seen it) and "agent_result" (once agent_execute returned). Never read here; the
    normal path is unchanged."""
    if partial is not None:
        partial.clear()
    plan = make_plan(plan_fields)
    ep = expert_prior or propose(backend, plan_fields, case_id)
    U.reset_run(case_id)
    U.set_run_label(case_id)          # every real HTTP request / cache hit below is attributed to this run
    health = {}

    def gate(res):
        """The evaluation's gate between retrieval and Stage 4: budget exhaustion raises, a degraded run is not assessed."""
        if partial is not None:
            partial["retrieval"] = res
        health["api"] = U.run_usage(case_id)
        ex = U.exhausted()
        if ex:   # budget ran out but the pipeline swallowed it (strict mode off): this run must never be scored
            raise U.OpenAlexBudgetExhausted(U.exhausted_message(ex), headers=ex.get("headers"), reset_s=ex.get("reset_s"))
        health["axis_errors"] = [str(l.error)[:120] for l in res.axis_logs if l.error]
        # Any failed HTTP request (429 / network) means some searches returned nothing for reasons unrelated
        # to retrieval quality: the run is degraded and excluded from recall counts (never a "miss").
        health["degraded"] = bool(health["api"]["failed_requests"] or health["axis_errors"])
        return bool(stage4 and not health["degraded"])      # a degraded run is discarded/retried; do not spend LLM calls on it

    # ---- retrieval + Stage 3, the gate, and (if it says yes) Stage 4: the LLM relevance evaluation of the Stage-3-retained papers ----
    agent_result = agent_execute(plan, ep, backend=backend, config=AgentConfig(budgets=budgets, stage4=stage4_config), case_id=case_id,
                                 user_request=user_request, gate=gate)
    if partial is not None:
        partial["agent_result"] = agent_result
    res, s4 = agent_result.retrieval, agent_result.assessment
    api, axis_errors, degraded = health["api"], health["axis_errors"], health["degraded"]
    s4_usage = None
    if s4 is not None:
        t_s4 = agent_result.assess_started_at
        rows = [r for r in read_usage_log() if r.get("ts", 0) >= t_s4 and str(r.get("label", "")).startswith("stage4")]
        s4_usage = {"llm_calls": len(rows), "input_tokens": sum(r.get("input_tokens", 0) for r in rows),
                    "output_tokens": sum(r.get("output_tokens", 0) for r in rows),
                    "cost_usd": round(sum(r.get("estimated_cost_usd", 0.0) for r in rows), 4)}

    ranked = rank_for_display(res.retained)
    verdicts = {v.title: v for v in res.stage3_verdicts}
    tinfo = {}
    for label, dois in targets.items():
        dois = [dois] if isinstance(dois, str) else dois
        flat = [p for p in res.direct_papers if _match(p, dois)]
        trans = [p for p in res.transfer_papers if _match(p, dois)]
        in_3a = any("transfer_direct" in (p.query_categories or []) for p in trans)
        in_3b = any("structural" in (p.query_categories or []) for p in trans)
        any_hit = (flat + trans)[0] if (flat or trans) else None
        v = verdicts.get(any_hit.title) if any_hit else None
        rank = next((i for i, p in enumerate(ranked, 1) if _match(p, dois)), None)
        kept_paper = next((p for p in res.retained if _match(p, dois)), None)
        st4, outcome = None, None
        if any_hit is None:
            outcome = "retrieval_miss"
        elif kept_paper is None:
            outcome = "stage3_drop"
        elif s4 is None:
            outcome = "retained_stage4_not_run"
        else:
            a = s4.by_key().get(_s4_key(kept_paper))
            if a is None or a.status == "error":
                outcome = "stage4_error"
            else:
                outcome = ("recommended" if a.recommended else
                           "stage4_missing_evidence" if a.status == "missing_evidence" else
                           "stage4_not_deep_evaluated_cap" if a.status == "not_evaluated_cap" else
                           "stage4_background_only" if a.final_grade == 1 else "stage4_rejected")
                st4 = {"final_grade": a.final_grade, "status": a.status, "rank": a.rank,
                       "n_ranked": sum(1 for x in s4.assessments if x.final_grade is not None),
                       "tier1_decision": a.screen.decision if a.screen else None,
                       "tier1_confidence": a.screen.confidence if a.screen else None,
                       "tier1_reason": a.screen.reason if a.screen else None,
                       "tier2_evaluated": a.judgment is not None, "abstract_source": a.abstract_source,
                       "deep_grade": a.judgment.grade if a.judgment else None,
                       "evidence_type": a.judgment.evidence_type if a.judgment else None,
                       "supports_applicability": a.judgment.supports_applicability if a.judgment else None,
                       "computational": (a.judgment.computational if a.judgment else (a.screen.computational if a.screen else None)),
                       "explanation": a.judgment.explanation if a.judgment else (a.screen.reason if a.screen else None),
                       "quote_grounded": a.quote_grounded, "flags": a.flags}
        tinfo[label] = {
            "retained": kept_paper is not None, "outcome": outcome, "stage4": st4,
            "in_pool": any_hit is not None, "in_flat": bool(flat), "in_3a": in_3a, "in_3b": in_3b,
            "3b_and_flat": bool(in_3b and flat),
            "stage3_keep": (v.keep if v else None),
            "classifier_score": (v.classifier_score if v else None),
            "final_rank": rank,
            "categories": sorted({c for p in flat + trans for c in (p.query_categories or [])}),
            "found_by_queries": sorted({q for p in flat + trans for q in (p.found_by_queries or [])}),
        }
    flat_queries = sorted({q for p in res.direct_papers for q in (p.found_by_queries or [])})
    all_dois = {l: ([d] if isinstance(d, str) else d) for l, d in targets.items()}
    qlist = [("flat", q) for q in flat_queries]
    for l in res.axis_logs:
        qlist += [("3A", q) for q in l.direct_queries] + [("hop1", q) for q in l.hop1_queries] + [("hop2", q) for q in l.hop2_queries]
    query_ranks = {label: [{"route": r, "query": q, "cached": cached_target_ranks(q, dois)} for r, q in qlist]
                   for label, dois in all_dois.items()}
    rec = {
        "case_id": case_id, "input": plan_fields,
        "expert_prior": {
            "transfer_needed": ep.transfer_needed,
            "axes": [{"basis": a.basis, "execution_mode": a.execution_mode, "confidence": a.confidence,
                      "scientific_class": a.scientific_class} for a in ep.transfer_axes],
        },
        "queries": {
            "flat_route": flat_queries,
            "axes": [{"basis": l.axis.basis, "mode": l.axis.execution_mode, "executed": l.executed,
                      "skip_reason": l.skip_reason, "direct_queries": l.direct_queries,
                      "hop1_queries": l.hop1_queries, "hop2_queries": l.hop2_queries,
                      "related_systems": [getattr(i, "label", str(i)) for i in l.related_system_identities],
                      "n_papers": l.n_papers_found, "error": l.error} for l in res.axis_logs],
        },
        "query_target_ranks": query_ranks,
        "stage4": (None if s4 is None else {
            "workload": s4.workload, "llm_usage": s4_usage,
            "assessments": [{"rank": a.rank, "title": a.title, "year": a.year, "doi": a.doi, "routes": a.routes,
                             "status": a.status, "tier1_decision": a.screen.decision if a.screen else None,
                             "tier1_confidence": a.screen.confidence if a.screen else None,
                             "abstract_source": a.abstract_source, "final_grade": a.final_grade,
                             "evidence_type": a.judgment.evidence_type if a.judgment else None,
                             "explanation": (a.judgment.explanation if a.judgment else (a.screen.reason if a.screen else a.error)),
                             "flags": a.flags} for a in s4.assessments]}),
        "api_usage": api, "axis_errors": axis_errors, "degraded": degraded,
        "pool": len(res.direct_papers) + len(res.transfer_papers),
        "retained": len(res.retained), "rejected": len(res.rejected),
        "targets": tinfo,
        "top_final": [{"rank": i, "title": p.title, "doi": getattr(p, "doi", None)} for i, p in enumerate(ranked[:top_n], 1)],
    }
    return rec, agent_result


def results_of(agent_result, partial: dict) -> tuple:
    """-> (retrieval result, Stage-4 result) of a run, for a record: taken from the AgentResult when run_plan returned one, otherwise from what `partial`
    (the dict given to run_plan) caught before an exception -- None for whatever had not completed."""
    done = agent_result if agent_result is not None else partial.get("agent_result")
    if done is not None:
        return done.retrieval, done.assessment
    return partial.get("retrieval"), None


def run_plan_checked(backend, plan_fields: dict, case_id: str, targets: dict, max_tries: int = 3,
                     wait_s: int = 30, expert_prior=None, budgets=None, **kw) -> tuple:
    """-> (rec, agent_result) of the LAST attempt.  run_plan, retried (same ExpertPrior, so only the network conditions change) while the run is
    degraded by FAILED OpenAlex requests (e.g. transient network errors). Budget exhaustion is never
    retried: OpenAlexBudgetExhausted propagates to the caller. A run still degraded after max_tries
    is returned flagged and must be excluded from recall counts."""
    ep = expert_prior or propose(backend, plan_fields, case_id)
    total = {"http_requests": 0, "cache_hits": 0, "credits_charged": 0}
    rec, agent_result = None, None
    for attempt in range(1, max_tries + 1):
        rec, agent_result = run_plan(backend, plan_fields, case_id, targets, expert_prior=ep, budgets=budgets, **kw)
        for k in total:
            total[k] += rec["api_usage"][k]
        rec["attempts"] = attempt
        rec["api_usage_all_attempts"] = dict(total)
        if not rec["degraded"]:
            return rec, agent_result
        print(f"  [degraded run: {rec['api_usage']['failed_requests']} failed OpenAlex requests, "
              f"axis_errors={len(rec['axis_errors'])}] attempt {attempt}/{max_tries}", flush=True)
        if attempt < max_tries:
            time.sleep(wait_s)
    return rec, agent_result
