"""Paper evaluation run of CGScout, full pipeline (ScientificPlanner + ExpertPrior replayed from plans_frozen/ -> retrieval -> Stage 3 -> Stage 4 v2).
Port of the held-out (NEW37) runner (same flow, same checkpoint / retry / budget rules); the run's data directory is a parameter, nothing is hard-coded.
usage:  CGSCOUT_BENCH_DIR=<run dir> python run_benchmark.py CUR [--pids a,b] [--expect-rows N]
Run directory contents:  queries.csv (Paper ID, Latest query (v2), DOI, OpenAlex ID), plans_frozen/ (freeze_plans.py), cache_work/ (created), results/CUR/ (created).
Pre-specified rules (fixed before the run; nothing is changed after seeing results):
  INPUT     only the frozen query text reaches CGScout; seed title/DOI/OpenAlex ID are used ONLY after retrieval to locate the seed.
  CONFIG    model gpt-5.6-luna, effort medium in every stage (harness wrapper lifts explicit 'low' to 'medium', counted); deep_cap 500; switches explicit (arms.py).
  CLARIFY   one fixed generic reply, max 2 rounds (done in freeze_plans.py); if the planner still does not proceed the query is NOT_RUN_CLARIFICATION.
  RETRY     only infrastructure failures (degraded OpenAlex run, transient errors), max 3 passes; LLM 4xx refusals are recorded REFUSED, never retried.
  OPENALEX  strict budget mode; sleeps to 00:00 UTC when the daily credits run out; an interrupted query is discarded and re-run.
  COST      LLM cost from the usage log per query window; hard stop when the batch total reaches CGSCOUT_STOP_USD (default 5.0; legacy name NEW37_STOP_USD also read).
  CACHE     fresh cache_work/ inside the run directory.   CHECKPOINT  results/CUR/{PID}.json = finished query (never re-run).
"""
from __future__ import annotations

import csv, hashlib, json, os, subprocess, sys, time, traceback
from pathlib import Path

import arms as AR
from recorder import Recorder

# --- run configuration (set by configure(); module-level so that run_query() and the offline verification can reuse it) ----------------------------------
ARM_NAME = None
ROOT = RES = INPUT = USAGE_LOG = LOG = None
BUDGETS = None
STOP_USD = DAILY_CREDITS = MIN_CREDITS = DEEP_CAP = None
CLARIFY_REPLY = "I have no further details beyond my question as written; please proceed with your best reasonable interpretation."
# `harness` (and with it cgscout.llm, which reads CGSCOUT_USAGE_LOG_PATH at import time) is imported inside configure(), AFTER the environment is set.
RL = U = None
RP = TB = TerminologyExpansion = get_backend = backend_label = read_usage_log = Stage4Config = None


def configure(arm_name):
    """Environment + imports that must be in place before any pipeline module reads them (cache dir, usage log, model, effort)."""
    global ARM_NAME, ROOT, RES, INPUT, USAGE_LOG, LOG, BUDGETS, STOP_USD, DAILY_CREDITS, MIN_CREDITS, DEEP_CAP
    global RL, U, RP, TB, TerminologyExpansion, get_backend, backend_label, read_usage_log, Stage4Config
    ARM_NAME, ROOT, INPUT = arm_name, AR.ROOT, AR.INPUT
    os.environ["CGSCOUT_CACHE_DIR"] = str(AR.SHARED_CACHE)           # shared by every arm: identical queries are fetched once
    os.environ["CGSCOUT_USAGE_LOG_PATH"] = str(ROOT / f"llm_usage_full_{ARM_NAME}.jsonl")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["CGSCOUT_LLM_MODEL"] = "gpt-5.6-luna"
    os.environ["CGSCOUT_LLM_EFFORT"] = "medium"
    os.environ.setdefault("OPENALEX_STRICT_BUDGET", "1")
    import harness as _RL
    import cgscout.retrieval.direct.route_plan as _RP
    import cgscout.retrieval.orchestrator as _TB
    from cgscout.retrieval.related_systems.terminology import TerminologyExpansion as _TE
    from cgscout.llm import get_backend as _gb, backend_label as _bl, read_usage_log as _rul
    from cgscout.assessment.stage4 import Stage4Config as _S4C
    RL, U = _RL, _RL.U
    RP, TB, TerminologyExpansion, get_backend, backend_label, read_usage_log, Stage4Config = _RP, _TB, _TE, _gb, _bl, _rul, _S4C
    BUDGETS = AR.budgets_for(ARM_NAME, RP)
    # Hop-1's terminology expansion is an UNCACHED live LLM call whose output only feeds the banned-name list of Hop-1 discovery.  To keep Hop-1/Hop-2
    # identical across arms it is memoised per scientific_class in terminology_memo.json: the first arm that needs it records it, the others replay it.
    memo_path = ROOT / "terminology_memo.json"
    orig_expand = TB.expand_terminology

    def _memo_expand(system_term, *, call):
        memo = json.loads(memo_path.read_text()) if memo_path.exists() else {}
        if system_term in memo:
            return TerminologyExpansion.model_validate(memo[system_term])
        r = orig_expand(system_term, call=call)
        memo[system_term] = r.model_dump()
        tmp = memo_path.with_suffix(".tmp"); tmp.write_text(json.dumps(memo, ensure_ascii=False, indent=1)); tmp.replace(memo_path)
        return r

    TB.expand_terminology = _memo_expand
    RES = ROOT / "results" / ARM_NAME
    STOP_USD = float(os.environ.get("CGSCOUT_STOP_USD", os.environ.get("NEW37_STOP_USD", "5.0")))     # hard LLM-spend ceiling per arm (approved value is set at launch)
    DAILY_CREDITS = 10000
    MIN_CREDITS = int(os.environ.get("CGSCOUT_MIN_CREDITS", os.environ.get("NEW37_MIN_CREDITS", "400")))            # do not START a query with less than this left (a query uses ~150)
    DEEP_CAP = 500
    USAGE_LOG = Path(os.environ["CGSCOUT_USAGE_LOG_PATH"])
    LOG = ROOT / f"run_log_full_{ARM_NAME}.txt"


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


class EffortUnifiedBackend:
    """Harness wrapper: lifts every explicit effort='low' request to 'medium' (config unification requested by the user)."""
    def __init__(self, inner):
        self._inner = inner; self.overrides = 0; self.calls = 0
    def structured(self, *, effort=None, **kw):
        self.calls += 1
        if effort == "low":
            effort = "medium"; self.overrides += 1
        return self._inner.structured(effort=effort, **kw)
    def __getattr__(self, k):
        return getattr(self._inner, k)


def norm_doi(d):
    d = (d or "").strip().lower()
    for p in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "http://dx.doi.org/", "doi:"):
        if d.startswith(p): d = d[len(p):]
    return d.rstrip(".,;")


def is_seed(p, seed_doi, seed_oa):
    d = norm_doi(getattr(p, "doi", None))
    oa = (getattr(getattr(p, "external_ids", None), "openalex_id", None) or "").split("/")[-1]
    return bool((d and d == seed_doi) or (oa and oa == seed_oa))


def pkey(p):
    return norm_doi(getattr(p, "doi", None)) or (getattr(p, "title", "") or "").lower()


def llm_window(t0, t1):
    rows = [r for r in read_usage_log() if t0 - 0.5 <= r.get("ts", 0) <= t1 + 0.5]
    by = {}
    for r in rows:
        k = str(r.get("label", ""))
        b = by.setdefault(k, {"calls": 0, "input": 0, "cached": 0, "output": 0, "cost": 0.0})
        b["calls"] += 1; b["input"] += r.get("input_tokens", 0); b["cached"] += r.get("cached_input_tokens", 0)
        b["output"] += r.get("output_tokens", 0); b["cost"] += r.get("estimated_cost_usd", 0.0)
    tot = {"calls": len(rows), "input_tokens": sum(r.get("input_tokens", 0) for r in rows), "cached_input_tokens": sum(r.get("cached_input_tokens", 0) for r in rows),
           "output_tokens": sum(r.get("output_tokens", 0) for r in rows), "cost_usd": round(sum(r.get("estimated_cost_usd", 0.0) for r in rows), 6)}
    return tot, {k: {**v, "cost": round(v["cost"], 6)} for k, v in by.items()}


def total_spent():
    return sum(json.loads(p.read_text()).get("llm", {}).get("cost_usd", 0) for p in RES.glob("*.json"))


def sleep_until(ts, why):
    log(f"PAUSE ({why}); resuming at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))}")
    while time.time() < ts:
        time.sleep(min(300, max(5, ts - time.time())))
    U.clear_exhausted()
    try:
        U.budget_status(force=True)
    except Exception as e:  # noqa: BLE001
        log(f"budget re-check failed: {e!r}")
    log("RESUME")


def credits_since(ts):
    n = 0
    try:
        for raw in open(U.ledger_path()):
            try:
                e = json.loads(raw)
            except ValueError:
                continue
            if e.get("ts", 0) >= ts and e.get("status") == 200:
                n += int(e.get("credits_charged") or 0)
    except FileNotFoundError:
        pass
    return n


def next_utc_midnight():
    now = time.time(); return now - (now % 86400) + 86400 + 60


def wait_for_budget():
    while True:
        ex = U.exhausted()
        if ex:
            sleep_until(max(ex.get("reset_at", 0), time.time() + 60) + 30, "OpenAlex budget exhausted"); continue
        try:
            b = U.budget_status()
        except Exception as e:  # noqa: BLE001
            log(f"budget check error {e!r}; waiting 120s"); time.sleep(120); continue
        if b.get("state") == "exhausted":
            sleep_until(next_utc_midnight(), "OpenAlex budget exhausted (live check)"); continue
        rem = b.get("remaining_credits")
        used = credits_since(max(time.time() - (time.time() % 86400), float(os.environ.get("LEDGER_COUNT_FROM_TS", 0))))          # this run's own ledger (fresh cache dir); the API does not state remaining credits
        if rem is None and used > DAILY_CREDITS - MIN_CREDITS:
            sleep_until(next_utc_midnight(), f"{used} credits used today by this run (daily free quota {DAILY_CREDITS})"); continue
        if rem is not None and rem < MIN_CREDITS:
            sleep_until(next_utc_midnight(), f"only {rem} OpenAlex credits left (< {MIN_CREDITS})"); continue
        return rem


def run_query(row, backend_w, s4cfg):
    pid, query = row["Paper ID"], row["Latest query (v2)"]
    seed_doi, seed_oa = norm_doi(row["DOI"]), row["OpenAlex ID"]
    run_label = f"cgscout_{pid}"
    t0 = time.time()
    rec = {"pid": pid, "query": query, "ts_start": time.strftime("%Y-%m-%d %H:%M:%S"), "clarifications": [], "code": f"cgscout_{ARM_NAME}", "arm": ARM_NAME}
    stash = {}
    orig_ret, orig_s4 = RL.run_expert_prior_retrieval, RL.run_stage4_v2
    def ret_w(*a, **k):
        r = orig_ret(*a, **k); stash["res"] = r; return r
    def s4_w(*a, **k):
        r = orig_s4(*a, **k); stash["s4"] = r; return r
    RL.run_expert_prior_retrieval, RL.run_stage4_v2 = ret_w, s4_w
    try:
        fp = AR.frozen_plan(pid)                            # PHASE 0 output: one ScientificPlanner + one ExpertPrior generation per case, identical in every arm
        if fp.get("status") != "READY":
            rec["status"] = "NOT_RUN_CLARIFICATION"; rec["planner"] = {"question": fp.get("last_question")}
            raise StopIteration                              # handled below: recorded, not an error
        rec["plans_frozen_from"] = f"{AR.PLANS.name}/{pid}.json (created {fp['created']})"
        if True:
            rec["planner"] = dict(fp["planner"])
            eff = {k: rec["planner"][k] for k in ("system", "target_component", "scientific_objective", "search_context", "search_axes")}
            ep_fixed = AR.fixed_expert_prior(pid)
            # exact record of every search sent (or answered from a cache) and the route that asked for it
            from cgscout.retrieval.direct.from_planner import evidence_plan_from_planner_output
            from cgscout.planning.scientific_planner import PlannerOutput
            _po = PlannerOutput(**{k: fp["planner"].get(k) for k in ("system", "target_component", "scientific_objective", "search_context", "search_axes")}, clarification_needed=False)
            _plan = RP.build_route_retrieval_plan(evidence_plan_from_planner_output(_po), BUDGETS) if BUDGETS else RP.build_route_retrieval_plan(evidence_plan_from_planner_output(_po))
            _roles = {}
            for _q in _plan.route_queries:
                _roles.setdefault(_q.query, []).append([_q.route, _q.query_role])
            REC = Recorder(lambda: U.run_usage(run_label), lambda p_: is_seed(p_, seed_doi, seed_oa), _roles).install()
            _orig_run_plan = RL.run_plan
            def _run_plan_clear(*a, **k):                    # a degraded run is retried: keep only the last attempt's requests
                REC.records.clear(); return _orig_run_plan(*a, **k)
            RL.run_plan = _run_plan_clear
            try:
                run = RL.run_plan_checked(backend_w, eff, run_label, {"seed": [seed_doi]}, user_request=fp.get("user_request", query), stage4=True, stage4_config=s4cfg,
                                          expert_prior=ep_fixed, budgets=BUDGETS)
            finally:
                RL.run_plan = _orig_run_plan; REC.uninstall()
                rec["requests"] = REC.records
                rec["request_summary"] = {k: {"total": sum(1 for r in REC.records if r["kind"] == k), "http": sum(1 for r in REC.records if r["kind"] == k and r["served_from"] == "http"),
                                              "seed_found_by": [r["n"] for r in REC.records if r["kind"] == k and r["seed_rank"]]} for k in sorted({r["kind"] for r in REC.records})}
            u = dict(run["api_usage"]); u.update(run["api_usage_all_attempts"])
            rec["api_usage"] = u
            rec["degraded"] = bool(run.get("degraded")); rec["attempts"] = run.get("attempts")
            rec["status"] = "DEGRADED" if run.get("degraded") else "OK"
            res, s4 = stash.get("res"), stash.get("s4")
            rec["expert_prior"] = run["expert_prior"]; rec["queries"] = run["queries"]
            n_q = {"flat": len(run["queries"]["flat_route"]),
                   "3A": sum(len(a["direct_queries"]) for a in run["queries"]["axes"]),
                   "hop1": sum(len(a["hop1_queries"]) for a in run["queries"]["axes"]),
                   "hop2": sum(len(a["hop2_queries"]) for a in run["queries"]["axes"])}
            rec["n_openalex_query_strings"] = n_q
            if res is not None:
                direct, trans = list(res.direct_papers), list(res.transfer_papers)
                verdict = {pkey_v: v for v in res.stage3_verdicts for pkey_v in [(v.title or "").lower()]}
                kept = {pkey(p) for p in res.retained}
                pool = {}
                for src, lst in (("flat", direct), ("transfer", trans)):
                    for p in lst:
                        k = pkey(p); e = pool.setdefault(k, {"title": p.title, "doi": getattr(p, "doi", None), "year": p.year, "routes": set(), "categories": set()})
                        e["routes"].add(src); e["categories"].update(p.query_categories or [])
                for k, e in pool.items():
                    e["routes"] = sorted(e["routes"]); e["categories"] = sorted(e["categories"]); e["stage3_kept"] = k in kept
                rec["pool"] = list(pool.values())
                seed_in_pool = [p for p in direct + trans if is_seed(p, seed_doi, seed_oa)]
                sp = seed_in_pool[0] if seed_in_pool else None
                v = verdict.get((sp.title or "").lower()) if sp else None
                rec["counts"] = {"pool_unique": len(pool), "stage3_retained": len(res.retained), "stage3_rejected": len(res.rejected),
                                 "flat_papers": len(direct), "transfer_papers": len(trans),
                                 "need_semantic_papers": sum(1 for p in direct if "need_semantic" in (p.query_categories or [])),
                                 "abstract_recovered_in_stage3": sum(1 for v_ in res.stage3_verdicts if getattr(v_, "abstract_recovered", False))}
                seed = {"in_pool": sp is not None, "routes": sorted({"flat" if p in direct else "transfer" for p in seed_in_pool}),
                        "categories": sorted({c for p in seed_in_pool for c in (p.query_categories or [])}),
                        "stage3_kept": (any(is_seed(p, seed_doi, seed_oa) for p in res.retained) if sp else None),
                        "stage3_reason": (getattr(v, "reason", None) if v else None), "classifier_score": (getattr(v, "classifier_score", None) if v else None), "abstract_recovered": (getattr(v, "abstract_recovered", None) if v else None)}
            else:
                seed = {"in_pool": None}
            if s4 is not None:
                A = s4.assessments
                rec["workload"] = s4.workload
                rec["counts"]["stage4_evaluated_tier1"] = s4.workload.get("n_screened")
                rec["counts"]["stage4_tier2_deep"] = s4.workload.get("n_deep_evaluated")
                rec["counts"]["stage4_graded"] = sum(1 for a in A if a.final_grade is not None)
                rec["counts"]["stage4_recommended_ge2"] = sum(1 for a in A if (a.final_grade or 0) >= 2)
                rec["ranked_list"] = [{"rank": a.rank, "title": a.title, "doi": a.doi, "year": a.year, "routes": a.routes, "status": a.status, "final_grade": a.final_grade,
                                       "tier1": (a.screen.decision if a.screen else None), "tier1_conf": (a.screen.confidence if a.screen else None),
                                       "evidence_type": (a.judgment.evidence_type if a.judgment else None),
                                       "explanation": (a.judgment.explanation if a.judgment else (a.screen.reason if a.screen else a.error)), "flags": a.flags,
                                       "is_seed": bool(norm_doi(a.doi) == seed_doi)} for a in A]
                sa = next((a for a in A if norm_doi(a.doi) == seed_doi), None)
                if sa is None and sp is not None:   # DOI missing on the assessment: fall back to the paper key
                    sa = next((a for a in A if a.key == next((pkey(p) for p in seed_in_pool), None)), None)
                seed.update({"stage4_status": sa.status if sa else None, "stage4_tier1": (sa.screen.decision if sa and sa.screen else None),
                             "final_grade": sa.final_grade if sa else None, "rank": sa.rank if sa else None,
                             "n_ranked": rec["counts"]["stage4_graded"]})
            rec["seed"] = seed
            rk = seed.get("rank")
            rec["hit10"] = int(rk is not None and rk <= 10); rec["hit20"] = int(rk is not None and rk <= 20)
            rec["deep_cap_binding"] = bool((rec.get("workload") or {}).get("deep_cap_binding"))
    except StopIteration:
        pass
    except U.OpenAlexBudgetExhausted:
        raise
    except Exception as e:  # noqa: BLE001
        code = getattr(e, "status_code", None)
        rec["status"] = "REFUSED" if (code is not None and 400 <= code < 500) else "ERROR"
        rec["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        if rec["status"] == "ERROR":
            rec["traceback"] = traceback.format_exc()[-1500:]
    finally:
        RL.run_expert_prior_retrieval, RL.run_stage4_v2 = orig_ret, orig_s4
    t1 = time.time()
    rec["llm"], rec["llm_by_label"] = llm_window(t0, t1)
    rec["seconds"] = round(t1 - t0, 1); rec["ts_end"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return rec


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).resolve().parent, check=True).stdout.strip()
    except Exception:   # noqa: BLE001 -- not a checkout / git missing: provenance is simply absent
        return None


def main():
    arm = sys.argv[1]
    configure(arm)
    rows = list(csv.DictReader(open(INPUT, encoding="utf-8-sig")))
    expect = int(sys.argv[sys.argv.index("--expect-rows") + 1]) if "--expect-rows" in sys.argv else None
    assert len({r["Paper ID"] for r in rows}) == len(rows) and (expect is None or len(rows) == expect), (len(rows), expect)
    hits_, misses_ = AR.case_ids(); only = set(hits_ + misses_)
    if "--pids" in sys.argv: only &= set(sys.argv[sys.argv.index("--pids") + 1].split(","))
    RES.mkdir(parents=True, exist_ok=True)
    RL.harness_guard(len(rows) * 16, what="cgscout_benchmark", check_budget=False)
    backend = get_backend(); bw = EffortUnifiedBackend(backend)
    s4cfg = Stage4Config(screen_effort="medium", extract_effort="medium", judge_effort="medium", deep_cap=DEEP_CAP, screen_abstract_chars=AR.ARMS[ARM_NAME]["screen_chars"])
    import cgscout
    cfg = {"model": backend.model, "default_effort": backend.default_effort, "stage4_config": {k: getattr(s4cfg, k) for k in ("screen_effort", "extract_effort", "judge_effort", "deep_cap", "screen_batch_size", "reject_min_confidence", "workers")},
           "changes_vs_production_defaults": {"stage4.screen_effort": "low->medium", "stage4.extract_effort": "low->medium", "hop2_search_anchor effort": "low->medium (harness backend wrapper)",
                                              "stage4.deep_cap": "80->500 (safety valve disabled in practice)", "CGSCOUT_CACHE_DIR": "fresh cache_work/ for this run"},
           "clarify_reply": CLARIFY_REPLY, "stop_usd": STOP_USD, "input_sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
           "cgscout_version": cgscout.__version__, "git_commit": _git_commit()}
    cfg.update({"arm": ARM_NAME, "arm_label": AR.ARMS[ARM_NAME]["label"], "budgets": AR.ARMS[ARM_NAME]["budgets"], "classifier": AR.ARMS[ARM_NAME].get("classifier")})
    (ROOT / f"run_config_{ARM_NAME}.json").write_text(json.dumps(cfg, indent=1))
    log(f"LLM backend: {backend_label(backend)}; efforts unified to medium; deep_cap={DEEP_CAP}; stop at ${STOP_USD}")
    for pass_no in range(1, 4):
        todo = [r for r in rows if (not only or r["Paper ID"] in only)
                and not ((RES / f"{r['Paper ID']}.json").exists() and json.loads((RES / f"{r['Paper ID']}.json").read_text())["status"] in ("OK", "REFUSED", "NOT_RUN_CLARIFICATION"))]
        if not todo:
            break
        log(f"=== pass {pass_no}: {len(todo)} queries to run ===")
        for r in todo:
            pid = r["Paper ID"]
            if total_spent() >= STOP_USD:
                log(f"STOP: spend ${total_spent():.2f} >= ${STOP_USD}; no new query started"); return
            while True:
                rem = wait_for_budget()
                try:
                    rec = run_query(r, bw, s4cfg)
                    break
                except U.OpenAlexBudgetExhausted as e:
                    log(f"{pid}: OpenAlex budget exhausted mid-query ({str(e)[:100]}); attempt discarded, will re-run after the reset")
                    ex = U.exhausted()
                    sleep_until(max((ex or {}).get("reset_at", 0), time.time()) + 90, "mid-query exhaustion")
            rec["pass"] = pass_no; rec["effort_overrides_total_so_far"] = bw.overrides
            tmp = (RES / f"{pid}.tmp"); tmp.write_text(json.dumps(rec, ensure_ascii=False, default=str)); tmp.replace(RES / f"{pid}.json")
            s = rec.get("seed", {}) or {}
            log(f"{pid} {rec['status']} {rec['seconds']}s ${rec['llm']['cost_usd']:.3f} oa_req={(rec.get('api_usage') or {}).get('http_requests')} "
                f"pool={(rec.get('counts') or {}).get('pool_unique')} kept={(rec.get('counts') or {}).get('stage3_retained')} seed(in_pool={s.get('in_pool')},kept={s.get('stage3_kept')},rank={s.get('rank')}) "
                f"| total ${total_spent():.2f}")
    log("DONE")


if __name__ == "__main__":
    RL.run_guarded(main)
