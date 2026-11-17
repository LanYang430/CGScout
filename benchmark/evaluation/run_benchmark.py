"""Paper evaluation run of CGScout, full pipeline (ScientificPlanner + ExpertPrior replayed from plans_frozen/ -> retrieval -> Stage 3 -> Stage 4 v2).
Port of the held-out (NEW37) runner (same flow, same checkpoint / retry / budget rules); the run's data directory is a parameter, nothing is hard-coded.
usage:  CGSCOUT_BENCH_DIR=<run dir> CGSCOUT_STOP_USD=<usd> python run_benchmark.py CUR --expect-rows N          (full run: every row of queries.csv)
        CGSCOUT_BENCH_DIR=<run dir> python run_benchmark.py CUR --pids a,b [--expect-rows N]                       (subset, e.g. a smoke test)
Run directory contents:  queries.csv (Paper ID, Latest query (v2), DOI, OpenAlex ID), plans_frozen/ (freeze_plans.py), cache_work/ (created), results/CUR/ (created).
Exit code:  0 every selected query has a final result | 2 not started (bad command line / input / plans; nothing spent) | 3 OpenAlex budget (run_guarded)
            | 4 STARTED BUT INCOMPLETE: spend ceiling reached, or queries still ERROR / DEGRADED after 3 passes -- never reported as success.
Pre-specified rules (fixed before the run; nothing is changed after seeing results):
  INPUT     only the frozen query text reaches CGScout; seed title/DOI/OpenAlex ID are used ONLY after retrieval to locate the seed.
  SELECT    a full run needs --expect-rows (queries.csv must have exactly that many rows) and an explicit CGSCOUT_STOP_USD; --pids must be non-empty and all known.
  CONFIG    every setting comes from cgscout.config.PAPER (model, effort, Stage-4 settings, retrieval switches; harness wrapper lifts explicit 'low' to the profile effort, counted);
            run_config_<arm>.json records the full effective values; each start is appended to run_starts_<arm>.jsonl.
  PLANS     plans_frozen/<pid>.json must carry the provenance of the P1/P2 code that is running (provenance.py); stale, foreign or missing plans stop the run before any spend.
  CLARIFY   one fixed generic reply, max 2 rounds (done in freeze_plans.py); if the planner still does not proceed the query is NOT_RUN_CLARIFICATION (a final result).
  RETRY     only infrastructure failures (degraded OpenAlex run, transient errors), max 3 passes; LLM 4xx refusals are recorded REFUSED, never retried.
  OPENALEX  strict budget mode; sleeps to 00:00 UTC when the daily credits run out; an interrupted query is discarded and re-run.
  COST      LLM cost from the usage log per query window; no new query starts once the arm's total reaches CGSCOUT_STOP_USD (legacy name NEW37_STOP_USD also read;
            default 5.0 applies to --pids runs only).  A run cut short by the ceiling exits with code 4; raise the ceiling and start again to resume.
  CACHE     fresh cache_work/ inside the run directory.   CHECKPOINT  results/CUR/{PID}.json with status OK / REFUSED / NOT_RUN_CLARIFICATION = finished query (never re-run).
"""
from __future__ import annotations

import argparse, collections, dataclasses, hashlib, inspect, json, os, sys, time, traceback
from pathlib import Path

import arms as AR
import provenance as PV
from arms import PreflightError
from recorder import Recorder

# --- run configuration (set by configure(); module-level so that run_query() and the offline verification can reuse it) ----------------------------------
ARM_NAME = None
ROOT = RES = INPUT = USAGE_LOG = LOG = None
BUDGETS = None
STOP_USD = DAILY_CREDITS = MIN_CREDITS = DEEP_CAP = None
STOP_EXPLICIT = False        # True when CGSCOUT_STOP_USD (or the legacy NEW37_STOP_USD) is set in the environment
PROFILE = None               # the cgscout.config entry of the arm: the single source of every setting below
EXIT_OK, EXIT_NOT_STARTED, EXIT_INCOMPLETE = 0, 2, 4          # (3 = OpenAlex budget, exit code of RL.run_guarded)
DONE_STATUSES = ("OK", "REFUSED", "NOT_RUN_CLARIFICATION")   # a checkpoint with one of these is a final result; DEGRADED / ERROR are retried
CLARIFY_REPLY = "I have no further details beyond my question as written; please proceed with your best reasonable interpretation."
# `harness` (and with it cgscout.llm, which reads CGSCOUT_USAGE_LOG_PATH at import time) is imported inside configure(), AFTER the environment is set.
RL = U = None
RP = TB = TerminologyExpansion = get_backend = backend_label = read_usage_log = Stage4Config = None


def configure(arm_name):
    """Environment + imports that must be in place before any pipeline module reads them (cache dir, usage log, model, effort)."""
    global ARM_NAME, ROOT, RES, INPUT, USAGE_LOG, LOG, BUDGETS, STOP_USD, DAILY_CREDITS, MIN_CREDITS, DEEP_CAP, STOP_EXPLICIT, PROFILE
    global RL, U, RP, TB, TerminologyExpansion, get_backend, backend_label, read_usage_log, Stage4Config
    ARM_NAME, ROOT, INPUT = arm_name, AR.ROOT, AR.INPUT
    PROFILE = AR.profile_for(arm_name)
    os.environ["CGSCOUT_CACHE_DIR"] = str(AR.SHARED_CACHE)           # shared by every arm: identical queries are fetched once
    os.environ["CGSCOUT_USAGE_LOG_PATH"] = str(ROOT / f"llm_usage_full_{ARM_NAME}.jsonl")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["CGSCOUT_LLM_MODEL"] = PROFILE["llm"]["model"]
    os.environ["CGSCOUT_LLM_EFFORT"] = PROFILE["llm"]["effort"]
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
    STOP_EXPLICIT = "CGSCOUT_STOP_USD" in os.environ or "NEW37_STOP_USD" in os.environ
    STOP_USD = float(os.environ.get("CGSCOUT_STOP_USD", os.environ.get("NEW37_STOP_USD", "5.0")))     # hard LLM-spend ceiling per arm; a full run requires it to be set explicitly
    DAILY_CREDITS = 10000
    MIN_CREDITS = int(os.environ.get("CGSCOUT_MIN_CREDITS", os.environ.get("NEW37_MIN_CREDITS", "400")))            # do not START a query with less than this left (a query uses ~150)
    DEEP_CAP = PROFILE["stage4"]["deep_cap"]
    USAGE_LOG = Path(os.environ["CGSCOUT_USAGE_LOG_PATH"])
    LOG = ROOT / f"run_log_full_{ARM_NAME}.txt"


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


class EffortUnifiedBackend:
    """Harness wrapper: lifts every explicit effort='low' request to the profile effort ('medium' for PAPER; config unification requested by the user)."""
    def __init__(self, inner, lift_to="medium", enabled=True):
        self._inner = inner; self.overrides = 0; self.calls = 0; self._lift_to = lift_to; self._enabled = enabled
    def structured(self, *, effort=None, **kw):
        self.calls += 1
        if self._enabled and effort == "low":
            effort = self._lift_to; self.overrides += 1
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


def effective_config(backend, s4cfg):
    """Every setting that determines what this run does, with the values actually in force: profile values plus the library defaults the profile does not name."""
    from cgscout.screening import sim_classifier as SC
    import cgscout.assessment.stage4 as S4M

    def entry_defaults(fn):
        return {n: p.default for n, p in inspect.signature(fn).parameters.items()
                if p.default is not inspect.Parameter.empty and isinstance(p.default, (bool, int, float, str, type(None)))}
    retrieval_defaults = entry_defaults(TB.run_expert_prior_retrieval)
    return {
        "profile": AR.ARMS[ARM_NAME]["profile"], "profile_label": PROFILE["label"],
        "llm": {"model": backend.model, "default_effort": backend.default_effort, "lift_explicit_low_effort": PROFILE["llm"]["lift_low_effort_to_medium"],
                "lift_target_effort": PROFILE["llm"]["effort"]},
        "budgets": BUDGETS.model_dump() if BUDGETS is not None else None,
        "stage4": dataclasses.asdict(s4cfg),
        "stage3_classifier": {"profile": PROFILE["stage3_classifier"], "file": SC.MODEL_PATH.name, "sha256": PV.sha256_file(SC.MODEL_PATH),
                              "threshold": retrieval_defaults.get("stage3_classifier_threshold")},
        "entry_defaults": {"run_expert_prior_retrieval": retrieval_defaults, "run_stage4_v2": entry_defaults(S4M.run_stage4_v2)},
        "openalex": {"strict_budget": os.environ.get("OPENALEX_STRICT_BUDGET"), "min_credits_to_start_a_query": MIN_CREDITS, "daily_credits": DAILY_CREDITS},
    }


def differs_from_library_defaults(s4cfg):
    out = {}
    if BUDGETS is not None:
        d = RP.RouteRetrievalBudgets()
        out["budgets"] = {k: {"default": getattr(d, k), "effective": getattr(BUDGETS, k)} for k in type(d).model_fields if getattr(d, k) != getattr(BUDGETS, k)}
    d4 = Stage4Config()
    out["stage4"] = {f.name: {"default": getattr(d4, f.name), "effective": getattr(s4cfg, f.name)} for f in dataclasses.fields(d4) if getattr(d4, f.name) != getattr(s4cfg, f.name)}
    return out


def verify_plans(rows, selected):
    """Every selected case needs a frozen plan made by the P1/P2 code that is about to run (provenance.py); otherwise nothing is started."""
    res = PV.check_plans(AR.PLANS, {r["Paper ID"]: r for r in rows}, selected, model=PROFILE["llm"]["model"], effort=PROFILE["llm"]["effort"])
    if res["missing"] or res["problems"]:
        raise PreflightError(PV.format_plan_problems(res, AR.PLANS))
    return res


def write_run_config(backend, s4cfg, rows, selected, full, expect_rows, plans):
    import cgscout
    eff = effective_config(backend, s4cfg)
    eff_sha = hashlib.sha256(json.dumps(eff, sort_keys=True, default=str).encode()).hexdigest()
    g = PV.git_state()
    cfg = {"model": backend.model, "default_effort": backend.default_effort, "stage4_config": {k: getattr(s4cfg, k) for k in ("screen_effort", "extract_effort", "judge_effort", "deep_cap", "screen_batch_size", "reject_min_confidence", "workers")},
           "changes_vs_production_defaults": {"stage4.screen_effort": "low->medium", "stage4.extract_effort": "low->medium", "hop2_search_anchor effort": "low->medium (harness backend wrapper)",
                                              "stage4.deep_cap": "80->500 (safety valve disabled in practice)", "CGSCOUT_CACHE_DIR": "fresh cache_work/ for this run"},
           "clarify_reply": CLARIFY_REPLY, "stop_usd": STOP_USD, "stop_usd_explicit": STOP_EXPLICIT, "input_sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
           "cgscout_version": cgscout.__version__, "git_commit": g["commit"], "git_dirty": g["dirty"], "code_sha256": PV.package_sha256(), "argv": sys.argv[1:]}
    cfg.update({"arm": ARM_NAME, "arm_label": AR.ARMS[ARM_NAME]["label"], "profile": AR.ARMS[ARM_NAME]["profile"], "budgets": PROFILE["budgets"], "classifier": PROFILE["stage3_classifier"],
                "effective_config": eff, "effective_config_sha256": eff_sha, "differs_from_library_defaults": differs_from_library_defaults(s4cfg),
                "selection": {"full_run": full, "n_rows": len(rows), "n_selected": len(selected), "expect_rows": expect_rows, "pids": None if full else selected},
                "plans_frozen": {"dir": str(AR.PLANS), **{k: plans[k] for k in ("checked", "usable", "p1p2_source_sha256", "git_commits", "git_dirty", "created_range")}}})
    (ROOT / f"run_config_{ARM_NAME}.json").write_text(json.dumps(cfg, indent=1, default=str))
    # run_config_<arm>.json is overwritten by every (re)start; the start log keeps one line per start, so a resumed run shows whether code / configuration / plans changed in between
    starts = ROOT / f"run_starts_{ARM_NAME}.jsonl"
    prev = None
    if starts.exists():
        lines = [ln for ln in starts.read_text().splitlines() if ln.strip()]
        prev = json.loads(lines[-1]) if lines else None
    entry = {"started": cfg["started"], "effective_config_sha256": eff_sha, "code_sha256": cfg["code_sha256"], "git_commit": g["commit"], "git_dirty": g["dirty"],
             "script_sha256": cfg["script_sha256"], "plans_p1p2_source_sha256": plans["p1p2_source_sha256"], "selection": cfg["selection"], "stop_usd": STOP_USD, "argv": sys.argv[1:]}
    with starts.open("a") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    if prev:
        for key in ("effective_config_sha256", "code_sha256"):
            if prev.get(key) != entry[key]:
                log(f"WARNING: {key} differs from the previous start of this run ({prev.get('started')}): results of this arm now mix settings/code -- see {starts.name}")
    return cfg


def _run(args):
    rows = AR.read_rows()
    selected, full = AR.select_pids([r["Paper ID"] for r in rows], args.pids, args.expect_rows)
    if STOP_USD <= 0:
        raise PreflightError(f"spend ceiling must be positive (CGSCOUT_STOP_USD={STOP_USD})")
    if full and not STOP_EXPLICIT:
        raise PreflightError("a full run needs an explicit spend ceiling: set CGSCOUT_STOP_USD=<usd>.  The built-in default of $5 is only meant for --pids smoke tests and would stop a full run early "
                             "(NEW37 averaged about $0.09 per case).")
    plans = verify_plans(rows, selected)
    only = set(selected)
    RES.mkdir(parents=True, exist_ok=True)
    RL.harness_guard(len(rows) * 16, what="cgscout_benchmark", check_budget=False)
    backend = get_backend(); bw = EffortUnifiedBackend(backend, lift_to=PROFILE["llm"]["effort"], enabled=PROFILE["llm"]["lift_low_effort_to_medium"])
    s4cfg = Stage4Config(**PROFILE["stage4"])
    write_run_config(backend, s4cfg, rows, selected, full, args.expect_rows, plans)
    log(f"LLM backend: {backend_label(backend)}; efforts unified to {PROFILE['llm']['effort']}; deep_cap={DEEP_CAP}; stop at ${STOP_USD}")
    log(f"selected {len(selected)} of {len(rows)} cases ({'full run' if full else '--pids'}); {plans['usable']} plans verified against the P1/P2 code in use")

    def done(pid):
        f = RES / f"{pid}.json"
        if not f.exists():
            return False
        try:
            return json.loads(f.read_text())["status"] in DONE_STATUSES
        except (ValueError, KeyError):
            return False

    def finish(reason):
        left = [p for p in selected if not done(p)]
        counts = collections.Counter(json.loads((RES / f"{p}.json").read_text()).get("status") for p in selected if (RES / f"{p}.json").exists())
        log(f"SUMMARY: {len(selected) - len(left)} of {len(selected)} selected cases have a final result; statuses {dict(counts)}; spent ${total_spent():.2f}")
        if left:
            msg = f"INCOMPLETE: {len(left)} of {len(selected)} selected cases have no final result ({reason}): {', '.join(left[:20])}{' ...' if len(left) > 20 else ''}"
            log(msg); print(f"run_benchmark.py: {msg}", file=sys.stderr)
            return EXIT_INCOMPLETE
        log("DONE")
        return EXIT_OK

    for pass_no in range(1, 4):
        todo = [r for r in rows if r["Paper ID"] in only and not done(r["Paper ID"])]
        if not todo:
            break
        log(f"=== pass {pass_no}: {len(todo)} queries to run ===")
        for r in todo:
            pid = r["Paper ID"]
            if total_spent() >= STOP_USD:
                log(f"STOP: spend ${total_spent():.2f} >= ${STOP_USD}; no new query started")
                return finish(f"spend ceiling ${STOP_USD} reached")
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
    return finish("queries still ERROR / DEGRADED after 3 passes")


def parse_args(argv):
    ap = argparse.ArgumentParser(prog="run_benchmark.py", allow_abbrev=False, description="CGScout paper evaluation run; the rules are listed at the top of this file.")
    ap.add_argument("arm", choices=sorted(AR.ARMS), help="evaluation arm (profile)")
    ap.add_argument("--pids", default=None, help="comma-separated Paper IDs to run (all must exist in queries.csv); omit for a full run")
    ap.add_argument("--expect-rows", type=int, default=None, help="number of cases queries.csv must contain; required for a full run (e.g. 200)")
    return ap.parse_args(argv)


def main(argv=None):
    """Returns the process exit code (see the module docstring); configure() runs first because it sets the environment the pipeline modules read at import."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        configure(args.arm)
        rc = RL.run_guarded(_run, args)
    except PreflightError as e:
        print(f"run_benchmark.py: not started: {e}", file=sys.stderr)
        return EXIT_NOT_STARTED
    return EXIT_OK if rc is None else rc


if __name__ == "__main__":
    sys.exit(main())
