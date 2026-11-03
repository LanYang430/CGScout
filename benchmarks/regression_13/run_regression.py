"""13-case regression suite runner. Production code is called unmodified.

  run     --label L [--cases 1,3,11 | --all] [--repeats N] [--replay-plans L0]
  summary --label L
  compare --before L0 --after L1

Every run writes results/<label>/case<id>_rep<k>.json containing: the raw
ScientificPlanner output (or the clarification question -- never a
fabricated answer), the ExpertPriorPlanner axes, every generated query
(flat route / 3A / Hop-1 / Hop-2), per-target recall + final rank + route
attribution by source list, and the top final papers.

--replay-plans L0 reuses the stored ScientificPlanner outputs of an earlier
label, so a downstream-only change can be compared against the same planner
input (separating planner variance from downstream effects).

Before wiring any core change into production: run --all, then `compare`
against the pre-change label. Cross-system targets (cases 11-13) only count
as a 3B contribution when in_3b is true.
"""
import argparse, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RESULTS = HERE / "results"

sys.path.insert(0, str(HERE / "canonical_validation"))
import regression_lib as RL  # noqa: E402
from canon_lib import apply_option_a  # noqa: E402
from llm_backend import get_backend, backend_label  # noqa: E402
from planner_session import PlannerSession  # noqa: E402
U = RL.U   # beam_agent.sources.openalex_usage (budget / cache / ledger)


def load_cases(sel):
    cases = json.loads((HERE / "cases.json").read_text())["cases"]
    return [c for c in cases if sel is None or c["id"] in sel]


def route_label(t):
    if not t["in_pool"]:
        return "-"
    parts = []
    if t["in_flat"]: parts.append("flat")
    if t["in_3a"]: parts.append("3A")
    if t["in_3b"]: parts.append("3B")
    return "+".join(parts) or "transfer(other)"


def failure_step(rec, t):
    if rec.get("status") == "awaiting_clarification":
        return "intake: clarification question, not run"
    if not t["in_pool"]:
        return "retrieval_miss (never in pool)"
    if t["stage3_keep"] is False:
        return "stage3_drop"
    if t["final_rank"] is None:
        return "not_in_final_output"
    return ""


def _usage_line(u: dict) -> str:
    rem = u.get("remaining_last")
    return (f"OpenAlex: {u['http_requests']} HTTP request(s) ({u['http_200']} ok, {u['failed_requests']} failed), "
            f"{u['cache_hits']} cache hit(s), {u['credits_charged']} credits charged"
            f"{' (price assumed)' if u.get('credits_assumed_requests') else ''}, "
            f"remaining {rem if rem is not None else 'n/a'}")


def cmd_run(a):
    sel = None if a.all or not a.cases else {int(x) for x in a.cases.split(",")}
    todo = [c for c in load_cases(sel) if c.get("status") != "TBD"]
    RL.harness_guard(len(todo) * a.repeats * a.searches_per_run, max_parallel=a.max_parallel,
                     check_budget=not a.no_budget_check, what=f"regression:{a.label}", refresh=a.refresh_budget)   # may refuse to start
    label_dir = RESULTS / a.label
    label_dir.mkdir(parents=True, exist_ok=True)
    backend = get_backend()
    print(f"LLM backend: {backend_label(backend)}  label={a.label}", flush=True)
    aborted = None
    for c in load_cases(sel):
        if aborted:
            break
        if c.get("status") == "TBD":
            print(f"\ncase {c['id']}: definition TBD -- skipped (pending)", flush=True)
            continue
        targets = {k: v["dois"] for k, v in c["targets"].items()}
        for rep in range(1, a.repeats + 1):
            out = label_dir / f"case{c['id']}_rep{rep}.json"
            run_label = f"reg_{a.label}_c{c['id']}_r{rep}"
            print(f"\n{'='*90}\ncase {c['id']} {c['short']} rep{rep}", flush=True)
            rec = {"case": c["id"], "short": c["short"], "rep": rep, "query": c["query"],
                   "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
            try:
                if a.replay_plans:
                    fp = RESULTS / a.replay_plans / f"case{c['id']}_rep{rep}.json"
                    if not fp.exists():   # fewer stored plans than repeats: reuse rep1's plan
                        fp = RESULTS / a.replay_plans / f"case{c['id']}_rep1.json"
                    old = json.loads(fp.read_text())
                    planner = old.get("planner_raw") or old["planner"]
                else:
                    sess = PlannerSession(backend=backend, case_id=run_label)
                    t = sess.submit_user_message(c["query"])
                    if t.status == "awaiting_user":
                        planner = {"status": "awaiting_clarification", "question": t.display_question_to_user}
                    else:
                        p = t.plan
                        planner = {"status": "ok", "system": p.system, "target_component": p.target_component,
                                   "scientific_objective": p.scientific_objective,
                                   "search_context": p.search_context, "search_axes": p.search_axes}
                rec["planner"] = planner
                rec["transform"] = a.transform
                if planner["status"] != "ok":
                    rec["status"] = "awaiting_clarification"
                else:
                    rec["status"] = "ran"
                    eff = apply_option_a(planner) if a.transform == "A" else planner
                    rec["planner_effective"] = {k: eff[k] for k in ("system", "target_component", "scientific_objective", "search_context")}
                    rec["run"] = RL.run_plan_checked(backend, eff, run_label, targets, user_request=c["query"], stage4=not a.no_stage4)
                    u = dict(rec["run"]["api_usage"])
                    u.update(rec["run"]["api_usage_all_attempts"])      # requests / hits / credits summed over retries
                    rec["api_usage"] = u
                    print("  " + _usage_line(u), flush=True)
            except U.OpenAlexBudgetExhausted as e:   # BaseException by design -- the pipeline cannot swallow it
                rec["status"] = "aborted_budget"
                rec["abort"] = {"message": str(e)[:400], "reset_s": e.reset_s, "api_usage": U.run_usage(run_label)}
                aborted = rec
                print(RL.abort_message(e), flush=True)
            except Exception as e:  # noqa: BLE001
                import traceback; traceback.print_exc()
                rec["status"] = "error"; rec["error"] = f"{type(e).__name__}: {e}"[:400]
            out.write_text(json.dumps(rec, indent=2, ensure_ascii=False))
            print(f"  -> {rec['status']}", flush=True)
            if aborted:
                break
    cmd_summary(argparse.Namespace(label=a.label))
    print(f"\nOpenAlex used today by this machine (all processes): {U.used_today_utc()}")
    if aborted:
        print(f"STOPPED at case {aborted['case']} rep{aborted['rep']} -- remaining cases were NOT run and "
              f"nothing was scored as a miss.")
        sys.exit(3)


def _load(label):
    recs = {}
    for f in sorted((RESULTS / label).glob("case*_rep*.json")):
        r = json.loads(f.read_text())
        recs[(r["case"], r["rep"])] = r
    return recs


def _scoreable(r):
    """Only a completed, non-degraded run says anything about retrieval quality."""
    return r["status"] == "ran" and not r["run"].get("degraded")


OUTCOME_TEXT = {
    "retrieval_miss": "STAGE 2: retrieval miss (never in the candidate pool)",
    "stage3_drop": "STAGE 3: dropped by the cheap filter",
    "stage4_rejected": "STAGE 4: relevance rejection (grade 0, positive evidence of irrelevance)",
    "stage4_missing_evidence": "STAGE 4: missing_evidence (internal; no abstract after the Europe PMC fallback) -- NOT a relevance failure",
    "stage4_background_only": "STAGE 4: judged background only (grade 1) -- not recommended",
    "stage4_not_deep_evaluated_cap": "STAGE 4: screened >=2 but cut by the deep-evaluation cap",
    "stage4_error": "STAGE 4: LLM error -- NOT evaluated (excluded, not a rejection)",
    "retained_stage4_not_run": "retained by Stage 3 (Stage 4 not run)",
    "recommended": "recommended",
}


def _evaluable(r):
    """A run is scoreable if it completed, was not degraded, and (when Stage 4 ran) had no Stage-4 errors on the target."""
    return r["status"] == "ran" and not r["run"].get("degraded")


def _final_text(t):
    o, st4 = t.get("outcome"), t.get("stage4") or {}
    if o == "retrieval_miss":
        return "retrieval miss"
    if o == "stage3_drop":
        return "dropped at Stage 3"
    if not st4:
        return {"stage4_error": "STAGE 4 LLM ERROR (not evaluated)", "retained_stage4_not_run": "Stage 4 not run"}.get(o, str(o))
    status, g = st4.get("status"), st4.get("final_grade")
    if status == "missing_evidence":
        return "missing_evidence (internal)"
    if status == "not_evaluated_cap":
        return "not evaluated (cap)"
    if status == "tier1_rejected":
        return "grade 0 (rejected at Tier 1)"
    return f"grade {g}" + (" (background)" if g == 1 else " (not relevant)" if g == 0 else "")


def _tier1_text(t):
    st4 = t.get("stage4") or {}
    if not st4:
        return "-"
    d = st4.get("tier1_decision")
    if d is None and st4.get("status") == "missing_evidence":
        return "skipped"
    return {"pass": "pass", "reject": "reject", "insufficient_evidence": "insufficient"}.get(d, str(d)) + (
        f" ({st4['tier1_confidence']})" if st4.get("tier1_confidence") and d != "insufficient_evidence" else "")


def cmd_summary(a):
    cases = {c["id"]: c for c in load_cases(None)}
    recs = _load(a.label)
    print(f"\nSUMMARY  label={a.label}\n")
    hdr = (f"{'Case':>4}  {'Target paper':32s} {'Retrieved?':10s} {'S3 kept?':8s} {'Tier 1':14s} {'Tier 2?':7s} "
           f"{'Final Stage 4 grade/status':30s} {'Rank':>9s} {'#Rec':>4}  route")
    print(hdr); print("-" * len(hdr))
    n = {"targets": 0, "pool": 0, "kept": 0, "rec": 0, "insuff": 0, "s4_err": 0, "bg": 0, "rej": 0, "cap": 0}
    ranks = []
    excluded, tot = [], {"http_requests": 0, "cache_hits": 0, "credits_charged": 0}
    llm = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    fp = {"retained": 0, "t1_pass": 0, "t1_reject": 0, "t1_insuff": 0, "deep": 0, "recommended": 0, "background": 0, "rejected": 0, "cap_binding": 0, "runs": 0, "final_insuff": 0, "abs_lookup": 0, "abs_recovered": 0}
    for (cid, rep), r in sorted(recs.items()):
        c = cases[cid]
        if r["status"] == "awaiting_clarification":
            print(f"{cid:>4}  {'(all targets)':32s} -- CLARIFICATION STOP: not run, not a miss")
            excluded.append((cid, rep, "clarification")); continue
        if r["status"] in ("aborted_budget", "error"):
            why = "OpenAlex BUDGET EXHAUSTED -- aborted" if r["status"] == "aborted_budget" else f"harness/pipeline error: {r.get('error','')[:70]}"
            print(f"{cid:>4}  {'(all targets)':32s} -- {why} -- EXCLUDED")
            excluded.append((cid, rep, r["status"])); continue
        run = r["run"]
        u = run.get("api_usage_all_attempts") or {}
        for k in tot:
            tot[k] += u.get(k, 0)
        if run.get("degraded"):
            print(f"{cid:>4}  {'(all targets)':32s} -- DEGRADED (OpenAlex requests failed) -- EXCLUDED, not a miss")
            excluded.append((cid, rep, "degraded")); continue
        s4 = run.get("stage4")
        if s4:
            w = s4["workload"]; fp["runs"] += 1
            fp["retained"] += w["n_retained"]; fp["t1_pass"] += w.get("n_tier1_pass", w.get("n_deep_candidates", 0))
            fp["t1_reject"] += w.get("n_tier1_rejected", 0); fp["t1_insuff"] += w.get("n_tier1_insufficient", 0); fp["final_insuff"] += w.get("n_missing_evidence", w.get("n_insufficient_evidence", 0)); fp["abs_lookup"] += w.get("n_abstract_lookup_attempted", 0); fp["abs_recovered"] += w.get("n_abstract_recovered_europepmc", 0)
            fp["deep"] += w["n_deep_evaluated"]; fp["recommended"] += w["n_recommended"]; fp["background"] += w["n_background"]
            fp["rejected"] += w["n_rejected"]; fp["cap_binding"] += int(w["deep_cap_binding"])
            for k in llm:
                llm[k] += (s4.get("llm_usage") or {}).get(k, 0)
        for label, t in run["targets"].items():
            kind = c["targets"][label]["kind"]
            st4 = t.get("stage4") or {}
            nrec = s4["workload"]["n_recommended"] if s4 else "-"
            route = "+".join(x for x, k in (("flat", "in_flat"), ("3A", "in_3a"), ("3B", "in_3b")) if t[k]) or "-"
            if kind == "cross_system" and t.get("outcome") == "recommended" and not t["in_3b"]:
                route += " [no 3B credit]"
            rk = f"{st4['rank']}/{st4['n_ranked']}" if st4.get("rank") else "-"
            t2 = "yes" if st4.get("tier2_evaluated") else ("-" if not st4 else "no")
            print(f"{cid:>4}  {label[:32]:32s} {'yes' if t['in_pool'] else 'NO':10s} {'yes' if t.get('retained') else ('NO' if t['in_pool'] else '-'):8s} "
                  f"{_tier1_text(t):14s} {t2:7s} {_final_text(t):30s} {rk:>9s} {str(nrec):>4}  {route}")
            n["targets"] += 1; n["pool"] += t["in_pool"]; n["kept"] += bool(t.get("retained"))
            o = t.get("outcome")
            n["rec"] += o == "recommended"; n["insuff"] += o == "stage4_missing_evidence"; n["s4_err"] += o == "stage4_error"
            n["bg"] += o == "stage4_background_only"; n["rej"] += o == "stage4_rejected"; n["cap"] += o == "stage4_not_deep_evaluated_cap"
            if o == "recommended" and st4.get("rank"):
                ranks.append(st4["rank"])
    N = n["targets"]
    bucket = lambda lo, hi: sum(1 for x in ranks if lo <= x <= hi)
    print(f"\nSHORT SUMMARY over {N} target rows (completed, non-degraded runs only)")
    print(f"  target retrieved recall (Stage 2):        {n['pool']}/{N}")
    print(f"  Stage 3 survival (of retrieved):          {n['kept']}/{n['pool']}   ({n['kept']}/{N} of all targets)")
    print(f"  Stage 4 grade >= 2 (recommended):         {n['rec']}/{n['kept']} of Stage-3-kept   ({n['rec']}/{N} of all targets)")
    print(f"  missing_evidence (internal, not a failure): {n['insuff']}    | background grade 1: {n['bg']} | rejected grade 0: {n['rej']} | cap-cut: {n['cap']} | LLM error: {n['s4_err']}")
    print(f"  final rank of recommended targets:        rank 1: {bucket(1,1)} | 2-3: {bucket(2,3)} | 4-5: {bucket(4,5)} | 6-10: {bucket(6,10)} | >10: {bucket(11,10**6)}   (ranks: {sorted(ranks)})")
    if excluded:
        print(f"  excluded from the counts (NOT misses):    {excluded}")
    if fp["runs"]:
        print(f"\nStage 4 workload over {fp['runs']} runs: {fp['retained']} Stage-3-kept -> Tier 1: {fp['t1_pass']} pass / {fp['t1_reject']} reject / {fp['t1_insuff']} insufficient (continued to Tier 2); "
              f"{fp['deep']} Tier-2 evaluated (cap binding in {fp['cap_binding']} runs); recommended {fp['recommended']}, background {fp['background']}, rejected(grade 0) {fp['rejected']}, "
              f"missing_evidence {fp['final_insuff']}; Europe PMC abstract fallback: {fp['abs_recovered']}/{fp['abs_lookup']} recovered")
        print(f"  LLM: {llm['llm_calls']} calls, {llm['input_tokens']} in / {llm['output_tokens']} out tokens, ${llm['cost_usd']:.3f}")
    print(f"OpenAlex in this label's runs: {tot['http_requests']} HTTP requests, {tot['cache_hits']} cache hits, {tot['credits_charged']} credits")


def _success(t):
    """Stage-4 era success = recommended; falls back to 'retained' for runs recorded without Stage 4."""
    if t.get("outcome") is not None:
        return t["outcome"] == "recommended" if t["outcome"] != "retained_stage4_not_run" else True
    return t.get("final_rank") is not None


def cmd_compare(a):
    b, af = _load(a.before), _load(a.after)
    print(f"COMPARE before={a.before} after={a.after}   (only runs scoreable on BOTH sides; success = recommended when Stage 4 ran, else retained)")
    regress = 0
    for k in sorted(set(b) | set(af)):
        rb, ra = b.get(k), af.get(k)
        if not rb or not ra:
            print(f"case {k[0]} rep{k[1]}: only in {'before' if rb else 'after'}"); continue
        if not (_evaluable(rb) and _evaluable(ra)):
            print(f"case {k[0]} rep{k[1]}: not comparable (before={rb['status']}, after={ra['status']})"); continue
        tb, ta = rb["run"]["targets"], ra["run"]["targets"]
        for label in sorted(set(tb) | set(ta)):
            x, y = tb.get(label), ta.get(label)
            fx, fy = bool(x and _success(x)), bool(y and _success(y))
            tag = "REGRESSION" if fx and not fy else ("recovered" if fy and not fx else "same")
            regress += tag == "REGRESSION"
            print(f"case {k[0]:>2} rep{k[1]} {label[:30]:30s} {tag:10s} {(x or {}).get('outcome')} -> {(y or {}).get('outcome')}")
    print(f"\nregressions (success -> not success): {regress}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run"); r.add_argument("--label", required=True); r.add_argument("--cases"); r.add_argument("--all", action="store_true")
    r.add_argument("--repeats", type=int, default=1); r.add_argument("--replay-plans")
    r.add_argument("--transform", choices=["none", "A"], default="none")
    r.add_argument("--searches-per-run", type=int, default=30, help="estimated UNCACHED OpenAlex searches per case-run (budget pre-flight)")
    r.add_argument("--max-parallel", type=int, default=1, help="max benchmark processes sharing the OpenAlex budget")
    r.add_argument("--refresh-budget", action="store_true", help="re-check the budget live (ignore a remembered exhaustion), e.g. after adding funds")
    r.add_argument("--no-stage4", action="store_true", help="skip Stage 4 (Stage-3 retained list only)")
    r.add_argument("--no-budget-check", action="store_true"); r.set_defaults(fn=lambda a: RL.run_guarded(cmd_run, a))
    s = sp.add_parser("summary"); s.add_argument("--label", required=True); s.set_defaults(fn=cmd_summary)
    c = sp.add_parser("compare"); c.add_argument("--before", required=True); c.add_argument("--after", required=True); c.set_defaults(fn=cmd_compare)
    a = p.parse_args(); a.fn(a)
