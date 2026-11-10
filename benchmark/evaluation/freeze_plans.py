"""PHASE 0 -- generate and FREEZE the planning decisions, once per case, with the current LLM (LLM calls only: no OpenAlex request, no retrieval).
For each case of queries.csv (CGSCOUT_BENCH_DIR):  ScientificPlanner on the raw query (same clarification rule as Run 1: ONE fixed generic reply, max 2 rounds; if the
planner still does not proceed the whole conversation is repeated, at most 3 conversations -- the only criterion is 'did it proceed', never the content)
-> ExpertPriorPlanner.propose() on that plan.  Both outputs are saved in full (plans_frozen/<pid>.json) and reused unchanged by every arm.
Every plan file carries a provenance block (provenance.py): sha256 of the P1/P2 source modules, model, effort, git commit.  Existing files are never overwritten, but they are
CHECKED: if any existing plan was made by different P1/P2 code (or has no provenance, or another query text) nothing is generated and the command exits with code 2 -- plans of
two code versions never end up in one directory.  Re-running after an interruption (same code) skips finished cases and continues.
usage: CGSCOUT_BENCH_DIR=<run dir> python freeze_plans.py --expect-rows N [--dry-run]       (every row of queries.csv)
       CGSCOUT_BENCH_DIR=<run dir> python freeze_plans.py --pids B083,B205 [--dry-run]        (some cases)
Exit code: 0 all selected plans are frozen and consistent | 2 refused (bad command line / input / stale plans); --dry-run uses the same codes and never calls the LLM."""
import argparse, sys, os, json, csv, time

import arms as AR
import provenance as PV
from arms import PreflightError

CLARIFY_REPLY = "I have no further details beyond my question as written; please proceed with your best reasonable interpretation."
EXIT_OK, EXIT_REFUSED = 0, 2


class EffortUnifiedBackend:           # same wrapper as the Run-1 / v3 harness: explicit effort='low' requests are lifted to the profile effort ('medium')
    def __init__(self, inner, lift_to="medium", enabled=True): self._inner = inner; self._lift_to = lift_to; self._enabled = enabled
    def structured(self, *, effort=None, **kw): return self._inner.structured(effort=self._lift_to if (self._enabled and effort == "low") else effort, **kw)
    def __getattr__(self, k): return getattr(self._inner, k)


def parse_args(argv):
    ap = argparse.ArgumentParser(prog="freeze_plans.py", allow_abbrev=False, description="Freeze P1 + P2 plans once per case (see the top of this file).")
    ap.add_argument("--pids", default=None, help="comma-separated Paper IDs; omit for every row of queries.csv")
    ap.add_argument("--expect-rows", type=int, default=None, help="number of cases queries.csv must contain; required when --pids is omitted")
    ap.add_argument("--dry-run", action="store_true", help="report what exists / is stale / is to do; no LLM call, nothing written")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        return _main(args)
    except PreflightError as e:
        print(f"freeze_plans.py: refused: {e}", file=sys.stderr)
        return EXIT_REFUSED


def _main(args):
    profile = AR.profile_for("CUR")["llm"]
    rows_list = AR.read_rows(); rows = {r["Paper ID"]: r for r in rows_list}
    PIDS, _full = AR.select_pids([r["Paper ID"] for r in rows_list], args.pids, args.expect_rows)
    # existing plans must come from the P1/P2 code that is importable now, for the same query text, with the profile's model / effort
    chk = PV.check_plans(AR.PLANS, rows, [p for p in PIDS if (AR.PLANS / f"{p}.json").exists()], model=profile["model"], effort=profile["effort"])
    todo = [p for p in PIDS if not (AR.PLANS / f"{p}.json").exists()]
    if args.dry_run:
        for pid in PIDS:
            state = "TO DO" if pid in todo else ("STALE: " + "; ".join(chk["problems"][pid]) if pid in chk["problems"] else "frozen, provenance OK")
            print(pid, state, "|", rows[pid]["Latest query (v2)"][:110])
        print("planned LLM calls: per case 1-3 ScientificPlanner conversations (1-3 turns each) + 1 ExpertPrior; OpenAlex requests: 0")
        print(f"{len(todo)} to do, {len(chk['problems'])} stale, {chk['usable']} frozen and consistent with the current P1/P2 code")
        return EXIT_REFUSED if chk["problems"] else EXIT_OK
    if chk["problems"]:
        raise PreflightError(PV.format_plan_problems(chk, AR.PLANS))
    AR.PLANS.mkdir(parents=True, exist_ok=True)
    if not todo:
        for pid in PIDS: print(pid, "already frozen")
        print(f"all {len(PIDS)} selected plans are frozen and consistent with the current P1/P2 code")
        return EXIT_OK
    os.environ.update({"CGSCOUT_LLM_MODEL": profile["model"], "CGSCOUT_LLM_EFFORT": profile["effort"], "CGSCOUT_CACHE_DIR": str(AR.ROOT / "cache_phase0"),
                       "CGSCOUT_USAGE_LOG_PATH": str(AR.ROOT / "llm_usage_phase0.jsonl"), "PYTHONDONTWRITEBYTECODE": "1"})
    import harness as RL
    from cgscout.llm import get_backend, backend_label, read_usage_log
    from cgscout.planning.session import PlannerSession

    backend = get_backend(); bw = EffortUnifiedBackend(backend, lift_to=profile["effort"], enabled=profile["lift_low_effort_to_medium"])
    print("LLM backend:", backend_label(backend))
    for pid in PIDS:
        out = AR.PLANS / f"{pid}.json"
        if out.exists():
            print(pid, "already frozen"); continue
        q = rows[pid]["Latest query (v2)"]; t0 = time.time(); attempts = []; ready = None
        for n in range(1, 4):
            sess = PlannerSession(backend=bw, case_id=f"plan_{pid}_{n}")
            t = sess.submit_user_message(q); cl = []
            for _ in range(2):
                if t.status != "awaiting_user": break
                cl.append({"question": t.display_question_to_user, "reply": CLARIFY_REPLY}); t = sess.submit_user_message(CLARIFY_REPLY)
            attempts.append({"n": n, "status": t.status, "clarifications": cl})
            if t.status == "ready": ready = t.plan; break
        rec = {"pid": pid, "query": q, "attempts": attempts, "created": time.strftime("%F %T"), "model": backend.model}
        if ready is None:
            rec.update(status="NOT_READY", last_question=attempts[-1]["clarifications"][-1]["question"] if attempts[-1]["clarifications"] else None)
        else:
            fields = {k: getattr(ready, k) for k in ("system", "target_component", "scientific_objective", "search_context")}
            ep = RL.propose(bw, fields, f"plan_{pid}")
            rec.update(status="READY", planner=ready.model_dump(), expert_prior=ep.model_dump())
        rows_u = [r for r in read_usage_log() if r.get("ts", 0) >= t0 - 0.5]
        rec["llm_cost_usd"] = round(sum(r.get("estimated_cost_usd", 0.0) for r in rows_u), 6); rec["llm_calls"] = len(rows_u)
        rec["provenance"] = PV.stamp(model=backend.model, effort=profile["effort"])
        tmp = out.with_suffix(".tmp"); tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1)); tmp.replace(out)
        print(f"{pid} {rec['status']} attempts={len(attempts)} calls={rec['llm_calls']} ${rec['llm_cost_usd']:.4f}", flush=True)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
