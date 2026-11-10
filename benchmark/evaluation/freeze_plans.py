"""PHASE 0 -- generate and FREEZE the planning decisions, once per case, with the current LLM (LLM calls only: no OpenAlex request, no retrieval).
For each case of queries.csv (CGSCOUT_BENCH_DIR):  ScientificPlanner on the raw query (same clarification rule as Run 1: ONE fixed generic reply, max 2 rounds; if the
planner still does not proceed the whole conversation is repeated, at most 3 conversations -- the only criterion is 'did it proceed', never the content)
-> ExpertPriorPlanner.propose() on that plan.  Both outputs are saved in full (plans_frozen/<pid>.json) and reused unchanged by every arm.
Existing files are never overwritten.   usage: CGSCOUT_BENCH_DIR=<run dir> python freeze_plans.py [--dry-run] [--pids B083,B205]"""
import sys, os, json, csv, time
from pathlib import Path
import arms as AR
DRY = "--dry-run" in sys.argv
hits, misses = AR.case_ids(); PIDS = hits + misses
if "--pids" in sys.argv: PIDS = [p for p in PIDS if p in sys.argv[sys.argv.index("--pids") + 1].split(",")]
rows = {r["Paper ID"]: r for r in csv.DictReader(open(AR.INPUT, encoding="utf-8-sig"))}
CLARIFY_REPLY = "I have no further details beyond my question as written; please proceed with your best reasonable interpretation."
AR.PLANS.mkdir(exist_ok=True)
if DRY:
    for pid in PIDS:
        print(pid, "exists" if (AR.PLANS / f"{pid}.json").exists() else "TO DO", "|", rows[pid]["Latest query (v2)"][:110])
    print("planned LLM calls: per case 1-3 ScientificPlanner conversations (1-3 turns each) + 1 ExpertPrior; OpenAlex requests: 0"); sys.exit(0)
os.environ.update({"CGSCOUT_LLM_MODEL": "gpt-5.6-luna", "CGSCOUT_LLM_EFFORT": "medium", "CGSCOUT_CACHE_DIR": str(AR.ROOT / "cache_phase0"),
                   "CGSCOUT_USAGE_LOG_PATH": str(AR.ROOT / "llm_usage_phase0.jsonl"), "PYTHONDONTWRITEBYTECODE": "1"})
import harness as RL
from cgscout.llm import get_backend, backend_label, read_usage_log
from cgscout.planning.session import PlannerSession


class EffortUnifiedBackend:           # same wrapper as the Run-1 / v3 harness: explicit effort='low' requests are lifted to 'medium'
    def __init__(self, inner): self._inner = inner
    def structured(self, *, effort=None, **kw): return self._inner.structured(effort="medium" if effort == "low" else effort, **kw)
    def __getattr__(self, k): return getattr(self._inner, k)


backend = get_backend(); bw = EffortUnifiedBackend(backend)
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
    tmp = out.with_suffix(".tmp"); tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1)); tmp.replace(out)
    print(f"{pid} {rec['status']} attempts={len(attempts)} calls={rec['llm_calls']} ${rec['llm_cost_usd']:.4f}", flush=True)
