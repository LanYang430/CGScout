"""Controlled diagnosis of the Case 6 canonical-render recall drop. Offline;
production untouched (budgets are passed as an argument, not edited).

Two suspects are separated by sharing ExpertPrior output between arms:
  E1  ExpertPlanner sees a different INPUT (canonical vs raw) and therefore
      picks different scientific_class values;
  E2  the OBJECTIVE TEXT that 3A templates / flat route paste into queries
      differs.
Per record (same ScientificPlanner output), ExpertPrior is proposed twice:
E_raw (from raw fields) and E_can (from canonical-rendered fields). Arms:
  R   : raw plan       + E_raw      (current production behaviour)
  C1  : canonical plan + E_raw      (E2 only -- classes held fixed)
  C2  : canonical plan + E_can      (E1 + E2 -- the full canonical path)
Each at depth 20 (production default) and depth 50 (max_papers_per_query=50,
max_papers_per_route=150 so the per-route cap does not bind).
Usage: case6_diagnosis.py <comma-separated record indexes into the first 5 ok records>
"""
import sys, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import regression_lib as RL  # noqa: E402
from route_retrieval_plan import RouteRetrievalBudgets  # noqa: E402
from llm_backend import get_backend, backend_label  # noqa: E402

RES = HERE / "results"
TARGETS = {"Protein-RNA vesicles (JACS Au 2026)": ["10.1021/jacsau.6c00322"]}
B20 = None
B50 = RouteRetrievalBudgets(max_papers_per_query=50, max_papers_per_route=150)


def main():
    idxs = [int(x) for x in sys.argv[1].split(",")]
    RL.harness_guard(len(idxs) * 6 * 25, what="case6_diagnosis")   # 6 runs per record, ~25 uncached searches each
    backend = get_backend()
    print(f"LLM backend: {backend_label(backend)} records={idxs}", flush=True)
    canon = json.loads((RES / "co_canon.json").read_text())["case6_protein_rna_vesicles"]
    ok = [r for r in canon if r["status"] == "ok"][:5]
    for i in idxs:
        row = ok[i]
        raw_f = {k: row["raw"][k] for k in ("system", "target_component", "scientific_objective", "search_context")}
        can_f = row["C"]["rendered"]
        e_raw = RL.propose(backend, raw_f, f"d6_{i}_eraw")
        e_can = RL.propose(backend, can_f, f"d6_{i}_ecan")
        classes = {"E_raw": [(a.basis, a.execution_mode, a.scientific_class) for a in e_raw.transfer_axes],
                   "E_can": [(a.basis, a.execution_mode, a.scientific_class) for a in e_can.transfer_axes]}
        print(f"\nrec{i} classes: {json.dumps(classes, ensure_ascii=False)}", flush=True)
        jobs = [("R", 20, raw_f, e_raw, B20), ("C1", 20, can_f, e_raw, B20), ("C2", 20, can_f, e_can, B20),
                ("R", 50, raw_f, e_raw, B50), ("C1", 50, can_f, e_raw, B50), ("C2", 50, can_f, e_can, B50)]
        out = {"rec": i, "rep": row["rep"], "classes": classes, "raw_plan": raw_f, "canonical_plan": can_f,
               "raw_objective": raw_f["scientific_objective"], "canonical_objective": can_f["scientific_objective"], "arms": []}
        for arm, depth, fields, ep, bud in jobs:
            try:
                r = RL.run_plan_checked(backend, fields, f"d6_{i}_{arm}_{depth}", TARGETS, expert_prior=ep, budgets=bud)
                t = list(r["targets"].values())[0]
                rec = {"arm": arm, "depth": depth, "found": t["final_rank"] is not None, "rank": t["final_rank"],
                       "in_pool": t["in_pool"], "route": [k for k in ("flat", "3a", "3b") if t["in_" + k]],
                       "pool": r["pool"], "queries": r["queries"], "degraded": r["degraded"],
                       "attempts": r["attempts"], "api_usage": r["api_usage_all_attempts"],
                       "target": t, "query_target_ranks": r["query_target_ranks"], "axes_run": r["expert_prior"]["axes"]}
            except Exception as e:  # noqa: BLE001
                import traceback; traceback.print_exc()
                rec = {"arm": arm, "depth": depth, "error": str(e)}
            out["arms"].append(rec)
            print(f"  rec{i} {arm:2s} depth{depth}: " + (f"found={rec['found']} rank={rec['rank']} pool={rec['in_pool']} route={rec['route']}" if "error" not in rec else rec["error"]), flush=True)
            (RES / f"diag6_rec{i}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    RL.run_guarded(main)
