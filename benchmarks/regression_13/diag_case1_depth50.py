"""Case 1 diagnosis (read-only wrt production): where does the Mpipi-Helix target rank in each 3A query's own
result list at the pipeline's depth (20) versus depth 50?  Uses the stored queries of three end-to-end draws.
Every OpenAlex request goes through the request cache + usage ledger (label-attributed); a failed request is
reported as failed, never as 'target absent'.  Usage: diag_case1_depth50.py [--labels a,b,c]"""
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import regression_lib as RL  # noqa: E402
from beam_agent.sources.openalex import search_openalex  # noqa: E402

U = RL.U
TARGET = ["10.1021/jacs.6c01403"]


def rank_in(papers):
    return next((i for i, p in enumerate(papers, 1) if any(t in (getattr(p, "doi", None) or "").lower() for t in TARGET)), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="baseline_prod_20261006,s4_full13,s4_tier1v2")
    a = ap.parse_args()
    RL.harness_guard(30, what="diag_case1_depth50")
    out = {}
    for lab in a.labels.split(","):
        run = json.loads((HERE / "results" / lab / "case1_rep1.json").read_text())["run"]
        rows = []
        for ax in run["queries"]["axes"]:
            for q in ax["direct_queries"]:
                U.set_run_label(f"diag_case1_{lab}")
                p20 = search_openalex(q, limit=20, mode="semantic")
                p50 = search_openalex(q, limit=50, mode="semantic")
                rows.append({"axis": ax["basis"], "query": q, "n20": len(p20), "rank20": rank_in(p20), "n50": len(p50), "rank50": rank_in(p50)})
        out[lab] = rows
        print(f"\n== draw {lab}  (3A semantic queries, rank of the target inside each query's own list)")
        for r in rows:
            print(f"  {r['axis'][:22]:22s} depth20: {str(r['rank20']):>5s}   depth50: {str(r['rank50']):>5s}   {r['query'][:100]}")
        best20 = [r["rank20"] for r in rows if r["rank20"]]; best50 = [r["rank50"] for r in rows if r["rank50"]]
        print(f"  -> queries finding it: depth20 {len(best20)}/{len(rows)}   depth50 {len(best50)}/{len(rows)}   best50 rank: {min(best50) if best50 else None}")
    (HERE / "results" / "diag_case1_depth50.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("\nusage:", U.used_today_utc())


if __name__ == "__main__":
    RL.run_guarded(main)
