"""Paired offline control for canonical OPTION A (entity/part normalization of `system` only; the
original scientific_objective is kept verbatim; 3A/3B untouched -- production code is not modified).

Per case and per draw, from the SAME stored ScientificPlanner output, ExpertPrior is proposed from
the raw plan (E_raw) and from the Option-A plan (E_A). Arms:
  R   : raw plan      + E_raw   (current production behaviour)
  A1  : A plan        + E_raw   (isolates the effect of the changed system/context text on the
                                  flat route and 3A -- ExpertPrior is held fixed, so 3A queries are
                                  identical to R's and are served from the cache for free)
  A2  : A plan        + E_A     (the full Option-A path: ExpertPrior also sees the A text)
A run with any failed OpenAlex request is flagged `degraded` and excluded from scoring; budget
exhaustion aborts the whole script (RL.run_guarded) -- neither is ever counted as a miss.
Usage: option_a_paired.py [--cases 1,2,3,...] [--draws 2] [--plans-from LABEL] [--out LABEL]
"""
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "canonical_validation"))
import regression_lib as RL  # noqa: E402
from canon_lib import apply_option_a  # noqa: E402
from llm_backend import get_backend, backend_label  # noqa: E402

U = RL.U
RESULTS = HERE / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="1,2,3,4,7,8,10")
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--plans-from", default="baseline_prod_20261006")
    ap.add_argument("--out", default="optA_paired")
    a = ap.parse_args()
    cases = {c["id"]: c for c in json.loads((HERE / "cases.json").read_text())["cases"]}
    ids = [int(x) for x in a.cases.split(",")]
    RL.harness_guard(len(ids) * a.draws * 3 * 15, what=f"optionA:{a.out}")
    out_dir = RESULTS / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = get_backend()
    print(f"LLM backend: {backend_label(backend)}", flush=True)

    for cid in ids:
        c = cases[cid]
        targets = {k: v["dois"] for k, v in c["targets"].items()}
        src = RESULTS / a.plans_from / f"case{cid}_rep1.json"
        planner = json.loads(src.read_text())["planner"]
        if planner.get("status") != "ok":
            print(f"case {cid}: stored plan is a clarification -- skipped", flush=True)
            continue
        raw = {k: planner[k] for k in ("system", "target_component", "scientific_objective", "search_context")}
        plan_a = apply_option_a(planner)
        a_fields = {k: plan_a[k] for k in raw}
        changed = a_fields != raw
        print(f"\n{'=' * 90}\ncase {cid} {c['short']}  A changes the plan: {changed}\n  raw system={raw['system']!r} ctx={raw['search_context']}\n  A   system={a_fields['system']!r} ctx={a_fields['search_context']}", flush=True)
        for draw in range(1, a.draws + 1):
            e_raw = RL.propose(backend, raw, f"optA_{cid}_{draw}_eraw")
            e_a = RL.propose(backend, a_fields, f"optA_{cid}_{draw}_ea")
            rec = {"case": cid, "short": c["short"], "draw": draw, "plan_changed_by_A": changed,
                   "raw_plan": raw, "A_plan": a_fields,
                   "expert_prior": {"E_raw": [(x.basis, x.execution_mode, x.confidence, x.scientific_class) for x in e_raw.transfer_axes],
                                    "E_A": [(x.basis, x.execution_mode, x.confidence, x.scientific_class) for x in e_a.transfer_axes]},
                   "arms": {}}
            for arm, fields, ep in (("R", raw, e_raw), ("A1", a_fields, e_raw), ("A2", a_fields, e_a)):
                r = RL.run_plan_checked(backend, fields, f"optA_{cid}_{draw}_{arm}", targets, expert_prior=ep)
                rec["arms"][arm] = r
                tt = r["targets"]
                summ = {k[:16]: (v["final_rank"], "+".join(x for x, f in (("flat", "in_flat"), ("3A", "in_3a"), ("3B", "in_3b")) if v[f]) or "-")
                        for k, v in tt.items()}
                u = r["api_usage_all_attempts"]
                print(f"  draw{draw} {arm:2s}: {'DEGRADED ' if r['degraded'] else ''}{summ}   [{u['http_requests']} HTTP, {u['cache_hits']} cache hits, {u['credits_charged']} credits]", flush=True)
                (out_dir / f"case{cid}_draw{draw}.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False, default=str))
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    RL.run_guarded(main)
