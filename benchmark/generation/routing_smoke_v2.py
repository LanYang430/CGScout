"""Re-run of the routing smoke test AFTER the 3B generalization (axis-aligned relationship validation, concrete-entity
grounding, abstract ExpertPrior classes), on the CURRENT production code.  Q15/16/18/19/20 + the old Pertactin/EspP 3B
regression cases (regression_13 cases 11, 12).  In this process only: Stage 3 pass-through, canonical-method verification
skipped, no Stage 4.  Three layers are reported separately:
  L1 discovery produces concrete systems   (raw extracted mentions -> kept by the membership check)
  L2 they survive grounding/identity and Hop-2 runs
  L3 Hop-2 brings back the known seed / targets
One draw per case.  Output: data/routing_smoke_v2.json"""
from __future__ import annotations

import json
import re
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_backend  # noqa: E402,F401
import regression_lib as RL  # noqa: E402
import cgscout.retrieval.orchestrator as TB  # noqa: E402
import cgscout.retrieval.related_systems.extraction as TE  # noqa: E402
from cgscout.planning.session import PlannerSession  # noqa: E402
from cgscout.llm import get_backend  # noqa: E402

L3_PICKS = {"Q15": 1, "Q16": 2, "Q18": 1, "Q19": 1, "Q20": 2}
ACCEPT = {
    "Q15": r"\bMscS\b|\bMscL\b|\bMSL\d*\b|small[- ]conductance",
    "Q16": r"\b[357]0S\b|bacterial ribosom|(E\. ?coli|Escherichia coli|Thermus|bacterial)[^|]{0,40}ribosom",
    "Q18": r"flagell",
    "Q19": r"RNA polymerase|\bPol II\b|polymerase II|\bRNAP\b|sigma|σ|TFIIB|\bTBP\b|\bTFB\b|holoenzyme",
    "Q20": r"dynamin|\bDrp1\b|DNM1L|\bOPA1\b|mitofusin|\bMfn\d?\b|\bFzo1\b|atlastin",
    "C11": r"EspP|\bPet\b|autotransporter|Hbp|pertactin",
    "C12": r"pertactin|autotransporter|\bPet\b|Hbp|EspP",
}


def main():
    pilot = {r["case_id"]: r for r in map(json.loads, (HERE / "data" / "pilot_v1.jsonl").read_text().splitlines())}
    seeds = {c["case_id"]: c for c in json.loads((HERE / "data" / "pilot_seeds_final.json").read_text())}
    cases79 = {c["id"]: c for c in json.loads((HERE / "data" / "regression_13_cases.json").read_text())["cases"]}
    jobs = [(cid, pilot[cid]["query_candidates"][k - 1], [seeds[cid]["doi"]], "L3") for cid, k in L3_PICKS.items()]
    jobs += [(f"C{i}", cases79[i]["query"], [d for t in cases79[i]["targets"].values() for d in t["dois"]], "regression") for i in (11, 12)]

    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    repeat = int(sys.argv[sys.argv.index("--repeat") + 1]) if "--repeat" in sys.argv else 1
    suffix = sys.argv[sys.argv.index("--suffix") + 1] if "--suffix" in sys.argv else ""
    if only:
        jobs = [j for j in jobs if j[0] in only.split(",")] * repeat
    TB.filter_papers = lambda papers, **kw: (list(papers), [], [])                  # Stage 3 pass-through (this process only)
    log, lock = [], threading.Lock()
    orig_member = TE._membership_is_grounded

    def counting(m, title, abstract):                                              # instrument the new validation (read-only)
        ok = orig_member(m, title, abstract)
        with lock:
            log.append({"name": m.name, "type": m.entity_type, "kept": ok, "membership_quote": m.class_membership_quote})
        return ok
    TE._membership_is_grounded = counting
    backend = get_backend()
    out = []

    def go():
        RL.harness_guard(len(jobs) * 14, what="bench_v1_routing_smoke_v2")
        for cid, q, dois, kind in jobs:
            label = f"routing_v2_{cid}{suffix}_{int(time.time())}"
            RL.U.set_run_label(label)
            with lock:
                log.clear()
            t0 = time.time()
            row = {"case": cid, "kind": kind, "query": q, "target_dois": dois}
            sess = PlannerSession(backend=backend, case_id=label)
            turn = sess.submit_user_message(q)
            plan = turn.plan
            row["planner"] = {"status": turn.status, "system": plan.system, "scientific_objective": plan.scientific_objective, "clarification_question": plan.clarification_question}
            if turn.status != "ready":
                out.append(row); print(cid, "CLARIFICATION", flush=True); continue
            fields = {"system": plan.system, "target_component": plan.target_component, "scientific_objective": plan.scientific_objective, "search_context": plan.search_context}
            ep = RL.propose(backend, fields, label)
            row["expert_prior_axes"] = [{"basis": a.basis, "mode": a.execution_mode, "confidence": a.confidence, "scientific_class": a.scientific_class, "anchors": a.retrieval_anchors} for a in ep.transfer_axes]
            RL.U.reset_run(label); RL.U.set_run_label(label)
            res = TB.run_expert_prior_retrieval(RL.make_plan(fields), ep, backend=backend, case_id=label)
            tb = [l for l in res.axis_logs if l.axis.execution_mode == "related_system_discovery" and l.executed]
            row["three_b_triggered"] = bool(tb)
            row["axis_logs_3b"] = [{"basis": l.axis.basis, "class": l.axis.scientific_class, "hop1_queries": l.hop1_queries, "hop2_queries": l.hop2_queries,
                                    "identities": [{"label": i.label, "grounding": i.grounding, "type": i.entity_type, "anchor": i.query_anchor} for i in l.related_system_identities],
                                    "n_papers": l.n_papers_found, "error": l.error, "hop2_plan": getattr(l, "hop2_plan", [])} for l in tb]
            with lock:
                ev = list(log)
            raw_names = sorted({e["name"] for e in ev}); kept_names = sorted({e["name"] for e in ev if e["kept"]})
            idents = [i for l in row["axis_logs_3b"] for i in l["identities"]]
            grounded = [i for i in idents if i["grounding"] != "insufficient"]
            rx = re.compile(ACCEPT[cid], re.I)
            row["layer1_discovery"] = {"raw_mentions": len(ev), "raw_unique_systems": raw_names, "after_membership_validation": len(kept_names), "kept_systems": kept_names,
                                       "dropped_by_validation": sorted({e["name"] for e in ev if not e["kept"]})}
            row["layer2_grounding"] = {"identities_total": len(idents), "identities_grounded": len(grounded), "grounded": [(i["label"], i["grounding"]) for i in idents],
                                       "hop2_ran": any(l["hop2_queries"] for l in row["axis_logs_3b"]), "hop2_queries": [q for l in row["axis_logs_3b"] for q in l["hop2_queries"]]}
            row["acceptable_X"] = {"raw": [n for n in raw_names if rx.search(n)], "after_validation": [n for n in kept_names if rx.search(n)], "after_grounding": [i["label"] for i in grounded if rx.search(i["label"])]}
            flat = [p for p in res.direct_papers if any((p.doi or "").lower() == d.lower() for d in dois)]
            trans = [p for p in res.transfer_papers if any((p.doi or "").lower() == d.lower() for d in dois)]
            row["layer3_seed"] = {"found_by_3B": sorted({(p.doi or "").lower() for p in trans if "structural" in (p.query_categories or [])}),
                                  "found_by_3A": sorted({(p.doi or "").lower() for p in trans if "transfer_direct" in (p.query_categories or [])}),
                                  "found_by_flat": sorted({(p.doi or "").lower() for p in flat}), "pool": {"direct": len(res.direct_papers), "transfer": len(res.transfer_papers)}}
            row["api_usage"] = RL.U.run_usage(label); row["seconds"] = round(time.time() - t0, 1)
            out.append(row)
            print(f"{cid}: 3B={row['three_b_triggered']} raw={len(ev)}→kept {len(kept_names)}→grounded {len(grounded)} hop2={row['layer2_grounding']['hop2_ran']} 3B-found={row['layer3_seed']['found_by_3B']}", flush=True)
            (HERE / "data" / f"routing_smoke_v2{suffix}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    RL.run_guarded(go)
    (HERE / "data" / f"routing_smoke_v2{suffix}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
