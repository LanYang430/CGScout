"""Routing smoke test for the six hand-built L3 queries (Q15-Q20): does the CURRENT BEAM route them to
related_system_discovery (3B)?  Runs the unmodified Planner -> ExpertPrior -> run_expert_prior_retrieval.  In THIS process only:
Stage 3 is a pass-through (no classifier / embedding calls) and canonical-method verification is skipped -- both are
irrelevant to routing and cost credits.  No Stage 4, no OpenAlex baseline, no routing logic touched.  One draw per case.
Everything BEAM-side is read-only; results go to data/routing_smoke.json."""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_backend  # noqa: E402,F401
import regression_lib as RL  # noqa: E402
import cgscout.retrieval.orchestrator as TB  # noqa: E402
from cgscout.planning.session import PlannerSession  # noqa: E402
from cgscout.llm import get_backend  # noqa: E402

PICKS = {"Q15": 1, "Q16": 2, "Q17": 2, "Q18": 1, "Q19": 1, "Q20": 2}
# the pre-designed source system X, plus names that would count as a scientifically equivalent related system
EXPECTED = {
    "Q15": ("MscS", ["mscs", "mscl", "small conductance", "mechanosensitive channel of small", "msc"]),
    "Q16": ("E. coli / bacterial 70S ribosome", ["e. coli ribosome", "escherichia coli ribosome", "70s", "bacterial ribosome", "thermus thermophilus ribosome", "e. coli 70s"]),
    "Q17": ("YidC", ["yidc", "oxa1", "insertase", "oxa"]),
    "Q18": ("bacterial flagellum", ["flagell", "flagellin", "hook"]),
    "Q19": ("Pol II / homologous transcription machinery", ["pol ii", "polymerase ii", "rna polymerase", "tbp", "tfiib", "general transcription factor", "preinitiation"]),
    "Q20": ("dynamin", ["dynamin", "drp1", "opa1", "mitofusin", "mfn", "fzo1", "atlastin", "dnm1"]),
}
SEEDS = {c["case_id"]: c for c in json.loads((HERE / "data" / "pilot_seeds_final.json").read_text())}


def main():
    recs = {r["case_id"]: r for r in map(json.loads, (HERE / "data" / "pilot_v1.jsonl").read_text().splitlines())}
    TB.filter_papers = lambda papers, **kw: (list(papers), [], [])                  # Stage 3 pass-through (this process only)
    backend = get_backend()
    out = []

    def go():
        RL.harness_guard(6 * 14, what="bench_v1_routing_smoke")
        RL.U.set_run_label("bench_v1_routing_smoke")
        for cid, k in PICKS.items():
            seed, rec = SEEDS[f"Q{int(cid[1:]):02d}" if False else cid], recs[cid]
            q = rec["query_candidates"][k - 1]
            label = f"routing_smoke_{cid}"
            t0 = time.time()
            row = {"case": cid, "candidate": k, "query": q, "seed_doi": seed["doi"], "expected_X": EXPECTED[cid][0]}
            sess = PlannerSession(backend=backend, case_id=label)
            turn = sess.submit_user_message(q)
            plan = turn.plan
            row["planner"] = {"status": turn.status, "system": plan.system, "target_component": plan.target_component,
                              "scientific_objective": plan.scientific_objective, "search_context": plan.search_context,
                              "clarification_question": plan.clarification_question}
            if turn.status != "ready":
                row["note"] = "planner asked for clarification -- not run further"
                out.append(row); print(cid, "CLARIFICATION", flush=True); continue
            ep = RL.propose(backend, {"system": plan.system, "target_component": plan.target_component,
                                      "scientific_objective": plan.scientific_objective, "search_context": plan.search_context}, label)
            row["expert_prior"] = {"transfer_needed": ep.transfer_needed,
                                   "axes": [{"basis": a.basis, "scientific_class": a.scientific_class, "confidence": a.confidence,
                                             "execution_mode": a.execution_mode} for a in ep.transfer_axes]}
            RL.U.reset_run(label); RL.U.set_run_label(label)
            res = TB.run_expert_prior_retrieval(RL.make_plan({"system": plan.system, "target_component": plan.target_component,
                                                              "scientific_objective": plan.scientific_objective, "search_context": plan.search_context}),
                                                ep, backend=backend, case_id=label)
            logs = []
            for l in res.axis_logs:
                logs.append({"basis": l.axis.basis, "mode": l.axis.execution_mode, "confidence": l.axis.confidence, "executed": l.executed,
                             "skip_reason": l.skip_reason, "error": l.error, "n_papers": l.n_papers_found,
                             "direct_queries": l.direct_queries, "hop1_queries": l.hop1_queries, "hop2_queries": l.hop2_queries,
                             "related_systems": [{"label": i.label, "grounding": i.grounding, "query_anchor": i.query_anchor, "n_support": i.n_support}
                                                 for i in l.related_system_identities]})
            row["axis_logs"] = logs
            tb = [l for l in logs if l["mode"] == "related_system_discovery" and l["executed"]]
            row["three_b_triggered"] = bool(tb)
            hop1 = [s["label"] for l in tb for s in l["related_systems"]]
            row["hop1_related_systems"] = hop1
            names, syn = " | ".join(hop1).lower(), EXPECTED[cid][1]
            row["expected_X_exact"] = any(sy in names for sy in syn if len(sy) >= 4 and sy not in ("tbp",))      # synonym match only; a generic label is not X
            row["equivalent_system_matches"] = sorted({s for s in syn if s in names})
            doi = seed["doi"].lower()
            flat = [p for p in res.direct_papers if (p.doi or "").lower() == doi]
            trans = [p for p in res.transfer_papers if (p.doi or "").lower() == doi]
            row["seed_in_pool"] = {"flat": bool(flat), "3A": any("transfer_direct" in (p.query_categories or []) for p in trans),
                                   "3B": any("structural" in (p.query_categories or []) for p in trans)}
            row["seed_found_by_queries"] = sorted({qq for p in flat + trans for qq in (p.found_by_queries or [])})
            ranks = []
            for l in tb:
                for qq in l["hop1_queries"] + l["hop2_queries"]:
                    for c in RL.cached_target_ranks(qq, [doi]):
                        if c["rank"] is not None:
                            ranks.append({"query": qq[:140], "mode": c["mode"], "depth": c["limit"], "rank": c["rank"]})
            row["seed_rank_in_3b_queries"] = ranks
            row["pool_sizes"] = {"direct_papers": len(res.direct_papers), "transfer_papers": len(res.transfer_papers)}
            row["api_usage"] = RL.U.run_usage(label)
            row["seconds"] = round(time.time() - t0, 1)
            out.append(row)
            print(f"{cid}: 3B={row['three_b_triggered']} modes={[a['execution_mode'] for a in row['expert_prior']['axes']]} hop1={len(hop1)} seed_in_pool={row['seed_in_pool']}", flush=True)
            (HERE / "data" / "routing_smoke.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
        from generate import cost_since
    RL.run_guarded(go)
    (HERE / "data" / "routing_smoke.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
