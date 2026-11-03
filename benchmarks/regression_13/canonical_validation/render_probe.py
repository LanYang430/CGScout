"""Isolation probe: same scientific_class, same record, only the OBJECTIVE
RENDERING differs. No LLM calls. For every (record, scientific_class) seen in
the retrieval runs, build the 3 production 3A templates
(direct_concept_semantic_intent.render_direct_concept_queries) from:
  raw      : the Planner's original objective sentence
  labeled  : the current canonical render ("...; goal: ...; constraints: ...")
  natural  : canonical fields recomposed as a plain sentence
             (goal, then constraints, comma-joined; no labels/parentheses)
and report the best OpenAlex rank@20 of the target across the 3 templates.
"""
import sys, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import regression_lib as RL  # (also puts mellea_integration + repo paths on sys.path)
sys.path.insert(0, str(HERE.parents[2]))
from direct_concept_semantic_intent import render_direct_concept_queries  # noqa: E402
from beam_agent.sources.openalex import search_openalex  # noqa: E402

import os
LIMIT = int(os.environ.get("PROBE_LIMIT", "20"))
RES = HERE / "results"
TARGETS = {"case6": ["10.1021/jacsau.6c00322"],
           "case9": ["10.7554/elife.105565", "10.1101/2024.05.26.595895"]}


def natural_objective(canon):
    goal = canon.get("modeling_goal") or ""
    parts = [goal] if goal else [p.get("qualifier") or p["head"].replace("_", " ") for p in canon["process"]]
    parts += canon.get("constraints") or []
    return ", ".join(x for x in parts if x)


def natural_entity_objective(canon, norm):
    """natural sentence + the subject re-attached: '<goal> of <system>, <constraints>'"""
    goal = canon.get("modeling_goal") or ", ".join(p.get("qualifier") or p["head"].replace("_", " ") for p in canon["process"])
    subj = norm["system_text"]
    base = goal if subj.lower() in goal.lower() else f"{goal} of {subj}"
    return ", ".join([base] + (canon.get("constraints") or []))


def best_rank(queries, dois):
    best = None
    for q in queries:
        try:
            papers = search_openalex(q, limit=LIMIT, mode="semantic")
        except Exception as e:  # noqa: BLE001
            print("   search error:", e); continue
        for i, p in enumerate(papers, 1):
            if any(d.lower() in (getattr(p, "doi", None) or "").lower() for d in dois):
                best = i if best is None else min(best, i)
                break
    return best


def main():
    RL.harness_guard(200, what="render_probe")   # direct search_openalex calls: cached by the client, budget-checked here
    canon = json.loads((RES / "co_canon.json").read_text())
    tally = {}
    for case, short in (("case6", "case6_protein_rna_vesicles"), ("case9", "case9_protein_dna")):
        runs = json.loads((RES / f"co_retrieval_C_{case}.json").read_text())
        rows = {r["rep"]: r for r in canon[short] if r["status"] == "ok"}
        seen = set()
        print(f"\n===== {case}")
        for r in runs:
            if "error" in r: continue
            row = rows[r["rep"]]
            for a in r["expert_prior"]["axes"]:
                cls = a["scientific_class"]
                if (r["rep"], cls) in seen or a["execution_mode"] != "direct_concept_search":
                    continue
                seen.add((r["rep"], cls))
                objs = {"raw": row["raw"]["scientific_objective"],
                        "labeled": row["C"]["rendered"]["scientific_objective"],
                        "natural": natural_objective(row["C"]["canon"]),
                        "nat+subj": natural_entity_objective(row["C"]["canon"], row["norm"])}
                res = {k: best_rank(render_direct_concept_queries(cls, v), TARGETS[case]) for k, v in objs.items()}
                for k, v in res.items():
                    tally.setdefault((case, k), []).append(v)
                print(f" rep{r['rep']} class={cls!r:60s} raw={res['raw']} labeled={res['labeled']} natural={res['natural']} nat+subj={res['nat+subj']}")
        print(f" nat+subj example (rep{min(rows)}): {natural_entity_objective(rows[min(rows)]['C']['canon'], rows[min(rows)]['norm'])!r}")
    print("\n===== SUMMARY: best-template rank@20 (None=miss), count of (class,record) combos hit")
    for (case, k), v in sorted(tally.items()):
        h20 = sum(x is not None and x <= 20 for x in v)
        print(f" {case} {k:8s}: hit@20 {h20}/{len(v)}  hit@{LIMIT} {sum(x is not None for x in v)}/{len(v)}   ranks={v}")


if __name__ == "__main__":
    RL.run_guarded(main)
