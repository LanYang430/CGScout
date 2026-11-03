"""Case 6 canonical-vs-raw diagnosis report (reads results/diag6_rec*.json; no network, no LLM).

Arms (per record, same ScientificPlanner output): R = raw plan + E_raw; C1 = canonical plan + E_raw
(classes held fixed -> isolates the objective TEXT); C2 = canonical plan + E_can (full canonical path,
adds the effect of ExpertPrior seeing different input). Loss attribution at depth 20:
  R hit, C1 miss            -> lost at the QUERY TEXT built from the canonical objective (E2)
  R hit, C1 hit, C2 miss    -> lost at the EXPERT-PRIOR CLASS choice (E1)
  R miss                    -> raw already misses: no canonical regression to explain (boundary/class luck)
Runs flagged degraded (unresolved failed OpenAlex requests) are excluded, never counted as misses.
"""
import glob, json, sys
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"


def arm(rec, name, depth):
    return next((a for a in rec["arms"] if a["arm"] == name and a["depth"] == depth and "error" not in a), None)


def best_query_rank(a, depth):
    """Per 3A query: target rank in that query's cached result list at this arm's depth (None = cached, absent)."""
    out = []
    for q in next(iter(a["query_target_ranks"].values())):
        if q["route"] != "3A":
            continue
        hit = next((c for c in q["cached"] if c["mode"] == "semantic" and c["limit"] == depth), None)
        out.append((q["query"], hit["rank"] if hit else "n/c"))
    return out


def cell(a):
    if a is None:
        return "  -  "
    if a.get("degraded"):
        return "DEGR "
    return f"{a['rank']}" if a["found"] else "miss"


def main():
    recs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(RES / "diag6_rec*.json")))]
    print("CASE 6 (protein-RNA vesicles) -- final rank of the target after Stage 3; 'miss' = never in the pool\n")
    print(f"{'rec':>3} | {'R@20':>5} {'C1@20':>6} {'C2@20':>6} | {'R@50':>5} {'C1@50':>6} {'C2@50':>6} | route(s) of hits")
    attrib = {"E2 (query text)": 0, "E1 (expert-prior classes)": 0, "raw already misses": 0, "both hit / no loss": 0, "not comparable (degraded)": 0}
    for r in recs:
        cells = [arm(r, n, d) for d in (20, 50) for n in ("R", "C1", "C2")]
        routes = sorted({x for a in cells if a and a.get("found") and not a.get("degraded") for x in a["route"]})
        print(f"{r['rec']:>3} | " + " ".join(f"{cell(a):>5}" for a in cells[:3]) + " | " + " ".join(f"{cell(a):>5}" for a in cells[3:]) + f" | {routes}")
        R, C1, C2 = (arm(r, n, 20) for n in ("R", "C1", "C2"))
        if any(a is None or a.get("degraded") for a in (R, C1, C2)):
            attrib["not comparable (degraded)"] += 1
        elif not R["found"]:
            attrib["raw already misses"] += 1
        elif not C1["found"]:
            attrib["E2 (query text)"] += 1
        elif not C2["found"]:
            attrib["E1 (expert-prior classes)"] += 1
        else:
            attrib["both hit / no loss"] += 1
    print("\nloss attribution at depth 20 (records):", attrib)

    print("\n" + "=" * 100 + "\nSTEP-BY-STEP, per record\n" + "=" * 100)
    for r in recs:
        print(f"\n--- record {r['rec']} (planner rep {r['rep']})")
        print("  [1] planner -> canonical text")
        print(f"      raw       system={r['raw_plan']['system']!r} context={r['raw_plan']['search_context']}")
        print(f"                objective={r['raw_objective']!r}")
        print(f"      canonical system={r['canonical_plan']['system']!r} context={r['canonical_plan']['search_context']}")
        print(f"                objective={r['canonical_objective']!r}")
        print("  [2] expert-prior axes (class names)")
        print(f"      E_raw: {[c[2] for c in r['classes']['E_raw']]}")
        print(f"      E_can: {[c[2] for c in r['classes']['E_can']]}")
        for name in ("R", "C1", "C2"):
            a = arm(r, name, 20)
            if not a or a.get("degraded"):
                continue
            print(f"  [3] 3A queries of {name}@20 -> target rank inside each query's own top-20 (n/c = not cached)")
            for q, rk in best_query_rank(a, 20):
                print(f"        rank={str(rk):>4}  {q[:150]}")
            t = a["target"]
            print(f"  [4] {name}@20 result: in_pool={t['in_pool']} flat={t['in_flat']} 3A={t['in_3a']} 3B={t['in_3b']} final_rank={t['final_rank']}  found_by={len(t['found_by_queries'])} quer{'y' if len(t['found_by_queries'])==1 else 'ies'}")
            u = a["api_usage"]
            print(f"      api: {u['http_requests']} HTTP, {u['cache_hits']} cache hits, {u['credits_charged']} credits")


if __name__ == "__main__":
    main()
