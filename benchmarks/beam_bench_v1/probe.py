"""OpenAlex semantic probe for the queries the human selected.  For each row of the pilot CSV that has a
`selected_query` (or a `selected_candidate` 1-3), run the raw query once through OpenAlex semantic search (depth 50)
and record the rank of the seed DOI.  The rank is only a DESCRIPTION of empirical difficulty: it never keeps, drops
or re-labels a case (L1/L2/L3 come from how the case was constructed).
Usage: probe.py [--in data/pilot_v0.csv] [--out data/pilot_v0_probed]"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_backend  # noqa: E402,F401
from generate import CSV_COLUMNS  # noqa: E402

DEPTH = 50
BANDS = [(5, "1-5"), (20, "6-20"), (DEPTH, "21-50")]


def chosen_query(row: dict) -> str:
    q = (row.get("selected_query") or "").strip()
    if q:
        return q
    k = (row.get("selected_candidate") or "").strip()
    return (row.get(f"query_candidate_{k}") or "").strip() if k in ("1", "2", "3") else ""


def rank_of(doi: str, papers) -> int | None:
    d = doi.lower()
    return next((i for i, p in enumerate(papers, 1) if (getattr(p, "doi", None) or "").lower() == d), None)


def band(rank: int | None) -> str:
    if rank is None:
        return f"miss(>{DEPTH})"
    return next(label for cap, label in BANDS if rank <= cap)


def probe_rows(rows: list[dict], search_fn) -> list[dict]:
    out = []
    for r in rows:
        r = dict(r)
        q = chosen_query(r)
        if q:
            rk = rank_of(r["seed_doi"], search_fn(q, DEPTH))
            r["selected_query"] = q
            r["openalex_semantic_rank"] = "" if rk is None else rk
            r["openalex_semantic_hit"] = rk is not None
            r["openalex_rank_band"] = band(rk)
        else:
            r["openalex_semantic_hit"] = r["openalex_rank_band"] = ""
        out.append(r)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(HERE / "data" / "pilot_v0.csv"))
    ap.add_argument("--out", default=str(HERE / "data" / "pilot_v0_probed")); a = ap.parse_args()
    sys.path.insert(0, str(HERE.parent / "regression_13"))
    import regression_lib as RL
    from beam_agent.sources.openalex import search_openalex
    rows = list(csv.DictReader(open(a.inp, encoding="utf-8")))
    todo = [r for r in rows if chosen_query(r)]
    if not todo:
        sys.exit("no row has a selected_query / selected_candidate yet -- fill the CSV first")

    def go():
        RL.harness_guard(len(todo), what="bench_v1_probe")
        RL.U.set_run_label("bench_v1_probe")
        probed = probe_rows(rows, lambda q, k: search_openalex(q, limit=k, mode="semantic"))
        cols = CSV_COLUMNS + ["openalex_semantic_hit", "openalex_rank_band"]
        out = Path(a.out)
        with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader(); w.writerows(probed)
        out.with_suffix(".jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in probed) + "\n")
        for r in probed:
            if r.get("openalex_rank_band") != "":
                print(f"{r['case_id']} {r['case_type']} rank={r['openalex_semantic_rank'] or '-'} band={r['openalex_rank_band']}")
        print("OpenAlex usage:", RL.U.run_usage("bench_v1_probe"))
    RL.run_guarded(go)


if __name__ == "__main__":
    main()
