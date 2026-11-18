"""OpenAlex Semantic Search baseline on the 200 benchmark queries (diagnostic run; nothing else is run).

Request (exactly the endpoint/param the repo's search_openalex(mode="semantic") uses):
    GET https://api.openalex.org/works?search.semantic=<Latest query (v2)>&per_page=50&select=...
No filters (no year / publication type / CG term), no title / DOI / OpenAlex-ID / seed metadata in the request, no reranking,
no LLM.  Throttle (1 request/s across processes), usage ledger, budget latch and exhaustion detection are the repo's own
(cgscout.sources.openalex_usage).  This script differs from search_openalex() only in that it keeps the RAW response
(rank order, relevance_score, headers) instead of normalising it to Paper objects, which drop the score.

Seed matching happens only AFTER retrieval: OpenAlex ID first, then normalised DOI.  A title-only look-alike in the top 50
(e.g. another version of the same work) is reported separately and NOT counted as a hit.

    python openalex_semantic.py --only B001,B002        # technical check
    python openalex_semantic.py                          # all 200 (cached queries are free)
    python openalex_semantic.py --report-only            # rebuild reports from the cache
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(os.environ.get("CGSCOUT_BENCH_DIR", ".")).resolve()           # run directory
from cgscout.sources import openalex_usage as usage  # noqa: E402

INPUT = Path(os.environ.get("CGSCOUT_QUERIES_FILE", ROOT / "BEAM_200_queries_and_papers_latest.xlsx"))
RAW = ROOT / "raw"
RAW.mkdir(exist_ok=True)
URL = "https://api.openalex.org/works"
SELECT = "id,doi,title,publication_year,type,relevance_score"
TOP_N = 50
KS = (5, 10, 20, 50)
LABEL = "openalex_semantic_v2_baseline"


def norm_doi(d):
    if not d:
        return None
    d = str(d).strip().lower()
    d = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", d)
    return d or None


def norm_oa(i):
    return str(i).replace("https://openalex.org/", "").strip() if i else None


def norm_title(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def fetch(pid: str, query: str, log) -> dict:
    """One semantic search; cached on disk only if HTTP 200.  Returns the stored record."""
    path = RAW / f"{pid}.json"
    if path.exists():
        rec = json.loads(path.read_text())
        rec["from_cache"] = True
        return rec
    params = {"search.semantic": query, "per_page": TOP_N, "select": SELECT}
    sent = dict(params)                                   # what we record (never the key)
    key = os.environ.get("OPENALEX_API_KEY")
    if key:
        params["api_key"] = key
    last_err, last_status = None, None
    for attempt in range(3):
        if usage.exhausted():
            return {"pid": pid, "status": "NOT_EVALUATED", "error": "budget exhausted (latched)", "http_status": 429}
        usage.throttle_semantic()
        t0 = time.time()
        try:
            r = requests.get(URL, params=params, timeout=40)
            el = time.time() - t0
            hdr = {k.lower(): v for k, v in r.headers.items()}
            rl = {k: hdr[k] for k in hdr if k.startswith("x-ratelimit") or k == "retry-after"}
            usage.record_request(mode="semantic", limit=TOP_N, status=r.status_code, headers=rl, elapsed_s=el, query=query)
            log.write(json.dumps({"ts": round(time.time(), 3), "pid": pid, "attempt": attempt, "status": r.status_code,
                                  "sent_ts": round(t0, 3), "elapsed_s": round(el, 3), "ratelimit": rl}) + "\n"); log.flush()
            last_status = r.status_code
            if r.status_code == 200:
                data = r.json()
                rec = {"pid": pid, "status": "OK", "http_status": 200, "query": query, "request_params": sent, "request_url": URL,
                       "retrieved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "elapsed_s": round(el, 3),
                       "ratelimit_headers": rl, "meta": data.get("meta"), "results": data.get("results", [])}
                path.write_text(json.dumps(rec, ensure_ascii=False))
                rec["from_cache"] = False
                return rec
            if r.status_code == 429 and usage.is_budget_exhausted_response(rl):
                usage.mark_exhausted(rl, f"{pid}")
                return {"pid": pid, "status": "NOT_EVALUATED", "error": "HTTP 429 budget exhausted", "http_status": 429}
            time.sleep(2 ** attempt)
        except requests.RequestException as e:
            last_err = str(e)[:200]
            usage.record_request(mode="semantic", limit=TOP_N, status=None, headers={}, elapsed_s=time.time() - t0, query=query, error=last_err)
            log.write(json.dumps({"ts": round(time.time(), 3), "pid": pid, "attempt": attempt, "status": None, "error": last_err}) + "\n")
            time.sleep(2 ** attempt)
    return {"pid": pid, "status": "NOT_EVALUATED", "error": last_err or f"HTTP {last_status}", "http_status": last_status}


def match(rec: dict, row: pd.Series) -> dict:
    seed_oa, seed_doi, seed_title = norm_oa(row["OpenAlex ID"]), norm_doi(row["DOI"]), norm_title(row["Paper title"])
    rank, by, title_only = None, None, None
    for i, w in enumerate(rec["results"], 1):
        if rank is None and seed_oa and norm_oa(w.get("id")) == seed_oa:
            rank, by = i, "openalex_id"
        if rank is None and seed_doi and norm_doi(w.get("doi")) == seed_doi:
            rank, by = i, "doi"
        if rank is None and title_only is None and seed_title and norm_title(w.get("title")) == seed_title:
            title_only = i
    return {"seed_rank": rank, "matched_by": by, "title_only_rank": title_only if rank is None else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only"); ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    df = pd.read_excel(INPUT, sheet_name="Latest 200", dtype=str).fillna("")
    assert len(df) == int(os.environ.get("CGSCOUT_EXPECT_ROWS", "200")) and df["Paper ID"].is_unique
    sha = hashlib.sha256(INPUT.read_bytes()).hexdigest()
    os.environ.setdefault("OPENALEX_STRICT_BUDGET", "0")      # exhaustion becomes NOT_EVALUATED, never a silent miss
    usage.set_run_label(LABEL)
    todo = df if not a.only else df[df["Paper ID"].isin(a.only.split(","))]
    log = open(ROOT / "request_log.jsonl", "a")
    t_start = time.time()
    if not a.report_only:
        uncached = [p for p in todo["Paper ID"] if not (RAW / f"{p}.json").exists()]
        need = len(uncached) * usage.CREDITS_PER_SEARCH_DEFAULT
        st = usage.budget_status()                              # one live check (10 credits) unless a recent response is known
        print(f"budget: state={st['state']} remaining={st.get('remaining_credits')} limit={st.get('limit_credits')} authenticated={st.get('authenticated')}; "
              f"{len(uncached)} uncached searches -> ~{need} credits", flush=True)
        if st["state"] != "ok" or (st.get("remaining_credits") is not None and st["remaining_credits"] < need):
            sys.exit(f"STOP: budget not sufficient ({st}); nothing was run.")
        for k, (_, row) in enumerate(todo.iterrows(), 1):
            rec = fetch(row["Paper ID"], row["Latest query (v2)"], log)
            if rec["status"] != "OK":
                print(f"  {row['Paper ID']} NOT_EVALUATED: {rec.get('error')}", flush=True)
            elif k % 20 == 0 or a.only:
                print(f"  [{k}/{len(todo)}] {row['Paper ID']} ok{' (cache)' if rec.get('from_cache') else ''}", flush=True)
    log.close()

    # ------------------------------------------------ reports (always rebuilt from raw/)
    rows, flat = [], []
    for _, row in df.iterrows():
        p = RAW / f"{row['Paper ID']}.json"
        base = {"Paper ID": row["Paper ID"], "Provisional level": row["Provisional level"], "Query mode": row["Query mode"],
                "Latest query (v2)": row["Latest query (v2)"], "Seed title": row["Paper title"], "Seed DOI": row["DOI"], "Seed OpenAlex ID": row["OpenAlex ID"]}
        if not p.exists():
            rows.append({**base, "Status": "NOT_EVALUATED", **{f"hit@{k}": None for k in KS}, "Seed rank (1-50)": None, "Matched by": "", "Title-only look-alike rank": None,
                         "Top-10 (rank | OpenAlex ID | DOI | title)": ""}); continue
        rec = json.loads(p.read_text())
        m = match(rec, row)
        r = m["seed_rank"]
        top10 = " || ".join(f"{i}. {norm_oa(w.get('id'))} | {norm_doi(w.get('doi')) or '-'} | {(w.get('title') or '')[:90]}" for i, w in enumerate(rec["results"][:10], 1))
        rows.append({**base, "Status": "EVALUATED", **{f"hit@{k}": int(r is not None and r <= k) for k in KS}, "Seed rank (1-50)": r if r else "NOT_IN_TOP50",
                     "Matched by": m["matched_by"] or "", "Title-only look-alike rank": m["title_only_rank"], "Results returned": len(rec["results"]),
                     "Top-10 (rank | OpenAlex ID | DOI | title)": top10})
        for i, w in enumerate(rec["results"], 1):
            flat.append({"Paper ID": row["Paper ID"], "rank": i, "OpenAlex ID": norm_oa(w.get("id")), "DOI": norm_doi(w.get("doi")), "title": w.get("title"),
                         "year": w.get("publication_year"), "type": w.get("type"), "relevance_score": w.get("relevance_score"),
                         "is_seed": int(r == i)})
    res = pd.DataFrame(rows)
    ev = res[res["Status"] == "EVALUATED"]
    N, nn = len(res), len(ev)

    def rate(d, k):
        return (d[f"hit@{k}"].sum() / len(d)) if len(d) else float("nan")

    lv_order = ["L1", "L2", "L1/L2", "L1/L2_PROPOSED_RECLASSIFIED_FROM_L3", "L3_PROVISIONAL"]
    perlevel = []
    for g, d in [("ALL", ev)] + [(l, ev[ev["Provisional level"] == l]) for l in lv_order] + [(f"mode={m}", ev[ev["Query mode"] == m]) for m in ("SOURCE_X", "TARGET_Y", "REVIEWED_DRAFT")]:
        perlevel.append({"Group": g, "Evaluated (denominator)": len(d), "Total in group": int((res["Provisional level"] == g).sum()) if g in lv_order else (N if g == "ALL" else int((res["Query mode"] == g.split('=')[1]).sum())),
                         **{f"Success@{k}": round(rate(d, k), 4) if len(d) else None for k in KS}, **{f"hits@{k}": int(d[f"hit@{k}"].sum()) for k in KS},
                         "NOT_IN_TOP50": int((d["Seed rank (1-50)"] == "NOT_IN_TOP50").sum())})
    per_level = pd.DataFrame(perlevel)
    ledger = usage.run_usage(LABEL)
    meta = {"input_file": str(INPUT), "input_sha256": sha, "sheet": "Latest 200", "query_column": "Latest query (v2)",
            "endpoint": URL, "param": "search.semantic", "per_page": TOP_N, "select": SELECT, "filters": "none", "reranking": "none",
            "throttle_s": usage.SEMANTIC_MIN_INTERVAL_S, "python": platform.python_version(), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "report_built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "wall_seconds_this_invocation": round(time.time() - t_start, 1),
            "evaluated": nn, "not_evaluated": N - nn, "usage_this_invocation": ledger, "usage_today_this_machine": usage.used_today_utc(),
            "raw_files": len(list(RAW.glob("*.json"))),
            "retrieved_at_utc_range": [min((json.loads(p.read_text()).get("retrieved_at_utc", "") for p in RAW.glob("*.json")), default=""),
                                       max((json.loads(p.read_text()).get("retrieved_at_utc", "") for p in RAW.glob("*.json")), default="")]}
    fails = int(((res["Status"] != "EVALUATED")).sum())
    overall = {"N total": N, "Evaluated": nn, "Not evaluated": N - nn, **{f"Success@{k}": round(rate(ev, k), 4) for k in KS},
               **{f"hits@{k}": int(ev[f"hit@{k}"].sum()) for k in KS}, "NOT_IN_TOP50": int((ev["Seed rank (1-50)"] == "NOT_IN_TOP50").sum()), "API failures (queries)": fails,
               "title-only look-alikes in top50 (not counted)": int(ev["Title-only look-alike rank"].notna().sum()), "matched by openalex_id": int((ev["Matched by"] == "openalex_id").sum()),
               "matched by doi only": int((ev["Matched by"] == "doi").sum())}
    out = ROOT / "BEAM_openalex_semantic_v2_results.xlsx"
    summary_df = pd.DataFrame(list(overall.items()), columns=["Metric", "Value"])
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        summary_df.to_excel(w, sheet_name="Overall summary", index=False)
        per_level.to_excel(w, sheet_name="Per-level", index=False)
        res.to_excel(w, sheet_name="Per-query (200)", index=False)
        pd.DataFrame(flat).to_excel(w, sheet_name="Top50 raw (flat)", index=False)
        pd.DataFrame(list(meta.items()), columns=["Key", "Value"]).astype(str).to_excel(w, sheet_name="Run metadata", index=False)
        for ws in w.book.worksheets:
            ws.freeze_panes = "A2"
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = min(60, max(10, max(len(str(c.value or "")) for c in list(col)[:100]) + 2))
    res.to_csv(ROOT / "per_query_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(flat).to_csv(ROOT / "top50_raw_flat.csv", index=False, encoding="utf-8-sig")
    per_level.to_csv(ROOT / "per_level_summary.csv", index=False)
    (ROOT / "run_metadata.json").write_text(json.dumps({**meta, "overall": overall}, indent=1, ensure_ascii=False, default=str))
    print(json.dumps(overall, indent=1)); print(per_level.to_string(index=False))


if __name__ == "__main__":
    main()
