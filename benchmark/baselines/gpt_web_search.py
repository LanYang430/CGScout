"""GPT + built-in Web Search baseline, RUN 1 (budget-limited): gpt-5.6-luna, reasoning effort medium, web_search, tool_choice="auto",
max_tool_calls=3, on the 200 frozen v2 queries.  One stateless Responses-API request per query (no history, no previous_response_id).

Pre-specified rules (fixed before the run):
  PROMPT   the fixed Top-20 prompt below, unchanged from the 13-case pilot.  Input = the frozen v2 query text only.
  CALLS    attempted = every web_search_call item in the response; completed = items with status == "completed" (only these count as searches
           actually performed).  completed > 3 is flagged ANOMALY.  budget_exhausted = completed == 3.
  RETRY    NO retry based on content or seed hit.  A request is repeated (max 3 tries) ONLY if it failed at transport level without any response
           (timeout / 5xx / connection).  OpenAI 400 refusals (e.g. bio_policy) are NOT retried: status REFUSED, excluded from the main
           denominator and reported separately (+ sensitivity: counted as misses).
  FORMAT   no retry; if the parsed list is not exactly ranks 1..20 it is deterministically repaired (sorted/truncated/padded with null slots) and flagged.
  VERIFY   same as the GPT-only baseline (OpenAlex single-DOI lookup, Crossref title lookup); a DOI contained in the cited source_url is used
           when the slot's doi is null.  Hit = verified identity has the seed's OpenAlex ID or DOI.  VERIFY_ERROR slots -> query stays
           NOT_EVALUATED (never silently counted as a miss) until verification can be completed.
  COST     tokens: repo Luna price table; web search: $0.01 per COMPLETED call (user-confirmed list price); an upper bound that also bills
           attempted-but-not-completed calls is reported.  STOP: no new request is started once the (upper-bound) spend >= $12.
  CHECKPOINT  raw/{PID}.json exists  =>  query done, never re-run.  Existing results elsewhere are never touched.

    python gpt_web_search.py [--only B001,B002]      generate (resumable) + verify + report
    python gpt_web_search.py --report-only
  (one script for every query set: the nine earlier copies differed only in the lines made parameters here -- CGSCOUT_QUERIES_FILE, CGSCOUT_EXPECT_ROWS,
   CGSCOUT_STOP_USD; run directory = CGSCOUT_BENCH_DIR.)
"""
from __future__ import annotations

import argparse, concurrent.futures as cf, hashlib, importlib.util, json, os, platform, re, sys, time
from pathlib import Path
from typing import Optional

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel

ROOT = Path(os.environ.get("CGSCOUT_BENCH_DIR", ".")).resolve()           # run directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpt_only as G                                                       # reuse the SAME verifier / matching helpers
RAW, VER, HTTP = ROOT / "raw", ROOT / "verify_cache", ROOT / "http_cache"
for d in (RAW, VER, HTTP):
    d.mkdir(exist_ok=True)
G.HTTP, G.VER = HTTP, VER          # all new cache files go to THIS run's folder

MODEL, EFFORT, MAX_TOOL_CALLS, N_SLOTS = "gpt-5.6-luna", "medium", 3, 20
PRICE = {"input": 0.20, "cached_input": 0.02, "output": 1.20}      # USD per 1M tokens (repo Luna table)
SEARCH_USD = 0.01
STOP_USD = float(os.environ.get("CGSCOUT_STOP_USD", "12.0"))
INPUT = Path(os.environ.get("CGSCOUT_QUERIES_FILE", ROOT / "BEAM_200_queries_and_papers_latest.xlsx"))


class Slot(BaseModel):
    rank: int
    title: Optional[str]
    doi: Optional[str]
    source_url: Optional[str]


class Papers(BaseModel):
    papers: list[Slot]


INSTR = ("You are helping a computational biophysicist find scientific literature. Use web search as you see fit to find papers that would best help with the user's question. "
         "Return exactly 20 ranked slots (rank 1-20), ordered by how useful each paper is for the user's scientific question (NOT by search-result order). "
         "For each slot give the paper's title, its DOI, and a source URL. If you cannot find a reliable paper for a slot, set that slot's title, doi and source_url to null; never invent a paper or a DOI. "
         "Output only the JSON object that matches the schema, no explanation.")
PROMPT_SHA = hashlib.sha256(INSTR.encode()).hexdigest()
client = OpenAI().with_options(timeout=900)


def doi_from_url(u):
    m = re.search(r"10\.\d{4,9}/[^\s?#\"'<>]+", u or "")
    return G.norm_doi(m.group(0)) if m else None


def summarize(resp_dict: dict) -> dict:
    calls = []
    for it in resp_dict.get("output", []):
        if it.get("type") == "web_search_call":
            a = it.get("action") or {}
            calls.append({"id": it.get("id"), "status": it.get("status"), "action": a.get("type"),
                          "query_or_url": a.get("query") or a.get("url"), "n_sources": len(a.get("sources") or [])})
    u = resp_dict.get("usage") or {}
    cached = ((u.get("input_tokens_details") or {}).get("cached_tokens")) or 0
    tin, tout = u.get("input_tokens", 0), u.get("output_tokens", 0)
    tok_cost = max(tin - cached, 0) / 1e6 * PRICE["input"] + cached / 1e6 * PRICE["cached_input"] + tout / 1e6 * PRICE["output"]
    attempted = len(calls)
    completed = sum(1 for c in calls if c["status"] == "completed")
    return {"calls": calls, "attempted": attempted, "completed": completed, "input_tokens": tin, "cached_tokens": cached, "output_tokens": tout,
            "token_cost_usd": round(tok_cost, 6), "search_cost_usd": round(completed * SEARCH_USD, 4),
            "total_cost_usd": round(tok_cost + completed * SEARCH_USD, 6),
            "total_cost_upper_usd": round(tok_cost + attempted * SEARCH_USD, 6)}


def generate(pid: str, query: str) -> dict:
    path = RAW / f"{pid}.json"
    if path.exists():
        return json.loads(path.read_text())
    tries, t_start, rec = [], time.time(), None
    for k in range(3):
        t0 = time.time()
        try:
            resp = client.responses.parse(model=MODEL, instructions=INSTR, input=query, text_format=Papers, reasoning={"effort": EFFORT},
                                          tools=[{"type": "web_search"}], tool_choice="auto", max_tool_calls=MAX_TOOL_CALLS,
                                          include=["web_search_call.action.sources"])
            d = resp.model_dump(mode="json")
            s = summarize(d)
            slots = [x.model_dump() for x in resp.output_parsed.papers] if resp.output_parsed else None
            repaired = ""
            if slots is not None and [x["rank"] for x in slots] != list(range(1, N_SLOTS + 1)):
                by = {}
                for x in sorted(slots, key=lambda z: z["rank"]):
                    if 1 <= x["rank"] <= N_SLOTS and x["rank"] not in by:
                        by[x["rank"]] = x
                slots = [by.get(r, {"rank": r, "title": None, "doi": None, "source_url": None}) for r in range(1, N_SLOTS + 1)]
                repaired = "sorted/truncated/padded by rank"
            rec = {"pid": pid, "query": query, "status": "OK" if slots is not None else "NO_PARSED_OUTPUT", "response_status": d.get("status"),
                   "incomplete_details": d.get("incomplete_details"), "response_id": d.get("id"), "format_repair": repaired,
                   "anomaly_completed_gt_3": s["completed"] > MAX_TOOL_CALLS, "budget_exhausted": s["completed"] >= MAX_TOOL_CALLS,
                   "slots": slots, **s, "seconds": round(time.time() - t0, 1), "tries": len(tries) + 1, "raw_response": d}
            break
        except Exception as e:  # noqa: BLE001
            msg = repr(e)[:400]
            tries.append({"try": k, "error": msg})
            code = getattr(e, "status_code", None)
            if code is not None and 400 <= code < 500:          # content-level refusal / bad request: record, never retry
                rec = {"pid": pid, "query": query, "status": "REFUSED_OR_REJECTED", "error": msg, "http_status": code, "slots": None,
                       "attempted": 0, "completed": 0, "input_tokens": 0, "cached_tokens": 0, "output_tokens": 0, "token_cost_usd": 0.0,
                       "search_cost_usd": 0.0, "total_cost_usd": 0.0, "total_cost_upper_usd": 0.0, "calls": [], "seconds": round(time.time() - t0, 1)}
                break
            time.sleep(5 * (k + 1))
    if rec is None:
        rec = {"pid": pid, "query": query, "status": "TRANSPORT_FAILED", "error": tries[-1]["error"] if tries else "", "slots": None, "attempted": 0, "completed": 0,
               "input_tokens": 0, "cached_tokens": 0, "output_tokens": 0, "token_cost_usd": 0.0, "search_cost_usd": 0.0, "total_cost_usd": 0.0,
               "total_cost_upper_usd": 0.0, "calls": []}
    rec["failed_tries_before"] = tries
    rec["model"], rec["effort"], rec["max_tool_calls"], rec["prompt_sha256"] = MODEL, EFFORT, MAX_TOOL_CALLS, PROMPT_SHA
    rec["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if rec["status"] == "TRANSPORT_FAILED":      # no checkpoint: this query was never answered; may be started again on the next invocation
        (ROOT / "transport_failures.jsonl").open("a").write(json.dumps({"pid": pid, "error": rec["error"], "utc": rec["generated_utc"]}) + "\n")
        return rec
    tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(rec, ensure_ascii=False)); tmp.replace(path)
    return rec


def spent(upper=True) -> float:
    k = "total_cost_upper_usd" if upper else "total_cost_usd"
    return sum(json.loads(p.read_text()).get(k, 0) for p in RAW.glob("*.json"))


def verify(pid: str, rec: dict):
    path = VER / f"{pid}.json"
    if path.exists():
        return json.loads(path.read_text())
    slots = []
    for s in rec["slots"]:
        s2 = dict(s)
        if not s2.get("doi") and s2.get("source_url"):
            s2["doi"] = doi_from_url(s2["source_url"])
        slots.append(s2)
    res = None
    for rnd in range(4):
        res = [G.verify_slot(s) for s in slots] if rnd == 0 else [(v if v["status"] != "VERIFY_ERROR" else G.verify_slot(s)) for v, s in zip(res, slots)]
        if not any(v["status"] == "VERIFY_ERROR" for v in res):
            break
        time.sleep(3 * (rnd + 1))
    if not any(v["status"] == "VERIFY_ERROR" for v in res):
        path.write_text(json.dumps(res, ensure_ascii=False))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only"); ap.add_argument("--report-only", action="store_true"); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    df = (pd.read_csv(INPUT, dtype=str, encoding="utf-8-sig") if INPUT.suffix == ".csv" else pd.read_excel(INPUT, sheet_name="Latest 200", dtype=str)).fillna("")
    assert len(df) == int(os.environ.get("CGSCOUT_EXPECT_ROWS", "200")) and df["Paper ID"].is_unique
    sha = hashlib.sha256(INPUT.read_bytes()).hexdigest()
    todo = df if not a.only else df[df["Paper ID"].isin(a.only.split(","))]
    stopped = None
    if not a.report_only:
        pending = [r for _, r in todo.iterrows() if not (RAW / f"{r['Paper ID']}.json").exists()]
        print(f"{len(pending)} to run; spent so far (upper bound) ${spent():.3f}; stop at ${STOP_USD}", flush=True)
        for b0 in range(0, len(pending), 8):
            if spent() >= STOP_USD:
                stopped = f"stop threshold ${STOP_USD} reached before {pending[b0]['Paper ID']}"
                print("STOP:", stopped, flush=True); break
            with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
                list(ex.map(lambda r: generate(r["Paper ID"], r["Latest query (v2)"]), pending[b0:b0 + 8]))
            print(f"  done {min(b0 + 8, len(pending))}/{len(pending)}  spent est ${spent(False):.3f} (upper ${spent():.3f})", flush=True)
        have = [r for _, r in todo.iterrows() if (RAW / f"{r['Paper ID']}.json").exists()]
        with cf.ThreadPoolExecutor(max_workers=6) as ex:
            list(ex.map(lambda r: (lambda rec: verify(r["Paper ID"], rec) if rec["slots"] else None)(json.loads((RAW / f"{r['Paper ID']}.json").read_text())), have))
        print("verified", len(have), flush=True)
    build_report(df, sha, stopped)


def build_report(df, sha, stopped):
    rows, slots_flat = [], []
    for _, row in df.iterrows():
        pid = row["Paper ID"]
        base = {"Paper ID": pid, "Provisional level": row["Provisional level"], "Query mode": row["Query mode"], "Latest query (v2)": row["Latest query (v2)"],
                "Seed title": row["Paper title"], "Seed DOI": row["DOI"], "Seed OpenAlex ID": row["OpenAlex ID"]}
        if not (RAW / f"{pid}.json").exists():
            rows.append({**base, "Status": "NOT_RUN", "Hit@10": None, "Hit@20": None}); continue
        rec = json.loads((RAW / f"{pid}.json").read_text())
        cm = {"Attempted search calls": rec["attempted"], "Completed search calls": rec["completed"], "Budget exhausted (completed=3)": rec.get("budget_exhausted"),
              "Anomaly completed>3": rec.get("anomaly_completed_gt_3"), "Token cost USD": rec["token_cost_usd"], "Search cost USD (completed)": rec["search_cost_usd"],
              "Total cost USD": rec["total_cost_usd"], "Total cost USD (upper, attempted billed)": rec["total_cost_upper_usd"],
              "Input tokens": rec["input_tokens"], "Output tokens": rec["output_tokens"], "Response status": rec.get("response_status", rec["status"])}
        if rec["status"] != "OK" or not rec["slots"]:
            rows.append({**base, **cm, "Status": rec["status"], "Hit@10": None, "Hit@20": None, "Note": (rec.get("error") or "")[:200]}); continue
        vp = VER / f"{pid}.json"
        if not vp.exists():
            rows.append({**base, **cm, "Status": "NOT_EVALUATED_VERIFICATION_INCOMPLETE", "Hit@10": None, "Hit@20": None}); continue
        ver = json.loads(vp.read_text())
        seed_oa, seed_doi = row["OpenAlex ID"], G.norm_doi(row["DOI"])
        rank, by, bad = None, None, None
        for v, s in zip(ver, rec["slots"]):
            ok = v["status"] in ("VERIFIED_BY_DOI", "VERIFIED_BY_TITLE")
            hit_id = (v.get("oa_id") == seed_oa) if v.get("oa_id") else False
            hit_doi = (v.get("resolved_doi") == seed_doi) if v.get("resolved_doi") else False
            if ok and (hit_id or hit_doi) and rank is None:
                rank, by = v["rank"], ("openalex_id" if hit_id else "doi")
            if (not ok) and (G.norm_doi(s.get("doi")) == seed_doi or doi_from_url(s.get("source_url")) == seed_doi) and bad is None:
                bad = v["rank"]
            slots_flat.append({"Paper ID": pid, "rank": v["rank"], "title": s.get("title"), "stated_doi": s.get("doi"), "source_url": s.get("source_url"),
                               "verification_status": v["status"], "resolved_doi": v.get("resolved_doi"), "oa_id": v.get("oa_id"), "note": v.get("note"),
                               "is_seed": int(rank == v["rank"] and by is not None)})
        st = [v["status"] for v in ver]
        rows.append({**base, **cm, "Status": "EVALUATED", "Hit@10": int(rank is not None and rank <= 10), "Hit@20": int(rank is not None and rank <= 20),
                     "Seed rank (1-20)": rank if rank else "NOT_IN_LIST", "Matched by": by or "", "DOI-equal but identity unverified (not counted) rank": bad,
                     "Non-null slots": sum(x != "EMPTY_SLOT" for x in st), "Verified slots": sum(x.startswith("VERIFIED") for x in st),
                     "Unverified slots": st.count("UNVERIFIED"), "Empty slots": st.count("EMPTY_SLOT"), "Full 20 slots (all non-null)": all(x != "EMPTY_SLOT" for x in st),
                     "Format repair": rec.get("format_repair", "")})
    res = pd.DataFrame(rows); ev = res[res["Status"] == "EVALUATED"]
    lv = ["L1", "L2", "L1/L2", "L1/L2_PROPOSED_RECLASSIFIED_FROM_L3", "L3_PROVISIONAL"]
    groups = [("ALL", ev)] + [(l, ev[ev["Provisional level"] == l]) for l in lv] + [(f"mode={m}", ev[ev["Query mode"] == m]) for m in ("SOURCE_X", "TARGET_Y", "REVIEWED_DRAFT")]
    per_level = pd.DataFrame([{"Group": g, "Evaluated": len(d), "Hit@10": round(d["Hit@10"].mean(), 4) if len(d) else None, "Hit@20": round(d["Hit@20"].mean(), 4) if len(d) else None,
                               "hits@10": int(d["Hit@10"].sum()), "hits@20": int(d["Hit@20"].sum())} for g, d in groups])
    ran = res[res["Status"] != "NOT_RUN"]
    ov = {"N total": len(res), "Evaluated": len(ev), "Refused/rejected (4xx)": int((res["Status"] == "REFUSED_OR_REJECTED").sum()), "Not run": int((res["Status"] == "NOT_RUN").sum()),
          "Verification incomplete": int((res["Status"] == "NOT_EVALUATED_VERIFICATION_INCOMPLETE").sum()),
          "Hit@10": round(ev["Hit@10"].mean(), 4) if len(ev) else None, "Hit@20": round(ev["Hit@20"].mean(), 4) if len(ev) else None,
          "hits@10": int(ev["Hit@10"].sum()) if len(ev) else 0, "hits@20": int(ev["Hit@20"].sum()) if len(ev) else 0,
          "Hit@20 sensitivity (refused counted as miss)": round(ev["Hit@20"].sum() / max(len(ev) + int((res["Status"] == "REFUSED_OR_REJECTED").sum()), 1), 4),
          "queries with completed=3 (budget exhausted)": int(ran["Budget exhausted (completed=3)"].fillna(False).astype(bool).sum()) if "Budget exhausted (completed=3)" in ran else 0,
          "ANOMALY completed>3": int(ran["Anomaly completed>3"].fillna(False).astype(bool).sum()) if "Anomaly completed>3" in ran else 0,
          "queries where attempted>completed": int((ran["Attempted search calls"] > ran["Completed search calls"]).sum()) if "Attempted search calls" in ran else 0,
          "mean completed calls": round(ran["Completed search calls"].mean(), 3) if "Completed search calls" in ran else None,
          "mean attempted calls": round(ran["Attempted search calls"].mean(), 3) if "Attempted search calls" in ran else None,
          "queries with all 20 slots non-null": int(ev["Full 20 slots (all non-null)"].sum()) if len(ev) else 0,
          "queries with format repair": int((ev["Format repair"] != "").sum()) if len(ev) else 0,
          "total cost USD (completed searches)": round(float(ran["Total cost USD"].sum()), 4) if "Total cost USD" in ran else 0,
          "total cost USD (upper bound)": round(float(ran["Total cost USD (upper, attempted billed)"].sum()), 4) if "Total cost USD" in ran else 0,
          "token cost USD": round(float(ran["Token cost USD"].sum()), 4) if "Token cost USD" in ran else 0, "stopped": stopped}
    meta = {"model": MODEL, "reasoning_effort": EFFORT, "tools": "web_search", "tool_choice": "auto", "max_tool_calls": MAX_TOOL_CALLS, "slots": N_SLOTS,
            "prompt_sha256": PROMPT_SHA, "input_sha256": sha, "stop_threshold_usd": STOP_USD, "search_price_usd_per_completed_call": SEARCH_USD,
            "python": platform.python_version(), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "report_built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with pd.ExcelWriter(ROOT / "BEAM_gpt_web_search_run1_results.xlsx", engine="openpyxl") as w:
        pd.DataFrame(list(ov.items()), columns=["Metric", "Value"]).astype(str).to_excel(w, sheet_name="Overall summary", index=False)
        per_level.to_excel(w, sheet_name="Per-level", index=False)
        res.to_excel(w, sheet_name="Per-query (200)", index=False)
        pd.DataFrame(slots_flat).to_excel(w, sheet_name="Top-20 slots + verification", index=False)
        pd.DataFrame(list(meta.items()), columns=["Key", "Value"]).astype(str).to_excel(w, sheet_name="Run metadata", index=False)
        pd.DataFrame([{"Instructions": INSTR}]).to_excel(w, sheet_name="Prompt", index=False)
        for ws in w.book.worksheets:
            ws.freeze_panes = "A2"
    res.to_csv(ROOT / "per_query_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(slots_flat).to_csv(ROOT / "all_slots_verified.csv", index=False, encoding="utf-8-sig")
    per_level.to_csv(ROOT / "per_level_summary.csv", index=False)
    (ROOT / "run_metadata.json").write_text(json.dumps({**meta, "overall": ov}, indent=1, default=str))
    print(json.dumps(ov, indent=1, default=str)); print(per_level.to_string(index=False))


if __name__ == "__main__":
    main()
