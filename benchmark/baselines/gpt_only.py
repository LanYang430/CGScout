"""GPT-only baseline (gpt-5.6-luna, reasoning effort = medium, no tools / retrieval / web) on the 200 frozen v2 queries.

Each query -> ONE structured call that must return a fixed 20-slot ranked list {rank, title, doi, relevance_reason}.
The ranking is the model's own; it is never re-sorted, filtered or replaced after verification.

Fixed rules (set before any output was seen):
  FORMAT   A response is valid iff it parses against the schema, has exactly 20 slots and ranks are exactly 1..20 in order.
           Invalid -> retry the SAME prompt (max 3 attempts total).  If still invalid -> deterministic repair
           (sort by rank if ranks are a permutation of 1..20; truncate to ranks 1..20; pad missing ranks with null slots) and
           record `format_repair`.  Retries/repairs are triggered ONLY by format problems, never by quality or by a seed miss.
  VERIFY   After generation, every non-null slot is identity-checked (OpenAlex single-entity lookups = 0 credits, Crossref):
           VERIFIED_BY_DOI    DOI resolves in OpenAlex and its title matches the stated title (similarity >= 0.85)
           VERIFIED_BY_TITLE  stated DOI missing/wrong, but the stated title matches a Crossref record (similarity >= 0.95)
           UNVERIFIED         neither (treated as a non-hit; slot and rank are KEPT)
  MATCH    Hit iff the verified identity has the seed's OpenAlex ID or DOI (same rule as the OpenAlex baseline).  A slot whose
           DOI equals the seed DOI but whose stated title does not match is reported separately and NOT counted.
  METRICS  Hit@10 and Hit@20 over queries with a valid generation (denominator reported).

    python gpt_only.py --only B001,B002,B128,B133,B142     # 5-query check
    python gpt_only.py                                      # remaining queries (cached ones are free)
    python gpt_only.py --report-only
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import platform
import re
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from pydantic import BaseModel

ROOT = Path(os.environ.get("CGSCOUT_BENCH_DIR", ".")).resolve()           # run directory: outputs, caches and (by default) the input workbook live here
os.environ["BEAM_USAGE_LOG_PATH"] = str(ROOT / "llm_usage_gpt_only.jsonl")   # absolute: the backend default is cwd-relative
from cgscout.llm import OpenAILunaBackend  # noqa: E402

INPUT = Path(os.environ.get("CGSCOUT_QUERIES_FILE", ROOT / "BEAM_200_queries_and_papers_latest.xlsx"))   # frozen query workbook (sheet "Latest 200")
RAW, VER = ROOT / "raw", ROOT / "verify_cache"
HTTP = ROOT / "http_cache"
for d in (RAW, VER, HTTP):
    d.mkdir(exist_ok=True)
MODEL, EFFORT, N_SLOTS, LABEL = "gpt-5.6-luna", "medium", 20, "gpt_only_baseline_v2"
BUDGET_CAP_USD = float(os.environ.get("BUDGET_CAP_USD", "10"))
USAGE_LOG = ROOT / "llm_usage_gpt_only.jsonl"

SYSTEM_PROMPT = """\
You are a literature-recommendation assistant for computational biophysicists who choose coarse-grained (CG) or other simulation models.
You will be given one researcher's question. Recommend scientific papers, from your own knowledge only, that would best help this researcher. You have no search tools.

Return exactly 20 ranked slots, numbered 1 to 20 (rank 1 = most useful to this researcher). For each slot give:
- title: the paper's title
- doi: its DOI; use null if you are not sure of the exact DOI (never guess a DOI)
- relevance_reason: one sentence on why it is relevant
Recommend only papers you believe really exist. If you cannot name a real, relevant paper for a slot, set title, doi and relevance_reason of that slot to null instead of inventing one.
Output only the JSON object that matches the schema."""
PROMPT_SHA = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()


class Slot(BaseModel):
    rank: int
    title: Optional[str]
    doi: Optional[str]
    relevance_reason: Optional[str]


class Recommendation(BaseModel):
    papers: list[Slot]


def norm_doi(d):
    if not d:
        return None
    d = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", str(d).strip().lower()).rstrip(".,;")
    return d if d.startswith("10.") else None


def norm_title(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def sim(a, b):
    from difflib import SequenceMatcher
    a, b = norm_title(a), norm_title(b)
    if not a or not b:
        return 0.0
    s = SequenceMatcher(None, a, b).ratio()
    return max(s, 0.97) if (a.startswith(b) or b.startswith(a)) and min(len(a), len(b)) >= 25 else s


def get_json(url, params=None):
    """cached GET (HTTP 200 only); OpenAlex key added at request time, never stored"""
    params = dict(params or {})
    key = hashlib.sha256((url + json.dumps(params, sort_keys=True)).encode()).hexdigest()
    cp = HTTP / f"{key}.json"
    if cp.exists():
        return json.loads(cp.read_text()), 200
    q = dict(params)
    if "openalex.org" in url and os.environ.get("OPENALEX_API_KEY"):
        q["api_key"] = os.environ["OPENALEX_API_KEY"]
    status = None
    for a in range(3):
        try:
            r = requests.get(url, params=q, timeout=40, headers={"User-Agent": "BEAM-benchmark/1.0"})
            status = r.status_code
            if status == 200:
                cp.write_text(r.text)
                return r.json(), 200
            if status == 404:
                return None, 404
            time.sleep(1.5 * (a + 1))
        except requests.RequestException:
            time.sleep(1.5 * (a + 1))
    return None, status


# ----------------------------------------------------------------- generation
def validate(parsed: Recommendation) -> list[str]:
    errs = []
    ranks = [s.rank for s in parsed.papers]
    if len(parsed.papers) != N_SLOTS:
        errs.append(f"length_{len(parsed.papers)}")
    if ranks != list(range(1, N_SLOTS + 1)):
        errs.append("ranks_not_1_to_20_in_order")
    return errs


def repair(parsed: Recommendation) -> tuple[list[dict], str]:
    slots = sorted((s.model_dump() for s in parsed.papers), key=lambda s: s["rank"])
    by = {}
    for s in slots:
        if 1 <= s["rank"] <= N_SLOTS and s["rank"] not in by:
            by[s["rank"]] = s
    out = [by.get(r, {"rank": r, "title": None, "doi": None, "relevance_reason": None}) for r in range(1, N_SLOTS + 1)]
    return out, "sorted/truncated/padded by rank"


def call_cost(pid: str, t0: float) -> dict:
    tok_in = tok_out = cached = 0
    cost = 0.0
    if USAGE_LOG.exists():
        for line in USAGE_LOG.read_text().splitlines()[-400:]:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("label") == LABEL and r.get("case_id") == pid and r.get("ts", 0) >= t0 - 1:
                tok_in += r["input_tokens"]; tok_out += r["output_tokens"]; cached += r.get("cached_input_tokens", 0); cost += r["estimated_cost_usd"]
    return {"input_tokens": tok_in, "cached_input_tokens": cached, "output_tokens": tok_out, "cost_usd": round(cost, 6)}


def spent_so_far() -> float:
    tot = 0.0
    if USAGE_LOG.exists():
        for line in USAGE_LOG.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("label") == LABEL:
                tot += r["estimated_cost_usd"]
    return tot


def generate(pid: str, query: str, backend) -> dict:
    path = RAW / f"{pid}.json"
    if path.exists():
        return json.loads(path.read_text())
    attempts, t_start = [], time.time()
    final, repaired = None, None
    for k in range(3):
        t0 = time.time()
        try:
            parsed = backend.structured(system=SYSTEM_PROMPT, user=query, schema=Recommendation, label=LABEL, effort=EFFORT,
                                        purpose="gpt_only_baseline", case_id=pid, timeout=300)
            errs = validate(parsed)
            attempts.append({"attempt": k, "format_errors": errs, "seconds": round(time.time() - t0, 1), "parsed": parsed.model_dump(), **call_cost(pid, t0)})
            if not errs:
                final = [s.model_dump() for s in parsed.papers]
                break
        except Exception as e:  # noqa: BLE001
            attempts.append({"attempt": k, "error": repr(e)[:300], "seconds": round(time.time() - t0, 1), **call_cost(pid, t0)})
    if final is None:
        last = next((a for a in reversed(attempts) if "parsed" in a), None)
        if last:
            final, repaired = repair(Recommendation(**last["parsed"]))
    rec = {"pid": pid, "query": query, "model": MODEL, "effort": EFFORT, "prompt_sha256": PROMPT_SHA, "status": "OK" if final else "GENERATION_FAILED",
           "format_repair": repaired, "n_attempts": len(attempts), "attempts": attempts, "slots": final,
           "cost_usd": round(sum(a.get("cost_usd", 0) for a in attempts), 6), "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    path.write_text(json.dumps(rec, ensure_ascii=False))
    return rec


# ----------------------------------------------------------------- verification
def oa_by_doi(doi):
    d, st = get_json(f"https://api.openalex.org/works/https://doi.org/{doi}", {"select": "id,doi,title,publication_year"})
    if st not in (200, 404):
        raise RuntimeError(f"openalex HTTP {st}")
    return (d if st == 200 else None), st


def crossref_by_title(title):
    """best title match among journal articles / proceedings / preprints (query.title first, then query.bibliographic)"""
    best = None
    for field in ("query.title", "query.bibliographic"):
        d, st = get_json("https://api.crossref.org/works", {field: title, "rows": 10, "select": "DOI,title,issued",
                                                            "filter": "type:journal-article,type:proceedings-article,type:posted-content"})
        if st != 200:
            raise RuntimeError(f"crossref HTTP {st}")          # a failed lookup is NOT evidence that the paper does not exist
        for m in ((d or {}).get("message") or {}).get("items", []):
            t = (m.get("title") or [""])[0]
            s_ = sim(title, t)
            if best is None or s_ > best[0]:
                best = (s_, m.get("DOI"), t)
        if best and best[0] >= 0.95:
            break
    return best


def verify_slot(slot: dict) -> dict:
    try:
        return _verify_slot(slot)
    except Exception as e:  # noqa: BLE001
        return {"rank": slot["rank"], "title": slot.get("title"), "stated_doi": slot.get("doi"), "status": "VERIFY_ERROR", "oa_id": None, "resolved_doi": None, "note": str(e)[:120]}


def _verify_slot(slot: dict) -> dict:
    out = {"rank": slot["rank"], "title": slot["title"], "stated_doi": slot["doi"], "status": "EMPTY_SLOT", "oa_id": None, "resolved_doi": None, "note": ""}
    title, doi = slot.get("title"), norm_doi(slot.get("doi"))
    if not title and not doi:
        return out
    out["status"] = "UNVERIFIED"
    if doi:
        w, st = oa_by_doi(doi)
        if w:
            s = sim(title, w.get("title"))
            out.update(oa_title=w.get("title"), title_sim=round(s, 3))
            if s >= 0.85 or (not title):
                out.update(status="VERIFIED_BY_DOI", oa_id=w["id"].split("/")[-1], resolved_doi=norm_doi(w.get("doi")))
                return out
            out["note"] = "DOI exists but belongs to a different title"
        else:
            out["note"] = f"DOI not found (HTTP {st})"
    if title:
        b = crossref_by_title(title)
        if b and b[0] >= 0.95 and b[1]:
            rd = norm_doi(b[1])
            w, _ = oa_by_doi(rd)
            out.update(status="VERIFIED_BY_TITLE", resolved_doi=rd, oa_id=w["id"].split("/")[-1] if w else None, title_sim=round(b[0], 3),
                       note=(out["note"] + "; " if out["note"] else "") + ("stated DOI correct" if doi == rd else "DOI corrected by title match"))
    return out


def verify(pid: str, rec: dict) -> list[dict]:
    path = VER / f"{pid}.json"
    if path.exists():
        return json.loads(path.read_text())
    for rnd in range(4):
        res = [verify_slot(s) for s in rec["slots"]] if rnd == 0 else [(v if v["status"] != "VERIFY_ERROR" else verify_slot(s)) for v, s in zip(res, rec["slots"])]
        if not any(v["status"] == "VERIFY_ERROR" for v in res):
            break
        time.sleep(3 * (rnd + 1))
    if not any(v["status"] == "VERIFY_ERROR" for v in res):   # only complete verifications are cached
        path.write_text(json.dumps(res, ensure_ascii=False))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only"); ap.add_argument("--report-only", action="store_true"); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    df = pd.read_excel(INPUT, sheet_name="Latest 200", dtype=str).fillna("")
    assert len(df) == int(os.environ.get("CGSCOUT_EXPECT_ROWS", "200")) and df["Paper ID"].is_unique
    sha = hashlib.sha256(INPUT.read_bytes()).hexdigest()
    todo = df if not a.only else df[df["Paper ID"].isin(a.only.split(","))]
    stopped = None
    if not a.report_only:
        backend = OpenAILunaBackend(model=MODEL, default_effort=EFFORT)
        pending = [r for _, r in todo.iterrows() if not (RAW / f"{r['Paper ID']}.json").exists()]
        print(f"{len(pending)} to generate; spent so far on this baseline ${spent_so_far():.3f} (cap ${BUDGET_CAP_USD})", flush=True)
        for b0 in range(0, len(pending), 8):
            if spent_so_far() >= BUDGET_CAP_USD:
                stopped = f"budget cap ${BUDGET_CAP_USD} reached before {pending[b0]['Paper ID']}"
                print("STOP:", stopped, flush=True); break
            with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
                list(ex.map(lambda r: generate(r["Paper ID"], r["Latest query (v2)"], backend), pending[b0:b0 + 8]))
            print(f"  generated {min(b0 + 8, len(pending))}/{len(pending)}  spent ${spent_so_far():.3f}", flush=True)
        # verification (0 OpenAlex credits; Crossref is free)
        have = [r for _, r in todo.iterrows() if (RAW / f"{r['Paper ID']}.json").exists()]
        with cf.ThreadPoolExecutor(max_workers=6) as ex:
            list(ex.map(lambda r: verify(r["Paper ID"], json.loads((RAW / f"{r['Paper ID']}.json").read_text())), have))
        print("verified", len(have), "queries", flush=True)

    # ------------------------------------------------ report (always rebuilt from raw/ + verify_cache/)
    rows, slots_flat = [], []
    for _, row in df.iterrows():
        pid = row["Paper ID"]
        base = {"Paper ID": pid, "Provisional level": row["Provisional level"], "Query mode": row["Query mode"], "Latest query (v2)": row["Latest query (v2)"],
                "Seed title": row["Paper title"], "Seed DOI": row["DOI"], "Seed OpenAlex ID": row["OpenAlex ID"]}
        if not (RAW / f"{pid}.json").exists() or not (VER / f"{pid}.json").exists():
            rows.append({**base, "Status": "NOT_EVALUATED", "Hit@10": None, "Hit@20": None}); continue
        rec = json.loads((RAW / f"{pid}.json").read_text())
        if rec["status"] != "OK":
            rows.append({**base, "Status": "NOT_EVALUATED", "Hit@10": None, "Hit@20": None, "Note": "generation failed"}); continue
        ver = json.loads((VER / f"{pid}.json").read_text())
        seed_oa, seed_doi = row["OpenAlex ID"], norm_doi(row["DOI"])
        rank, by, doi_eq_title_bad = None, None, None
        for v, s in zip(ver, rec["slots"]):
            ok = v["status"] in ("VERIFIED_BY_DOI", "VERIFIED_BY_TITLE")
            hit_id = (v.get("oa_id") == seed_oa) if v.get("oa_id") else False
            hit_doi = (v.get("resolved_doi") == seed_doi) if v.get("resolved_doi") else False
            if ok and (hit_id or hit_doi) and rank is None:
                rank, by = v["rank"], ("openalex_id" if hit_id else "doi")
            if (not ok) and norm_doi(s.get("doi")) == seed_doi and doi_eq_title_bad is None:
                doi_eq_title_bad = v["rank"]
            slots_flat.append({"Paper ID": pid, "rank": v["rank"], "title": s.get("title"), "stated_doi": s.get("doi"), "status": v["status"], "resolved_doi": v.get("resolved_doi"),
                               "oa_id": v.get("oa_id"), "note": v.get("note"), "is_seed": int(rank == v["rank"] and by is not None)})
        st = [v["status"] for v in ver]
        nonnull = sum(s != "EMPTY_SLOT" for s in st)
        rows.append({**base, "Status": "EVALUATED", "Hit@10": int(rank is not None and rank <= 10), "Hit@20": int(rank is not None and rank <= 20),
                     "Seed rank (1-20)": rank if rank else "NOT_IN_LIST", "Matched by": by or "", "DOI-equal but title mismatch (not counted) rank": doi_eq_title_bad,
                     "Non-null slots": nonnull, "Verified slots": sum(s.startswith("VERIFIED") for s in st), "Verified by DOI": st.count("VERIFIED_BY_DOI"),
                     "Verified by title only": st.count("VERIFIED_BY_TITLE"), "Unverified (possibly fabricated)": st.count("UNVERIFIED"), "Empty slots": st.count("EMPTY_SLOT"), "Verification errors": st.count("VERIFY_ERROR"),
                     "Format repair": rec["format_repair"] or "", "Attempts": rec["n_attempts"], "Cost USD": rec["cost_usd"]})
    res = pd.DataFrame(rows)
    ev = res[res["Status"] == "EVALUATED"]
    lv = ["L1", "L2", "L1/L2", "L1/L2_PROPOSED_RECLASSIFIED_FROM_L3", "L3_PROVISIONAL"]
    groups = [("ALL", ev)] + [(l, ev[ev["Provisional level"] == l]) for l in lv] + [(f"mode={m}", ev[ev["Query mode"] == m]) for m in ("SOURCE_X", "TARGET_Y", "REVIEWED_DRAFT")]
    per_level = pd.DataFrame([{"Group": g, "Evaluated (denominator)": len(d), "Hit@10": round(d["Hit@10"].mean(), 4) if len(d) else None, "Hit@20": round(d["Hit@20"].mean(), 4) if len(d) else None,
                               "hits@10": int(d["Hit@10"].sum()), "hits@20": int(d["Hit@20"].sum())} for g, d in groups])
    totals = {k: int(ev[k].sum()) for k in ("Non-null slots", "Verified slots", "Verified by DOI", "Verified by title only", "Unverified (possibly fabricated)", "Empty slots", "Verification errors")} if len(ev) else {}
    cost = round(sum(json.loads(p.read_text())["cost_usd"] for p in RAW.glob("*.json")), 4)
    overall = {"N total": len(res), "Evaluated": len(ev), "Not evaluated": len(res) - len(ev), "Hit@10": round(ev["Hit@10"].mean(), 4) if len(ev) else None,
               "Hit@20": round(ev["Hit@20"].mean(), 4) if len(ev) else None, "hits@10": int(ev["Hit@10"].sum()) if len(ev) else 0, "hits@20": int(ev["Hit@20"].sum()) if len(ev) else 0,
               **totals, "share of non-null slots unverified": round(totals["Unverified (possibly fabricated)"] / max(totals["Non-null slots"], 1), 4) if totals else None,
               "queries with format repair": int((ev["Format repair"] != "").sum()) if len(ev) else 0, "queries needing >1 attempt": int((ev["Attempts"] > 1).sum()) if len(ev) else 0,
               "total cost USD (all attempts)": cost, "stopped": stopped}
    meta = {"model": MODEL, "reasoning_effort": EFFORT, "tools": "none", "slots": N_SLOTS, "prompt_sha256": PROMPT_SHA, "input_sha256": sha, "budget_cap_usd": BUDGET_CAP_USD,
            "python": platform.python_version(), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "report_built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with pd.ExcelWriter(ROOT / "BEAM_gpt_only_luna_v2_results.xlsx", engine="openpyxl") as w:
        pd.DataFrame(list(overall.items()), columns=["Metric", "Value"]).astype(str).to_excel(w, sheet_name="Overall summary", index=False)
        per_level.to_excel(w, sheet_name="Per-level", index=False)
        res.to_excel(w, sheet_name="Per-query (200)", index=False)
        pd.DataFrame(slots_flat).to_excel(w, sheet_name="All 20 slots + verification", index=False)
        pd.DataFrame(list(meta.items()), columns=["Key", "Value"]).astype(str).to_excel(w, sheet_name="Run metadata", index=False)
        pd.DataFrame([{"System prompt": SYSTEM_PROMPT}]).to_excel(w, sheet_name="Prompt", index=False)
        for ws in w.book.worksheets:
            ws.freeze_panes = "A2"
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = min(60, max(10, max(len(str(c.value or "")) for c in list(col)[:100]) + 2))
    res.to_csv(ROOT / "per_query_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(slots_flat).to_csv(ROOT / "all_slots_verified.csv", index=False, encoding="utf-8-sig")
    per_level.to_csv(ROOT / "per_level_summary.csv", index=False)
    (ROOT / "run_metadata.json").write_text(json.dumps({**meta, "overall": overall}, indent=1, default=str))
    print(json.dumps(overall, indent=1, default=str)); print(per_level.to_string(index=False))


if __name__ == "__main__":
    main()
