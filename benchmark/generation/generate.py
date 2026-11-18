"""Pilot generator: seed abstract -> need card (stage A) -> 3 blind query candidates (stage B); L3 adds a sister-system
step.  Writes data/pilot_v0.jsonl (everything) and data/pilot_v0.csv (for the human pick).  Benchmark code only:
nothing here touches CGScout's pipeline.

  D/H (L1, L2):  A sees ONLY the abstract (no title / methods / discussion).  B sees ONLY the need card.
                 L1 uses `specific_need`, L2 uses `abstracted_need`.
  L3:            A as above.  A proposer sees only the seed's specific need card and proposes 2-3 related-but-different
                 systems Y; each Y gets an independent "is there already a direct CG paper for Y?" check (one OpenAlex
                 semantic search, seed excluded, + one LLM judgement of the top-10).  B sees only Y's description and
                 the seed's ABSTRACTED need; the seed system X's name must not appear in the query.
Usage: generate.py [--only P01,P02] [--out data/pilot_v0]
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Callable, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_backend  # noqa: E402,F401  (sets sys.path for CGScout modules)
import leakage as LK  # noqa: E402
import prompts as P  # noqa: E402
from schemas import (DirectCheck, NeedCard, NeedCards, QueryCandidates, SisterProposals)  # noqa: E402

CSV_COLUMNS = ["case_id", "case_type", "seed_doi", "seed_title", "seed_year", "seed_abstract", "need_card",
               "query_candidate_1", "query_candidate_2", "query_candidate_3", "selected_query", "selected_candidate",
               "transfer_source_system", "transfer_target_system", "verified_no_direct", "leakage_flags",
               "openalex_semantic_rank", "generator_model", "domain", "sister_summary"]


def card_text(c: NeedCard) -> str:
    return " | ".join([c.system_class, c.scientific_process, c.phenomenon_or_goal, "; ".join(c.important_constraints),
                       c.required_model_capability, "; ".join(c.transferable_features)])


WORDS_MIN, WORDS_MAX = 15, 30
PERSONA_RE = re.compile(r"^\s*(as an?\b|being an?\b|i am an?\b|i'?m an?\b|speaking as\b)|early[- ]career|\bphd student\b|\bgraduate student\b", re.I)


def format_flags(q: str) -> list[str]:
    """Format rules from the short-query convention: 15-30 words, no role opening."""
    n = len(q.split())
    fl = []
    if not WORDS_MIN <= n <= WORDS_MAX:
        fl.append(f"format_length:{n}_words")
    if PERSONA_RE.search(q):
        fl.append("format_persona_opening")
    return fl


def _call(backend, system, user, schema, label, case_id, effort="low"):
    last = None
    for _ in range(2):
        try:
            return backend.structured(system=system, user=user, schema=schema, label=label, effort=effort,
                                      purpose="bench_v1_pilot", case_id=case_id, timeout=120)
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.0)
    raise last  # type: ignore[misc]


# ------------------------------------------------------------------ stage A
def stage_a(seed: dict, backend, rare: set[str], case_id: str) -> tuple[NeedCards, list[str]]:
    user = "ABSTRACT:\n" + seed["abstract"].strip()
    flags: list[str] = []
    for attempt in range(2):
        cards = _call(backend, P.STAGE_A_SYSTEM, user, NeedCards, "bench_stageA", case_id, effort="medium")
        flags = []
        for name, c in (("specific", cards.specific_need), ("abstracted", cards.abstracted_need)):
            allowed = LK.tokens(cards.specific_need.system_class) if name == "specific" else []
            for f in LK.check_text(card_text(c), title=seed["title"], abstract=seed["abstract"], authors=seed.get("authors", []),
                                   rare=rare, allowed_words=allowed, is_card=True):
                flags.append(f"{name}_card:{f}")
        if not flags:
            break
        user = ("ABSTRACT:\n" + seed["abstract"].strip() + "\n\nYour previous cards were rejected for these reasons: "
                + "; ".join(flags) + ". Rewrite both cards without them.")
    return cards, flags


# ------------------------------------------------------------------ stage B
def _query_flags(q: str, seed: dict, rare: set[str], need_text: str, allowed: list[str], x_terms: list[str] = (),
                 y_text: str = "") -> list[str]:
    fl = LK.check_text(q, title=seed["title"], abstract=seed["abstract"], authors=seed.get("authors", []), rare=rare,
                       card_text=need_text, allowed_words=allowed)
    qt = set(LK.tokens(q))
    yt = set(LK.tokens(y_text))
    for t in x_terms:
        if t in qt and t not in yt:
            fl.append(f"x_system_term:{t!r}")
    return fl + format_flags(q)


def stage_b(system_prompt: str, user: str, seed: dict, rare: set[str], need_text: str, allowed: list[str], backend,
            case_id: str, x_terms: list[str] = (), y_text: str = "") -> tuple[list[str], list[list[str]]]:
    """3 blind candidates.  If none of them is flag-free, regenerate once telling the model which strings to avoid;
    whatever the last attempt gives is kept WITH its flags (the human sees them; nothing is silently dropped)."""
    best = ([], [])
    for attempt in range(2):
        out = _call(backend, system_prompt, user, QueryCandidates, "bench_stageB", case_id)
        qs = [q.strip() for q in out.queries if q and q.strip()][:3]
        if len(qs) < 3 and attempt == 0:
            continue
        flags = [_query_flags(q, seed, rare, need_text, allowed, x_terms, y_text) for q in qs]
        if not best[0] or sum(map(len, flags)) < sum(map(len, best[1])):
            best = (qs, flags)
        if any(len(f) == 0 for f in flags):
            break
        bad = sorted({x.split(":", 1)[1] for f in flags for x in f if not x.startswith("format_")})
        fmt = sorted({x for f in flags for x in f if x.startswith("format_")})
        fb = []
        if bad:
            fb.append("your previous questions contained: " + "; ".join(bad) + " -- do not use these")
        if fmt:
            fb.append("format problems: " + ", ".join(fmt) + " -- each question must be 15-30 words with no role opening")
        user = user + "\n\nREWRITE: " + "; ".join(fb) + "."
    return best


# ------------------------------------------------------------------ L3
def x_system_terms(seed: dict) -> list[str]:
    """Distinctive words of the seed's own system name (from the tag step), lower-case, >=4 letters, not generic."""
    generic = {"protein", "proteins", "membrane", "membranes", "model", "models", "system", "systems", "complex", "complexes",
               "lipid", "lipids", "peptide", "peptides", "dna", "rna", "channel", "channels", "virus", "viral", "enzyme"}
    return sorted({t for t in LK.tokens(seed.get("named_system", "")) if len(t) >= 4 and t not in generic and t not in LK.STOP})


def direct_check(y, seed: dict, backend, search_fn, case_id: str) -> dict:
    """Is there already a direct CG paper for target system Y?  One semantic OpenAlex search (seed excluded) + one judgement."""
    q = f"coarse-grained model simulation of {y.system_name}"
    papers = [p for p in search_fn(q, 10) if (p.doi or "").lower() != seed["doi"].lower()][:10]
    if not papers:
        return {"query": q, "n_results": 0, "verified_no_direct": True, "direct": [], "note": "no results"}
    listing = "\n".join(f"[{i}] {p.title}\n    {(p.abstract or '(no abstract)')[:600]}" for i, p in enumerate(papers, 1))
    chk = _call(backend, P.DIRECT_CHECK_SYSTEM, f"TARGET SYSTEM: {y.system_name} -- {y.description}\n\nPAPERS\n{listing}",
                DirectCheck, "bench_directcheck", case_id)
    hits = [(it.index, it.reason) for it in chk.items if it.is_direct_cg_for_target and 1 <= it.index <= len(papers)]
    return {"query": q, "n_results": len(papers), "verified_no_direct": not hits,
            "direct": [{"title": papers[i - 1].title, "doi": papers[i - 1].doi, "reason": r} for i, r in hits],
            "top3_titles": [p.title for p in papers[:3]]}


def l3_track(seed: dict, cards: NeedCards, backend, search_fn, rare, case_id: str) -> dict:
    props = _call(backend, P.SISTER_SYSTEM, "NEED CARD OF SOURCE SYSTEM X:\n" + cards.specific_need.model_dump_json(indent=1),
                  SisterProposals, "bench_sisters", case_id, effort="medium").proposals[:3]
    seed_abs_t = " ".join(LK.tokens(seed["abstract"]))
    rows = []
    for y in props:
        row = {"system_name": y.system_name, "description": y.description, "shared_feature": y.shared_feature,
               "key_difference": y.key_difference, "eliminated": None, "check": None}
        if " ".join(LK.tokens(y.system_name)) in seed_abs_t and len(LK.tokens(y.system_name)) >= 1:
            row["eliminated"] = "named_in_seed_abstract (not a different system)"
        else:
            row["check"] = direct_check(y, seed, backend, search_fn, case_id)
            if not row["check"]["verified_no_direct"]:
                row["eliminated"] = "has_direct_cg_literature"
        rows.append(row)
    ok = [r for r in rows if r["eliminated"] is None]
    fallback = [r for r in rows if r["eliminated"] == "has_direct_cg_literature"]
    chosen = (ok or fallback or [None])[0]
    return {"proposals": rows, "chosen": chosen, "verified_no_direct": bool(ok),
            "n_proposed": len(rows), "n_eliminated_direct": sum(r["eliminated"] == "has_direct_cg_literature" for r in rows),
            "n_eliminated_same": sum((r["eliminated"] or "").startswith("named_in_seed") for r in rows)}


# ------------------------------------------------------------------ one case
def run_case(seed: dict, backend, search_fn: Callable, rare: set[str], model_label: str) -> dict:
    cid = f"bench_pilot_{seed['case_id']}"
    cards, card_flags = stage_a(seed, backend, rare, cid)
    ctype = seed["case_type"]
    rec = {"case_id": seed["case_id"], "case_type": ctype, "domain": seed["domain"], "seed_doi": seed["doi"],
           "seed_title": seed["title"], "seed_year": seed["year"], "seed_abstract": seed["abstract"],
           "seed_named_system": seed.get("named_system", ""), "seed_paper_type": seed.get("paper_type"),
           "need_card": cards.model_dump(), "card_flags": card_flags, "generator_model": model_label,
           "prompt_version": P.PROMPT_VERSION, "transfer_source_system": "", "transfer_target_system": "",
           "verified_no_direct": None, "sister_proposals": None, "selected_query": "", "selected_candidate": "",
           "openalex_semantic_rank": None}
    if ctype in ("L1", "L2"):
        need = cards.specific_need if ctype == "L1" else cards.abstracted_need
        user = "NEED CARD\n" + need.model_dump_json(indent=1)
        allowed = LK.tokens(cards.specific_need.system_class) if ctype == "L1" else []
        qs, flags = stage_b(P.STAGE_B_SYSTEM, user, seed, rare, card_text(need), allowed, backend, cid)
    else:
        if seed.get("l3_target"):                 # target system Y was fixed and verified beforehand (l3_verify.py / l3_find_more.py)
            t = seed["l3_target"]
            y = {"system_name": t["system_name"], "description": t["description"], "shared_feature": t["shared_feature"]}
            rec["l3_evidence"] = t.get("evidence")
            rec["transfer_source_system"] = t.get("source_system") or seed.get("named_system") or cards.specific_need.system_class
            rec["verified_no_direct"] = t.get("verified_no_direct")
            rec["sister_summary"] = None
        else:
            l3 = l3_track(seed, cards, backend, search_fn, rare, cid)
            rec["sister_proposals"] = l3["proposals"]
            rec["sister_summary"] = {k: l3[k] for k in ("n_proposed", "n_eliminated_direct", "n_eliminated_same", "verified_no_direct")}
            y = l3["chosen"]
            if y is None:
                rec.update(query_candidates=[], leakage_flags=[], error="no usable sister system")
                return rec
            rec["transfer_source_system"] = seed.get("named_system") or cards.specific_need.system_class
            rec["verified_no_direct"] = l3["verified_no_direct"]
        rec["transfer_target_system"] = y["system_name"]
        ab = cards.abstracted_need
        user = (f"THE RESEARCHER'S OWN SYSTEM: {y['system_name']} -- {y['description']}\n"
                f"PROPERTY IT HAS (mention only if natural): {y['shared_feature']}\n"
                f"MODELING NEED: process: {ab.scientific_process}; goal: {ab.phenomenon_or_goal}; "
                f"required capability: {ab.required_model_capability}; constraints: {'; '.join(ab.important_constraints) or '-'}; "
                f"transferable features: {'; '.join(ab.transferable_features) or '-'}")
        y_text = f"{y['system_name']} {y['description']} {y['shared_feature']}"
        qs, flags = stage_b(P.STAGE_B_L3_SYSTEM, user, seed, rare, user, LK.tokens(y["system_name"]), backend, cid,
                            x_terms=x_system_terms(seed), y_text=y_text)
    rec["query_candidates"] = qs
    rec["leakage_flags"] = flags
    return rec


# ------------------------------------------------------------------ output
def to_csv_row(r: dict) -> dict:
    qs = (r.get("query_candidates") or []) + ["", "", ""]
    sister = r.get("sister_summary")
    return {"case_id": r["case_id"], "case_type": r["case_type"], "seed_doi": r["seed_doi"], "seed_title": r["seed_title"],
            "seed_year": r["seed_year"], "seed_abstract": r["seed_abstract"], "need_card": json.dumps(r["need_card"], ensure_ascii=False),
            "query_candidate_1": qs[0], "query_candidate_2": qs[1], "query_candidate_3": qs[2],
            "selected_query": r.get("selected_query", ""), "selected_candidate": r.get("selected_candidate", ""),
            "transfer_source_system": r.get("transfer_source_system", ""), "transfer_target_system": r.get("transfer_target_system", ""),
            "verified_no_direct": "" if r.get("verified_no_direct") is None else r["verified_no_direct"],
            "leakage_flags": json.dumps({"cards": r.get("card_flags", []), "candidates": r.get("leakage_flags", [])}, ensure_ascii=False),
            "openalex_semantic_rank": "" if r.get("openalex_semantic_rank") is None else r["openalex_semantic_rank"],
            "generator_model": r.get("generator_model", ""), "domain": r.get("domain", ""),
            "sister_summary": json.dumps(sister, ensure_ascii=False) if sister else ""}


def write_outputs(records: list[dict], out: Path) -> None:
    out.with_suffix(".jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
    with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in records:
            w.writerow(to_csv_row(r))


def cost_since(t0: float) -> dict:
    from cgscout.llm import read_usage_log
    rows = [r for r in read_usage_log() if r.get("ts", 0) >= t0 and str(r.get("label", "")).startswith("bench_")]
    return {"llm_calls": len(rows), "input_tokens": sum(r.get("input_tokens", 0) for r in rows),
            "output_tokens": sum(r.get("output_tokens", 0) for r in rows),
            "cost_usd": round(sum(r.get("estimated_cost_usd", 0.0) for r in rows), 4)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only"); ap.add_argument("--out", default=str(HERE / "data" / "pilot_v1"))
    ap.add_argument("--seeds", default=str(HERE / "data" / "pilot_seeds_final.json")); a = ap.parse_args()
    import regression_lib as RL                              # budget guard + usage ledger (same one the CGScout harness uses)
    from cgscout.sources.openalex import search_openalex
    seeds = json.loads(Path(a.seeds).read_text())
    if a.only:
        seeds = [s for s in seeds if s["case_id"] in set(a.only.split(","))]
    pool = json.loads((HERE / "data" / "seed_pool.json").read_text())
    rare = LK.build_rare_words(c["abstract"] for c in pool)
    backend = gen_backend.get_generator_backend()
    model_label = gen_backend.label(backend)
    out = Path(a.out)
    done = {}
    if out.with_suffix(".jsonl").exists():
        done = {json.loads(l)["case_id"]: json.loads(l) for l in out.with_suffix(".jsonl").read_text().splitlines() if l.strip()}
    n_l3 = sum(1 for s in seeds if s["case_type"] == "L3" and s["case_id"] not in done and not s.get("l3_target"))   # only unverified L3 seeds need OpenAlex

    def go():
        RL.harness_guard(n_l3 * 3, what="bench_v1_pilot", check_budget=n_l3 > 0)
        RL.U.set_run_label("bench_v1_pilot")
        sfn = lambda q, k: search_openalex(q, limit=k, mode="semantic")  # noqa: E731
        t0 = time.time()
        for s in seeds:
            if s["case_id"] in done:
                continue
            done[s["case_id"]] = run_case(s, backend, sfn, rare, model_label)
            write_outputs([done[k] for k in sorted(done)], out)
            print(f"{s['case_id']} {s['case_type']} done", flush=True)
        write_outputs([done[k] for k in sorted(done)], out)
        print("LLM cost this run:", cost_since(t0))
        print("OpenAlex usage:", RL.U.run_usage("bench_v1_pilot"))
    RL.run_guarded(go)


if __name__ == "__main__":
    main()
