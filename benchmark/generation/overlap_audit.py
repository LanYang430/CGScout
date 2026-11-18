"""Overlap audit: which curated-pool papers are safe ("clean_for_pilot") to use as benchmark seeds?
Evidence, all from files (no memory):
  * the user's Stage-3 classifier datasets (training_ready 557 rows, external_test 519 rows, silver/89 sets; not in the repository: the directory is
    --stage3-data-dir or $CGSCOUT_STAGE3_DATA_DIR);
  * the exact rows the production trainer really used (sim_classifier_train.load_train_rows(), after its DOI/title cleaning);
  * the `case` column of those files = which seed/case the candidates were retrieved for;
  * the 13 dev/smoke cases (data/regression_13_cases.json).
A paper is NOT clean if it (a) is a dev-13 target, (b) appears as a row in a classifier train/test file, or (c) was the
seed of a case that produced classifier data.  DOIs for papers not found in the CSVs are resolved with OpenAlex title
search (cached, label-attributed)."""
from __future__ import annotations

import csv
import difflib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_backend  # noqa: E402,F401
import stage3_data  # noqa: E402
from curated_pool import CURATED  # noqa: E402

FILES = {"train557": "beam_stage3_training_ready_yes_no.csv", "test519": "beam_stage3_external_test_clean_519.csv",
         "labels89": "beam_stage3_training_labels_89.csv", "silver91": "beam_stage3_silver_labels_91_web_reviewed.csv"}


def norm(t: str) -> str:
    t = (t or "").lower().replace("α", "alpha").replace("β", "beta").replace("δ", "d")
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def best_title_match(title: str, rows: list[dict], thr: float = 0.9):
    nt = norm(title)
    best = (0.0, None)
    for r in rows:
        s = difflib.SequenceMatcher(None, nt, norm(r["title"])).ratio()
        if s > best[0]:
            best = (s, r)
    return best if best[0] >= thr else (best[0], None)


def load_all(data_dir: str | None = None):
    dl = stage3_data.data_dir(data_dir)
    data = {k: list(csv.DictReader(open(dl / f, encoding="utf-8"))) for k, f in FILES.items()}
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        import sim_classifier_train as T
        used = T.load_train_rows()
    return data, {(r["doi"] or r["title"]).lower() for r in used}


def dev13():
    cases = json.loads((HERE / "data" / "regression_13_cases.json").read_text())["cases"]
    return [{"case": c["id"], "label": lab, "doi": d.lower()} for c in cases for lab, t in c["targets"].items() for d in t["dois"]]


def resolve_openalex(title: str, year: int):
    from cgscout.sources.openalex import search_openalex
    best = (0.0, None)
    for p in search_openalex(title, limit=5, mode="lexical"):
        s = difflib.SequenceMatcher(None, norm(title), norm(p.title)).ratio()
        if s > best[0]:
            best = (s, p)
    return best


def main():
    import argparse
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--stage3-data-dir", help=f"directory with the labelled Stage-3 CSV exports (default: ${stage3_data.ENV_NAME}; they are not in the repository)")
    a = ap.parse_args()
    stage3_data.data_dir(a.stage3_data_dir)             # fail fast, before any other import or work
    import regression_lib as RL
    data, used_keys = load_all(a.stage3_data_dir)
    all_rows = [r for rows in data.values() for r in rows]
    dev = dev13()
    dev_dois = {d["doi"] for d in dev}
    case_labels = {k: sorted({r["case"] for r in rows}) for k, rows in data.items()}
    out = []

    def go():
        RL.harness_guard(len(CURATED), what="bench_v1_overlap_audit")
        RL.U.set_run_label("bench_v1_overlap_audit")
        for cid, title, year, dom, seed_case in CURATED:
            rec = {"id": cid, "title": title, "year": year, "domain": dom, "doi": None, "openalex_id": None, "resolved_by": None,
                   "classifier_rows": [], "seed_of_classifier_case": seed_case, "seed_case_confirmed": None, "dev13": False}
            s, m = best_title_match(title, all_rows)
            if m is not None:
                rec["doi"], rec["resolved_by"] = (m["doi"] or "").lower() or None, f"classifier csv (title ratio {s:.2f})"
            if not rec["doi"]:
                s2, p = resolve_openalex(title, year)
                if p is not None and s2 >= 0.9 and p.doi:
                    rec.update(doi=p.doi.lower(), openalex_id=getattr(p.external_ids, "openalex_id", None), resolved_by=f"openalex (title ratio {s2:.2f})",
                               abstract_chars=len(p.abstract or ""), venue=p.venue, oa_type=p.oa_type, found_year=p.year)
                else:
                    rec["resolved_by"] = f"UNRESOLVED (best ratio {max(s2, s):.2f})"
            for k, rows in data.items():                       # exact-paper overlap: by DOI, else by title
                for r in rows:
                    same = (rec["doi"] and (r["doi"] or "").lower() == rec["doi"]) or difflib.SequenceMatcher(None, norm(title), norm(r["title"])).ratio() >= 0.93
                    if same:
                        rec["classifier_rows"].append({"file": k, "case": r["case"], "label": r.get("simulation_by_authors"),
                                                       "in_final_training_set": (k == "train557" and (r["doi"] or r["title"]).lower() in used_keys)})
            if seed_case:                                       # confirm: the seed paper should be a row of its own case
                rec["seed_case_confirmed"] = any(x["case"] == seed_case for x in rec["classifier_rows"])
            rec["dev13"] = bool(rec["doi"] and rec["doi"] in dev_dois)
            reasons = []
            if rec["dev13"]:
                reasons.append("dev13_target")
            if rec["classifier_rows"]:
                reasons.append("row_in_classifier_data:" + ",".join(sorted({f"{x['file']}/{x['case'].split(':')[0]}" for x in rec["classifier_rows"]})))
            if seed_case:
                reasons.append(f"seed_of_classifier_case:{seed_case.split(':')[0]}")
            rec["status"] = "clean_for_pilot" if not reasons else "excluded"
            rec["exclusion_reasons"] = reasons
            out.append(rec)
    RL.run_guarded(go)
    (HERE / "data" / "curated_pool_audit.json").write_text(json.dumps({"case_labels": case_labels, "dev13": dev, "papers": out}, indent=1, ensure_ascii=False))
    for r in out:
        print(f"{r['id']:>2} {r['status']:15s} {r['domain'][:9]:9s} {str(r['doi'])[:34]:34s} {r['title'][:52]:52s} {'; '.join(r['exclusion_reasons'])[:90]}")
    print("\nclean:", sum(r["status"] == "clean_for_pilot" for r in out), "| excluded:", sum(r["status"] == "excluded" for r in out),
          "| unresolved DOI:", sum(r["doi"] is None for r in out))
    print("seed-case confirmations (seed paper present as a row of its own case):",
          {r["id"]: r["seed_case_confirmed"] for r in out if r["seed_of_classifier_case"]})


if __name__ == "__main__":
    main()
