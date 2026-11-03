"""Independent verification of hand-proposed L3 candidates (seed X -> target system Y).  Benchmark code only.
For each candidate:
  1. the seed DOI must resolve (OpenAlex title search) to the claimed title, and not overlap the classifier data / dev-13 / curated audit;
  2. direct-CG evidence check for Y: 3 queries (2 semantic + 1 lexical, top 20 each; lexical because exact names such as
     'MSL10' / 'Alb3' are poorly served by semantic search), seed excluded, de-duplicated, judged by the same
     DIRECT_CHECK prompt the pilot generator uses (direct = a CG / multiscale-CG model OF Y itself);
  3. a transfer rationale (axis, feature, why the seed is useful for Y, confidence, caveat) from the seed abstract.
Nothing is generated and no query is written.  Usage: l3_verify.py [--only 1,2]"""
from __future__ import annotations

import difflib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_backend  # noqa: E402,F401
import prompts as P  # noqa: E402
from overlap_audit import norm  # noqa: E402
from schemas import DirectCheck, TransferRationale  # noqa: E402

CANDIDATES = [
    {"n": 1, "short": "MscS -> MSL10", "seed_title": "Coarse-Grained Model for Mechanosensitive Ion Channels", "seed_doi": "10.1021/jp9053567",
     "Y": "Arabidopsis MSL10 (MscS-like mechanosensitive channel)", "Y_objective": "membrane-tension (stretch) gating of the channel",
     "claimed_axis": "family/function transfer",
     "queries": [("semantic", "coarse-grained simulation of the plant mechanosensitive channel MSL10"),
                 ("semantic", "MscS-like mechanosensitive channel membrane tension gating molecular dynamics simulation"),
                 ("lexical", "MSL10 coarse-grained simulation")]},
    {"n": 2, "short": "E. coli 70S ribosome -> chloroplast 70S ribosome", "seed_title": "Key Intermolecular Interactions in the E. coli 70S Ribosome Revealed by Coarse-Grained Analysis",
     "seed_doi": "10.1021/ja2028487", "Y": "chloroplast (plastid) 70S ribosome", "Y_objective": "inter-subunit interactions and collective motions of the ribosome",
     "claimed_axis": "close structural homology",
     "queries": [("semantic", "coarse-grained model of the chloroplast ribosome intersubunit interactions and collective motions"),
                 ("semantic", "plastid 70S ribosome conformational dynamics molecular dynamics simulation"),
                 ("lexical", "chloroplast ribosome coarse-grained")]},
    {"n": 3, "short": "YidC -> Alb3", "seed_title": "Monitoring the binding and insertion of a single transmembrane protein by an insertase",
     "seed_doi": "10.1038/s41467-021-27315-3", "Y": "chloroplast Alb3 insertase (YidC/Oxa1 family)", "Y_objective": "binding and insertion of a transmembrane protein by the insertase",
     "claimed_axis": "family/function transfer",
     "queries": [("semantic", "coarse-grained simulation of Alb3 insertase binding and insertion of a transmembrane protein"),
                 ("semantic", "YidC Oxa1 Alb3 insertase membrane protein insertion molecular dynamics"),
                 ("lexical", "Alb3 insertase coarse-grained simulation")]},
    {"n": 4, "short": "bacterial flagellum -> archaellum", "seed_title": "Coarse-Grained Molecular Dynamics Simulations of a Rotating Bacterial Flagellum",
     "seed_doi": "10.1529/biophysj.106.093443", "Y": "archaeal archaellum (rotating helical filament)", "Y_objective": "rotation, torque transmission and polymorphism of the filament",
     "claimed_axis": "physical-process transfer",
     "queries": [("semantic", "coarse-grained simulation of the archaellum rotating helical filament"),
                 ("semantic", "archaellum filament rotation torque mechanics molecular dynamics"),
                 ("lexical", "archaellum coarse-grained simulation")]},
    {"n": 5, "short": "eukaryotic nucleosome assembly -> archaeal histone-DNA", "seed_title": "The kinetic landscape of nucleosome assembly: A coarse-grained molecular dynamics study",
     "seed_doi": "10.1371/journal.pcbi.1009253", "Y": "archaeal histone-DNA complex (hypernucleosome)", "Y_objective": "histone-DNA wrapping and assembly",
     "claimed_axis": "close structural homology",
     "queries": [("semantic", "coarse-grained simulation of archaeal histone DNA wrapping and hypernucleosome assembly"),
                 ("semantic", "archaeal chromatin histone DNA assembly molecular dynamics"),
                 ("lexical", "archaeal histone hypernucleosome coarse-grained")]},
    {"n": 6, "short": "clathrin/AP2 -> COPI coatomer", "seed_title": "Clathrin Assembly Regulated by Adaptor Proteins in Coarse-Grained Models",
     "seed_doi": "10.1016/j.bpj.2016.06.003", "Y": "COPI coatomer coat", "Y_objective": "self-assembly of the coat and coat-induced membrane curvature",
     "claimed_axis": "physical-process / mesoscale transfer",
     "queries": [("semantic", "coarse-grained simulation of COPI coatomer coat assembly on a membrane"),
                 ("semantic", "COPI coat self-assembly membrane curvature mesoscale model"),
                 ("lexical", "COPI coatomer coarse-grained")]},
]


def resolve_seed(c, search):
    best = (0.0, None)
    for p in search(c["seed_title"], 5, "lexical"):
        s = difflib.SequenceMatcher(None, norm(c["seed_title"]), norm(p.title)).ratio()
        if s > best[0]:
            best = (s, p)
    s, p = best
    return {"title_ratio": round(s, 2), "found_title": p.title if p else None, "found_doi": (p.doi or "").lower() if p else None,
            "doi_matches_claim": bool(p and (p.doi or "").lower() == c["seed_doi"].lower()), "year": p.year if p else None,
            "venue": p.venue if p else None, "abstract": (p.abstract or "") if p else ""}


def verify(c, backend, search, epmc, audit_dois: dict):
    cid = f"bench_l3verify_{c['n']}"
    seed = resolve_seed(c, search)
    if len(seed["abstract"]) < 200 and seed["found_doi"]:
        from beam_agent.models import Paper
        e = epmc(Paper(title=c["seed_title"], doi=seed["found_doi"]))
        if e and e.get("abstract"):
            seed["abstract"], seed["abstract_source"] = e["abstract"], "europepmc"
    hits: dict[str, object] = {}
    per_query = []
    for mode, q in c["queries"]:
        res = search(q, 20, mode)
        per_query.append({"mode": mode, "query": q, "n_results": len(res)})
        for p in res:
            key = (p.doi or p.title).lower()
            if key != c["seed_doi"].lower() and key not in hits:
                hits[key] = p
    papers = list(hits.values())
    direct = []
    for start in range(0, len(papers), 20):
        batch = papers[start:start + 20]
        listing = "\n".join(f"[{i}] {p.title}\n    {(p.abstract or '(no abstract)')[:500]}" for i, p in enumerate(batch, 1))
        chk = backend.structured(system=P.DIRECT_CHECK_SYSTEM, label="bench_l3verify_direct", effort="low", purpose="bench_v1_l3verify", case_id=cid, timeout=120,
                                 user=f"TARGET SYSTEM: {c['Y']} -- objective: {c['Y_objective']}\n\nPAPERS\n{listing}", schema=DirectCheck)
        for it in chk.items:
            if it.is_direct_cg_for_target and 1 <= it.index <= len(batch):
                p = batch[it.index - 1]
                direct.append({"title": p.title, "doi": p.doi, "year": p.year, "reason": it.reason})
    rat = None
    if seed["abstract"]:
        rat = backend.structured(system=P.TRANSFER_RATIONALE_SYSTEM, label="bench_l3verify_rationale", effort="medium", purpose="bench_v1_l3verify", case_id=cid, timeout=120,
                                 user=f"SEED ABSTRACT (system X):\n{seed['abstract']}\n\nTARGET SYSTEM Y: {c['Y']}\nTarget objective: {c['Y_objective']}", schema=TransferRationale)
    return {"n": c["n"], "source_seed_X": {"title": c["seed_title"], "doi": c["seed_doi"], **{k: seed[k] for k in ("title_ratio", "found_title", "doi_matches_claim", "year", "venue")},
                                           "abstract_chars": len(seed["abstract"]), "overlap": audit_dois.get(c["seed_doi"].lower(), [])},
            "target_system_Y": c["Y"], "claimed_axis": c["claimed_axis"],
            "transfer": rat.model_dump() if rat else None,
            "direct_CG_search_queries": per_query, "n_unique_papers_judged": len(papers),
            "direct_CG_hits_for_Y": direct, "verified_no_direct": not direct}


def classifier_overlap() -> dict:
    """seed DOI -> list of places it already appears (classifier files / dev-13 / earlier pilot / curated pool)."""
    import csv
    out: dict[str, list] = {}
    dl = Path("~/Downloads")
    for f in ("beam_stage3_training_ready_yes_no.csv", "beam_stage3_external_test_clean_519.csv", "beam_stage3_training_labels_89.csv", "beam_stage3_silver_labels_91_web_reviewed.csv"):
        for r in csv.DictReader(open(dl / f, encoding="utf-8")):
            out.setdefault((r["doi"] or "").lower(), []).append(f"{f.split('_')[2]}:{r['case'].split(':')[0]}")
    cases = json.loads((gen_backend.ROOT / "benchmarks" / "regression_13" / "cases.json").read_text())["cases"]
    for c in cases:
        for t in c["targets"].values():
            for d in t["dois"]:
                out.setdefault(d.lower(), []).append(f"dev13:case{c['id']}")
    for s in json.loads((HERE / "data" / "pilot_seeds.json").read_text()):
        out.setdefault(s["doi"].lower(), []).append("earlier_pilot_" + s["case_id"])
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--only"); a = ap.parse_args()
    sys.path.insert(0, str(gen_backend.ROOT / "benchmarks" / "regression_13"))
    import regression_lib as RL
    from beam_agent.sources.openalex import search_openalex
    sys.path.insert(0, str(gen_backend.MI))
    import stage4_v2 as S4
    cands = [c for c in CANDIDATES if not a.only or str(c["n"]) in a.only.split(",")]
    backend = gen_backend.get_generator_backend()
    ov = classifier_overlap()
    results = []

    def go():
        RL.harness_guard(len(cands) * 4, what="bench_v1_l3verify")
        RL.U.set_run_label("bench_v1_l3verify")
        t0 = time.time()
        for c in cands:
            results.append(verify(c, backend, lambda q, k, m: search_openalex(q, limit=k, mode=m), S4.europepmc_abstract, ov))
            print(f"candidate {c['n']} {c['short']}: no_direct={results[-1]['verified_no_direct']} hits={len(results[-1]['direct_CG_hits_for_Y'])}", flush=True)
        from generate import cost_since
        print("LLM:", cost_since(t0)); print("OpenAlex:", RL.U.run_usage("bench_v1_l3verify"))
    RL.run_guarded(go)
    (HERE / "data" / "l3_verification.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
