"""READ-ONLY Hop-1 audit of the routing smoke test (Q15, Q16, Q19, Q20): why did related-system discovery end at 0 systems?
Everything is reconstructed from what the smoke run already left on disk -- no OpenAI call, no OpenAlex request, no CGScout file touched:
  * the Hop-1 sentence + axis       <- data/routing_smoke.json
  * the 15 papers Hop-1 retrieved    <- OpenAlex request cache (same sentence, semantic, limit 15)
  * the ORIGINAL per-paper LLM extractions <- .cache/hop1_extract_rich, matched to usage-log rows by write time, then to papers by evidence quote
  * the relationship-fidelity gate / grounding / cleanup / identity steps re-applied with CGScout's own functions to those extractions.
Caveat: ExpertPrior's retrieval_anchors were not saved by the smoke run, so `also_known_as` (used only to ban class-like names) is approximated by
{class + objective}; the stored structural_term_matched values come from the original run."""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_backend  # noqa: E402,F401
import cgscout.retrieval.related_systems.discovery as RSD  # noqa: E402
import cgscout.retrieval.related_systems.extraction as TE  # noqa: E402
from cgscout.retrieval.related_systems.postprocess import build_related_system_identities, cleanup_discovered_systems  # noqa: E402
from cgscout import cache as C  # noqa: E402
from cgscout.models import Paper  # noqa: E402
from cgscout.sources import openalex_usage as U  # noqa: E402
from cgscout.sources.openalex import _SELECT  # noqa: E402

CASES = ["Q15", "Q16", "Q19", "Q20"]
X_PATTERNS = {
    "Q15": ("MscS (and other MscS-family channels)", r"\bMscS\b|\bMscL\b|small[- ]conductance|\bMSL\d*\b|MscS-like"),
    "Q16": ("bacterial / E. coli 70S ribosome", r"E\. ?coli[^.]{0,60}ribosom|Escherichia coli[^.]{0,60}ribosom|bacterial ribosom|\b70S\b|Thermus thermophilus[^.]{0,40}ribosom"),
    "Q19": ("Pol II / bacterial-type / homologous RNAP", r"RNA polymerase II|\bPol II\b|polymerase II|bacterial RNA polymerase|E\. ?coli RNA polymerase|σ70|sigma ?70|TFIIB|\bTBP\b|\bTFB\b|archaeal RNA polymerase|\bRNAP\b"),
    "Q20": ("dynamin (and dynamin-family GTPases)", r"dynamin|\bDrp1\b|DNM1L|\bOPA1\b|mitofusin|\bMfn\d?\b|\bFzo1\b|atlastin"),
}


def usage_rows(case_id):
    return [json.loads(l) for l in open(gen_backend.ROOT / ".cache" / "llm_usage.jsonl")
            if f'"case_id": "{case_id}"' in l and '"label": "hop1_extract_rich"' in l]


def hop1_cached_papers(cid):
    """The Hop-1 request's cached result list, found via the usage ledger (semantic, limit 15, this run's label) and the cache
    file written right after that request -- a direct key lookup missed, so match by time instead."""
    rows = [json.loads(l) for l in open(gen_backend.ROOT / ".cache" / "openalex_usage.jsonl") if f'"label": "routing_smoke_{cid}"' in l]
    r = next(x for x in rows if x["mode"] == "semantic" and x["limit"] == 15)
    t = r["ts"] + r["elapsed_s"]
    files = sorted((abs(os.path.getmtime(f) - t), f) for f in glob.glob(str(gen_backend.ROOT / ".cache" / "openalex_request" / "*.json")))
    for d, f in files[:3]:
        v = json.load(open(f))["value"]
        if len(v) == 15 and d <= 3.0:
            return v
    return []


def match_cache_files(rows):
    files = [(os.path.getmtime(f), f) for f in glob.glob(str(gen_backend.ROOT / ".cache" / "hop1_extract_rich" / "*.json"))]
    used, out = set(), []
    for r in rows:
        t = r["ts"] + r["latency_s"]
        cands = sorted((abs(m - t), f) for m, f in files if f not in used and abs(m - t) <= 1.5)
        if cands:
            used.add(cands[0][1]); out.append(cands[0][1])
    return out


def main():
    smoke = {r["case"]: r for r in json.loads((HERE / "data" / "routing_smoke.json").read_text())}
    seeds = {c["case_id"]: c for c in json.loads((HERE / "data" / "pilot_seeds_final.json").read_text())}
    report = []
    for cid in CASES:
        r = smoke[cid]
        axis = next(a for a in r["axis_logs"] if a["mode"] == "related_system_discovery")
        cls, basis, sentence = next(a["scientific_class"] for a in r["expert_prior"]["axes"] if a["execution_mode"] == "related_system_discovery"), axis["basis"], axis["hop1_queries"][0]
        objective = r["planner"]["scientific_objective"]
        cached = hop1_cached_papers(cid)
        papers = [Paper.model_validate(d) for d in cached]
        rows = usage_rows(f"routing_smoke_{cid}")
        files = match_cache_files(rows)
        ext = [json.load(open(f))["value"] for f in files]
        # assign recovered raw extractions to papers by evidence-quote containment
        by_title, unmapped_nonempty = {}, 0
        for e in ext:
            if not e["systems"]:
                continue
            placed = False
            for p in papers:
                tn = RSD._norm(RSD._paper_text(p))
                if any(RSD._norm(m["evidence_quote"]) in tn for m in e["systems"]):
                    by_title.setdefault(p.title, []).extend(e["systems"]); placed = True; break
            unmapped_nonempty += (not placed)
        n_empty = sum(1 for e in ext if not e["systems"])
        # ---- stage 1: raw candidates; stage 2: relationship-fidelity gate (distinctive_structure only)
        gate_rows = []
        for t, ms in by_title.items():
            for m in ms:
                kept = True if basis != "distinctive_structure" else TE._quote_supports_structural_relationship(m["evidence_quote"], m["structural_term_matched"])
                gate_rows.append({"paper": t, "name": m["name"], "entity_type": m["entity_type"], "quote": m["evidence_quote"],
                                  "structural_term_matched": m["structural_term_matched"], "gate_kept": kept,
                                  "gate_reason": ("basis is not distinctive_structure: gate not applied" if basis != "distinctive_structure" else
                                                  ("kept: class term in quote" if m["structural_term_matched"] else "kept: shape morpheme in quote") if kept else
                                                  "DROPPED: quote has neither the class/alias phrase nor a shape-specific morpheme")})

        def extract_fn(title, abstract, structural_class, aka=None):
            ms = [TE.RichSystemMention(**{k: m[k] for k in ("name", "evidence_quote", "structural_term_matched", "entity_type", "identity_context")})
                  for m in by_title.get(title, [])]
            if basis == "distinctive_structure":
                ms = [m for m in ms if TE._quote_supports_structural_relationship(m.evidence_quote, m.structural_term_matched)]
            return TE.RichSystemExtraction(systems=ms)
        disc = RSD.discover_related_systems(cls, [f"{cls} {objective}"], search_fn=lambda q: papers, extract_fn=extract_fn,
                                            max_discovered=150, max_papers_to_extract=150, max_workers=1)
        discovered = disc.discovered_systems
        cleaned, _ = cleanup_discovered_systems(discovered, clean_objective=objective) if discovered else ([], [])
        idents = build_related_system_identities(cleaned, scientific_class=cls, basis=basis) if cleaned else []
        xname, xrx = X_PATTERNS[cid]
        x_in_papers = [{"rank": i, "title": p.title, "year": p.year, "hits": sorted(set(re.findall(xrx, (p.title or "") + " " + (p.abstract or ""), re.I)))[:4]}
                       for i, p in enumerate(papers, 1) if re.search(xrx, (p.title or "") + " " + (p.abstract or ""), re.I)]
        x_in_raw = [g for g in gate_rows if re.search(xrx, g["name"] + " " + g["quote"], re.I)]
        x_in_disc = [d.name for d in discovered if re.search(xrx, d.name, re.I)]
        seed_in_hop1 = [i for i, p in enumerate(papers, 1) if (p.doi or "").lower() == seeds[cid]["doi"].lower()]
        rec = {"case": cid, "expected_X": xname, "seed_doi": seeds[cid]["doi"], "class": cls, "basis": basis, "objective": objective, "hop1_sentence": sentence,
               "n_retrieved": len(papers), "n_extraction_rows_matched": len(files), "n_empty_extractions": n_empty, "n_nonempty_unmapped": unmapped_nonempty,
               "papers": [{"rank": i, "year": p.year, "title": p.title, "has_abstract": bool(p.abstract),
                           "extracted": [{"name": m["name"], "type": m["entity_type"]} for m in by_title.get(p.title, [])]} for i, p in enumerate(papers, 1)],
               "stage_counts": {"papers_retrieved": len(papers), "raw_candidate_mentions": len(gate_rows),
                                "after_relationship_gate": sum(g["gate_kept"] for g in gate_rows),
                                "after_grounding_discovered_systems": len(discovered), "after_cleanup": len(cleaned),
                                "identities_grounded": sum(i.grounding != "insufficient" for i in idents), "identities_total": len(idents)},
               "gate_rows": gate_rows, "grounding_dropped": [{"name": d.name, "cleaned": d.cleaned_name, "reason": d.reason, "quote": d.evidence_quote[:160]} for d in disc.dropped],
               "discovered": [d.name for d in discovered],
               "identities": [{"label": i.label, "grounding": i.grounding, "query_anchor": i.query_anchor} for i in idents],
               "X_in_hop1_papers": x_in_papers, "X_in_raw_extraction": [{"name": g["name"], "gate_kept": g["gate_kept"]} for g in x_in_raw],
               "X_in_discovered": x_in_disc, "seed_in_hop1_top15_rank": seed_in_hop1}
        report.append(rec)
    (HERE / "data" / "hop1_audit.json").write_text(json.dumps({"morphemes_used_by_gate": sorted(TE._SPECIFIC_STRUCTURAL_MORPHEMES), "cases": report}, indent=1, ensure_ascii=False))
    print("morphemes:", sorted(TE._SPECIFIC_STRUCTURAL_MORPHEMES))
    for rec in report:
        print("\n" + "=" * 100); print(rec["case"], "| basis:", rec["basis"], "| class:", rec["class"]); print("  Hop-1 sentence:", rec["hop1_sentence"])
        print("  stage counts:", rec["stage_counts"], "| extraction rows matched:", rec["n_extraction_rows_matched"], "| empty:", rec["n_empty_extractions"], "| nonempty-unmapped:", rec["n_nonempty_unmapped"])
        for p in rec["papers"]:
            print(f"   [{p['rank']:>2}] {p['year']} {p['title'][:88]}  ->  {[e['name'][:30] for e in p['extracted']] or '-'}")
        print("  X present in retrieved papers:", [(x["rank"], x["hits"]) for x in rec["X_in_hop1_papers"]] or "NONE")
        print("  X in raw extraction:", rec["X_in_raw_extraction"] or "none", "| X in discovered:", rec["X_in_discovered"] or "none", "| seed in Hop-1 top15:", rec["seed_in_hop1_top15_rank"] or "no")
        for g in rec["gate_rows"]:
            print(f"   GATE {'KEPT   ' if g['gate_kept'] else 'DROPPED'} {g['name'][:40]:40s} [{g['entity_type'][:22]}] {g['gate_reason'][:70]}")
        for d in rec["grounding_dropped"]:
            print(f"   GROUNDING-DROP {d['name'][:40]:40s} :: {d['reason']}")
        print("  discovered:", rec["discovered"], "| identities:", rec["identities"])


if __name__ == "__main__":
    main()
