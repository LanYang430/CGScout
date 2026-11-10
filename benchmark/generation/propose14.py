"""14 L1/L2 seed proposals for the 20-case pilot: up to 6 from the user's curated pool (clean_for_pilot only, capped per
domain bucket) + 8 from a NEW application-oriented, family-diverse OpenAlex search.  Nothing is generated.
Caps (so no single class dominates): membrane/Martini <= 3, IDP/condensate <= 3 (both from the curated pool), the rest
from other systems; at most 1 new seed that mentions Martini.  Exclusions: any DOI in the classifier train/test/label files,
the dev-13 targets, the 30 curated papers themselves (for the new search), and papers without a usable abstract."""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_backend  # noqa: E402,F401
import seeds as SD  # noqa: E402

QUERIES = [
    ("protein_binding_awsem", "AWSEM coarse-grained simulation of protein-protein binding and association mechanism"),
    ("dna_nanotech_oxdna", "oxDNA coarse-grained simulation of DNA nanostructures and DNA origami"),
    ("nucleosome_dna_3spn", "3SPN coarse-grained simulation of nucleosome dynamics and protein-DNA search"),
    ("rna_riboswitch", "coarse-grained simulation of RNA folding riboswitch ribozyme dynamics"),
    ("virus_capsid_genome", "coarse-grained simulation of virus capsid assembly with genome encapsidation"),
    ("cytoskeleton_motor", "coarse-grained simulation of microtubule actin kinesin dynein mechanics"),
    ("enzyme_go_model", "structure-based Go-model simulation of enzyme conformational change and catalysis"),
    ("unres_protein", "UNRES coarse-grained simulation of protein conformational dynamics"),
    ("sirah_system", "SIRAH coarse-grained simulation of protein-DNA or membrane system"),
    ("dpd_nanoparticle", "dissipative particle dynamics simulation of lipid nanoparticle or polymer micelle"),
    ("chromatin_polymer", "coarse-grained polymer model of chromatin loop extrusion and genome organization"),
    ("fibrous_protein", "coarse-grained simulation of collagen intermediate filament or fibril mechanics"),
    ("protein_surface", "coarse-grained simulation of protein adsorption on a surface or nanoparticle"),
    ("molecular_machine", "coarse-grained simulation of ribosome or molecular machine conformational dynamics"),
]
FAMILY = [("martini", r"martini"), ("hps_calvados_mpipi", r"\bHPS\b|calvados|mpipi|hydrophobicity scale"), ("awsem", r"awsem"),
          ("oxdna_3spn", r"oxdna|3spn|\boxrna\b"), ("unres_opep_cabs", r"unres|opep|cabs"), ("sirah", r"sirah"),
          ("go_structure_based", r"g[oō]-?like|structure-based|go model|elastic network"), ("dpd_polymer", r"dissipative particle|polymer model|bead-spring")]
BUCKET_CURATED = {2: "membrane", 9: "membrane", 14: "membrane", 21: "idp_condensate", 22: "idp_condensate", 23: "idp_condensate"}


def family(text: str) -> str:
    return next((n for n, rx in FAMILY if re.search(rx, text, re.I)), "other/bespoke")


def overlap_dois() -> set[str]:
    dl = Path("~/Downloads")
    out = set()
    for f in ("beam_stage3_training_ready_yes_no.csv", "beam_stage3_external_test_clean_519.csv", "beam_stage3_training_labels_89.csv", "beam_stage3_silver_labels_91_web_reviewed.csv"):
        out |= {(r["doi"] or "").lower() for r in csv.DictReader(open(dl / f, encoding="utf-8"))}
    out |= SD.dev_dois()
    out |= {p["doi"] for p in json.loads((HERE / "data" / "curated_pool_audit.json").read_text())["papers"] if p["doi"]}
    return out


def cached_abstract(doi: str) -> str:
    import glob
    for f in glob.glob(str(gen_backend.ROOT / ".cache" / "openalex_request" / "*.json")):
        try:
            v = json.load(open(f))["value"]
        except Exception:  # noqa: BLE001
            continue
        if isinstance(v, list):
            for p in v:
                if isinstance(p, dict) and (p.get("doi") or "").lower() == doi.lower() and p.get("abstract"):
                    return p["abstract"]
    return ""


def new_search(limit: int = 25) -> list[dict]:
    import regression_lib as RL
    from cgscout.sources.openalex import search_openalex
    SD.DEV_DOMAIN = re.compile(r"condensate|phase separation|liquid[- ]liquid", re.I)      # folding is allowed here; condensates are covered by the curated IDP slots
    RL.harness_guard(len(QUERIES), what="bench_v1_propose14")
    RL.U.set_run_label("bench_v1_propose14")
    bad, seen, out = overlap_dois(), set(), []
    for dom, q in QUERIES:
        for p in search_openalex(q, limit=limit, mode="semantic"):
            d = (p.doi or "").lower()
            if d and d not in seen and d not in bad and SD.passes(p, set()):
                seen.add(d)
                out.append({"domain": dom, "doi": p.doi, "title": p.title, "year": p.year, "venue": p.venue, "authors": p.authors,
                            "abstract": p.abstract, "oa_type": p.oa_type})
    return out


def main():
    audit = {p["id"]: p for p in json.loads((HERE / "data" / "curated_pool_audit.json").read_text())["papers"]}
    import regression_lib as RL
    pool = RL.run_guarded(new_search)
    backend = gen_backend.get_generator_backend()
    SD.tag_pool(pool, backend)
    for c in pool:
        c["family"] = family(c["title"] + " " + c["abstract"])
        c["martini"] = c["family"] == "martini" or bool(re.search("martini", c["title"] + c["abstract"], re.I))
    # rank: application first, specific system, non-Martini, recent
    def score(c):
        return (c.get("paper_type") != "application", not c.get("system_specific"), c["martini"], -int((c["year"] or 0) >= 2022), -(c["year"] or 0), c["doi"])
    new, used_dom, n_mart = [], set(), 0
    for c in sorted(pool, key=score):
        if c["domain"] in used_dom or len(new) >= 8:
            continue
        if c["martini"] and n_mart >= 1:
            continue
        n_mart += int(c["martini"])
        used_dom.add(c["domain"]); new.append(c)
    cur = []
    for cid, bucket in BUCKET_CURATED.items():
        a = audit[cid]
        ab = cached_abstract(a["doi"])
        cur.append({"source": "curated", "curated_id": cid, "bucket": bucket, "doi": a["doi"], "title": a["title"], "year": a["year"], "venue": a.get("venue"),
                    "abstract": ab, "abstract_chars": len(ab), "audit_status": a["status"]})
    props = cur + [{"source": "new_search", "bucket": c["domain"], **{k: c[k] for k in ("doi", "title", "year", "venue", "abstract", "paper_type", "system_specific", "named_system", "family", "martini")}} for c in new]
    for p in props:
        p.setdefault("family", family(p["title"] + " " + p.get("abstract", "")))
        p["martini"] = p.get("martini") or p["family"] == "martini" or bool(re.search("martini", p["title"] + p.get("abstract", ""), re.I))
    (HERE / "data" / "seed_proposal_14.json").write_text(json.dumps({"proposals": props, "new_search_pool_size": len(pool)}, indent=1, ensure_ascii=False))
    print("new-search pool after exclusions:", len(pool), "| picked", len(new))
    for p in props:
        print(f"{p['source'][:3]} {p['bucket'][:20]:20s} {p['year']} {p['family'][:12]:12s} {'M' if p['martini'] else ' '} {str(p.get('paper_type', '?'))[:11]:11s} {p['doi']:30s} {p['title'][:62]}")


if __name__ == "__main__":
    main()
