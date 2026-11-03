"""Assemble the FINAL 20 pilot seeds (6 L1 + 8 L2 + 6 L3) from decisions already made:
  L1/L2: the 14 proposals in data/seed_proposal_14.json (kept as is; L1 = named-system papers).
  L3:    #1-#4 verified earlier (data/l3_verification.json) + Pol II initiation -> archaeal RNAP and dynamin -> yeast Mgm1
         (data/l3_finalists.json).  Target systems Y are FIXED here (not re-proposed); their text deliberately avoids the
         seed system's name so the L3 query cannot leak X.
Abstracts / authors / venue come from the OpenAlex request cache (already fetched), never from a new request."""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_backend  # noqa: E402

L1 = ["10.1038/s41467-024-49396-6", "10.1038/s42003-025-09161-x", "10.1038/s41467-024-53136-1",
      "10.52396/justc-2023-0064", "10.1039/c4cp06100d", "10.1063/1.4954992"]
L2 = ["10.1038/s41467-025-60593-9", "10.1038/s41467-024-47602-z", "10.1038/s41467-025-58445-7", "10.1063/5.0246977",
      "10.3390/v15081672", "10.1093/nar/gkx378", "10.3389/fpls.2013.00555", "10.1021/jp409326f"]
BUCKET = {"10.1038/s41467-024-49396-6": "membrane", "10.1038/s42003-025-09161-x": "membrane", "10.1038/s41467-024-53136-1": "idp_condensate",
          "10.52396/justc-2023-0064": "molecular_machine", "10.1039/c4cp06100d": "cytoskeleton", "10.1063/1.4954992": "rna",
          "10.1038/s41467-025-60593-9": "membrane", "10.1038/s41467-024-47602-z": "idp_condensate", "10.1038/s41467-025-58445-7": "idp_condensate",
          "10.1063/5.0246977": "chromatin_dna", "10.3390/v15081672": "virus_capsid", "10.1093/nar/gkx378": "dna_nanotech",
          "10.3389/fpls.2013.00555": "membrane", "10.1021/jp409326f": "protein_surface"}
# L3: (seed doi, source system X, domain, target Y).  Y text never names X.
L3 = [
    ("10.1021/jp9053567", "MscS", "ion_channel", {"system_name": "Arabidopsis MSL10",
        "description": "a plant ion channel that opens when the membrane is stretched",
        "shared_feature": "tension-dependent gating of a mechanosensitive channel"}),
    ("10.1021/ja2028487", "E. coli 70S ribosome", "ribonucleoprotein", {"system_name": "chloroplast 70S ribosome",
        "description": "the plastid ribosome, a two-subunit bacterial-type ribosome with plastid-specific additions",
        "shared_feature": "inter-subunit contacts and collective motions of a two-subunit ribosome"}),
    ("10.1038/s41467-021-27315-3", "YidC", "membrane_insertion", {"system_name": "chloroplast Alb3 insertase",
        "description": "a thylakoid-membrane insertase that inserts transmembrane proteins into the membrane",
        "shared_feature": "binding and insertion of a single transmembrane protein by an insertase"}),
    ("10.1529/biophysj.106.093443", "bacterial flagellum", "motility_filament", {"system_name": "archaeal archaellum",
        "description": "the rotating helical motility filament of archaea",
        "shared_feature": "rotation and torque transmission of a long helical protein filament"}),
    ("10.3389/fmolb.2021.772486", "RNA polymerase II transcription initiation complex", "transcription", {"system_name": "Sulfolobus RNA polymerase",
        "description": "the archaeal RNA polymerase, which initiates transcription with TBP- and TFB-type factors",
        "shared_feature": "promoter DNA opening during transcription initiation"}),
    ("10.1021/acs.jpclett.5c02867", "dynamin", "membrane_remodeling", {"system_name": "yeast Mgm1",
        "description": "a yeast mitochondrial GTPase that assembles on the inner membrane and remodels it",
        "shared_feature": "nucleotide-dependent assembly that constricts a membrane"}),
]


def cached_paper(doi: str) -> dict | None:
    best = None
    for f in glob.glob(str(gen_backend.ROOT / ".cache" / "openalex_request" / "*.json")):
        try:
            v = json.load(open(f))["value"]
        except Exception:  # noqa: BLE001
            continue
        if isinstance(v, list):
            for p in v:
                if isinstance(p, dict) and (p.get("doi") or "").lower() == doi.lower() and (best is None or len(p.get("abstract") or "") > len(best.get("abstract") or "")):
                    best = p
    return best


def main():
    ver = {r["source_seed_X"]["doi"].lower(): r for r in json.loads((HERE / "data" / "l3_verification.json").read_text())}
    fin = {r["source_seed_X"]["doi"].lower(): r for r in json.loads((HERE / "data" / "l3_finalists.json").read_text())}
    props = {p["doi"].lower(): p for p in json.loads((HERE / "data" / "seed_proposal_14.json").read_text())["proposals"]}
    seeds, missing = [], []
    order = [(d, "L1") for d in L1] + [(d, "L2") for d in L2] + [(t[0], "L3") for t in L3]
    for n, (doi, ctype) in enumerate(order, 1):
        p = cached_paper(doi)
        src = "openalex"
        if p is not None and len(p.get("abstract") or "") < 300:     # OpenAlex lacks the abstract -> the same Europe PMC fallback Stage 4 uses (cached)
            sys.path.insert(0, str(gen_backend.MI))
            import stage4_v2 as S4
            from beam_agent.models import Paper
            e = S4.europepmc_abstract(Paper(title=p["title"], doi=doi))
            if e and len(e.get("abstract") or "") >= 300:
                p = {**p, "abstract": e["abstract"]}
                src = "europepmc"
        if not p or len(p.get("abstract") or "") < 300:
            missing.append(doi)
            continue
        s = {"case_id": f"Q{n:02d}", "case_type": ctype, "doi": p["doi"], "title": p["title"], "year": p.get("year"), "venue": p.get("venue"),
             "authors": p.get("authors") or [], "abstract": p["abstract"], "abstract_source": src, "oa_type": p.get("oa_type")}
        if ctype == "L3":
            _, x, dom, y = next(t for t in L3 if t[0] == doi)
            ev = fin.get(doi.lower()) or ver.get(doi.lower())
            s.update(domain=dom, named_system=x, l3_target={**y, "source_system": x, "verified_no_direct": ev["verified_no_direct"],
                     "evidence": {"n_queries": len(ev["direct_CG_search_queries"]), "n_unique_papers_judged": ev["n_unique_papers_judged"],
                                  "direct_hits": ev["direct_CG_hits_for_Y"], "transfer_axis": (ev.get("transfer") or {}).get("transfer_axis"),
                                  "confidence": (ev.get("transfer") or {}).get("confidence"), "caveat": (ev.get("transfer") or {}).get("caveat")}})
        else:
            s.update(domain=BUCKET[doi], source=props[doi.lower()]["source"] if doi.lower() in props else None)
        seeds.append(s)
    (HERE / "data" / "pilot_seeds_final.json").write_text(json.dumps(seeds, indent=1, ensure_ascii=False))
    print(f"{len(seeds)} seeds written; missing from cache: {missing}")
    for s in seeds:
        print(f"{s['case_id']} {s['case_type']} {s['domain']:18s} {s['year']} {s['doi']:30s} {s['title'][:70]}")


if __name__ == "__main__":
    main()
