"""Seed-paper candidate search + selection for the pilot.  Candidates come from 16 neutral domain queries (NOT
BEAM's own queries) run once through OpenAlex semantic search; selection is rule-based and deterministic.

Hard filters: research article, DOI present, abstract >= 300 chars, CG/mesoscale modelling is in the abstract, not a
review/perspective, not one of the 13 dev-case targets, not in the dev domains (condensates / phase separation / folding).
Selection: one seed per domain first (most recent if it exists), then second picks from domains with the most
candidates until 20; at least 8 of 20 published in 2025-2026.  Output: data/seed_pool.json (every passing candidate;
also the background corpus for the rare-word leakage check) and data/pilot_seeds.json (the 20 seeds).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

DOMAINS = [
    ("membrane_protein_lipid", "coarse-grained molecular dynamics simulation of membrane protein lipid interactions"),
    ("membrane_remodeling", "coarse-grained simulation of lipid membrane curvature fusion pore vesicle"),
    ("chromatin_dna", "coarse-grained model of chromatin nucleosome DNA simulation"),
    ("rna", "coarse-grained model of RNA structure simulation"),
    ("protein_complex_assembly", "coarse-grained simulation of protein complex assembly oligomerization"),
    ("glycan_carbohydrate", "coarse-grained force field for carbohydrates glycans polysaccharides simulation"),
    ("enzyme_allostery", "coarse-grained simulation of enzyme conformational dynamics allostery"),
    ("cytoskeleton_motor", "coarse-grained simulation of actin microtubule filament motor protein"),
    ("virus_capsid", "coarse-grained simulation of viral capsid assembly"),
    ("peptide_biomaterial", "coarse-grained simulation of peptide self-assembly hydrogel nanostructure"),
    ("ligand_binding", "coarse-grained simulation of ligand binding to a protein receptor"),
    ("cell_mesoscale", "mesoscale coarse-grained model of cytoplasm crowding cell"),
    ("amyloid_aggregation", "coarse-grained simulation of amyloid fibril formation aggregation"),
    ("ion_channel_transporter", "coarse-grained simulation of ion channel or transporter conformational change"),
    ("ml_cg_forcefield", "machine learning coarse-grained force field for proteins"),
    ("antimicrobial_pore", "coarse-grained simulation of antimicrobial peptide membrane pore"),
]
# domains whose seeds usually have a specific biomolecular system that has plausible siblings -> L3 track
L3_PREFERRED = ["ion_channel_transporter", "enzyme_allostery", "virus_capsid", "rna", "protein_complex_assembly",
                "amyloid_aggregation", "cytoskeleton_motor", "membrane_protein_lipid"]
CG_RE = re.compile(r"coarse[- ]?grain|\bCG\b|mesoscale|martini|bead|multiscale|united[- ]atom", re.I)
SIM_RE = re.compile(r"simulat|molecular dynamics|\bMD\b|Monte Carlo", re.I)
CORE_RE = re.compile(r"coarse[- ]?grain|\bCG\b|martini|mesoscale", re.I)
BAD_TITLE = re.compile(r"\b(review|perspectives?|opinion|commentary|editorial|tutorial|roadmap|erratum|correction|appraisal|overview|"
                       r"advances|progress|challenges|survey|primer)\b", re.I)
# the first pass uses the plain domain query; later passes (only run while too few recent papers were found) append
# phrasing typical of recent work.  This only widens the candidate pool; selection rules are unchanged.
SUFFIXES = ["", " using Martini 3 or machine-learned coarse-grained models", " with AlphaFold-predicted structures and GPU-accelerated simulations"]
DEV_DOMAIN = re.compile(r"condensate|phase separation|liquid[- ]liquid|protein folding|folding", re.I)


def dev_dois() -> set[str]:
    cases = json.loads((HERE / "data" / "regression_13_cases.json").read_text())["cases"]
    return {d.lower() for c in cases for t in c["targets"].values() for d in t["dois"]}


def passes(p, dev: set[str]) -> bool:
    abs_ = (p.abstract or "").strip()
    return bool(p.doi and len(abs_) >= 300 and (p.oa_type == "article") and p.year and p.year >= 2012
                and CG_RE.search(abs_) and SIM_RE.search(abs_) and len(CORE_RE.findall((p.title or "") + " " + abs_)) >= 2
                and not BAD_TITLE.search(p.title or "")
                and not DEV_DOMAIN.search((p.title or "") + " " + abs_[:600]) and p.doi.lower() not in dev)


def search_pool(limit: int = 25, min_recent_in_pool: int = 14) -> list[dict]:
    import regression_lib as RL
    from cgscout.sources.openalex import search_openalex
    RL.harness_guard(len(DOMAINS) * len(SUFFIXES), what="bench_v1_seed_search")
    RL.U.set_run_label("bench_v1_seed_search")
    dev, out, seen = dev_dois(), [], set()
    for suffix in SUFFIXES:
        for dom, q in DOMAINS:
            for p in search_openalex(q + suffix, limit=limit, mode="semantic"):
                if p.doi and p.doi.lower() in seen:
                    continue
                if passes(p, dev):
                    seen.add(p.doi.lower())
                    out.append({"domain": dom, "doi": p.doi, "title": p.title, "year": p.year, "venue": p.venue,
                                "authors": p.authors, "abstract": p.abstract, "oa_type": p.oa_type, "found_by_pass": SUFFIXES.index(suffix)})
        if sum(1 for c in out if (c["year"] or 0) >= 2025) >= min_recent_in_pool:
            break
    return out


def tag_pool(pool: list[dict], backend) -> None:
    """One cheap LLM call per candidate: is it about a specific named system, and what kind of paper is it?"""
    from concurrent.futures import ThreadPoolExecutor
    from prompts import SEED_TAG_SYSTEM
    from schemas import SeedTag

    def one(c):
        if "system_specific" in c:
            return
        t = backend.structured(system=SEED_TAG_SYSTEM, user="ABSTRACT:\n" + c["abstract"], schema=SeedTag,
                               label="bench_seed_tag", effort="low", purpose="bench_v1", case_id="bench_seed_pool", timeout=90)
        c.update(system_specific=t.system_specific, named_system=t.named_system, paper_type=t.paper_type)
    with ThreadPoolExecutor(max_workers=6) as ex:
        list(ex.map(one, pool))


def select(pool: list[dict], n: int = 20, min_recent: int = 8) -> list[dict]:
    by_dom: dict[str, list[dict]] = {}
    for c in pool:
        by_dom.setdefault(c["domain"], []).append(c)
    for v in by_dom.values():
        # recent first (>=2025), then real applications over tool/method papers, then newest, DOI as the tie-break
        v.sort(key=lambda c: (-int((c["year"] or 0) >= 2025), c.get("paper_type") != "application", -(c["year"] or 0), c["doi"]))
    chosen: list[dict] = []
    for dom, _ in DOMAINS:                                            # first pass: one per domain
        if by_dom.get(dom):
            chosen.append(by_dom[dom].pop(0))
    rest = sorted((c for v in by_dom.values() for c in v), key=lambda c: (-len(by_dom[c["domain"]]), -(c["year"] or 0), c["doi"]))
    for c in rest:                                                    # second pass: fill up
        if len(chosen) >= n:
            break
        if sum(1 for x in chosen if x["domain"] == c["domain"]) < 2:
            chosen.append(c)
    chosen = chosen[:n]
    if sum(1 for c in chosen if (c["year"] or 0) >= 2025) < min_recent:     # swap old picks for recent ones
        recent = [c for c in rest if (c["year"] or 0) >= 2025 and c not in chosen]
        old = [c for c in chosen if (c["year"] or 0) < 2025]
        while recent and old and sum(1 for c in chosen if (c["year"] or 0) >= 2025) < min_recent:
            chosen.remove(old.pop())
            chosen.append(recent.pop(0))
    return chosen


def assign_types(seeds: list[dict], n_l3: int = 6, n_l1: int = 6) -> list[dict]:
    """6 L3 seeds from the L3-preferred domains, then L1 (6) / L2 (rest) alternating over the remaining seeds."""
    l3 = []
    for dom in L3_PREFERRED:                      # L3 needs a specific named system (otherwise sister proposals are vacuous)
        for s in seeds:
            if s["domain"] == dom and s.get("system_specific") and s.get("paper_type") in ("application", "method_or_forcefield") \
                    and s not in l3 and len(l3) < n_l3:
                l3.append(s)
                break
    for s in sorted(seeds, key=lambda x: (x.get("paper_type") != "application", x["doi"])):      # fill from any other specific system
        if len(l3) < n_l3 and s not in l3 and s.get("system_specific") and s.get("paper_type") in ("application", "method_or_forcefield"):
            l3.append(s)
    rest = [s for s in seeds if s not in l3]
    for s in seeds:
        s["case_type"] = "L3" if s in l3 else None
    # L1 ("direct": the query names the system) needs a specific named system; everything else is L2 (abstracted need)
    cand_l1 = sorted((s for s in rest if s.get("system_specific")), key=lambda x: (x.get("paper_type") != "application", x["doi"]))
    for s in rest:
        s["case_type"] = "L2"
    for s in cand_l1[:n_l1]:
        s["case_type"] = "L1"
    for i, s in enumerate(sorted(seeds, key=lambda x: x["doi"]), 1):
        s["case_id"] = f"P{i:02d}"
    return sorted(seeds, key=lambda s: s["case_id"])


def main():
    import regression_lib as RL
    from gen_backend import get_generator_backend
    pool_path = HERE / "data" / "seed_pool.json"
    pool = json.loads(pool_path.read_text()) if ("--reuse" in sys.argv and pool_path.exists()) else RL.run_guarded(search_pool)
    tag_pool(pool, get_generator_backend())
    (HERE / "data" / "seed_pool.json").write_text(json.dumps(pool, indent=1, ensure_ascii=False))
    seeds = assign_types(select(pool))
    (HERE / "data" / "pilot_seeds.json").write_text(json.dumps(seeds, indent=1, ensure_ascii=False))
    print(f"pool: {len(pool)} candidates in {len({c['domain'] for c in pool})}/{len(DOMAINS)} domains")
    print(f"selected {len(seeds)}: " + ", ".join(f"{t}={sum(1 for s in seeds if s['case_type']==t)}" for t in ('L1', 'L2', 'L3'))
          + f" | recent(>=2025)={sum(1 for s in seeds if (s['year'] or 0) >= 2025)}")
    for s in seeds:
        print(f"{s['case_id']} {s['case_type']} {s['domain']:24s} {s['year']} {s.get('paper_type','?')[:11]:11s} {'S' if s.get('system_specific') else '-'} {s['title'][:70]}")


if __name__ == "__main__":
    main()
