"""Find replacement L3 candidates (seed X -> target Y).  Constraints: seed = a CG APPLICATION paper from 2020 on about a
specific system, in a domain other than mechanosensitive channels / ribosomes / insertases / rotating filaments; Y = a
related system proposed from the seed abstract; Y's direct-CG literature is then checked with the same 3-query
(2 semantic + 1 lexical, top 20) + LLM judgement used for L3 #1-#6.  Benchmark code only; nothing is generated."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_backend  # noqa: E402,F401
import prompts as P  # noqa: E402
import seeds as SD  # noqa: E402
from schemas import SisterProposals  # noqa: E402

SEED_QUERIES = [
    ("crispr", "coarse-grained simulation of CRISPR-Cas9 DNA interrogation and R-loop formation"),
    ("cell_wall", "coarse-grained simulation of peptidoglycan bacterial cell wall mechanics"),
    ("dynamin_like", "coarse-grained simulation of mitochondrial dynamin-like GTPase membrane remodeling"),
    ("viral_glycoprotein", "coarse-grained simulation of viral fusion glycoprotein conformational change"),
    ("nanocage", "coarse-grained simulation of bacterial microcompartment shell or protein nanocage assembly"),
    ("lps_membrane", "coarse-grained simulation of lipopolysaccharide outer membrane asymmetric bilayer"),
    ("silk_elastin", "coarse-grained simulation of spider silk or elastin protein assembly and mechanics"),
    ("immune_recognition", "coarse-grained simulation of antibody or T-cell receptor peptide recognition"),
    ("biomass", "coarse-grained model of cellulose or lignin"),
    ("kinase", "coarse-grained simulation of protein kinase conformational allostery"),
]
SEED_QUERIES_R2 = [
    ("lipid_transfer", "coarse-grained simulation of bridge-like lipid transfer protein VPS13 lipid transport between membranes"),
    ("pore_forming", "coarse-grained simulation of pore-forming protein oligomerization gasdermin perforin toxin membrane pore"),
    ("escrt", "coarse-grained simulation of ESCRT-III polymer membrane scission"),
    ("snare_fusion", "coarse-grained simulation of SNARE-mediated membrane fusion"),
    ("dna_repair_search", "coarse-grained simulation of DNA repair protein or transcription factor sliding and target search on DNA"),
    ("atp_synthase", "coarse-grained simulation of ATP synthase or respiratory supercomplex in a membrane"),
]
SEED_QUERIES_R3 = [("cas9", "coarse-grained Cas9"), ("escrt", "coarse-grained ESCRT-III"), ("snare", "coarse-grained SNARE fusion"),
                   ("gasdermin", "coarse-grained gasdermin pore"), ("vps13", "coarse-grained lipid transfer protein"),
                   ("atp_synthase", "coarse-grained ATP synthase"), ("cohesin", "coarse-grained cohesin condensin"),
                   ("nuclear_pore", "coarse-grained nuclear pore complex"), ("polymerase", "coarse-grained RNA polymerase transcription"),
                   ("spliceosome", "coarse-grained chromatin remodeler"), ("kinesin", "coarse-grained kinesin dynein stepping"),
                   ("bam_bcs", "coarse-grained bacterial secretion system")]
EXCLUDE_TOPIC = re.compile(r"mechanosensitive|piezo|msc[sl]|ribosom|insertase|yidc|oxa1|sec61|flagell|archaell|pilus|pili\b|microtubule|condensate|phase separation", re.I)


def main():
    sys.path.insert(0, str(gen_backend.ROOT / "benchmarks" / "regression_13"))
    import regression_lib as RL
    from beam_agent.sources.openalex import search_openalex
    sys.path.insert(0, str(gen_backend.MI))
    import stage4_v2 as S4
    import propose14 as PR
    from l3_verify import verify, CANDIDATES
    backend = gen_backend.get_generator_backend()
    bad = PR.overlap_dois() | {c["seed_doi"].lower() for c in CANDIDATES}
    bad |= {p["doi"].lower() for p in json.loads((HERE / "data" / "seed_proposal_14.json").read_text())["proposals"]}
    SD.DEV_DOMAIN = re.compile(r"condensate|phase separation|liquid[- ]liquid", re.I)
    rnd3 = "--round3" in sys.argv
    rnd2 = "--round2" in sys.argv or rnd3
    queries = SEED_QUERIES_R3 if rnd3 else (SEED_QUERIES_R2 if rnd2 else SEED_QUERIES)
    seed_mode = "lexical" if rnd3 else "semantic"
    out = {"seeds": [], "pairs": []}

    def go():
        RL.harness_guard(len(queries) + 40, what="bench_v1_l3_find_more")
        RL.U.set_run_label("bench_v1_l3_find_more")
        search = lambda q, k, m: search_openalex(q, limit=k, mode=m)  # noqa: E731
        pool, seen = [], set()
        for dom, q in queries:
            for p in search_openalex(q, limit=25, mode=seed_mode):
                d = (p.doi or "").lower()
                if d and d not in seen and d not in bad and (p.year or 0) >= 2020 and SD.passes(p, set()) and not EXCLUDE_TOPIC.search(p.title + " " + (p.abstract or "")[:500]):
                    seen.add(d)
                    pool.append({"domain": dom, "doi": p.doi, "title": p.title, "year": p.year, "venue": p.venue, "abstract": p.abstract, "authors": p.authors})
        SD.tag_pool(pool, backend)
        best = {}
        for c in sorted(pool, key=lambda c: (c.get("paper_type") != "application", not c.get("system_specific"), -(c["year"] or 0))):
            if c.get("system_specific") and c.get("paper_type") in ("application", "method_or_forcefield") and c["domain"] not in best:
                best[c["domain"]] = c
        out["seeds"] = list(best.values())[:7]
        print(f"seed pool {len(pool)} -> {len(best)} domains with a usable application seed; checking {len(out['seeds'])}", flush=True)
        for s in out["seeds"]:
            props = backend.structured(system=P.SISTER_SYSTEM_V2, user="ABSTRACT:\n" + s["abstract"], schema=SisterProposals, label="bench_findmore_sisters",
                                       effort="medium", purpose="bench_v1_l3_find_more", case_id="bench_l3_more", timeout=120).proposals[:2]
            for y in props:
                name = y.system_name.strip()
                cand = {"n": f"{s['domain']}:{name}", "short": f"{s['title'][:40]} -> {name}", "seed_title": s["title"], "seed_doi": s["doi"], "Y": name,
                        "Y_objective": y.shared_feature, "claimed_axis": y.shared_feature,
                        "queries": [("semantic", f"coarse-grained model simulation of {name}"), ("semantic", f"{name} molecular dynamics simulation mechanism"),
                                    ("lexical", f"{name} coarse-grained")]}
                r = verify(cand, backend, search, S4.europepmc_abstract, {})
                r["seed_domain"], r["Y_proposal"] = s["domain"], y.model_dump()
                out["pairs"].append(r)
                print(f"{s['domain']:18s} {name[:34]:34s} no_direct={r['verified_no_direct']} hits={len(r['direct_CG_hits_for_Y'])} judged={r['n_unique_papers_judged']} q={[q['n_results'] for q in r['direct_CG_search_queries']]}", flush=True)
        print("OpenAlex:", RL.U.run_usage("bench_v1_l3_find_more"))
    RL.run_guarded(go)
    (HERE / "data" / ("l3_more_candidates_r3.json" if rnd3 else ("l3_more_candidates_r2.json" if rnd2 else "l3_more_candidates.json"))).write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
