"""Pure unit tests for related_system_hop2 -- query building + deterministic
bookkeeping. No LLM, no network: search_fn is a fake.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks" / "concept_resolver_reasoning"))

from related_system_discovery import DiscoveredSystem, SupportingPaper  # noqa: E402
from related_system_hop2 import (  # noqa: E402
    ROLE_SYSTEM_OBJECTIVE_METHODOLOGY,
    build_hop2_queries,
    discover_hop2,
    run_hop2,
    select_systems_for_hop2,
)
from beam_agent.models import Paper  # noqa: E402


def _sys(name, n_support=1, rank=0, paper_ref="10/x", quote="q", title="T"):
    sp = SupportingPaper(paper_ref=paper_ref, title=title, found_by_query="beta-helix",
                         retrieval_rank=rank, evidence_quote=quote,
                         structural_term_matched="beta-helix", system_name=name)
    return DiscoveredSystem(name=name, norm_name=name.lower(), supporting_papers=[sp],
                            n_support=n_support, first_seen_rank=rank)


def _paper(title, abstract="", doi=None, year=2020):
    return Paper(title=title, abstract=abstract, doi=doi, year=year)


def _search(mapping):
    return lambda q: mapping.get(q, [])


# --- query building: ONE query per system (was three bounded roles) ------

def test_one_query_per_system_exact_string():
    qs, _ = build_hop2_queries([_sys("EspP")], "beta-helix protein", "protein folding")
    assert len(qs) == 1
    assert qs[0].role == ROLE_SYSTEM_OBJECTIVE_METHODOLOGY
    assert qs[0].query == "EspP protein folding coarse-grained model"


def test_objective_is_expected_pre_cleaned_by_the_caller():
    """build_hop2_queries() no longer cleans the objective itself -- that's
    route_retrieval_plan.py's job now (objective_normalization.
    normalize_objective(), called before this function). Passing an
    already-clean objective like "folding" (not a raw sentence) produces
    the query verbatim, uninspected."""
    qs, _ = build_hop2_queries([_sys("Pertactin")], "beta-helix", "folding")
    assert qs[0].query == "Pertactin folding coarse-grained model"


def test_no_terminology_expansion_and_no_class_terms_in_queries():
    qs, _ = build_hop2_queries([_sys("CdrA")], "beta-helix protein", "protein folding",
                               also_known_as=["beta-solenoid", "autotransporter passenger domain"])
    joined = " ".join(q.query for q in qs).lower()
    assert "beta-helix" not in joined and "beta-solenoid" not in joined
    assert "autotransporter" not in joined


def test_query_degrades_to_methodology_only_when_objective_missing():
    qs, _ = build_hop2_queries([_sys("EspP")], "beta-helix protein", "")
    assert len(qs) == 1
    assert qs[0].role == ROLE_SYSTEM_OBJECTIVE_METHODOLOGY  # same role covers every query now
    assert qs[0].query == "EspP coarse-grained model"


def test_no_query_at_all_when_both_objective_and_methodology_are_empty():
    qs, _ = build_hop2_queries([_sys("EspP")], "beta-helix protein", "", methodology_phrase="")
    assert qs == []


def test_provenance_chain_is_carried_verbatim():
    s = _sys("HET-s(218-289)", paper_ref="10.1126/science.1151839",
             quote="HET-s(218-289) forms a left-handed beta solenoid", title="Amyloid fibrils...")
    qs, _ = build_hop2_queries([s], "beta-helix protein", "protein folding",
                               also_known_as=["beta-solenoid"])
    p = qs[0].provenance
    assert p.structural_class == "beta-helix protein"
    assert p.also_known_as == ["beta-solenoid"]
    assert p.discovered_system == "HET-s(218-289)"
    assert p.supporting_paper_ref == "10.1126/science.1151839"
    assert p.evidence_quote == "HET-s(218-289) forms a left-handed beta solenoid"


def test_cap_at_8_uses_support_then_rank_no_llm():
    systems = [_sys(f"S{i}", n_support=1, rank=i) for i in range(12)]
    systems[9] = _sys("S9", n_support=3, rank=9)          # more support -> first
    kept, note = select_systems_for_hop2(systems, max_systems=8)
    assert len(kept) == 8
    assert kept[0].name == "S9"
    assert kept[1].name == "S0"                            # then by rank
    assert "S9" in note and "Dropped" in note


def test_a_single_grounded_paper_is_enough():
    kept, _ = select_systems_for_hop2([_sys("EspP", n_support=1)], max_systems=8)
    assert [k.name for k in kept] == ["EspP"]
    # a system with no supporting paper never qualifies
    bare = DiscoveredSystem(name="ghost", norm_name="ghost", supporting_papers=[], n_support=0)
    kept2, _ = select_systems_for_hop2([bare], max_systems=8)
    assert kept2 == []


# --- retrieval bookkeeping ------------------------------------------

def test_run_hop2_classifies_and_compares_to_generic():
    # ONE query per system now -- the query itself can still return MULTIPLE
    # papers, so folding-evidence / cg-evidence / generic-overlap detection
    # is exercised the same way, just via one query's result list instead
    # of three separate queries' results.
    hits = [
        _paper("EspP folding kinetics", "The autotransporter EspP passenger domain folding was studied.", doi="10/a"),
        _paper("Coarse-grained model of EspP", "A coarse-grained model of the EspP beta-helix.", doi="10/b"),
    ]
    qs, note = build_hop2_queries([_sys("EspP", paper_ref="10/hop1")], "beta-helix protein", "protein folding")
    assert len(qs) == 1
    r = run_hop2(
        qs, search_fn=_search({qs[0].query: hits}),
        structural_class="beta-helix protein", scientific_objective="protein folding",
        generic_paper_refs={"10/a"},                      # 10/a already known to generic retrieval
        systems_available=1, system_selection_note=note,
    )
    assert r.systems_carried == 1
    assert r.n_papers_unique == 2                         # 10/a and 10/b
    assert r.n_overlap_with_generic == 1                  # 10/a
    assert r.n_hop2_only == 1 and r.hop2_only_refs == ["10/b"]
    s = r.per_system[0]
    assert s.has_folding_evidence and s.has_cg_evidence
    assert "10/a" in s.folding_evidence_refs
    assert "10/b" in s.cg_evidence_refs
    assert s.provenance.supporting_paper_ref == "10/hop1"


def test_absence_of_downstream_evidence_is_recorded_not_filled():
    # the one query returns papers that never mention the system -> noisy, no evidence
    junk = [_paper("Amyloid beta and Alzheimer disease", "unrelated review of amyloid-beta.", doi="10/j")]
    qs, note = build_hop2_queries([_sys("CdrA")], "beta-helix protein", "protein folding")
    r = run_hop2(qs, search_fn=lambda q: junk,
                 structural_class="beta-helix protein", scientific_objective="protein folding",
                 systems_available=1, system_selection_note=note)
    s = r.per_system[0]
    assert s.has_folding_evidence is False and s.has_cg_evidence is False
    assert s.system_name in r.systems_with_no_downstream_evidence
    assert len(s.noisy_queries) == 1                      # the one query drifted off-system


def test_empty_and_error_queries_are_tracked_by_source():
    # ONE query per system now -- exercise both empty-result and search-error
    # paths via TWO different systems (each contributes exactly one query)
    # rather than via different roles of the same system.
    def _sf(q):
        if q.startswith("CdrA"):
            raise RuntimeError("openalex 500")
        return []
    qs, note = build_hop2_queries([_sys("EspP"), _sys("CdrA")], "beta-helix protein", "protein folding")
    assert len(qs) == 2
    r = run_hop2(qs, search_fn=_sf, structural_class="beta-helix protein",
                 scientific_objective="protein folding", systems_available=2,
                 system_selection_note=note)
    espp_sr = next(s for s in r.per_system if s.system_name == "EspP")
    cdra_sr = next(s for s in r.per_system if s.system_name == "CdrA")
    assert espp_sr.empty_queries == ["EspP protein folding coarse-grained model"]
    assert len(cdra_sr.search_errors) == 1 and "openalex 500" in cdra_sr.search_errors[0]
    assert len(r.search_errors) == 1


def test_discover_hop2_end_to_end_shape():
    r = discover_hop2(
        [_sys("EspP", rank=1), _sys("CdrA", rank=2)],
        "beta-helix protein", "protein folding",
        search_fn=lambda q: [_paper(f"{q} paper", f"about {q}", doi=f"10/{hash(q) % 1000}")],
        also_known_as=["beta-solenoid"],
    )
    assert r.systems_carried == 2
    assert {s.system_name for s in r.per_system} == {"EspP", "CdrA"}
    assert all(len(s.queries) == 1 for s in r.per_system)   # was 3
    assert r.methodology_phrase == "coarse-grained model"
