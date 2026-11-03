"""Benchmark-generation code (benchmarks/beam_bench_v1): leakage checks, blind-generation wiring, L3 sister logic,
CSV schema and the probe.  A fake backend / fake search stand in for the LLM and OpenAlex -- no network."""
import csv
import json
import sys
from pathlib import Path

import pytest

BB = Path(__file__).resolve().parents[1] / "benchmarks" / "beam_bench_v1"
sys.path.insert(0, str(BB))

import generate as G  # noqa: E402
import leakage as LK  # noqa: E402
import probe as PR  # noqa: E402
import prompts as P  # noqa: E402
from beam_agent.models import Paper  # noqa: E402
from schemas import (DirectCheck, DirectCheckItem, NeedCard, NeedCards, QueryCandidates, SisterProposal,  # noqa: E402
                     SisterProposals)

ABSTRACT = ("We present the SIRAH2 force field extension and apply the Martini-free approach to a heavily glycosylated "
            "spike glycoprotein. Glycan flexibility and glycan-protein recognition are reproduced with microsecond "
            "coarse-grained simulations of the glycoprotein in solution.")
SEED = {"case_id": "P99", "case_type": "L1", "domain": "glycan", "doi": "10.1/seed", "title": "SIRAH glycosylation unlocks glycan flexibility",
        "year": 2024, "authors": ["Maria Garcia-Smithers", "Li Zhang"], "abstract": ABSTRACT, "named_system": "spike glycoprotein",
        "paper_type": "application"}
RARE = LK.build_rare_words([ABSTRACT] + ["protein folding simulation of a small domain"] * 40)


def card(system="a heavily glycosylated viral surface protein"):
    return NeedCard(system_class=system, scientific_process="glycan dynamics", phenomenon_or_goal="glycan flexibility and recognition",
                    important_constraints=["explicit glycans"], required_model_capability="residue-level CG with explicit glycans",
                    transferable_features=["dense glycan shield"])


class FakeBackend:
    name, model = "fake", "fake-1"

    def __init__(self, queries=None, proposals=None, direct=None):
        self.queries = queries or ["I want to simulate a glycoprotein where glycan flexibility matters, so which coarse-grained models could I realistically use for it?",
                                   "We need coarse-grained simulations of a heavily glycosylated protein, and which force fields can represent the glycans well enough?",
                                   "How can I model glycan-protein recognition with a coarse-grained approach, and which models would capture that interaction best?"]
        self.proposals = proposals
        self.direct = direct or {}
        self.calls = []
        self.users = []

    def structured(self, *, system, user, schema, label, effort=None, purpose=None, case_id=None, timeout=None):
        self.calls.append(label)
        self.users.append((label, user))
        if schema is NeedCards:
            return NeedCards(specific_need=card("a heavily glycosylated spike glycoprotein"), abstracted_need=card())
        if schema is QueryCandidates:
            return QueryCandidates(queries=list(self.queries))
        if schema is SisterProposals:
            return SisterProposals(proposals=self.proposals)
        if schema is DirectCheck:
            names = [l for l in user.splitlines() if l.startswith("TARGET SYSTEM:")][0]
            direct = any(k in names for k, v in self.direct.items() if v)
            n = user.count("\n[")  # papers listed
            return DirectCheck(items=[DirectCheckItem(index=i, is_direct_cg_for_target=(direct and i == 1), reason="r") for i in range(1, n + 1)])
        raise AssertionError(schema)


# ------------------------------------------------------------------------------------------ leakage checks
def flags(q, **kw):
    base = dict(title=SEED["title"], abstract=ABSTRACT, authors=SEED["authors"], rare=RARE)
    base.update(kw)
    return LK.check_text(q, **base)


def test_clean_query_has_no_flags():
    assert flags("I need CG models for a heavily glycosylated viral protein with flexible glycans. What could I use?") == []


def test_doi_author_and_model_name_are_flagged():
    f = flags("Is there a Smithers paper doi 10.1021/abc about SIRAH?")
    assert any(x.startswith("doi") for x in f) and any("author:'smithers'" in x for x in f) and any("model_name:'sirah'" in x for x in f)


def test_model_name_from_the_abstract_pattern_is_flagged_but_system_words_are_not():
    assert any("model_name:'sirah2'" in x for x in flags("Which SIRAH2 setup works?"))
    # a word the need card legitimately uses for the researcher's own system is not a 'model name'
    assert flags("Can the Spike model handle it?", allowed_words=["spike"]) == []


def test_title_ngram_overlap_is_flagged():
    f = flags("How does glycosylation unlocks glycan flexibility in practice, what CG options exist?")
    assert any(x.startswith("title_ngram") for x in f)


def test_copied_rare_phrase_is_flagged_unless_it_comes_from_the_card():
    q = "We need microsecond coarse-grained simulations of the glycoprotein in solution, what force field?"
    assert any(x.startswith("copied_phrase") for x in flags(q))
    assert not any(x.startswith("copied_phrase") for x in flags(q, card_text="microsecond coarse-grained simulations of the glycoprotein in solution"))


def test_card_copying_a_long_abstract_phrase_is_flagged():
    text = "Glycan flexibility and glycan-protein recognition are reproduced with microsecond coarse-grained simulations"
    assert any(x.startswith("copied_phrase") for x in LK.check_text(text, title=SEED["title"], abstract=ABSTRACT, authors=[], rare=RARE, is_card=True))


# ------------------------------------------------------------------------------------------ D/H wiring
def test_stage_a_sees_only_the_abstract_and_stage_b_only_the_card():
    be = FakeBackend()
    rec = G.run_case(SEED, be, lambda q, k: [], RARE, "fake:fake-1")
    a_user = next(u for l, u in be.users if l == "bench_stageA")
    b_user = next(u for l, u in be.users if l == "bench_stageB")
    assert a_user.startswith("ABSTRACT:") and SEED["title"] not in a_user and "10.1/seed" not in a_user and "Garcia" not in a_user
    assert "NEED CARD" in b_user and ABSTRACT[:40] not in b_user and SEED["title"] not in b_user and "Garcia" not in b_user
    assert len(rec["query_candidates"]) == 3 and rec["selected_query"] == "" and rec["generator_model"] == "fake:fake-1"


def test_l1_uses_the_specific_card_and_l2_the_abstracted_one():
    be1, be2 = FakeBackend(), FakeBackend()
    G.run_case(dict(SEED, case_type="L1"), be1, lambda q, k: [], RARE, "m")
    G.run_case(dict(SEED, case_type="L2"), be2, lambda q, k: [], RARE, "m")
    assert "spike glycoprotein" in next(u for l, u in be1.users if l == "bench_stageB")
    u2 = next(u for l, u in be2.users if l == "bench_stageB")
    assert "heavily glycosylated viral surface protein" in u2 and "spike glycoprotein" not in u2


def test_flagged_candidates_are_regenerated_once_then_kept_with_their_flags():
    bad = ["Does the SIRAH force field handle glycans well, and which coarse-grained alternatives would you suggest for this system?"] * 3
    be = FakeBackend(queries=bad)
    rec = G.run_case(SEED, be, lambda q, k: [], RARE, "m")
    assert be.calls.count("bench_stageB") == 2                      # one regeneration
    assert all(any("sirah" in x for x in f) for f in rec["leakage_flags"])   # still visible to the human
    assert len(rec["query_candidates"]) == 3


# ------------------------------------------------------------------------------------------ L3
def prop(name, shared="dense glycan shield"):
    return SisterProposal(system_name=name, description=f"{name}: a related viral protein", shared_feature=shared, key_difference="other virus")


def paper(title, doi):
    return Paper(title=title, doi=doi, abstract="x " * 30)


def l3_seed():
    return dict(SEED, case_type="L3")


def test_l3_prefers_a_sister_without_direct_cg_literature_and_counts_eliminations():
    be = FakeBackend(proposals=[prop("HIV-1 Env"), prop("Lassa GP"), prop("Nipah G")], direct={"HIV-1 Env": True})
    rec = G.run_case(l3_seed(), be, lambda q, k: [paper("p1", "10.1/a"), paper("p2", "10.1/b")], RARE, "m")
    assert rec["transfer_target_system"] == "Lassa GP" and rec["verified_no_direct"] is True
    assert rec["sister_summary"]["n_proposed"] == 3 and rec["sister_summary"]["n_eliminated_direct"] == 1
    assert [p["eliminated"] for p in rec["sister_proposals"]] == ["has_direct_cg_literature", None, None]


def test_l3_falls_back_to_a_sister_with_direct_literature_and_flags_it():
    be = FakeBackend(proposals=[prop("HIV-1 Env"), prop("Lassa GP")], direct={"HIV-1 Env": True, "Lassa GP": True})
    rec = G.run_case(l3_seed(), be, lambda q, k: [paper("p1", "10.1/a")], RARE, "m")
    assert rec["verified_no_direct"] is False and rec["transfer_target_system"] == "HIV-1 Env"


def test_l3_seed_paper_is_excluded_from_the_direct_check_and_a_sister_named_in_the_abstract_is_dropped():
    seen = []

    def search(q, k):
        seen.append(q)
        return [paper("seed itself", "10.1/seed")]                    # only the seed comes back -> nothing left to judge
    be = FakeBackend(proposals=[prop("spike glycoprotein"), prop("Lassa GP")])
    rec = G.run_case(l3_seed(), be, search, RARE, "m")
    assert rec["sister_proposals"][0]["eliminated"].startswith("named_in_seed_abstract")
    assert rec["sister_proposals"][1]["check"]["verified_no_direct"] is True and rec["sister_proposals"][1]["check"]["n_results"] == 0
    assert be.calls.count("bench_directcheck") == 0                    # nothing but the seed came back -> no judgement needed


def test_l3_stage_b_sees_the_sister_not_the_seed_system_and_x_name_is_flagged():
    q_leaky = ["How do I model the spike glycoprotein glycan shield with coarse-grained methods, and which models would capture its dynamics best?",
               "Which coarse-grained models could I use to study glycan dynamics on Lassa GP, and how well do they capture its flexibility overall?",
               "Which coarse-grained approach would you recommend for simulating glycan shielding on Lassa GP, and what are its main limitations here?"]
    be = FakeBackend(proposals=[prop("Lassa GP")], queries=q_leaky)
    rec = G.run_case(l3_seed(), be, lambda q, k: [], RARE, "m")
    b_user = [u for l, u in be.users if l == "bench_stageB"][0]
    assert "Lassa GP" in b_user and "spike glycoprotein" not in b_user and SEED["title"] not in b_user
    assert any("x_system_term:'spike'" in x for x in rec["leakage_flags"][0])             # the seed system leaked into candidate 1
    assert not any("x_system_term" in x for x in rec["leakage_flags"][1])
    assert rec["transfer_source_system"] == "spike glycoprotein" and rec["transfer_target_system"] == "Lassa GP"


# ------------------------------------------------------------------------------------------ outputs + probe
REQUIRED = ["case_id", "case_type", "seed_doi", "seed_title", "seed_year", "seed_abstract", "need_card", "query_candidate_1",
            "query_candidate_2", "query_candidate_3", "selected_query", "transfer_source_system", "transfer_target_system",
            "verified_no_direct", "leakage_flags", "openalex_semantic_rank"]


def test_csv_has_every_required_column_and_selected_query_is_blank(tmp_path):
    rec = G.run_case(SEED, FakeBackend(), lambda q, k: [], RARE, "m")
    G.write_outputs([rec], tmp_path / "x")
    rows = list(csv.DictReader(open(tmp_path / "x.csv", encoding="utf-8")))
    assert set(REQUIRED) <= set(rows[0]) and rows[0]["selected_query"] == "" and rows[0]["case_type"] == "L1"
    assert json.loads(rows[0]["need_card"])["specific_need"]["system_class"]
    assert json.loads((tmp_path / "x.jsonl").read_text().splitlines()[0])["case_id"] == "P99"


def test_probe_records_rank_but_never_drops_or_relabels_a_case():
    rows = [{"case_id": "P01", "case_type": "L2", "seed_doi": "10.1/Seed", "selected_query": "", "selected_candidate": "2",
             "query_candidate_1": "q1", "query_candidate_2": "q2", "query_candidate_3": "q3"},
            {"case_id": "P02", "case_type": "L3", "seed_doi": "10.1/other", "selected_query": "my own wording", "selected_candidate": ""},
            {"case_id": "P03", "case_type": "L1", "seed_doi": "10.1/x", "selected_query": "", "selected_candidate": ""}]
    asked = []

    def search(q, k):
        asked.append((q, k))
        return [paper("a", "10.1/zzz"), paper("b", "10.1/seed")] if q == "q2" else [paper("a", "10.1/zzz")]
    out = PR.probe_rows(rows, search)
    assert asked == [("q2", 50), ("my own wording", 50)]               # only rows the human filled; depth 50
    assert (out[0]["openalex_semantic_rank"], out[0]["openalex_rank_band"], out[0]["selected_query"]) == (2, "1-5", "q2")
    assert out[1]["openalex_semantic_rank"] == "" and out[1]["openalex_rank_band"].startswith("miss")
    assert out[2]["openalex_semantic_hit"] == "" and [r["case_type"] for r in out] == ["L2", "L3", "L1"]    # nothing relabelled, nothing dropped
    assert PR.band(30) == "21-50" and PR.band(5) == "1-5" and PR.band(6) == "6-20"


def test_prompts_forbid_the_leak_vectors():
    assert "DOI" in P.STAGE_A_SYSTEM and "author name" in P.STAGE_A_SYSTEM and "name of any CG model" in P.STAGE_A_SYSTEM
    assert "never seen any paper" in P.STAGE_B_SYSTEM and "never seen any paper" in P.STAGE_B_L3_SYSTEM


# ------------------------------------------------------------------------------------------ short-query format rules
def test_format_flags_catch_length_and_persona_openings():
    ok = "I want to simulate how a plant channel gates under membrane tension, so which coarse-grained models could capture that?"
    assert 15 <= len(ok.split()) <= 30 and G.format_flags(ok) == []
    assert G.format_flags("Which CG model fits a channel?") == ["format_length:6_words"]
    long_q = " ".join(["word"] * 41)
    assert G.format_flags(long_q) == ["format_length:41_words"]
    assert "format_persona_opening" in G.format_flags("As an early-career researcher, I want to model channel gating under membrane tension with coarse-grained methods, which models fit?")
    assert "format_persona_opening" in G.format_flags("I am an experimentalist who wants to model channel gating under membrane tension, which coarse-grained model would you suggest?")


def test_all_flagged_candidates_trigger_one_rewrite_that_names_the_format_problem():
    too_long = [" ".join(["coarse"] * 45)] * 3
    be = FakeBackend(queries=too_long)
    rec = G.run_case(SEED, be, lambda q, k: [], RARE, "m")
    assert be.calls.count("bench_stageB") == 2
    assert "format problems" in [u for l, u in be.users if l == "bench_stageB"][1]
    assert all("format_length:45_words" in f for f in rec["leakage_flags"])


def test_stage_b_prompts_state_the_short_query_convention():
    for sysmsg in (P.STAGE_B_SYSTEM, P.STAGE_B_L3_SYSTEM):
        assert "15 to 30 words" in sysmsg and "AT MOST TWO" in sysmsg and "NO opening that states who the researcher is" in sysmsg
    assert P.PROMPT_VERSION.endswith("-2") or "pilot-2" in P.PROMPT_VERSION


# ------------------------------------------------------------------------------------------ pre-specified L3 target
def l3_seed_with_target(verified=True):
    return dict(SEED, case_type="L3", l3_target={"system_name": "Lassa GP", "description": "a viral surface protein with a dense glycan shield",
                "shared_feature": "dense glycan shielding", "source_system": "spike glycoprotein", "verified_no_direct": verified,
                "evidence": {"n_unique_papers_judged": 90, "direct_hits": []}})


def test_prespecified_l3_target_skips_sister_proposal_and_direct_check_and_uses_the_given_y():
    be = FakeBackend()
    searched = []
    rec = G.run_case(l3_seed_with_target(), be, lambda q, k: searched.append(q) or [], RARE, "m")
    assert "bench_sisters" not in be.calls and "bench_directcheck" not in be.calls and searched == []       # no extra LLM / OpenAlex work
    assert rec["transfer_target_system"] == "Lassa GP" and rec["transfer_source_system"] == "spike glycoprotein"
    assert rec["verified_no_direct"] is True and rec["l3_evidence"]["n_unique_papers_judged"] == 90
    b_user = [u for l, u in be.users if l == "bench_stageB"][0]
    assert "Lassa GP" in b_user and "dense glycan shield" in b_user and "spike glycoprotein" not in b_user and SEED["title"] not in b_user


def test_final_seed_file_has_6_6_8_and_l3_targets_never_name_the_source_system():
    import finalize_pilot_seeds as F
    assert len(F.L1) == 6 and len(F.L2) == 8 and len(F.L3) == 6
    for doi, x, dom, y in F.L3:
        text = " ".join(LK.tokens(y["system_name"] + " " + y["description"] + " " + y["shared_feature"]))
        for tok in LK.tokens(x):
            if len(tok) >= 4 and tok not in LK.STOP and tok not in ("coli", "ribosome", "polymerase", "initiation", "complex", "transcription"):
                assert tok not in text.split(), (x, y["system_name"], tok)
