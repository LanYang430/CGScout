"""Unit tests for relevance_filter_handoff.py -- the methodology
surface-form normalization fix (coarse-grained / coarse grained / coarse
graining / coarse-graining), the route-aware CG/system-evidence gate, the
review filter, and the general-path classifier gate.

No network: every filter_papers() call here injects a fake `classify_fn`
(_fake_marker_classify_fn, or an inline lambda for the classifier-specific
tests below) instead of the real sim_classifier.default_classify_fn, which
would call the OpenAI embeddings API and load the persisted model
artifact. relevance_filter_handoff.py itself only needs beam_agent.
models/text_utils/concept_resolver's read-only cluster table, plus
review_filter.py (also network-free -- pure regex/field check).
"""


from cgscout.screening.filter import _METHODOLOGY_EQUIVALENCE, _STRUCTURAL_CG_MARKERS, _normalize_methodology_stem, filter_papers  # noqa: E402

from cgscout.models import Paper  # noqa: E402
from cgscout.text import stemmed_keywords  # noqa: E402


def _fake_marker_classify_fn(papers):
    """Network-free stand-in for sim_classifier.default_classify_fn,
    injected into every filter_papers() call in this file. Reproduces the
    OLD literal-marker check (score 1.0 if any _STRUCTURAL_CG_MARKERS
    substring is present in title+abstract), PLUS the methodology-
    equivalence stem hit ("coarse graining" etc. normalizing to the same
    canonical "coarse-grained" key -- see _build_methodology_equivalence())
    -- a real semantic classifier would obviously also recognize that
    surface form, and several tests below exist specifically to exercise
    it. Everything else keeps exercising the gate/route-combination logic
    exactly as it did before the classifier existed -- this file tests
    THAT logic, not the real classifier (which has its own dedicated
    validation, see sim_classifier_train.py's module docstring)."""
    out = []
    for p in papers:
        blob = f"{p.title or ''} {p.abstract or ''}".lower()
        marker_hit = any(m in blob for m in _STRUCTURAL_CG_MARKERS)
        stems = {_normalize_methodology_stem(s) for s in stemmed_keywords(blob, min_len=2)}
        methodology_hit = "coarse-grained" in stems
        out.append(1.0 if (marker_hit or methodology_hit) else 0.0)
    return out


def _paper(title, found_by_queries, categories=("system_with_methodology",), abstract=None):
    return Paper(title=title, abstract=abstract, sources=["openalex"], found_by_queries=list(found_by_queries), query_categories=list(categories))


def _kept_titles(papers):
    retained, _, _ = filter_papers(papers, classify_fn=_fake_marker_classify_fn)
    return {p.title for p in retained}


def test_coarse_grained_query_matches_coarse_graining_title():
    p = _paper("The power of coarse graining in biomolecular simulations", ["study folding using a coarse-grained model"])
    assert p.title in _kept_titles([p])


def test_coarse_grained_query_matches_coarse_grained_space_title():
    p = _paper("Coarse grained models of protein folding", ["study folding using a coarse-grained model"])
    assert p.title in _kept_titles([p])


def test_coarse_graining_query_matches_coarse_grained_title():
    p = _paper("A review of coarse grained protein models", ["coarse-graining approaches to folding"])
    assert p.title in _kept_titles([p])


def test_coarse_grained_hyphenated_title_matches_space_query():
    p = _paper("Coarse-grained simulation of membrane proteins", ["study folding using coarse grained models"])
    assert p.title in _kept_titles([p])


def test_cg_abbreviation_still_matches_via_existing_synonym():
    # "cg" is already a listed synonym in _METHODOLOGY_TERM_CLUSTERS --
    # confirms the equivalence map doesn't break the pre-existing mechanism.
    p = _paper("CG models for protein folding", ["study folding using a coarse-grained model"])
    assert p.title in _kept_titles([p])


def test_unrelated_paper_still_dropped():
    p = _paper("Coronavirus spike protein structural biology", ["study folding using a coarse-grained model"])
    assert p.title not in _kept_titles([p])


def test_beta_helix_bare_beta_still_not_matched():
    """The exact real regression case: "Amyloid beta" must NOT match a
    beta-helix query -- confirms the methodology equivalence map has no
    effect on structural terms."""
    p = _paper("Amyloid beta: structure, biology and structure-based therapeutic development", ["200-aa beta-helix protein folding"])
    assert p.title not in _kept_titles([p])


def test_beta_helix_exact_match_still_works():
    # Title avoids "protein" (also in the query) so this isolates the
    # beta-helix exact-hyphenated-match axis specifically, not an
    # incidental match on a shared generic word. Abstract added (this
    # fixture predates the general-path classifier gate below) purely so
    # this paper clears THAT unrelated, newer requirement too -- a title-
    # only fixture with zero simulation content is a genuine drop under
    # that gate now, which isn't the axis this test exists to check.
    p = _paper(
        "Structural basis of beta-helix architectural stability", ["200-aa beta-helix protein folding"],
        abstract="We use coarse-grained simulations to study the architectural stability of the fold.",
    )
    assert p.title in _kept_titles([p])


def test_beta_helix_space_variant_deliberately_NOT_normalized():
    # "beta helix" (space) is ALREADY a listed synonym of beta-helix in
    # _STRUCTURAL_TERM_CLUSTERS -- but relevance_filter_handoff.py doesn't
    # read that table at all (only _METHODOLOGY_TERM_CLUSTERS), so this
    # exercises the UNCHANGED strict compound-stem path, not the new
    # equivalence map. Title deliberately avoids any other word the query
    # also contains (e.g. "protein") so the only possible match axis is
    # beta/helix itself, isolating what this test actually checks. Must
    # NOT match -- confirming no accidental structural-term leniency was
    # introduced by this pass, at the cost of a symmetrical (but
    # out-of-scope, per instruction) gap: structural terms could suffer
    # the same hyphen/space mismatch this fix solved for methodology
    # wording, but that's deliberately not addressed in this pass.
    p = _paper("Beta helix architecture in extracellular matrix", ["200-aa beta-helix protein folding"])
    assert p.title not in _kept_titles([p])


def test_methodology_equivalence_map_has_no_beta_helix_entries():
    """Direct structural check: the equivalence map built from
    _METHODOLOGY_TERM_CLUSTERS must never contain a key/value derived from
    beta-helix (which lives only in the separate _STRUCTURAL_TERM_CLUSTERS,
    never imported by this module)."""
    from cgscout.screening.filter import _METHODOLOGY_EQUIVALENCE

    assert "beta-helix" not in _METHODOLOGY_EQUIVALENCE.values()
    assert "beta-heli" not in _METHODOLOGY_EQUIVALENCE  # beta-helix's own compound stem


# --- structural-route (Hop-2) strict gate: "Hop-1 broad, Hop-2 strict" ----
# Real scenario this was designed against: Pertactin's Hop-2 query
# "pertactin folding coarse-grained model" -- Martini 3 (CG evidence, no
# Pertactin) and the 2006 Junker experimental folding paper (Pertactin
# evidence, no CG) both cleared the OLD >=1-keyword-hit rule; only the
# target 2026 paper (both) should survive the new gate.

def _structural_paper(title, discovered_system, abstract="", query="pertactin folding coarse-grained model"):
    return Paper(
        title=title, abstract=abstract, sources=["openalex"],
        found_by_queries=[query], query_categories=["structural"],
        structural_discovered_system=discovered_system,
    )


def test_structural_gate_drops_cg_evidence_without_system_evidence():
    # Martini 3 -- real CG evidence, never mentions Pertactin at all
    p = _structural_paper(
        "Improving Martini 3 for Disordered and Multidomain Proteins", "Pertactin",
        abstract="We present improvements to the Martini 3 coarse-grained force field for disordered proteins.",
    )
    assert p.title not in _kept_titles([p])


def test_structural_gate_drops_system_evidence_without_cg_evidence():
    # the real 2006 Junker experimental folding paper -- real Pertactin
    # evidence, no coarse-grained/CG simulation evidence at all
    p = _structural_paper(
        "Pertactin beta-helix folding mechanism suggests common themes for the secretion "
        "and folding of autotransporter proteins", "Pertactin",
        abstract="We characterize the folding mechanism of the Pertactin passenger domain by "
                 "experimental biophysical methods.",
    )
    assert p.title not in _kept_titles([p])


def test_structural_gate_keeps_paper_with_both():
    # the actual target paper's real title/abstract shape -- both present
    p = _structural_paper(
        "Autotransporter folding avoids a kinetic trap during vectorial translocation "
        "across the bacterial outer membrane", "Pertactin",
        abstract="Using coarse-grained and all-atom simulations, we study folding of the "
                 "C-terminal segment of the Pertactin passenger domain.",
    )
    assert p.title in _kept_titles([p])


def test_structural_gate_accepts_cg_simulation_and_cg_model_phrasing():
    """The two phrasings explicitly named in the design decision, not
    covered by the hyphen/space "coarse-grain(ed)" forms alone."""
    p1 = _structural_paper("EspP folding", "EspP", abstract="EspP was studied using a CG simulation of its passenger domain.")
    p2 = _structural_paper("EspP folding", "EspP", abstract="A CG model of EspP reveals its folding pathway.")
    assert p1.title in _kept_titles([p1])
    assert p2.title in _kept_titles([p2])


def test_structural_gate_requires_all_words_of_a_multiword_system_name():
    # "P22 tailspike" -- only "P22" present, "tailspike" (the word itself)
    # never appears anywhere -> still drop
    p = _structural_paper(
        "P22 phage capsid assembly", "P22 tailspike",
        abstract="A coarse-grained model of P22 bacteriophage capsid assembly and maturation.",
    )
    assert p.title not in _kept_titles([p])


def test_structural_gate_does_not_apply_to_non_structural_routes():
    """The exact same title/abstract that WOULD fail the gate must still
    pass the ordinary >=1-keyword-hit rule when it's NOT a structural-route
    paper -- confirms the gate is scoped to Hop-2 only, per instruction."""
    p = Paper(
        title="Improving Martini 3 for Disordered and Multidomain Proteins",
        abstract="We present improvements to the Martini 3 coarse-grained force field.",
        sources=["openalex"],
        found_by_queries=["coarse-grained model"],
        query_categories=["modeling_requirement"],  # NOT structural
        structural_discovered_system=None,
    )
    assert p.title in _kept_titles([p])


def test_structural_gate_does_not_apply_when_discovered_system_missing():
    """A structural-route paper with no structural_discovered_system set
    (e.g. old provenance, or the field never got wired at some call site)
    fails open to the general rule -- never silently drops everything."""
    p = Paper(
        title="Improving Martini 3 for Disordered and Multidomain Proteins",
        abstract="We present improvements to the Martini 3 coarse-grained force field.",
        sources=["openalex"],
        found_by_queries=["Pertactin folding coarse-grained model"],
        query_categories=["structural"],
        structural_discovered_system=None,
    )
    assert p.title in _kept_titles([p])  # matches "coarse-grained" against its own query vocabulary, general rule only


# --- structural-route gate: EXISTENTIAL over all retrieving systems --------
# This session's fix for an order-dependent false drop: a paper found by
# BOTH 'EspP' and 'pertactin' Hop-2 queries used to be gated only against
# whichever system happened to be the first claimant (an accumulator/
# processing-order artifact, not a relevance judgment). Now the gate passes
# if ANY of the systems that actually, independently retrieved the paper has
# grounded lexical evidence -- restricted to structural_discovered_systems
# (the real retrieving set), never the full universe of Hop-1-discovered
# systems.

_TARGET_TITLE = ("Autotransporter folding avoids a kinetic trap during vectorial translocation "
                  "across the bacterial outer membrane")
_TARGET_ABSTRACT = (
    "A longstanding puzzle is why pertactin folds and is secreted in vivo within minutes but "
    "refolds in vitro over hours to days. We introduce BEAM, a multiscale coarse-grained "
    "framework applied to the pertactin passenger domain."
)


def _multi_structural_paper(discovered_systems, query="EspP folding coarse-grained model"):
    return Paper(
        title=_TARGET_TITLE, abstract=_TARGET_ABSTRACT, sources=["openalex"],
        found_by_queries=[query], query_categories=["structural"],
        structural_discovered_system=discovered_systems[0] if discovered_systems else None,  # first-claimant, as the real reattachment loop sets it
        structural_discovered_systems=discovered_systems,
    )


def test_espp_alone_fails_the_gate_pertactin_alone_passes():
    """Isolating each retrieving system individually, exactly as the user's
    regression scenario specifies."""
    espp_only = _multi_structural_paper(["EspP"])
    pertactin_only = _multi_structural_paper(["Pertactin"])
    assert espp_only.title not in _kept_titles([espp_only])       # "EspP" never appears in the text
    assert pertactin_only.title in _kept_titles([pertactin_only])  # "pertactin" appears repeatedly


def test_target_paper_survives_when_retrieved_by_both_espp_and_pertactin():
    """The real diagnosed case: EspP claimed it first (Hop-2 processing
    order, after EspP's n_support was boosted by this session's own
    duplicate/interaction merge), but Pertactin's query ALSO independently
    retrieved it -- the paper must now survive."""
    p = _multi_structural_paper(["EspP", "Pertactin"])
    kept, dropped, verdicts = filter_papers([p], classify_fn=_fake_marker_classify_fn)
    assert p.title in {k.title for k in kept}
    v = next(v for v in verdicts if v.title == p.title)
    assert v.keep is True
    assert "Pertactin" in v.reason or "structural gate" in v.reason


def test_existential_check_is_restricted_to_this_papers_own_retrieving_systems():
    """A noise paper found ONLY by a non-matching system ('SystemX', whose
    name never appears in its text) must NOT be let through just because
    some OTHER, unrelated Hop-1-discovered system (e.g. 'Pertactin') exists
    elsewhere in the run -- structural_discovered_systems must list ONLY
    what actually, independently retrieved THIS paper, never the full
    Hop-1-discovered-systems universe."""
    noise = Paper(
        title="A totally unrelated coarse-grained simulation study",
        abstract="This coarse-grained simulation study examines an unrelated membrane protein.",
        sources=["openalex"], found_by_queries=["SystemX folding coarse-grained model"],
        query_categories=["structural"],
        structural_discovered_system="SystemX",
        structural_discovered_systems=["SystemX"],  # ONLY SystemX actually retrieved this paper
    )
    assert noise.title not in _kept_titles([noise])


def test_falls_back_to_singular_field_when_plural_list_is_empty():
    """Backward compatibility: a Paper built the OLD way (only the singular
    field set, e.g. an older diagnostic/fixture that predates this list)
    must behave exactly as before -- single-system AND-gate, not silently
    fail open."""
    p = Paper(
        title="Improving Martini 3 for Disordered and Multidomain Proteins",
        abstract="We present improvements to the Martini 3 coarse-grained force field.",
        sources=["openalex"], found_by_queries=["Pertactin folding coarse-grained model"],
        query_categories=["structural"],
        structural_discovered_system="Pertactin",
        # structural_discovered_systems deliberately left unset (empty)
    )
    assert p.title not in _kept_titles([p])  # never mentions Pertactin -- same as the pre-existing single-field test above


# --- multi-route gate: keep if ANY applicable route independently passes --
#
# Real diagnosed false negative (this session, live TamAB case): a paper is
# correctly discovered via a flat direct_concept_search (query_categories
# includes "transfer_direct") -- its own provenance alone would clear the
# simple CG-evidence-only gate -- but Hop-2 ALSO (mis-)attached it to
# unrelated discovered systems, adding "structural" to its merged query_
# categories (transfer_retrieval_bridge.py's _merge_papers unions
# categories across every route that found the same paper_ref). The OLD
# if/elif branching judged such a paper ONLY by the stricter structural
# gate whenever "structural" was present at all, discarding it on a
# system-evidence failure it should never have been subject to, since its
# OWN flat-route provenance never required system evidence in the first
# place. None of these fixtures set `doi`, so fetch_methods_text()'s
# `if not doi: return None` guard keeps every case here network-free
# (matches the rest of this file's existing no-network convention).

def _multi_route_paper(title, discovered_system, abstract="", categories=("structural", "transfer_direct")):
    return Paper(
        title=title, abstract=abstract, sources=["openalex"],
        found_by_queries=["some unrelated system folding coarse-grained model",
                          "TamAB phospholipid capture coarse-grained model"],
        query_categories=list(categories),
        structural_discovered_system=discovered_system,
        structural_discovered_systems=[discovered_system],
    )


def test_multi_route_paper_kept_via_flat_route_despite_failing_structural_system_evidence():
    p = _multi_route_paper(
        "Structural basis of outer membrane biogenesis by the TamAB translocase",
        discovered_system="OmpC-Mla system",  # wrong/unrelated Hop-1 attribution -- structural gate WILL fail this
        abstract="We characterize phospholipid transport within the hydrophobic cavity of TamB "
                 "using coarse-grained Martini 3 simulations.",  # has CG evidence -- flat route alone should pass
    )
    assert p.title in _kept_titles([p])


def test_multi_route_paper_dropped_when_neither_route_has_cg_evidence():
    p = _multi_route_paper(
        "Structural basis of outer membrane biogenesis by the TamAB translocase",
        discovered_system="OmpC-Mla system",
        abstract="We characterize phospholipid transport within the hydrophobic cavity of TamB "
                 "using molecular dynamics simulations.",  # no CG marker anywhere -- both routes fail
    )
    retained, rejected, verdicts = filter_papers([p], fulltext_lookup_budget=0, classify_fn=_fake_marker_classify_fn)  # no network -- verify the abstract-only verdict itself
    assert p.title not in {r.title for r in retained}
    assert verdicts[0].missing_cg_evidence_only is True  # flat route needs nothing but CG evidence -- eligible for the fallback


def test_structural_only_route_still_fails_closed_when_system_evidence_missing():
    """Sanity guard: a paper found via structural route ONLY (no flat-route
    category at all) must be completely unaffected by this fix -- missing
    system evidence there is a genuine relevance miss, not CG-evidence-only,
    and must never become fallback-eligible."""
    p = _multi_route_paper(
        "Structural basis of outer membrane biogenesis by the TamAB translocase",
        discovered_system="OmpC-Mla system",
        abstract="We characterize phospholipid transport within the hydrophobic cavity of TamB "
                 "using coarse-grained Martini 3 simulations.",
        categories=("structural",),  # no "transfer_direct" -- flat route does not apply at all
    )
    retained, rejected, verdicts = filter_papers([p], fulltext_lookup_budget=0, classify_fn=_fake_marker_classify_fn)
    assert p.title not in {r.title for r in retained}
    assert verdicts[0].missing_cg_evidence_only is False  # system evidence is missing -- full text can't fix that


# --- review filter (this pass): runs first, before any scoring -----------

def test_review_flagged_by_openalex_type_is_dropped_before_classification():
    p = Paper(
        title="Recent advances in coarse-grained modeling of biomolecules",
        abstract="We summarize progress in the field.",
        sources=["openalex"], oa_type="review",
        found_by_queries=["coarse-grained model"], query_categories=["system_with_methodology"],
    )
    calls = []
    retained, rejected, verdicts = filter_papers(
        [p], classify_fn=lambda ps: calls.append(len(ps)) or [1.0] * len(ps),
    )
    assert p.title not in {r.title for r in retained}
    v = verdicts[0]
    assert v.review_flagged is True
    assert "type=='review'" in v.review_reason
    assert calls == []  # classifier never invoked -- review-flagged papers are dropped before scoring


def test_review_flagged_by_abstract_phrase_is_dropped():
    p = Paper(
        title="Coarse-grained models of biomolecular condensates",
        abstract="In this review, we discuss coarse-grained simulation approaches to phase separation.",
        sources=["openalex"], found_by_queries=["coarse-grained model"], query_categories=["system_with_methodology"],
    )
    retained, rejected, verdicts = filter_papers([p], classify_fn=_fake_marker_classify_fn)
    assert p.title not in {r.title for r in retained}
    assert verdicts[0].review_flagged is True
    assert "in this review" in verdicts[0].review_reason.lower()


def test_non_review_paper_is_not_flagged_and_reaches_classification():
    p = _paper(
        "Coarse-grained simulations of protein folding", ["coarse-grained model"],
        abstract="We perform coarse-grained molecular dynamics simulations of protein folding.",
    )
    retained, rejected, verdicts = filter_papers([p], classify_fn=_fake_marker_classify_fn)
    assert p.title in {r.title for r in retained}
    assert verdicts[0].review_flagged is False
    assert verdicts[0].classifier_score == 1.0


# --- general-path classifier gate (this pass): applies even with no ------
# structural/flat route provenance at all -- previously fully ungated on
# the CG-evidence axis, kept on >=1 keyword hit alone.

def test_general_path_paper_dropped_when_classifier_predicts_not_simulation():
    p = _paper(
        "Experimental characterization of a coarse-grained-model target system",
        ["coarse-grained model"],
        abstract="We use X-ray crystallography and NMR to determine the structure; no simulation was performed.",
    )
    retained, rejected, verdicts = filter_papers([p], classify_fn=lambda ps: [0.1] * len(ps))
    assert p.title not in {r.title for r in retained}
    v = verdicts[0]
    assert v.classifier_score == 0.1
    assert "classifier predicts" in v.reason


def test_general_path_paper_kept_when_classifier_predicts_simulation():
    p = _paper(
        "Computational study of a coarse-grained-model target system", ["coarse-grained model"],
        abstract="We run coarse-grained simulations of the target system.",
    )
    retained, rejected, verdicts = filter_papers([p], classify_fn=lambda ps: [0.9] * len(ps))
    assert p.title in {r.title for r in retained}
    assert verdicts[0].classifier_score == 0.9


def test_classifier_never_rescues_a_genuine_topical_miss():
    """filter_papers() batch-classifies every non-review-flagged candidate
    up front (for one batched embedding call), so classifier_score is
    still populated here -- but a high score can never override a paper
    that failed the general keyword-overlap score in the first place; the
    classifier has no topical-relevance signal, only a sim-vs-experiment
    one."""
    p = _paper("Totally unrelated topic", ["coarse-grained model"])  # zero keyword overlap
    retained, rejected, verdicts = filter_papers([p], classify_fn=lambda ps: [0.99] * len(ps))
    assert p.title not in {r.title for r in retained}
    assert verdicts[0].classifier_score == 0.99
    assert "no keyword overlap" in verdicts[0].reason


def test_classifier_threshold_is_configurable_not_hardcoded():
    p = _paper(
        "Coarse-grained-model study of a target system", ["coarse-grained model"],
        abstract="We use a coarse-grained approach.",
    )
    score = 0.4
    # below the 0.5 default -> dropped
    retained_default, _, _ = filter_papers([p], classify_fn=lambda ps: [score] * len(ps))
    assert p.title not in {r.title for r in retained_default}
    # lowering the threshold below the score -> kept, same paper, same score
    retained_lowered, _, verdicts = filter_papers([p], classify_fn=lambda ps: [score] * len(ps), classifier_threshold=0.3)
    assert p.title in {r.title for r in retained_lowered}
    assert verdicts[0].classifier_score == score


# --- P3: papers found by the `need_semantic` route are judged like the other flat routes (CG evidence alone) ---------------------

def test_need_semantic_route_uses_the_flat_cg_evidence_gate():
    cg = Paper(title="Coarse-grained Martini 3 simulation of an enzyme", abstract="We use coarse-grained Martini 3 simulations.",
               sources=["openalex"], found_by_queries=["Coarse-grained models of X to model Y."], query_categories=["need_semantic"])
    exp = Paper(title="Crystal structure of an enzyme", abstract="We solved the crystal structure by X-ray diffraction.",
                sources=["openalex"], found_by_queries=["Coarse-grained models of X to model Y."], query_categories=["need_semantic"])
    retained, rejected, verdicts = filter_papers([cg, exp], fulltext_lookup_budget=0, classify_fn=_fake_marker_classify_fn)
    assert cg.title in {r.title for r in retained}
    assert exp.title not in {r.title for r in retained}
    assert any("flat" in v.reason for v in verdicts if v.title == exp.title)


# --- Stage 3 abstract recovery (filter_papers(recover_abstracts=True)) -- injected lookup, no network ---------------

_RECOVERED_ABSTRACT = "We run coarse-grained molecular dynamics simulations of the target system and analyse its assembly pathway."


def _title_only_cg_paper():
    return Paper(title="Assembly of a target system", abstract=None, doi="10.1000/target", sources=["openalex"],
                 found_by_queries=["coarse-grained model of the target system"], query_categories=["system_with_methodology"])


def test_title_only_paper_is_scored_on_the_title_alone_by_default():
    p = _title_only_cg_paper()
    calls = []
    retained, rejected, verdicts = filter_papers([p], classify_fn=_fake_marker_classify_fn, abstract_lookup=lambda x: calls.append(x) or {"abstract": _RECOVERED_ABSTRACT})
    assert calls == []                                              # flag off -> no lookup at all
    assert verdicts[0].abstract_recovered is False
    assert p.title in {r.title for r in rejected}                   # the previous behaviour, unchanged


def test_recover_abstracts_lets_stage3_judge_title_plus_abstract():
    p = _title_only_cg_paper()
    retained, rejected, verdicts = filter_papers(
        [p], classify_fn=_fake_marker_classify_fn, fulltext_lookup_budget=0, recover_abstracts=True,
        abstract_lookup=lambda x: {"abstract": _RECOVERED_ABSTRACT, "via": "doi"})
    assert p.title in {r.title for r in retained}
    assert verdicts[0].abstract_recovered is True
    assert retained[0].abstract == _RECOVERED_ABSTRACT              # Stage 4 receives the abstract Stage 3 judged
    assert p.abstract is None                                       # the input object is not mutated


def test_recover_abstracts_only_looks_up_papers_without_a_usable_abstract():
    have = _paper("Coarse-grained model study", ["coarse-grained model"], abstract="We run coarse-grained simulations of the target system in detail.")
    short = _paper("Coarse-grained model note", ["coarse-grained model"], abstract="n/a")
    looked_up = []
    filter_papers([have, short], classify_fn=_fake_marker_classify_fn, fulltext_lookup_budget=0, recover_abstracts=True,
                  abstract_lookup=lambda x: looked_up.append(x.title) or None)
    assert looked_up == ["Coarse-grained model note"]


def test_recover_abstracts_keeps_the_paper_unchanged_when_lookup_fails_or_is_empty():
    for lookup in (lambda x: (_ for _ in ()).throw(RuntimeError("network down")), lambda x: None, lambda x: {"abstract": None}, lambda x: {"abstract": "too short"}):
        p = _title_only_cg_paper()
        retained, rejected, verdicts = filter_papers([p], classify_fn=_fake_marker_classify_fn, fulltext_lookup_budget=0, recover_abstracts=True, abstract_lookup=lookup)
        assert verdicts[0].abstract_recovered is False and p.title in {r.title for r in rejected}


def test_recovered_abstract_is_what_the_classifier_sees():
    seen = []
    p = _title_only_cg_paper()
    filter_papers([p], classify_fn=lambda ps: seen.append([x.abstract for x in ps]) or [0.9] * len(ps), fulltext_lookup_budget=0, recover_abstracts=True,
                  abstract_lookup=lambda x: {"abstract": _RECOVERED_ABSTRACT})
    assert seen == [[_RECOVERED_ABSTRACT]]
