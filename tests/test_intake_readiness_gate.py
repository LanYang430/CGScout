"""Pure unit tests for IntakeReadinessGate (v5). No LLM, no network.
The gap categorizer is passed in as a fake callable.

    .venv/bin/python -m pytest tests/test_intake_readiness_gate.py -q
"""
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent.parent / "benchmarks" / "concept_resolver_reasoning"
sys.path.insert(0, str(BENCH))

import run_abcd_test_v5 as v5  # noqa: E402
from intake_readiness_gate import (  # noqa: E402
    SPECIFICITY_GAP_PREFIX,
    SYSTEM_GAP_CHARACTERISTIC,
    run_intake_readiness_gate,
)


def _es(status, objective=None, gaps=None):
    return v5.EffectiveReasonerState(
        objective_status=status,
        scientific_objective=objective,
        objective_support_fact_ids=["f1"] if status == v5.ObjectiveStatus.EXPLICIT else [],
        gaps=gaps or [],
    )


def _gap(char, *, blocking=True, resolution=v5.ResolutionType.ASK_USER, effect="x"):
    return v5.Gap(characteristic=char, retrieval_effect=effect,
                  blocks_first_retrieval=blocking, resolution=resolution)


def _fixed_categorizer(mapping):
    """mapping: characteristic-substring -> category string."""
    def _c(gap):
        for sub, cat in mapping.items():
            if sub.lower() in gap.characteristic.lower():
                return cat
        return "other"
    return _c


# ---- inject / promote ------------------------------------------------------

def test_present_system_actionable_objective_is_ready():
    es = _es(v5.ObjectiveStatus.EXPLICIT, "folding")
    ns, audit = run_intake_readiness_gate(es, "present", "actionable_for_first_retrieval")
    assert audit["readiness_overrides"] == [] and audit["notes"] == []
    assert v5.controller_decide(ns.gaps).next_action == "READY"


def test_absent_system_injects_blocking_system_gap_first():
    es = _es(v5.ObjectiveStatus.EXPLICIT, "folding")
    ns, audit = run_intake_readiness_gate(es, "absent", "actionable_for_first_retrieval")
    assert ns.gaps[0].characteristic == SYSTEM_GAP_CHARACTERISTIC
    d = v5.controller_decide(ns.gaps)
    assert d.next_action == "ASK_USER" and "system" in d.question_to_user.lower()


def test_present_system_too_broad_objective_asks_refinement():
    es = _es(v5.ObjectiveStatus.EXPLICIT, "the interaction")
    ns, _ = run_intake_readiness_gate(es, "present", "too_broad", objective_label="the interaction")
    d = v5.controller_decide(ns.gaps)
    assert d.next_action == "ASK_USER"
    assert d.target_gap.characteristic == f"{SPECIFICITY_GAP_PREFIX} the interaction to study"


def test_too_broad_with_no_label_uses_generic_objective_gap():
    from intake_readiness_gate import GENERIC_OBJECTIVE_CHARACTERISTIC
    es = _es(v5.ObjectiveStatus.EXPLICIT, "simulate an IDP")
    ns, _ = run_intake_readiness_gate(es, "present", "too_broad", objective_label=None)
    d = v5.controller_decide(ns.gaps)
    assert d.next_action == "ASK_USER"
    assert d.target_gap.characteristic == GENERIC_OBJECTIVE_CHARACTERISTIC


def test_absent_system_and_too_broad_asks_system_first():
    es = _es(v5.ObjectiveStatus.EXPLICIT, "the interaction")
    ns, _ = run_intake_readiness_gate(es, "absent", "too_broad", objective_label="the interaction")
    blocking = [g for g in ns.gaps if g.blocks_first_retrieval]
    assert len(blocking) == 2
    assert v5.controller_decide(ns.gaps).target_gap.characteristic == SYSTEM_GAP_CHARACTERISTIC


def test_none_specificity_is_treated_as_actionable():
    es = _es(v5.ObjectiveStatus.EXPLICIT, "the interaction")
    ns, audit = run_intake_readiness_gate(es, "present", None)
    assert audit["notes"] == []
    assert v5.controller_decide(ns.gaps).next_action == "READY"


def test_missing_objective_with_system_present_leaves_b_guard_gap():
    obj_gap = _gap(v5.OBJECTIVE_GAP_CHARACTERISTIC)
    es = _es(v5.ObjectiveStatus.MISSING, None, gaps=[obj_gap])
    ns, audit = run_intake_readiness_gate(es, "present", None,
                                          categorize=_fixed_categorizer({"process or property": "scientific_objective"}))
    # spec is None (objective MISSING) -> NOT actionable -> not demoted
    assert audit["readiness_overrides"] == []
    assert v5.controller_decide(ns.gaps).target_gap.characteristic == v5.OBJECTIVE_GAP_CHARACTERISTIC


def test_missing_objective_and_absent_system_asks_system_first():
    obj_gap = _gap(v5.OBJECTIVE_GAP_CHARACTERISTIC)
    es = _es(v5.ObjectiveStatus.MISSING, None, gaps=[obj_gap])
    ns, _ = run_intake_readiness_gate(es, "absent", None)
    assert v5.controller_decide(ns.gaps).target_gap.characteristic == SYSTEM_GAP_CHARACTERISTIC


# ---- demote authority ----------------------------------------------------

def test_demote_b_system_gap_when_system_present():
    b_gap = _gap("please name the specific TIM-barrel fold protein")
    es = _es(v5.ObjectiveStatus.EXPLICIT, "thermal stability", gaps=[b_gap])
    cat = _fixed_categorizer({"specific": "system_identity"})
    ns, audit = run_intake_readiness_gate(es, "present", "actionable_for_first_retrieval", categorize=cat)
    assert len(audit["readiness_overrides"]) == 1
    assert audit["readiness_overrides"][0]["action"] == "DEMOTED"
    assert audit["readiness_overrides"][0]["category"] == "system_identity"
    assert v5.controller_decide(ns.gaps).next_action == "READY"
    # raw B gap preserved in audit, unmodified
    assert audit["raw_b_gaps"][0]["blocks_first_retrieval"] is True


def test_demote_b_objective_gap_when_objective_actionable():
    b_gap = _gap("which aspect of folding to focus on")
    es = _es(v5.ObjectiveStatus.EXPLICIT, "study folding", gaps=[b_gap])
    cat = _fixed_categorizer({"aspect of folding": "scientific_objective"})
    ns, audit = run_intake_readiness_gate(es, "present", "actionable_for_first_retrieval", categorize=cat)
    assert audit["readiness_overrides"][0]["category"] == "scientific_objective"
    assert v5.controller_decide(ns.gaps).next_action == "READY"


def test_do_not_demote_other_categories():
    ts_gap = _gap("what timescale is of interest")
    es = _es(v5.ObjectiveStatus.EXPLICIT, "folding", gaps=[ts_gap])
    cat = _fixed_categorizer({"timescale": "timescale"})
    ns, audit = run_intake_readiness_gate(es, "present", "actionable_for_first_retrieval", categorize=cat)
    assert audit["readiness_overrides"] == []
    assert v5.controller_decide(ns.gaps).next_action == "ASK_USER"


def test_do_not_demote_when_categorizer_unavailable():
    b_gap = _gap("please name the specific protein")
    es = _es(v5.ObjectiveStatus.EXPLICIT, "thermal stability", gaps=[b_gap])
    ns, audit = run_intake_readiness_gate(es, "present", "actionable_for_first_retrieval", categorize=None)
    assert audit["readiness_overrides"] == []
    assert v5.controller_decide(ns.gaps).next_action == "ASK_USER"


def test_do_not_demote_objective_gap_when_objective_too_broad():
    b_gap = _gap("which aspect of the interaction to focus on")
    es = _es(v5.ObjectiveStatus.EXPLICIT, "the interaction", gaps=[b_gap])
    cat = _fixed_categorizer({"aspect of the interaction": "scientific_objective"})
    ns, audit = run_intake_readiness_gate(es, "present", "too_broad", categorize=cat)
    assert audit["readiness_overrides"] == []
    # B's own gap stays blocking; gate does not also inject a duplicate
    assert sum(g.blocks_first_retrieval for g in ns.gaps) == 1
    assert v5.controller_decide(ns.gaps).next_action == "ASK_USER"


def test_demote_system_gap_then_no_injected_system_gap_when_absent_but_b_asked():
    # system ABSENT, B asked about system -> not demoted (present-only), and
    # the gate must not inject a SECOND system gap on top of B's.
    b_gap = _gap("which specific system do you mean")
    es = _es(v5.ObjectiveStatus.EXPLICIT, "folding", gaps=[b_gap])
    cat = _fixed_categorizer({"which specific system": "system_identity"})
    ns, audit = run_intake_readiness_gate(es, "absent", "actionable_for_first_retrieval", categorize=cat)
    assert audit["readiness_overrides"] == []
    assert sum(g.blocks_first_retrieval for g in ns.gaps) == 1


# ---- resolution=search demotion (category-independent) --------------------
#
# CG-model-recommendation scope decision (this session): a gap B marked
# resolution=search must never block first retrieval, regardless of gap
# category or whether a categorizer is even available -- final readiness
# is still decided purely by the existing system-presence + objective-
# specificity axes above, unchanged. No categorizer is passed in ANY of
# these tests (categorize=None) specifically to prove the search demotion
# does not depend on it, unlike the system_identity/scientific_objective
# demotions above.

def test_regression_1_phospho_idp_long_query_search_gap_demoted_to_ready():
    """Fixture is the REAL Gap B produced (this session's live diagnostic
    run) for: "I want to study how serine or threonine phosphorylation
    changes the conformational ensemble of an intrinsically disordered
    protein. Is there a CG model that explicitly supports phosphorylated
    residues?" -- system/objective were already grounded fine; this SEARCH
    gap alone is what used to send next_action=SEARCH (search_unsupported,
    a dead end) instead of READY."""
    search_gap = _gap(
        "specific coarse-grained model or models that support phosphorylated residues",
        blocking=True, resolution=v5.ResolutionType.SEARCH,
        effect="Identifying specific CG models that support phosphorylated residues "
               "would allow for a focused search on the capabilities and performance "
               "of these models in simulating the conformational changes induced by "
               "phosphorylation.",
    )
    es = _es(
        v5.ObjectiveStatus.EXPLICIT,
        "study how serine or threonine phosphorylation changes the conformational "
        "ensemble of an intrinsically disordered protein",
        gaps=[search_gap],
    )
    ns, audit = run_intake_readiness_gate(es, "present", "actionable_for_first_retrieval")
    assert v5.controller_decide(ns.gaps).next_action == "READY"
    assert len(audit["readiness_overrides"]) == 1
    override = audit["readiness_overrides"][0]
    assert override["action"] == "DEMOTED"
    assert override["category"] is None  # category-independent -- no categorizer was even passed
    assert "search" in override["reason"].lower()
    # original gap preserved verbatim in audit (still resolution=search, still blocking)
    assert audit["raw_b_gaps"][0]["resolution"] == "search"
    assert audit["raw_b_gaps"][0]["blocks_first_retrieval"] is True
    # working gap D's contract requires (blocks_first_retrieval=False <=> not_yet_needed)
    assert ns.gaps[0].blocks_first_retrieval is False
    assert ns.gaps[0].resolution == v5.ResolutionType.NOT_YET_NEEDED


def test_regression_2_plain_short_system_and_objective_query_is_ready():
    """No gaps at all -- the ordinary case a short, clean query produces.
    Confirms the new pass 0 is a strict no-op when there's nothing to
    demote (same outcome as test_present_system_actionable_objective_is_ready,
    kept here too as an explicit named regression point for this change)."""
    es = _es(v5.ObjectiveStatus.EXPLICIT, "folding")
    ns, audit = run_intake_readiness_gate(es, "present", "actionable_for_first_retrieval")
    assert audit["readiness_overrides"] == []
    assert v5.controller_decide(ns.gaps).next_action == "READY"


def test_regression_3_missing_system_still_asks_user_even_with_search_gap():
    """A resolution=search gap sitting alongside a genuinely absent system
    must NOT paper over the missing system -- readiness is still governed
    by the system-presence axis, unchanged."""
    search_gap = _gap(
        "specific coarse-grained model that supports phosphorylated residues",
        blocking=True, resolution=v5.ResolutionType.SEARCH,
    )
    es = _es(v5.ObjectiveStatus.EXPLICIT, "conformational ensemble changes", gaps=[search_gap])
    ns, audit = run_intake_readiness_gate(es, "absent", "actionable_for_first_retrieval")
    d = v5.controller_decide(ns.gaps)
    assert d.next_action == "ASK_USER"
    assert d.target_gap.characteristic == SYSTEM_GAP_CHARACTERISTIC
    # the search gap was still demoted (it's not what's asking the question)
    assert any(o["action"] == "DEMOTED" for o in audit["readiness_overrides"])


def test_regression_4_missing_actionable_objective_still_asks_user_even_with_search_gap():
    """Same, but for the objective axis: objective_status=MISSING (B never
    grounded a scientific_objective fact at all) must still ASK_USER, even
    with an unrelated resolution=search gap present."""
    obj_gap = _gap(v5.OBJECTIVE_GAP_CHARACTERISTIC)
    search_gap = _gap(
        "specific coarse-grained model that supports phosphorylated residues",
        blocking=True, resolution=v5.ResolutionType.SEARCH,
    )
    es = _es(v5.ObjectiveStatus.MISSING, None, gaps=[obj_gap, search_gap])
    ns, audit = run_intake_readiness_gate(es, "present", None)
    d = v5.controller_decide(ns.gaps)
    assert d.next_action == "ASK_USER"
    assert d.target_gap.characteristic == v5.OBJECTIVE_GAP_CHARACTERISTIC
    assert any(o["action"] == "DEMOTED" for o in audit["readiness_overrides"])


def test_search_demotion_does_not_need_a_categorizer():
    """Category-independent by construction -- must demote even when
    categorize=None (unlike the system_identity/scientific_objective
    demotions, which explicitly require a working categorizer)."""
    search_gap = _gap("some external fact", blocking=True, resolution=v5.ResolutionType.SEARCH)
    es = _es(v5.ObjectiveStatus.EXPLICIT, "folding", gaps=[search_gap])
    ns, audit = run_intake_readiness_gate(es, "present", "actionable_for_first_retrieval", categorize=None)
    assert len(audit["readiness_overrides"]) == 1
    assert v5.controller_decide(ns.gaps).next_action == "READY"


def test_non_blocking_search_gap_is_left_untouched():
    """A resolution=search gap that was never blocking in the first place
    (blocks_first_retrieval=False already) is passed through verbatim --
    nothing to demote, no spurious audit entry."""
    search_gap = _gap("some external fact", blocking=False, resolution=v5.ResolutionType.SEARCH)
    es = _es(v5.ObjectiveStatus.EXPLICIT, "folding", gaps=[search_gap])
    ns, audit = run_intake_readiness_gate(es, "present", "actionable_for_first_retrieval")
    assert audit["readiness_overrides"] == []
    assert ns.gaps[0].resolution == v5.ResolutionType.SEARCH  # untouched, not rewritten
    assert v5.controller_decide(ns.gaps).next_action == "READY"
