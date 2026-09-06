"""Pure unit tests for the objective-specificity normalization layer.
No LLM, no network -- runs under either venv.

    .venv/bin/python -m pytest tests/test_objective_normalization.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks" / "concept_resolver_reasoning"))

from objective_normalization import normalize_objective  # noqa: E402


def test_strip_framing_verb_and_inline_system_span():
    r = normalize_objective("study GPCR dimerization", "GPCR")
    assert r.residual == "dimerization"
    assert r.removed_system is True
    assert r.removed_prefixes == ["study"]
    assert r.deterministic_verdict is None


def test_simulate_x_with_span_removed_is_vacuous_too_broad():
    r = normalize_objective("simulate an intrinsically disordered protein", "intrinsically disordered protein")
    assert r.residual == ""
    assert r.deterministic_verdict == "too_broad"


def test_studying_the_wnt_wls_interaction_reduces_to_interaction_phrase():
    r = normalize_objective("studying the Wnt-Wls protein-protein interaction", "Wnt-Wls")
    assert r.residual == "protein-protein interaction"
    assert r.removed_system is True
    assert r.deterministic_verdict is None  # classifier then judges -> too_broad


def test_binding_interface_between_x_and_y_reduces_to_binding_interface():
    r = normalize_objective("binding interface between Wnt and Wls", "Wnt and Wls")
    assert r.residual == "binding interface"
    assert r.deterministic_verdict is None  # classifier -> actionable


def test_no_system_span_inside_objective_leaves_process_word():
    r = normalize_objective("study folding", None)
    assert r.residual == "folding"
    assert r.removed_system is False
    assert r.deterministic_verdict is None


def test_span_not_present_in_objective_is_not_removed():
    # A decomposed cleanly: objective value carries no system name, span is elsewhere
    r = normalize_objective("studying the protein-protein interaction", "Wnt-Wls")
    assert r.residual == "protein-protein interaction"
    assert r.removed_system is False


def test_the_study_of_x_folding():
    r = normalize_objective("the study of chignolin folding", "chignolin")
    assert r.residual == "folding"
    assert r.removed_prefixes == ["the study of"]


def test_behavior_of_the_ribosome_reduces_to_behavior():
    r = normalize_objective("the behavior of the ribosome", "ribosome")
    assert r.residual == "behavior"
    assert r.deterministic_verdict is None  # classifier -> too_broad


def test_conformational_change_in_adenylate_kinase():
    r = normalize_objective("conformational change in adenylate kinase", "adenylate kinase")
    assert r.residual == "conformational change"
    assert r.deterministic_verdict is None


def test_empty_objective_is_vacuous():
    assert normalize_objective("", None).deterministic_verdict == "too_broad"
    assert normalize_objective("the", None).deterministic_verdict == "too_broad"


def test_objective_that_is_only_the_system_name_is_vacuous():
    r = normalize_objective("chignolin", "chignolin")
    assert r.residual == ""
    assert r.deterministic_verdict == "too_broad"


def test_span_must_not_match_inside_a_longer_token():
    # span "a beta-helix protein" must NOT match the tail of
    # "200-aa beta-helix protein" and leave a "200-a" fragment
    r = normalize_objective("the folding process of a 200-aa beta-helix protein", "a beta-helix protein")
    assert "200-a " not in r.residual and not r.residual.endswith("200-a")
    assert "200-aa" in r.residual  # left intact because the span isn't a clean token run here
    assert r.removed_system is False


def test_clean_hyphenated_system_token_is_removed():
    r = normalize_objective("study Wnt-Wls binding", "Wnt-Wls")
    assert r.residual == "binding"
    assert r.removed_system is True


def test_span_removed_only_at_token_boundary_when_it_is_a_clean_run():
    r = normalize_objective("conformational change in adenylate kinase", "adenylate kinase")
    assert r.residual == "conformational change"
    assert r.removed_system is True
    assert r.deterministic_verdict is None  # classifier -> actionable (post prompt update)
