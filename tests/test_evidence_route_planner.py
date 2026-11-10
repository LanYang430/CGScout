"""Unit tests for this session's fix to
evidence_plan.extract_lexical_structural_feature(): Case B, a
fallback for when the grounded system span IS the bare structural-class
name itself (e.g. "beta helix"), with no class-head noun ("protein"/"fold"/
...) for the pre-existing Case A anchor-and-walk algorithm to anchor on.

Real case this was diagnosed against: the user message "hi I want to study
folding of beta helix" grounds SYSTEM_IDENTITY/session.system_span exactly
to "beta helix" -- no head noun -- so Case A returned None and the
structural branch never ran at all (0 related systems discovered).

Case B is edge-trimming ONLY (never scans mid-span, never parses a raw
sentence), using a deliberately NARROWER word set than the full
_CONNECTIVES (_EDGE_WRAPPER_TOKENS) -- pure empty wrapper words only.
Anything relational/derivational ("of"/"with"/"in"/"and"/"or"/"like"/
"based"/"containing"/"type"/"types"/"class") is excluded from trimming AND,
if it ends up surviving inside the trimmed core anyway, causes an outright
reject (None) rather than being silently included in the extracted span.
"""
from __future__ import annotations


from cgscout.retrieval.direct.evidence_plan import (  # noqa: E402
    _EDGE_WRAPPER_TOKENS,
    extract_lexical_structural_feature,
)
from cgscout.retrieval.related_systems.discovery import _CONNECTIVES  # noqa: E402


# --- Case A: pre-existing anchored behavior, MUST be byte-identical --------

def test_case_a_anchored_extraction_unchanged():
    assert extract_lexical_structural_feature("beta-helix protein") == "beta-helix"
    assert extract_lexical_structural_feature("TIM barrel enzyme") == "TIM barrel"
    assert extract_lexical_structural_feature("coiled-coil protein") == "coiled-coil"
    assert (extract_lexical_structural_feature("I have a beta-helix protein and want to study folding")
            == "beta-helix")


def test_case_a_named_proteins_without_any_morpheme_stay_none():
    for name in ("Pertactin", "Wnt-Wntless", "GPCR", "lysozyme", "EspP"):
        assert extract_lexical_structural_feature(name) is None


# --- Case B: the new fallback, positive cases -------------------------------

def test_bare_structural_term_now_extracts_the_real_diagnosed_case():
    """The exact real case: 'hi I want to study folding of beta helix'
    grounds session.system_span to 'beta helix' -- this must now extract."""
    assert extract_lexical_structural_feature("beta helix") == "beta helix"
    assert extract_lexical_structural_feature("beta-helix") == "beta-helix"


def test_bare_structural_term_with_leading_article_is_trimmed():
    assert extract_lexical_structural_feature("a beta helix") == "beta helix"
    assert extract_lexical_structural_feature("the amyloid fibril") == "amyloid fibril"


def test_other_bare_structural_terms():
    assert extract_lexical_structural_feature("coiled coil") == "coiled coil"
    assert extract_lexical_structural_feature("coiled-coil") == "coiled-coil"
    assert extract_lexical_structural_feature("TIM barrel") == "TIM barrel"
    assert extract_lexical_structural_feature("left-handed beta-helix") == "left-handed beta-helix"
    assert extract_lexical_structural_feature("membrane transporter") == "membrane transporter"


def test_hyphen_joined_compound_with_a_relational_suffix_is_one_token():
    """'beta-helix-like' is a single hyphen-joined token under the existing
    tokenizer (regex includes '-' in a word), so it isn't caught by the
    stray-connective check -- a pre-existing tokenization convention shared
    with Case A (Case A's own connective-stop check has the identical
    property), not something new introduced here. Documented, not silently
    assumed."""
    assert extract_lexical_structural_feature("beta-helix-like") == "beta-helix-like"


# --- Case B: negative cases -- must stay None -------------------------------

def test_named_proteins_still_none_under_case_b():
    """No morpheme anywhere -- Case B can't manufacture one either."""
    for name in ("Pertactin", "Wnt-Wntless", "GPCR", "lysozyme", "EspP"):
        assert extract_lexical_structural_feature(name) is None


def test_mid_span_conjunction_is_not_split():
    """Case B never scans/splits mid-span -- 'and' is not at an edge here,
    so nothing is trimmed and the 4-token core exceeds _MAX_MODIFIER_WORDS."""
    assert extract_lexical_structural_feature("beta helix and coiled coil") is None


def test_long_unanchored_sentence_stays_none_case_b_does_not_parse_sentences():
    """The critical negative case: Case B must NOT try to pull a structural
    substring out of a long, unanchored raw sentence -- it only accepts an
    already-short, already-mostly-clean grounded span. This function is
    NEVER handed a raw full user message in the real pipeline (only an
    already-grounded system_span), but the guard must hold regardless."""
    long_cases = [
        "hi I want to study folding of beta helix",
        "a very long unrelated sentence about something else entirely with no anchor at all",
        # deliberately no _CLASS_HEAD_WORDS token anywhere (so this reaches
        # Case B, not Case A) -- long, contains the morpheme "helix" buried
        # in the middle, no class-head noun anywhere for Case A to anchor
        # on either. Neither edge is a wrapper token, so nothing trims, and
        # the un-trimmed core is far longer than _MAX_MODIFIER_WORDS.
        "I am interested in learning more about how beta helix behave under various conditions",
    ]
    for text in long_cases:
        assert extract_lexical_structural_feature(text) is None, f"must not extract from: {text!r}"


def test_case_a_own_pre_existing_backward_walk_quirk_is_unaffected_by_case_b():
    """NOT a Case B regression -- Case A's own (unmodified) backward walk
    already, before this session's change, could include a non-connective
    filler word ('how') within its _MAX_MODIFIER_WORDS bound when a
    class-head word (here 'proteins') anchors it. Documented so this
    pre-existing behavior is never mistaken for something Case B
    introduced."""
    assert (extract_lexical_structural_feature(
        "I am interested in learning more about how beta helix proteins behave under various conditions")
        == "how beta helix")


def test_class_head_word_present_but_morpheme_only_after_it_stays_none():
    """Pre-existing Case A limitation (backward-walk only) -- Case B's
    length bound also correctly declines to rescue this; not a regression,
    both paths agree it's None."""
    assert extract_lexical_structural_feature("this enzyme has a beta sheet") is None


# --- Case B: the exact relational/derivational words the user flagged ------
# must NEVER be silently stripped from an edge, and must NEVER survive
# inside the extracted span either.

def test_relational_prepositions_at_an_edge_are_not_stripped_or_included():
    for text in ("beta helix of", "of beta helix", "with beta helix", "beta helix with",
                 "beta helix in", "in beta helix"):
        assert extract_lexical_structural_feature(text) is None, f"unexpected result for {text!r}"


def test_similarity_and_derivation_markers_are_not_stripped_or_included():
    """'like'/'based' change the CLAIM (resembles X / built on X) -- must
    never be silently dropped to produce a bare 'X'."""
    for text in ("beta helix like", "beta helix based"):
        assert extract_lexical_structural_feature(text) is None, f"unexpected result for {text!r}"


def test_classifier_suffixes_are_not_stripped_or_included():
    """'type'/'class' mean "a kind of X", not X itself."""
    for text in ("beta helix type", "beta helix class"):
        assert extract_lexical_structural_feature(text) is None, f"unexpected result for {text!r}"


def test_dangling_conjunctions_are_not_stripped_or_included():
    for text in ("beta helix and", "and beta helix", "beta helix or", "or beta helix"):
        assert extract_lexical_structural_feature(text) is None, f"unexpected result for {text!r}"


# --- _EDGE_WRAPPER_TOKENS itself: exactly the curated subset ---------------

def test_edge_wrapper_tokens_excludes_relational_and_derivational_words():
    excluded = {"and", "or", "of", "with", "in", "like", "based",
                "containing", "type", "types", "class"}
    assert _EDGE_WRAPPER_TOKENS.isdisjoint(excluded)


def test_edge_wrapper_tokens_is_a_strict_subset_of_connectives():
    assert _EDGE_WRAPPER_TOKENS <= _CONNECTIVES
    assert _EDGE_WRAPPER_TOKENS != _CONNECTIVES  # strictly narrower, not a mechanical full reuse


def test_edge_wrapper_tokens_keeps_pure_empty_wrappers():
    for word in ("the", "a", "an", "other", "others", "such", "as", "eg"):
        assert word in _EDGE_WRAPPER_TOKENS
