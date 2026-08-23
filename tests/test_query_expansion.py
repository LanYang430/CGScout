"""Unit tests for rule-based query expansion. No network calls."""

from beam_agent.query_expansion import RuleBasedQueryExpander


def test_first_query_is_original_verbatim():
    expander = RuleBasedQueryExpander()
    result = expander.expand("200-aa beta-helix protein folding coarse-grained model")
    assert result[0] == "200-aa beta-helix protein folding coarse-grained model"


def test_produces_more_than_one_query():
    expander = RuleBasedQueryExpander()
    result = expander.expand("200-aa beta-helix protein folding coarse-grained model")
    assert len(result) > 1


def test_a_variant_drops_the_size_qualifier():
    expander = RuleBasedQueryExpander()
    result = expander.expand("200-aa beta-helix protein folding coarse-grained model")
    assert any("200-aa" not in q for q in result[1:])


def test_synonym_variant_generated_for_coarse_grained():
    expander = RuleBasedQueryExpander()
    result = expander.expand("coarse-grained model for protein folding")
    assert any("CG" in q for q in result)


def test_dedupes_case_insensitive_duplicates():
    expander = RuleBasedQueryExpander()
    result = expander.expand("Folding Folding")
    lowered = [q.lower() for q in result]
    assert len(lowered) == len(set(lowered))


def test_respects_max_queries_cap():
    expander = RuleBasedQueryExpander(max_queries=2)
    result = expander.expand(
        "coarse-grained molecular dynamics protein folding 200-aa beta-helix model"
    )
    assert len(result) <= 2


def test_handles_query_with_no_expandable_content():
    # No size qualifiers, no synonyms, no stopwords to strip -- expansion
    # should still return at least the original query without crashing.
    expander = RuleBasedQueryExpander()
    result = expander.expand("xyz")
    assert result == ["xyz"]
