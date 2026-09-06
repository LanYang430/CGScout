"""Unit tests for pipeline.py's query-grouping + provenance-stamping helpers.

No network calls -- these are pure functions operating on PlannedQuery/Paper
objects built by hand. run_pipeline() itself (which does call the network) is
not exercised here; these two helpers are exactly the new logic this pass
added, so they're what's worth testing in isolation.
"""

from beam_agent.models import Paper
from beam_agent.pipeline import group_queries_by_normalized_key, stamp_provenance
from beam_agent.query_planner import PlannedQuery


def test_group_queries_dedupes_same_string_across_categories():
    queries = [
        PlannedQuery(query="beta-helix folding", category="system_specific"),
        PlannedQuery(query="  Beta-Helix   Folding  ", category="general_discovery"),  # same, case/whitespace differ
        PlannedQuery(query="coarse-grained model folding", category="general_discovery"),
    ]

    grouped = group_queries_by_normalized_key(queries)

    assert len(grouped) == 2
    original, categories = grouped["beta-helix folding"]
    assert original == "beta-helix folding"  # first-seen original string kept, not the reformatted duplicate
    assert categories == ["system_specific", "general_discovery"]

    original2, categories2 = grouped["coarse-grained model folding"]
    assert original2 == "coarse-grained model folding"
    assert categories2 == ["general_discovery"]


def test_group_queries_preserves_first_seen_order():
    queries = [
        PlannedQuery(query="first query", category="system_specific"),
        PlannedQuery(query="second query", category="general_discovery"),
        PlannedQuery(query="third query", category="general_discovery"),
    ]

    grouped = group_queries_by_normalized_key(queries)

    assert [v[0] for v in grouped.values()] == ["first query", "second query", "third query"]


def test_group_queries_skips_blank_query_strings():
    queries = [PlannedQuery(query="   ", category="system_specific"), PlannedQuery(query="real query", category="system_specific")]

    grouped = group_queries_by_normalized_key(queries)

    assert len(grouped) == 1
    assert list(grouped.values())[0][0] == "real query"


def test_stamp_provenance_adds_query_and_categories():
    papers = [Paper(title="A", sources=["openalex"]), Paper(title="B", sources=["openalex"])]

    stamp_provenance(papers, query="beta-helix folding", categories=["system_specific"])

    assert papers[0].found_by_queries == ["beta-helix folding"]
    assert papers[0].query_categories == ["system_specific"]
    assert papers[1].found_by_queries == ["beta-helix folding"]


def test_stamp_provenance_multiple_categories_no_duplicates():
    papers = [Paper(title="A", sources=["openalex"])]

    stamp_provenance(papers, query="q", categories=["system_specific", "general_discovery"])
    assert papers[0].query_categories == ["system_specific", "general_discovery"]

    # Stamping again with an already-present query/category must not duplicate.
    stamp_provenance(papers, query="q", categories=["system_specific"])
    assert papers[0].found_by_queries == ["q"]
    assert papers[0].query_categories == ["system_specific", "general_discovery"]


def test_stamp_provenance_different_queries_accumulate():
    papers = [Paper(title="A", sources=["openalex"])]

    stamp_provenance(papers, query="query one", categories=["system_specific"])
    stamp_provenance(papers, query="query two", categories=["general_discovery"])

    assert papers[0].found_by_queries == ["query one", "query two"]
    assert papers[0].query_categories == ["system_specific", "general_discovery"]
