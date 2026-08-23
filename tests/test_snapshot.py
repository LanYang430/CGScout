"""Unit tests for reproducibility snapshots. No network calls.

Builds a PipelineResult by hand (no live search) and checks that
save_run_snapshot() writes exactly the expected files with sane content, and
that run-id auto-numbering doesn't collide within the same day.
"""

from __future__ import annotations

import json

from beam_agent.models import Paper
from beam_agent.pipeline import PipelineResult
from beam_agent.snapshot import _next_run_id, save_run_snapshot


def _fake_result() -> PipelineResult:
    raw = [
        Paper(title="Paper A", year=2020, doi="10.1/a", sources=["semantic_scholar"]),
        Paper(title="Paper A", year=2020, doi="10.1/a", sources=["openalex"]),  # dup of A
        Paper(title="Paper B", year=2021, doi="10.1/b", sources=["openalex"]),
        Paper(title="Off-topic paper", year=2019, sources=["openalex"]),
    ]
    deduped = [raw[0].model_copy(update={"sources": ["semantic_scholar", "openalex"]}), raw[2]]
    retained = [
        deduped[0].model_copy(update={"relevance_score": 4.0, "matched_keywords": ["paper"]}),
        deduped[1].model_copy(update={"relevance_score": 2.0, "matched_keywords": ["paper"]}),
    ]
    rejected = [raw[3].model_copy(update={"relevance_score": 0.0, "matched_keywords": []})]

    return PipelineResult(
        user_query="test query",
        expanded_queries=["test query", "query test"],
        raw_papers=raw,
        deduped_papers=deduped,
        retained_papers=retained,
        rejected_papers=rejected,
        source_errors=["semantic_scholar['query test']: boom"],
        sources=("semantic_scholar", "openalex"),
        limit_per_query=15,
        expander_config={"class": "RuleBasedQueryExpander", "max_queries": 5},
        filter_config={"class": "RuleBasedRelevanceFilter", "min_score": 1.0},
    )


def test_save_run_snapshot_writes_all_expected_files(tmp_path):
    result = _fake_result()

    run_dir = save_run_snapshot(result, runs_dir=tmp_path)

    expected_files = {
        "run_metadata.json",
        "queries.json",
        "raw_results.json",
        "deduplicated.json",
        "retained.json",
        "rejected.json",
        "report.md",
    }
    assert {p.name for p in run_dir.iterdir()} == expected_files


def test_run_dir_is_under_the_given_runs_root_with_date_prefixed_id(tmp_path):
    result = _fake_result()

    run_dir = save_run_snapshot(result, runs_dir=tmp_path)

    assert run_dir.parent == tmp_path
    # e.g. "2026-08-23_001"
    date_part, _, seq_part = run_dir.name.partition("_")
    assert len(date_part) == 10  # YYYY-MM-DD
    assert seq_part.isdigit()


def test_run_metadata_has_required_fields(tmp_path):
    result = _fake_result()

    run_dir = save_run_snapshot(result, runs_dir=tmp_path)
    metadata = json.loads((run_dir / "run_metadata.json").read_text())

    assert metadata["user_query"] == "test query"
    assert "timestamp_utc" in metadata
    assert metadata["sources"] == ["semantic_scholar", "openalex"]
    assert metadata["limit_per_query"] == 15
    assert "pipeline_version" in metadata
    assert metadata["counts"] == {
        "expanded_queries": 2,
        "raw_candidates": 4,
        "deduplicated": 2,
        "retained": 2,
        "rejected": 1,
    }
    assert metadata["query_expander"]["class"] == "RuleBasedQueryExpander"
    assert metadata["relevance_filter"]["class"] == "RuleBasedRelevanceFilter"


def test_raw_results_json_preserves_literal_duplicates(tmp_path):
    result = _fake_result()

    run_dir = save_run_snapshot(result, runs_dir=tmp_path)
    raw = json.loads((run_dir / "raw_results.json").read_text())

    assert len(raw) == 4  # dedup has NOT been applied to raw_results.json


def test_second_run_same_day_gets_incremented_sequence(tmp_path):
    result = _fake_result()

    first = save_run_snapshot(result, runs_dir=tmp_path)
    second = save_run_snapshot(result, runs_dir=tmp_path)

    assert first != second
    first_seq = int(first.name.split("_")[-1])
    second_seq = int(second.name.split("_")[-1])
    assert second_seq == first_seq + 1


def test_next_run_id_starts_at_001_for_empty_dir(tmp_path):
    run_id = _next_run_id(tmp_path, today="2026-08-23")
    assert run_id == "2026-08-23_001"


def test_next_run_id_skips_past_existing_runs(tmp_path):
    (tmp_path / "2026-08-23_001").mkdir()
    (tmp_path / "2026-08-23_002").mkdir()
    (tmp_path / "2026-08-23_not_numeric_junk").mkdir()  # should be ignored, not crash

    run_id = _next_run_id(tmp_path, today="2026-08-23")
    assert run_id == "2026-08-23_003"


def test_explicit_run_id_collision_raises(tmp_path):
    result = _fake_result()
    save_run_snapshot(result, runs_dir=tmp_path, run_id="fixed-id")

    import pytest

    with pytest.raises(FileExistsError):
        save_run_snapshot(result, runs_dir=tmp_path, run_id="fixed-id")
