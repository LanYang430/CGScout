"""Unit tests for reproducibility snapshots. No network calls.

Builds a PipelineResult by hand (no live search) and checks that
save_run_snapshot() writes exactly the expected files with sane content, and
that run-id auto-numbering doesn't collide within the same day.
"""

from __future__ import annotations

import json

from beam_agent.concept_resolver import ConceptTerm, ScientificConceptProfile
from beam_agent.models import MethodEvidence, Paper, PaperEvidence, ProcessEvidenceItem
from beam_agent.pipeline import PipelineResult
from beam_agent.query_planner import PlannedQuery, QueryPlan
from beam_agent.snapshot import _next_run_id, save_run_snapshot


def _fake_result() -> PipelineResult:
    profile = ScientificConceptProfile(
        raw_query="test query",
        structural_terms=[ConceptTerm(term="beta-helix", synonyms=["beta-solenoid"])],
        resolver="BaselineConceptResolver",
    )
    plan = QueryPlan(
        profile=profile,
        queries=[
            PlannedQuery(query="test query", category="system_specific", rationale="original"),
            PlannedQuery(query="query test", category="general_discovery", rationale="broadened"),
        ],
        planner="BaselineQueryPlanner",
    )

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
    evidence_papers = [
        retained[0].model_copy(
            update={
                "evidence": PaperEvidence(
                    method=MethodEvidence(model_class=["coarse-grained"], study_role=["applied"]),
                    processes=[ProcessEvidenceItem(raw_text="folding simulated", normalized_label="folding")],
                    extractor="baseline",
                )
            }
        ),
        retained[1].model_copy(
            update={
                "evidence": PaperEvidence(
                    missing_information=["no abstract available; cannot extract without full text or an abstract"],
                    extractor="baseline",
                )
            }
        ),
    ]

    return PipelineResult(
        user_query="test query",
        concept_profile=profile,
        query_plan=plan,
        raw_papers=raw,
        deduped_papers=deduped,
        retained_papers=retained,
        rejected_papers=rejected,
        evidence_papers=evidence_papers,
        source_errors=["semantic_scholar['query test']: boom"],
        sources=("semantic_scholar", "openalex"),
        limit_per_query=15,
        resolver_config={"class": "BaselineConceptResolver"},
        planner_config={"class": "BaselineQueryPlanner", "max_queries_per_category": 5},
        filter_config={"class": "RuleBasedRelevanceFilter", "min_score": 1.0},
        extractor_config={"class": "BaselineEvidenceExtractor"},
    )


def test_save_run_snapshot_writes_all_expected_files(tmp_path):
    result = _fake_result()

    run_dir = save_run_snapshot(result, runs_dir=tmp_path)

    expected_files = {
        "run_metadata.json",
        "concept_profile.json",
        "queries.json",
        "raw_results.json",
        "deduplicated.json",
        "retained.json",
        "rejected.json",
        "paper_evidence.json",
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
        "planned_queries": 2,
        "unique_queries_searched": 2,
        "raw_candidates": 4,
        "deduplicated": 2,
        "retained": 2,
        "rejected": 1,
        "extracted": 2,
        "no_abstract": 1,
        "with_process_evidence": 1,
        "with_method_evidence": 1,
        "with_system_evidence": 0,
    }
    assert metadata["concept_resolver"]["class"] == "BaselineConceptResolver"
    assert metadata["query_planner"]["class"] == "BaselineQueryPlanner"
    assert metadata["relevance_filter"]["class"] == "RuleBasedRelevanceFilter"
    assert metadata["evidence_extractor"]["class"] == "BaselineEvidenceExtractor"


def test_concept_profile_json_round_trips():
    result = _fake_result()
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        run_dir = save_run_snapshot(result, runs_dir=Path(d))
        profile = json.loads((run_dir / "concept_profile.json").read_text())

    assert profile["raw_query"] == "test query"
    assert profile["structural_terms"][0]["term"] == "beta-helix"


def test_queries_json_preserves_categories():
    result = _fake_result()
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        run_dir = save_run_snapshot(result, runs_dir=Path(d))
        plan = json.loads((run_dir / "queries.json").read_text())

    categories = {q["category"] for q in plan["queries"]}
    assert categories == {"system_specific", "general_discovery"}


def test_paper_evidence_json_has_counts_and_papers():
    result = _fake_result()
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        run_dir = save_run_snapshot(result, runs_dir=Path(d))
        evidence = json.loads((run_dir / "paper_evidence.json").read_text())

    assert evidence["counts"]["extracted"] == 2
    assert evidence["counts"]["no_abstract"] == 1
    assert len(evidence["papers"]) == 2
    assert evidence["papers"][0]["evidence"]["method"]["model_class"] == ["coarse-grained"]


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
