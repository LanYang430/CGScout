"""Optional reproducibility snapshots for pipeline runs.

Nothing in this module runs automatically. `run_pipeline()` (pipeline.py) has
no knowledge this module exists and performs no file I/O of its own --
snapshotting is opt-in at the outermost layer only: the CLI's `--snapshot`
flag, or calling `save_run_snapshot()` yourself with a `PipelineResult`. That
keeps the low-level search/expand/dedupe/filter functions free of I/O side
effects and means snapshot logic lives in exactly one place instead of being
threaded through every function that happens to produce data worth keeping.

Snapshots are written *outside* the git repository by default -- a sibling of
the repo root, `../beam_agent_runs/` -- specifically so run outputs are never
accidentally tracked, committed, or pushed. Override with the `runs_dir`
argument, the `BEAM_AGENT_RUNS_DIR` environment variable, or (from the CLI)
`--snapshot-dir`.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from beam_agent import __version__ as PIPELINE_VERSION
from beam_agent.pipeline import PipelineResult

# This file lives at <repo_root>/beam_agent/snapshot.py, so parents[1] is
# <repo_root> and its parent is the sibling level "../beam_agent_runs/" is
# specified relative to.
_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = Path(
    os.environ.get("BEAM_AGENT_RUNS_DIR", str(_REPO_ROOT.parent / "beam_agent_runs"))
)


def _next_run_id(runs_dir: Path, *, today: str | None = None) -> str:
    """Pick the next `<date>_<NNN>` run id for today, e.g. '2026-08-23_001'.

    Scans existing subdirectories under `runs_dir` for today's date prefix and
    increments past the highest sequence number found, so concurrent/repeated
    runs on the same day never collide.
    """
    date_str = today or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prefix = f"{date_str}_"
    existing_seqs = []
    if runs_dir.exists():
        for entry in runs_dir.iterdir():
            if entry.is_dir() and entry.name.startswith(prefix):
                suffix = entry.name[len(prefix):]
                if suffix.isdigit():
                    existing_seqs.append(int(suffix))
    next_seq = max(existing_seqs, default=0) + 1
    return f"{prefix}{next_seq:03d}"


def _dump_json(path: Path, payload: object) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def _render_report(result: PipelineResult, run_id: str, timestamp_utc: str) -> str:
    lines = [
        f"# BEAM literature run `{run_id}`",
        "",
        f"- **Timestamp (UTC):** {timestamp_utc}",
        f"- **Query:** {result.user_query!r}",
        f"- **Sources:** {', '.join(result.sources)}",
        f"- **Limit per query:** {result.limit_per_query}",
        f"- **Pipeline version:** {PIPELINE_VERSION}",
        f"- **Query expander:** {result.expander_config}",
        f"- **Relevance filter:** {result.filter_config}",
        "",
        "## Expanded queries",
        "",
    ]
    lines += [f"- {q!r}" for q in result.expanded_queries]

    lines += [
        "",
        "## Counts",
        "",
        f"- Raw candidates: {result.raw_candidate_count}",
        f"- After dedup: {result.deduped_count}",
        f"- Retained: {result.filtered_count}",
        f"- Rejected: {len(result.rejected_papers)}",
        "",
    ]

    if result.source_errors:
        lines += ["## Source errors", ""]
        lines += [f"- {e}" for e in result.source_errors]
        lines.append("")

    lines += [
        "## Retained papers",
        "",
        "| Score | Year | Sources | Title | DOI |",
        "|---|---|---|---|---|",
    ]
    for p in result.retained_papers:
        tag = "+".join(p.sources)
        score = f"{p.relevance_score:.1f}" if p.relevance_score is not None else ""
        title = p.title.replace("|", "\\|")
        lines.append(f"| {score} | {p.year or ''} | {tag} | {title} | {p.doi or ''} |")

    if result.rejected_papers:
        lines += ["", "## Rejected papers", "", "| Score | Title |", "|---|---|"]
        for p in result.rejected_papers:
            score = f"{p.relevance_score:.1f}" if p.relevance_score is not None else ""
            title = p.title.replace("|", "\\|")
            lines.append(f"| {score} | {title} |")

    return "\n".join(lines) + "\n"


def save_run_snapshot(
    result: PipelineResult,
    *,
    runs_dir: Path | str | None = None,
    run_id: str | None = None,
) -> Path:
    """Write a full reproducibility snapshot of a pipeline run to disk.

    Creates `<runs_dir>/<run_id>/` (default run_id: today's UTC date plus the
    next sequence number, e.g. `2026-08-23_001`) containing:

        run_metadata.json   query, timestamp, sources, limits, version,
                             component config, counts at each stage
        queries.json         user_query + all expanded queries
        raw_results.json     every candidate before dedup (duplicates included)
        deduplicated.json    candidates after cross-source/cross-query dedup
        retained.json        candidates that passed the relevance filter
        rejected.json        candidates the relevance filter dropped
        report.md            human-readable summary of all of the above

    Returns the run directory path. Raises if `run_id` is given explicitly and
    already exists (auto-generated ids never collide by construction).
    """
    runs_root = Path(runs_dir) if runs_dir is not None else DEFAULT_RUNS_DIR
    runs_root.mkdir(parents=True, exist_ok=True)

    resolved_run_id = run_id or _next_run_id(runs_root)
    run_dir = runs_root / resolved_run_id
    run_dir.mkdir(parents=False, exist_ok=False)

    timestamp_utc = datetime.now(timezone.utc).isoformat()

    metadata = {
        "run_id": resolved_run_id,
        "timestamp_utc": timestamp_utc,
        "user_query": result.user_query,
        "sources": list(result.sources),
        "limit_per_query": result.limit_per_query,
        "pipeline_version": PIPELINE_VERSION,
        "query_expander": result.expander_config,
        "relevance_filter": result.filter_config,
        "counts": {
            "expanded_queries": len(result.expanded_queries),
            "raw_candidates": result.raw_candidate_count,
            "deduplicated": result.deduped_count,
            "retained": result.filtered_count,
            "rejected": len(result.rejected_papers),
        },
        "source_errors": result.source_errors,
    }

    _dump_json(run_dir / "run_metadata.json", metadata)
    _dump_json(
        run_dir / "queries.json",
        {"user_query": result.user_query, "expanded_queries": result.expanded_queries},
    )
    _dump_json(run_dir / "raw_results.json", [p.model_dump() for p in result.raw_papers])
    _dump_json(run_dir / "deduplicated.json", [p.model_dump() for p in result.deduped_papers])
    _dump_json(run_dir / "retained.json", [p.model_dump() for p in result.retained_papers])
    _dump_json(run_dir / "rejected.json", [p.model_dump() for p in result.rejected_papers])

    with open(run_dir / "report.md", "w") as f:
        f.write(_render_report(result, resolved_run_id, timestamp_utc))

    return run_dir
