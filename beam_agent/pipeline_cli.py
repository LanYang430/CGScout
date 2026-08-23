"""CLI for the Phase 1.5 pipeline: expand -> search -> dedupe -> filter.

    python -m beam_agent.pipeline_cli \
        "200-aa beta-helix protein folding coarse-grained model" \
        --limit-per-query 15 --out results.json

Prints the generated queries, raw/deduped/filtered counts, and the top
retained papers so results can be scanned by eye before anything downstream
(Crossref validation, full-text retrieval, LLM extraction) touches them.

Add --snapshot to also write a full reproducibility snapshot (see
snapshot.py) to a timestamped run directory outside the git repo. Off by
default -- a plain `--out results.json` run writes nothing but that one file.
"""

from __future__ import annotations

import argparse
import json
import sys

from beam_agent.pipeline import run_pipeline
from beam_agent.snapshot import DEFAULT_RUNS_DIR, save_run_snapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Expand a query, search Semantic Scholar + OpenAlex, dedupe, rule-filter."
    )
    parser.add_argument("query", help="Natural-language scientific question")
    parser.add_argument(
        "--limit-per-query",
        type=int,
        default=15,
        help="Max results per source per expanded query, before dedupe (default 15)",
    )
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    parser.add_argument(
        "--show", type=int, default=10, help="How many top retained papers to print (default 10)"
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Save a full reproducibility snapshot of this run (run_metadata.json, "
        "queries.json, raw/deduplicated/retained/rejected JSON, report.md) to a "
        f"timestamped directory under --snapshot-dir. Off by default.",
    )
    parser.add_argument(
        "--snapshot-dir",
        default=None,
        help="Snapshot output root (default: outside the repo at "
        f"{DEFAULT_RUNS_DIR}, or $BEAM_AGENT_RUNS_DIR if set). Only used with --snapshot.",
    )
    args = parser.parse_args(argv)

    result = run_pipeline(args.query, limit_per_query=args.limit_per_query)

    for err in result.source_errors:
        print(f"warning: {err}", file=sys.stderr)

    print(f"User query: {result.user_query!r}")
    print(f"\nExpanded into {len(result.expanded_queries)} queries:")
    for q in result.expanded_queries:
        print(f"  - {q!r}")

    print(f"\nRaw candidates (all queries x all sources, pre-dedup): {result.raw_candidate_count}")
    print(f"After cross-source/cross-query dedup: {result.deduped_count}")
    print(
        f"After relevance filter: {result.filtered_count} retained, "
        f"{result.deduped_count - result.filtered_count} rejected"
    )

    n_show = min(args.show, len(result.retained_papers))
    print(f"\nTop {n_show} retained papers:")
    for p in result.retained_papers[:n_show]:
        tag = "+".join(p.sources)
        score = p.relevance_score if p.relevance_score is not None else 0.0
        print(f"\n[score {score:.1f}] ({tag}) {p.year} | {p.title}")
        print(f"    doi: {p.doi}")
        print(f"    matched: {', '.join(p.matched_keywords) or '(none)'}")

    if args.out:
        payload = {
            "user_query": result.user_query,
            "expanded_queries": result.expanded_queries,
            "raw_candidate_count": result.raw_candidate_count,
            "deduped_count": result.deduped_count,
            "filtered_count": result.filtered_count,
            "source_errors": result.source_errors,
            "retained_papers": [p.model_dump() for p in result.retained_papers],
            "rejected_papers": [p.model_dump() for p in result.rejected_papers],
        }
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nWrote full result (including rejected papers) to {args.out}")

    if args.snapshot:
        run_dir = save_run_snapshot(result, runs_dir=args.snapshot_dir)
        print(f"\nSnapshot saved to {run_dir}")
        for name in (
            "run_metadata.json",
            "queries.json",
            "raw_results.json",
            "deduplicated.json",
            "retained.json",
            "rejected.json",
            "report.md",
        ):
            print(f"  - {name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
