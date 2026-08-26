"""CLI for the full pipeline:

    resolve -> plan -> search -> dedupe -> cheap filter -> evidence extraction -> ranking

    python -m beam_agent.pipeline_cli \
        "200-aa beta-helix protein folding coarse-grained model" \
        --limit-per-query 15 --out results.json

Prints the resolved concept profile, the planned queries (by category),
raw/deduped/filtered/extracted counts, and the top-ranked retained papers so
results can be scanned by eye before anything downstream (full-text
retrieval) touches them.

Ranking (Stage 4b/5, evidence_matching.py) always runs against the default
`FilterCriteria` for display purposes -- it's cheap, pure Python, and rerun
freely; a future flag can expose per-dimension required/preferred/optional
overrides without touching retrieval or the extractor.

Add --snapshot to also write a full reproducibility snapshot (see
snapshot.py) to a timestamped run directory outside the git repo. Off by
default -- a plain `--out results.json` run writes nothing but that one file.
"""

from __future__ import annotations

import argparse
import json
import sys

from beam_agent.evidence_extraction import BaselineEvidenceExtractor, OllamaPaperEvidenceExtractor
from beam_agent.evidence_matching import rank_papers
from beam_agent.pipeline import run_pipeline
from beam_agent.snapshot import DEFAULT_RUNS_DIR, save_run_snapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Resolve concepts, plan queries, search, dedupe, rule-filter, extract evidence, rank."
    )
    parser.add_argument("query", help="Natural-language scientific question")
    parser.add_argument(
        "--limit-per-query",
        type=int,
        default=15,
        help="Max results per source per planned query, before dedupe (default 15)",
    )
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    parser.add_argument(
        "--show", type=int, default=10, help="How many top-ranked retained papers to print (default 10)"
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Save a full reproducibility snapshot of this run (run_metadata.json, "
        "concept_profile.json, queries.json, raw/deduplicated/retained/rejected JSON, "
        "paper_evidence.json, report.md) to a timestamped directory under "
        "--snapshot-dir. Off by default.",
    )
    parser.add_argument(
        "--snapshot-dir",
        default=None,
        help="Snapshot output root (default: outside the repo at "
        f"{DEFAULT_RUNS_DIR}, or $BEAM_AGENT_RUNS_DIR if set). Only used with --snapshot.",
    )
    parser.add_argument(
        "--extractor",
        choices=["baseline", "ollama"],
        default="baseline",
        help="Stage 4a evidence extractor backend. 'baseline' (default) makes no LLM calls and "
        "every abstract-bearing paper comes back with an empty PaperEvidence. 'ollama' calls a "
        "local model through Ollama for real single-pass structured extraction.",
    )
    parser.add_argument(
        "--extractor-model",
        default="qwen3:8b",
        help="Model name for --extractor ollama (default: qwen3:8b).",
    )
    parser.add_argument(
        "--extractor-host",
        default="http://localhost:11434",
        help="Ollama host for --extractor ollama (default: http://localhost:11434).",
    )
    args = parser.parse_args(argv)

    if args.extractor == "ollama":
        extractor = OllamaPaperEvidenceExtractor(model=args.extractor_model, host=args.extractor_host)
    else:
        extractor = BaselineEvidenceExtractor()

    result = run_pipeline(args.query, limit_per_query=args.limit_per_query, evidence_extractor=extractor)

    for err in result.source_errors:
        print(f"warning: {err}", file=sys.stderr)

    profile = result.concept_profile
    print(f"User query: {result.user_query!r}")
    print("\nConcept profile:")
    for label, terms in (
        ("structural_terms", profile.structural_terms),
        ("related_families", profile.related_families),
        ("representative_systems", profile.representative_systems),
        ("scientific_tasks", profile.scientific_tasks),
        ("methodology_terms", profile.methodology_terms),
    ):
        rendered = ", ".join(
            f"{t.term} ({'/'.join(t.synonyms)})" if t.synonyms else t.term for t in terms
        ) or "(none)"
        print(f"  {label}: {rendered}")
    print(
        "  size_constraints: "
        + (", ".join(c.raw_text for c in profile.size_constraints) or "(none)")
    )
    print("  other_constraints: " + (", ".join(profile.other_constraints) or "(none)"))

    print(f"\nPlanned {len(result.query_plan.queries)} queries:")
    for pq in result.query_plan.queries:
        print(f"  [{pq.category}] {pq.query!r}")

    print(f"\nRaw candidates (all queries x all sources, pre-dedup): {result.raw_candidate_count}")
    print(f"After cross-source/cross-query dedup: {result.deduped_count}")
    print(
        f"After Stage 3 cheap filter: {result.filtered_count} retained, "
        f"{len(result.rejected_papers)} rejected"
    )

    summary = result.evidence_summary()
    print(
        "After Stage 4a evidence extraction: "
        f"{summary['extracted']} extracted ({summary['no_abstract']} skipped, no abstract), "
        f"{summary['with_process_evidence']} with process evidence, "
        f"{summary['with_method_evidence']} with method evidence, "
        f"{summary['with_system_evidence']} with system evidence"
    )

    ranked = rank_papers(profile, result.evidence_papers)
    n_show = min(args.show, len(ranked))
    print(f"\nTop {n_show} ranked papers (Stage 4b/5, default FilterCriteria):")
    for paper, match in ranked[:n_show]:
        tag = "+".join(paper.sources)
        score = paper.relevance_score if paper.relevance_score is not None else 0.0
        gate = "OK" if match.passes_required else f"FAILS REQUIRED: {', '.join(match.failed_required_dimensions)}"
        print(f"\n[stage3 score {score:.1f}] [rank score {match.score:.2f}] ({tag}) {paper.year} | {paper.title}")
        print(f"    doi: {paper.doi}")
        print(f"    gate: {gate}")
        for dim, dim_result in match.dimensions.items():
            print(f"    {dim}: {dim_result.match.value}" + (f" -- {dim_result.note}" if dim_result.note else ""))

    if args.out:
        payload = {
            "user_query": result.user_query,
            "concept_profile": profile.model_dump(),
            "query_plan": result.query_plan.model_dump(),
            "raw_candidate_count": result.raw_candidate_count,
            "deduped_count": result.deduped_count,
            "filtered_count": result.filtered_count,
            "source_errors": result.source_errors,
            "retained_papers": [p.model_dump() for p in result.retained_papers],
            "rejected_papers": [p.model_dump() for p in result.rejected_papers],
            "evidence_papers": [p.model_dump() for p in result.evidence_papers],
        }
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nWrote full result (including rejected papers) to {args.out}")

    if args.snapshot:
        run_dir = save_run_snapshot(result, runs_dir=args.snapshot_dir)
        print(f"\nSnapshot saved to {run_dir}")
        for name in (
            "run_metadata.json",
            "concept_profile.json",
            "queries.json",
            "raw_results.json",
            "deduplicated.json",
            "retained.json",
            "rejected.json",
            "paper_evidence.json",
            "report.md",
        ):
            print(f"  - {name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
