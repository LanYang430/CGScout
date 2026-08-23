"""Command-line entry point: run a search and write structured JSON to disk.

    python -m beam_literature.cli "beta-helix protein folding coarse-grained model" \
        --limit 20 --out results.json
"""

from __future__ import annotations

import argparse
import json
import sys

from beam_literature.search import search_and_dedupe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Search Semantic Scholar + OpenAlex, dedupe, write JSON."
    )
    parser.add_argument("query", help="Search query, e.g. a system + goal description")
    parser.add_argument(
        "--limit", type=int, default=20, help="Max results per source before dedupe (default 20)"
    )
    parser.add_argument(
        "--out", default="results.json", help="Output JSON file path (default results.json)"
    )
    args = parser.parse_args(argv)

    papers = search_and_dedupe(args.query, limit=args.limit)

    errors = getattr(search_and_dedupe, "last_errors", [])
    for err in errors:
        print(f"warning: {err}", file=sys.stderr)

    output = {
        "query": args.query,
        "n_results": len(papers),
        "source_errors": errors,
        "papers": [p.model_dump() for p in papers],
    }

    with open(args.out, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(papers)} deduplicated papers to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
