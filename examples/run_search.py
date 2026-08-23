"""Example: search for CG folding models on a beta-helix system and print a summary.

    python examples/run_search.py
"""

from beam_agent.search import search_and_dedupe

QUERY = "coarse-grained model protein folding beta-helix"


def main() -> None:
    papers = search_and_dedupe(QUERY, limit=20)

    print(f"Query: {QUERY!r}")
    print(f"{len(papers)} unique papers after dedup\n")

    for p in papers:
        oa = "OA" if p.is_open_access else "  "
        both = "+".join(p.sources)
        print(f"[{oa}] {p.year} | {both:<28} | {p.title[:90]}")
        if p.doi:
            print(f"       doi:{p.doi}")


if __name__ == "__main__":
    main()
