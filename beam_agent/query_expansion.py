"""Rule-based multi-query expansion: one scientific question -> several queries.

No LLM. Turns a single user query into a handful of complementary phrasings
using cheap, deterministic heuristics:

1. the original query, verbatim
2. a "core keywords" variant with filler words stripped (shorter, more
   precise -- helps on engines that weight exact phrase matches heavily)
3. a "broadened" variant with numeric size qualifiers removed (e.g. "200-aa",
   "24 kDa") -- these rarely appear verbatim in titles/abstracts, so keeping
   them in every query silently caps recall
4. up to a couple of synonym-substituted variants from a small domain table
   (coarse-grained/CG, protein folding/folding, ...)

The goal here is recall, not precision -- catching papers that one exact
phrasing would miss. Precision is RelevanceFilter's job, downstream.

`QueryExpander` is a Protocol specifically so an LLM-backed expander (e.g. one
that proposes topically distinct sub-questions instead of surface rewrites)
can be dropped in later without changing anything that calls it.
"""

from __future__ import annotations

import re
from typing import Protocol

from beam_agent.text_utils import core_keyword_string

_SIZE_QUALIFIER_RE = re.compile(
    r"\b\d+[\s-]?(?:aa|kda|residues?|amino[\s-]?acids?)\b", re.IGNORECASE
)

# Small, generic biophysics/CG-simulation synonym table. Deliberately not a
# list of specific model names (Martini, AWSEM, ...) -- that would bias
# recall toward whatever models we thought to hardcode, which is exactly the
# kind of guessing this module should avoid.
_SYNONYMS: dict[str, str] = {
    "coarse-grained": "CG",
    "coarse grained": "CG",
    "molecular dynamics": "MD simulation",
    "protein folding": "folding",
    "computational model": "simulation model",
}


class QueryExpander(Protocol):
    def expand(self, user_query: str) -> list[str]:
        """Return a list of search queries, original phrasing included."""
        ...


class RuleBasedQueryExpander:
    def __init__(self, max_queries: int = 5) -> None:
        self.max_queries = max_queries

    def expand(self, user_query: str) -> list[str]:
        query = user_query.strip()
        candidates = [query]

        core = core_keyword_string(query)
        if core:
            candidates.append(core)

        broadened = _SIZE_QUALIFIER_RE.sub("", query)
        broadened = " ".join(broadened.split())
        if broadened:
            candidates.append(broadened)

        candidates.extend(self._synonym_variants(query))

        return self._dedupe(candidates)[: self.max_queries]

    def _synonym_variants(self, query: str) -> list[str]:
        variants = []
        lowered = query.lower()
        for term, replacement in _SYNONYMS.items():
            if term in lowered:
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                variants.append(pattern.sub(replacement, query, count=1))
        return variants

    @staticmethod
    def _dedupe(queries: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for q in queries:
            key = " ".join(q.lower().split())
            if key and key not in seen:
                seen.add(key)
                out.append(q.strip())
        return out
