"""Rule-based relevance filtering: keyword-overlap scoring, no LLM.

Filters "obvious irrelevant" candidates -- papers with zero lexical connection
to the user's original question -- while staying conservative: anything with
at least one matching keyword survives. At this stage recall matters more
than precision; a later LLM-based filter (or a human) gets the final say on
borderline cases, so this module's job is only to remove the clear noise a
multi-query expansion pass inevitably drags in.

Score = title_weight * |query_stems & title_stems|
      + abstract_weight * |query_stems & abstract_stems|

Matching is done on `stem()`-reduced tokens rather than exact words, so
"protein"/"proteins" and "beta-helix"/"beta-helical" count as the same term
instead of silently missing each other (see text_utils.stem's docstring for
why a real suffix-stripping stemmer wouldn't catch that second case either).

Title matches count for more than abstract matches because titles are short
and every word in them is a deliberate choice; abstracts are noisier.

`RelevanceFilter` is a Protocol specifically so an LLM-backed filter (e.g. one
that actually reads the abstract and judges topical fit) can be dropped in
later without changing the pipeline that calls it.
"""

from __future__ import annotations

from typing import Protocol

from beam_agent.models import Paper
from beam_agent.text_utils import keywords, stem, stemmed_keywords


class RelevanceFilter(Protocol):
    def filter(
        self, user_query: str, papers: list[Paper]
    ) -> tuple[list[Paper], list[Paper]]:
        """Return (retained, rejected), each paper annotated with a relevance score.

        Both lists are sorted by score descending.
        """
        ...


class RuleBasedRelevanceFilter:
    def __init__(
        self,
        min_score: float = 1.0,
        title_weight: float = 2.0,
        abstract_weight: float = 1.0,
    ) -> None:
        self.min_score = min_score
        self.title_weight = title_weight
        self.abstract_weight = abstract_weight

    def score(self, user_query: str, paper: Paper) -> tuple[float, list[str]]:
        # Map each query term's stem back to the original (readable) word, so
        # a human scanning `matched_keywords` sees "beta-helix" rather than
        # the internal stem "beta-heli". First occurrence wins on collision.
        query_stem_map: dict[str, str] = {}
        for word in keywords(user_query):
            query_stem_map.setdefault(stem(word), word)

        title_stems = stemmed_keywords(paper.title)
        abstract_stems = stemmed_keywords(paper.abstract or "")

        matched_title = set(query_stem_map) & title_stems
        matched_abstract = set(query_stem_map) & abstract_stems

        score = self.title_weight * len(matched_title) + self.abstract_weight * len(
            matched_abstract
        )
        matched = sorted({query_stem_map[s] for s in matched_title | matched_abstract})
        return score, matched

    def filter(
        self, user_query: str, papers: list[Paper]
    ) -> tuple[list[Paper], list[Paper]]:
        retained: list[Paper] = []
        rejected: list[Paper] = []

        for paper in papers:
            score, matched = self.score(user_query, paper)
            annotated = paper.model_copy(
                update={"relevance_score": score, "matched_keywords": matched}
            )
            (retained if score >= self.min_score else rejected).append(annotated)

        retained.sort(key=lambda p: p.relevance_score or 0.0, reverse=True)
        rejected.sort(key=lambda p: p.relevance_score or 0.0, reverse=True)
        return retained, rejected
