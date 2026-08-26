"""Stage 3: cheap rule-based relevance filtering, keyword-overlap scoring, no LLM.

Filters "obvious irrelevant" candidates -- papers with zero lexical connection
to the resolved concept profile -- while staying conservative: anything with
at least one matching keyword survives. At this stage recall matters more
than precision; a later LLM-based stage (evidence_extraction.py, Stage 4a) or
a human gets the final say on borderline cases, so this module's job is only
to remove the clear noise a multi-query search pass inevitably drags in.

Matches against a `ScientificConceptProfile` rather than the raw query string
-- the match vocabulary is `profile.raw_query`'s own keywords UNION every
structural/methodology/task/family/system term's canonical form and synonyms.
This is what lets a paper that only ever says "CG model" (never "coarse-
grained") still score a hit against a query that said "coarse-grained": the
methodology_terms cluster for "coarse-grained" carries "cg" as a synonym, so
it's in the match vocabulary even though it never appears in the literal
query text. Before this, the filter only ever matched literal query text and
had exactly this blind spot.

Score = title_weight * |profile_stems & title_stems|
      + abstract_weight * |profile_stems & abstract_stems|

Matching is done on `stem()`-reduced tokens rather than exact words, so
"protein"/"proteins" and "beta-helix"/"beta-helical" count as the same term
instead of silently missing each other (see text_utils.stem's docstring for
why a real suffix-stripping stemmer wouldn't catch that second case either).
`min_term_len` defaults to 2 (lower than text_utils' usual default of 3)
specifically so short abbreviations pulled in via concept-term synonyms (CG,
MD, ML, PPI, IDP) survive into the match vocabulary instead of being filtered
out as "too short to be meaningful" -- a reasonable default for free text, but
wrong for a deliberately-chosen abbreviation.

Title matches count for more than abstract matches because titles are short
and every word in them is a deliberate choice; abstracts are noisier.

`RelevanceFilter` is a Protocol specifically so an LLM-backed filter (e.g. one
that actually reads the abstract and judges topical fit) can be dropped in
later without changing the pipeline that calls it.
"""

from __future__ import annotations

from typing import Protocol

from beam_agent.concept_resolver import ScientificConceptProfile
from beam_agent.models import Paper
from beam_agent.text_utils import keywords, stem, stemmed_keywords


class RelevanceFilter(Protocol):
    def filter(
        self, profile: ScientificConceptProfile, papers: list[Paper]
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
        min_term_len: int = 2,
    ) -> None:
        self.min_score = min_score
        self.title_weight = title_weight
        self.abstract_weight = abstract_weight
        self.min_term_len = min_term_len

    def _profile_match_vocabulary(self, profile: ScientificConceptProfile) -> set[str]:
        words = set(keywords(profile.raw_query, min_len=self.min_term_len))
        for term in profile.all_terms():
            words |= keywords(term.term, min_len=self.min_term_len)
            for synonym in term.synonyms:
                words |= keywords(synonym, min_len=self.min_term_len)
        return words

    def score(self, profile: ScientificConceptProfile, paper: Paper) -> tuple[float, list[str]]:
        # Map each match-vocabulary word's stem back to a readable original
        # word, so a human scanning `matched_keywords` sees "beta-helix"
        # rather than the internal stem "beta-heli". First occurrence wins.
        stem_map: dict[str, str] = {}
        for word in sorted(self._profile_match_vocabulary(profile)):
            stem_map.setdefault(stem(word), word)

        title_stems = stemmed_keywords(paper.title, min_len=self.min_term_len)
        abstract_stems = stemmed_keywords(paper.abstract or "", min_len=self.min_term_len)

        matched_title = set(stem_map) & title_stems
        matched_abstract = set(stem_map) & abstract_stems

        score = self.title_weight * len(matched_title) + self.abstract_weight * len(
            matched_abstract
        )
        matched = sorted({stem_map[s] for s in matched_title | matched_abstract})
        return score, matched

    def filter(
        self, profile: ScientificConceptProfile, papers: list[Paper]
    ) -> tuple[list[Paper], list[Paper]]:
        retained: list[Paper] = []
        rejected: list[Paper] = []

        for paper in papers:
            score, matched = self.score(profile, paper)
            annotated = paper.model_copy(
                update={"relevance_score": score, "matched_keywords": matched}
            )
            (retained if score >= self.min_score else rejected).append(annotated)

        retained.sort(key=lambda p: p.relevance_score or 0.0, reverse=True)
        rejected.sort(key=lambda p: p.relevance_score or 0.0, reverse=True)
        return retained, rejected
