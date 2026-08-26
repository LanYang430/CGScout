"""Stage 0: turn a user's scientific question into a structured concept profile.

Runs before query planning:

    user_query -> ConceptResolver.resolve() -> ScientificConceptProfile -> QueryPlanner

The point of this stage is to separate "what is this question actually about,
scientifically" from "what search strings should we run" -- the query planner
consumes the profile, not raw text, so query strategy and concept
understanding can evolve independently (see query_planner.py).

Categories captured, each a list of `ConceptTerm` (a canonical term plus
synonyms) except `size_constraints`:

- `structural_terms`     the protein/system's structural class, e.g. beta-helix
- `related_families`     broader biological families it belongs to, e.g.
                          autotransporter passenger domains
- `representative_systems`  specific proteins/systems, when identifiable
- `scientific_tasks`     the actual scientific goal, e.g. protein folding,
                          folding kinetics
- `methodology_terms`    the modeling/computational approach, e.g.
                          coarse-grained, molecular dynamics -- not one of the
                          four categories named in the original spec, but a
                          natural fifth bucket: it's neither a structural
                          property of the system nor the scientific task, and
                          the query planner needs it separately from both
- `size_constraints`     numeric constraints like "200-aa"
- `other_constraints`    leftover content words not confidently categorized
                          into any of the above (kept, not discarded, so
                          nothing silently vanishes)

`BaselineConceptResolver` is deliberately limited: it recognizes structural/
methodology/task terms via small synonym-cluster tables (same pattern as the
old query_expansion.py's synonym table -- generic vocabulary, not protein
names) and extracts size constraints via regex.

`related_families` and `representative_systems` come from a second, separate
mechanism: a small hand-curated *association* table keyed off already-matched
structural terms (e.g. `"beta-helix" -> ["autotransporter passenger domains",
"phage tailspike proteins"]`). This is real domain knowledge, not lexical
pattern-matching, so it's held to a higher bar of honesty about what it is:

- It is intentionally tiny (currently one structural term with associations)
  and exists to prove the architecture carries this kind of knowledge
  end-to-end into query planning -- it is NOT an attempt at a general protein
  or biological-family database, and adding entries for every structural
  motif this resolver recognizes would be exactly the kind of large hardcoded
  protein list this module should avoid.
- Every association-derived `ConceptTerm` is tagged `source="baseline-curated"`
  (distinct from the plain `"baseline"` used for lexical matches) and given
  `confidence < 1.0`, because it's an inference from the structural term, not
  something the user's own text said. A future LLM-backed resolver, or an
  external ontology/database check, should be expected to overwrite or
  corroborate these -- that's what the distinct provenance tag is for.

`confidence` and `source` on every `ConceptTerm` are the hook for later
external verification (e.g. an LLM-proposed or baseline-curated family
checked against a real ontology/database and re-tagged
`source="external:<db>"`) without needing a separate verifier protocol yet --
the schema already has room for it.

`ConceptResolver` is a Protocol specifically so an LLM-backed resolver can be
dropped in later without changing anything that calls it.
"""

from __future__ import annotations

import re
from typing import Protocol

from pydantic import BaseModel, Field

from beam_agent.text_utils import STOPWORDS, keywords, stem, tokenize

# Small, generic synonym-cluster tables -- vocabulary, not protein names.
# Each entry is (canonical_term, [synonyms]). A cluster "matches" a query if
# the canonical term OR any synonym appears in it as a substring.

_STRUCTURAL_TERM_CLUSTERS: list[tuple[str, list[str]]] = [
    ("beta-helix", ["beta-solenoid", "parallel beta-helix", "beta helix"]),
    ("alpha-helix", ["alpha helix"]),
    ("beta-sheet", ["beta sheet"]),
    ("beta-barrel", ["beta barrel"]),
    ("coiled-coil", ["coiled coil"]),
    ("intrinsically disordered", ["disordered protein", "idp"]),
]

_METHODOLOGY_TERM_CLUSTERS: list[tuple[str, list[str]]] = [
    ("coarse-grained", ["cg", "coarse grained"]),
    ("all-atom", ["all atom", "atomistic"]),
    ("molecular dynamics", ["md simulation", "md"]),
    ("machine learning", ["ml", "deep learning", "neural network"]),
]

_TASK_TERM_CLUSTERS: list[tuple[str, list[str]]] = [
    ("protein folding", ["folding"]),
    ("folding kinetics", ["folding kinetics", "folding rate", "kinetics of folding"]),
    ("protein structure prediction", ["structure prediction"]),
    ("protein-protein interaction", ["protein protein interaction", "ppi"]),
    ("aggregation", ["protein aggregation", "amyloid aggregation"]),
]

_SIZE_CONSTRAINT_RE = re.compile(
    r"\b(\d+)[\s-]?(aa|kda|residues?|amino[\s-]?acids?)\b", re.IGNORECASE
)

# Small, hand-curated associations from a *matched structural term* to
# biological families / representative systems known to exhibit it. Each
# value is a list of (canonical_term, [synonyms]) -- same shape as the
# lexical clusters above, but this table encodes domain knowledge, not
# vocabulary, hence the separate confidence/provenance treatment in
# _curated_associations() below. See the module docstring for why this stays
# intentionally tiny rather than growing into a protein database.
_STRUCTURAL_TERM_FAMILY_ASSOCIATIONS: dict[str, list[tuple[str, list[str]]]] = {
    "beta-helix": [
        ("autotransporter passenger domains", ["autotransporter passenger domain"]),
        ("phage tailspike proteins", ["bacteriophage tailspike proteins", "tailspike proteins"]),
    ],
}

_STRUCTURAL_TERM_SYSTEM_ASSOCIATIONS: dict[str, list[tuple[str, list[str]]]] = {
    "beta-helix": [
        ("pertactin", []),
        ("P22 tailspike", ["P22 tailspike protein", "bacteriophage P22 tailspike"]),
    ],
}

# Confidence assigned to every baseline-curated association: lower than the
# 1.0 used for direct lexical matches, since this is an inference ("beta-helix
# structures include autotransporter passenger domains") rather than
# something the user's own query text said.
_CURATED_ASSOCIATION_CONFIDENCE = 0.6


class ConceptTerm(BaseModel):
    term: str
    synonyms: list[str] = Field(default_factory=list)
    confidence: float = 1.0
    source: str = "baseline"


class SizeConstraint(BaseModel):
    value: int
    unit: str
    raw_text: str


class ScientificConceptProfile(BaseModel):
    raw_query: str
    structural_terms: list[ConceptTerm] = Field(default_factory=list)
    related_families: list[ConceptTerm] = Field(default_factory=list)
    representative_systems: list[ConceptTerm] = Field(default_factory=list)
    scientific_tasks: list[ConceptTerm] = Field(default_factory=list)
    methodology_terms: list[ConceptTerm] = Field(default_factory=list)
    size_constraints: list[SizeConstraint] = Field(default_factory=list)
    other_constraints: list[str] = Field(default_factory=list)
    resolver: str = "baseline"

    def all_terms(self) -> list[ConceptTerm]:
        """Every ConceptTerm across all categories, flattened."""
        return (
            self.structural_terms
            + self.related_families
            + self.representative_systems
            + self.scientific_tasks
            + self.methodology_terms
        )


class ConceptResolver(Protocol):
    def resolve(self, user_query: str) -> ScientificConceptProfile:
        ...


class BaselineConceptResolver:
    def resolve(self, user_query: str) -> ScientificConceptProfile:
        query = user_query.strip()

        structural_terms = self._match_clusters(query, _STRUCTURAL_TERM_CLUSTERS)
        methodology_terms = self._match_clusters(query, _METHODOLOGY_TERM_CLUSTERS)
        scientific_tasks = self._match_clusters(query, _TASK_TERM_CLUSTERS)
        size_constraints = self._extract_size_constraints(query)

        related_families = self._curated_associations(
            structural_terms, _STRUCTURAL_TERM_FAMILY_ASSOCIATIONS
        )
        representative_systems = self._curated_associations(
            structural_terms, _STRUCTURAL_TERM_SYSTEM_ASSOCIATIONS
        )

        matched = structural_terms + methodology_terms + scientific_tasks
        other_constraints = self._residual_terms(query, matched, size_constraints)

        return ScientificConceptProfile(
            raw_query=query,
            structural_terms=structural_terms,
            related_families=related_families,
            representative_systems=representative_systems,
            scientific_tasks=scientific_tasks,
            methodology_terms=methodology_terms,
            size_constraints=size_constraints,
            other_constraints=other_constraints,
            resolver="BaselineConceptResolver",
        )

    @staticmethod
    def _match_clusters(
        query: str, clusters: list[tuple[str, list[str]]]
    ) -> list[ConceptTerm]:
        lowered = query.lower()
        matches = []
        for canonical, synonyms in clusters:
            if any(form.lower() in lowered for form in [canonical, *synonyms]):
                matches.append(
                    ConceptTerm(
                        term=canonical, synonyms=synonyms, confidence=1.0, source="baseline"
                    )
                )
        return matches

    @staticmethod
    def _curated_associations(
        structural_terms: list[ConceptTerm],
        table: dict[str, list[tuple[str, list[str]]]],
    ) -> list[ConceptTerm]:
        """Look up each matched structural term in a small curated association
        table and return any hits as ConceptTerms, tagged with the distinct
        "baseline-curated" provenance (see module docstring)."""
        seen: set[str] = set()
        out: list[ConceptTerm] = []
        for structural_term in structural_terms:
            for canonical, synonyms in table.get(structural_term.term, []):
                if canonical in seen:
                    continue
                seen.add(canonical)
                out.append(
                    ConceptTerm(
                        term=canonical,
                        synonyms=synonyms,
                        confidence=_CURATED_ASSOCIATION_CONFIDENCE,
                        source="baseline-curated",
                    )
                )
        return out

    @staticmethod
    def _extract_size_constraints(query: str) -> list[SizeConstraint]:
        constraints = []
        for m in _SIZE_CONSTRAINT_RE.finditer(query):
            constraints.append(
                SizeConstraint(value=int(m.group(1)), unit=m.group(2).lower(), raw_text=m.group(0))
            )
        return constraints

    @staticmethod
    def _residual_terms(
        query: str, matched_terms: list[ConceptTerm], size_constraints: list[SizeConstraint]
    ) -> list[str]:
        accounted_words: set[str] = set()
        for term in matched_terms:
            accounted_words |= keywords(term.term, min_len=2)
            for syn in term.synonyms:
                accounted_words |= keywords(syn, min_len=2)
        for c in size_constraints:
            accounted_words |= keywords(c.raw_text, min_len=2)
        accounted_stems = {stem(w) for w in accounted_words}

        # Iterate tokenize() (ordered) rather than keywords() (a set, whose
        # iteration order isn't guaranteed stable across runs) so
        # other_constraints comes out in query order deterministically.
        seen: set[str] = set()
        residual: list[str] = []
        for word in tokenize(query):
            if word in STOPWORDS or len(word) < 2 or word in seen:
                continue
            if stem(word) not in accounted_stems:
                seen.add(word)
                residual.append(word)
        return residual
