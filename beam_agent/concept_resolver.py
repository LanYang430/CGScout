"""Stage 0: turn a user's scientific question into a structured concept profile.

Runs before query planning:

    user_query -> ConceptResolver.resolve() -> ScientificConceptProfile -> QueryPlanner

The point of this stage is to separate "what is this question actually about,
scientifically" from "what search strings should we run" -- the query planner
consumes the profile, not raw text, so query strategy and concept
understanding can evolve independently (see query_planner.py).

Two layers, deliberately different in how far they generalize:

Layer 0 -- lexical vocabulary matching (`BaselineConceptResolver`, no LLM):
- `structural_terms`, `methodology_terms`, `scientific_tasks` -- small
  synonym-cluster tables (generic vocabulary, not protein names -- "beta-helix"
  recognizes "beta-solenoid"/"parallel beta-helix" as surface synonyms, not as
  a claim about what else is related to it). Genuinely generalizes to any term
  actually in these tables.
- `size_constraints` -- numeric constraints like "200-aa", via regex.
- `other_constraints` -- leftover content words not confidently categorized
  into any of the above (kept, not discarded, so nothing silently vanishes).

Layer 1 -- scientific concept expansion (`expanded_concepts`, LLM-backed only,
see `LLMConceptResolver`): earlier versions of this module had a second,
separate mechanism here -- `related_families`/`representative_systems`
populated from a small hand-curated *association* table keyed off an exact
structural-term match (e.g. `"beta-helix" -> ["autotransporter passenger
domains", "phage tailspike proteins"]`). That table had exactly one key
("beta-helix") and could never produce anything for a system it wasn't
specifically pre-typed for -- not a mechanism, a single hardcoded worked
example. It has been removed entirely (see git history for the old
`_STRUCTURAL_TERM_FAMILY_ASSOCIATIONS`/`_STRUCTURAL_TERM_SYSTEM_ASSOCIATIONS`
tables and `_curated_associations()`).

`expanded_concepts` (a list of `ExpandedConcept`) replaces it: given the raw
query, an LLM is asked to propose scientifically related concepts worth
including in a literature search, and -- critically -- to explain, in its own
words, what kind of relationship each one has to the described system
(`relationship`) and why it's proposing it (`basis`). There is no fixed
taxonomy the model is asked to fill in; it decides how many concepts to
propose and what kinds of relationships are worth naming. Every concept is
tagged `source="llm-proposed"` -- deliberately not scored with a numeric
confidence (an LLM's self-reported confidence on its own scientific claims is
not a calibrated signal worth trusting as a filter or a rank). If a concept's
reliability needs to be judged later, that should come from something outside
the LLM's own opinion of itself -- e.g. corroboration against papers actually
retrieved using it, or human review -- not a number it made up about itself.

`ConceptResolver` is a Protocol so `LLMConceptResolver` can be dropped in (or
swapped for a different backend) without changing anything that calls it;
`BaselineConceptResolver` (Layer 0 only, no LLM, no `expanded_concepts`)
remains available as the no-LLM fallback/baseline.
"""

from __future__ import annotations

import json
import re
from typing import Optional, Protocol

import requests
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


class ConceptTerm(BaseModel):
    term: str
    synonyms: list[str] = Field(default_factory=list)
    confidence: float = 1.0
    source: str = "baseline"


class SizeConstraint(BaseModel):
    value: int
    unit: str
    raw_text: str


class ExpandedConcept(BaseModel):
    """One LLM-proposed, scientifically related concept worth including in a
    literature search -- see the module docstring for the full design
    rationale (this replaces the old hardcoded association table). No
    numeric confidence by design: an LLM's self-reported score on its own
    scientific claim is not a calibrated signal. If reliability needs
    judging later, it should come from outside the LLM's own opinion of
    itself -- e.g. corroborating it against papers actually retrieved using
    it, or human review.
    """

    term: str
    relationship: str = Field(
        description="What kind of relationship this concept has to the described system, in the model's own "
        "words -- not a fixed category. E.g. a structural/fold synonym, a broader or narrower fold family, a "
        "biological system class known to exhibit the described feature, a specific well-characterized example, "
        "or anything else with a genuine scientific basis."
    )
    basis: str = Field(
        description="Why this concept is being proposed -- a real structural/biological reason, not a "
        "restatement of the term."
    )
    source: str = "llm-proposed"


class ScientificConceptProfile(BaseModel):
    raw_query: str
    structural_terms: list[ConceptTerm] = Field(default_factory=list)
    related_families: list[ConceptTerm] = Field(default_factory=list)
    representative_systems: list[ConceptTerm] = Field(default_factory=list)
    scientific_tasks: list[ConceptTerm] = Field(default_factory=list)
    methodology_terms: list[ConceptTerm] = Field(default_factory=list)
    size_constraints: list[SizeConstraint] = Field(default_factory=list)
    other_constraints: list[str] = Field(default_factory=list)
    expanded_concepts: list[ExpandedConcept] = Field(
        default_factory=list,
        description="LLM-proposed scientifically related concepts (LLMConceptResolver only; always empty from "
        "BaselineConceptResolver). See ExpandedConcept -- replaces the old related_families/representative_systems "
        "hardcoded-association mechanism. related_families/representative_systems remain on this schema for "
        "backward compatibility with existing consumers (query_planner.py, evidence_matching.py) but are no "
        "longer populated by any resolver; downstream consumption of expanded_concepts is a separate, later step.",
    )
    resolver: str = "baseline"

    def all_terms(self) -> list[ConceptTerm]:
        """Every ConceptTerm across all categories, flattened. Does NOT
        include expanded_concepts -- those are ExpandedConcept, a different
        shape (term/relationship/basis, no synonyms/confidence), and are not
        yet consumed by anything that calls this method."""
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

        matched = structural_terms + methodology_terms + scientific_tasks
        other_constraints = self._residual_terms(query, matched, size_constraints)

        return ScientificConceptProfile(
            raw_query=query,
            structural_terms=structural_terms,
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


# ---------------------------------------------------------------------------
# LLM-backed concept expansion
# ---------------------------------------------------------------------------

_EXPANSION_SYSTEM_PROMPT = """\
You are a computational biophysicist helping plan a literature search for coarse-grained (CG) \
molecular simulation models relevant to a described system. Given the system description below, \
propose scientifically related concepts that would be worth including in a literature search, \
beyond directly searching for the system as described.

For each concept you propose, decide for yourself what kind of relationship it has to the described \
system, and explain it in your own words in the `relationship` field -- there is no fixed list of \
relationship types you must use. Examples of the KINDS of relationship you might find (not an \
exhaustive or required list, and you do not need to cover every kind): a structurally equivalent or \
closely related term/synonym, a broader or more specific structural/fold family, a biological system \
class known to exhibit this feature, a specific well-characterized example system, or something else \
entirely if you have a scientifically grounded reason. It is fine to propose concepts that highlight a \
meaningful DIFFERENCE or distinction worth being aware of (e.g. a related-but-structurally-distinct \
subtype), not only concepts that are straightforwardly similar.

You do not need to propose a fixed number of concepts. Propose only what you have a genuine, specific \
scientific basis for -- quality over coverage.

For every concept, `basis` must state WHY you are proposing it -- a real structural/biological reason, \
not a restatement of the term. If you are not confident of a real, correct, specific example (e.g. a \
named protein/system), do NOT invent one -- leave it out rather than guessing a plausible-sounding name \
that might not be real or might not actually be correct.

Do not assign a confidence or reliability score to any concept -- that is not part of your job here.

Respond with a single JSON object matching the given schema, and nothing else."""


class _ExpandedConceptCandidate(BaseModel):
    """LLM-facing shape for one expansion candidate -- deliberately missing
    `source` (stamped in code afterward; not the model's to decide) and any
    confidence field (see ExpandedConcept's docstring for why)."""

    term: str
    relationship: str
    basis: str


class _ExpansionResponse(BaseModel):
    expanded_concepts: list[_ExpandedConceptCandidate] = Field(default_factory=list)


def _expansion_output_schema() -> dict:
    schema = _ExpansionResponse.model_json_schema()
    # Drop class-level docstrings from the schema sent to the model -- same
    # reasoning as evidence_extraction.py's _llm_output_schema(): they
    # describe this codebase's internal design, not prompting instructions
    # the model needs; per-field descriptions (kept) do that job already.
    schema.pop("description", None)
    for defn in schema.get("$defs", {}).values():
        defn.pop("description", None)
    return schema


def _post_ollama_chat(
    *,
    session: requests.Session,
    host: str,
    model: str,
    timeout: float,
    system_prompt: str,
    user_message: str,
    schema: dict,
    think: bool = False,
) -> dict:
    """One Ollama /api/chat request with JSON-schema-constrained output. Same
    shape as evidence_extraction.py's helper of the same name -- duplicated
    rather than imported across modules to keep each module's LLM-calling
    code independent of the other's internals; if a third module needs this
    it's worth factoring out into a shared client module then, not before.
    """
    response = session.post(
        f"{host}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "format": schema,
            "stream": False,
            "think": think,
            "options": {"temperature": 0},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    content = response.json()["message"]["content"]
    return json.loads(content)


class LLMConceptResolver:
    """Stage 0 resolver backed by a local model served through Ollama.

    Layers on top of `BaselineConceptResolver` rather than replacing it:
    Layer 0 (structural/methodology/task synonym-cluster matching, size
    constraints, other_constraints) runs unchanged and deterministically;
    this class adds `expanded_concepts` via one LLM call against the raw
    query. `related_families`/`representative_systems` are never populated
    by this resolver -- see ScientificConceptProfile's docstring for why
    those fields stay on the schema but unused going forward.

    Default model `qwen3:8b`, `think=False` by default -- same choices as
    `OllamaPaperEvidenceExtractor`, for the same reasons (structured-
    extraction quality at this size tier; thinking-mode preamble buys
    nothing on a schema-constrained task like this one).
    """

    def __init__(
        self,
        model: str = "qwen3:8b",
        host: str = "http://localhost:11434",
        timeout: float = 120.0,
        *,
        think: bool = False,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.think = think
        self._session = session or requests.Session()
        self._deterministic = BaselineConceptResolver()

    @property
    def name(self) -> str:
        return f"LLMConceptResolver:{self.model}"

    def resolve(self, user_query: str) -> ScientificConceptProfile:
        profile = self._deterministic.resolve(user_query)
        expanded = self._expand_concepts(profile.raw_query)
        return profile.model_copy(update={"expanded_concepts": expanded, "resolver": self.name})

    def _expand_concepts(self, query: str) -> list[ExpandedConcept]:
        try:
            raw = _post_ollama_chat(
                session=self._session,
                host=self.host,
                model=self.model,
                timeout=self.timeout,
                system_prompt=_EXPANSION_SYSTEM_PROMPT,
                user_message=f"System description: {query}",
                schema=_expansion_output_schema(),
                think=self.think,
            )
            response = _ExpansionResponse.model_validate(raw)
        except Exception:  # noqa: BLE001 -- any failure degrades to no expansion, never crashes resolve()
            return []
        return [
            ExpandedConcept(term=c.term, relationship=c.relationship, basis=c.basis, source="llm-proposed")
            for c in response.expanded_concepts
        ]
