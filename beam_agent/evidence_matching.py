"""Stage 4b + 5: deterministic evidence organization, matching, and ranking.

Sits after single-pass evidence extraction (evidence_extraction.py):

    search -> dedupe -> cheap filter -> PaperEvidence extraction -> HERE -> full-text retrieval

Compares a paper's (cached, query-independent) `PaperEvidence` against the
query's `ScientificConceptProfile`, across all evidence groups extraction
produces -- method (`method_class`), system (`studied_systems`,
`family_fold`), and process (`task_process`) -- reusing the same stem-based
overlap machinery `RuleBasedRelevanceFilter` (Stage 3) uses, generalized to
structured evidence fields instead of raw title/abstract text. No LLM call
anywhere in this module; everything here is pure, unit-testable Python.

Per-dimension results are never collapsed into a single relevant/not_relevant
verdict -- that call belongs to the user, made explicit via `FilterCriteria`
(which dimensions are `required` vs merely `preferred` vs `optional`/
`irrelevant` for *this* analysis). BEAM organizes and ranks evidence; it does
not hand back an unexplainable final CG-model recommendation.

Recall-biased by construction, matching every earlier stage in this pipeline:
a dimension with `NO_EVIDENCE` (profile has nothing to compare, or the paper's
own text didn't say) never fails a `required` gate -- only an explicit
`MISMATCH` does. Missing information is not evidence of irrelevance.

Known gap: there is no dedicated `system_type`/`size`/`environment` dimension.
`SystemEvidence` v3 dropped those as separate structured fields (see
models.py's module docstring -- they were the source of concrete
misextractions: a topology count read as a residue count, a simulation
duration read as a residue count, a specific fold compressed into a coarse
bucket) in favor of one open-text `studied_systems` field. `match_studied_systems`
below is a plain stem-overlap comparison against that free text; it can
surface a topical match ("beta-helix" appearing in both the profile and a
paper's `studied_systems` entry) but cannot do numeric size comparison or
categorical system-type/environment filtering the way the old split fields
briefly could. Recovering that (e.g. a regex/NLP pass to pull a size back out
of `studied_systems` text, or a dedicated environment field grounded well
enough not to repeat the old conflation) is future work, not attempted here.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from beam_agent.concept_resolver import ConceptTerm, ScientificConceptProfile
from beam_agent.models import Paper, PaperEvidence
from beam_agent.text_utils import stemmed_keywords

__all__ = [
    "DimensionRequirement",
    "FilterCriteria",
    "DimensionMatch",
    "DimensionMatchResult",
    "EvidenceMatchResult",
    "match_evidence",
    "rank_papers",
]


class DimensionRequirement(str, Enum):
    REQUIRED = "required"
    PREFERRED = "preferred"
    OPTIONAL = "optional"
    IRRELEVANT = "irrelevant"


class FilterCriteria(BaseModel):
    """User-configurable per-dimension weighting for one analysis.

    Defaults favor recall: only `task_process` defaults to PREFERRED (it
    contributes to ranking); every other dimension defaults to OPTIONAL
    (recorded, never gates or ranks) until the user says otherwise. A future
    agent can translate a natural-language preference like "folding
    required, topology optional" or "folding AND beta-helix both required"
    directly into this object.
    """

    task_process: DimensionRequirement = DimensionRequirement.PREFERRED
    method_class: DimensionRequirement = DimensionRequirement.OPTIONAL
    studied_systems: DimensionRequirement = DimensionRequirement.OPTIONAL
    family_fold: DimensionRequirement = DimensionRequirement.OPTIONAL


class DimensionMatch(str, Enum):
    MATCHED = "matched"  # every comparable profile term is covered by the paper's own evidence
    PARTIAL = "partial"  # some, but not all, comparable profile terms are covered
    NO_EVIDENCE = "no_evidence"  # profile has nothing to compare, or the paper's text didn't say
    MISMATCH = "mismatch"  # both sides have something to say, and they don't overlap


class DimensionMatchResult(BaseModel):
    dimension: str
    match: DimensionMatch
    matched_terms: list[str] = Field(default_factory=list)
    note: Optional[str] = None


class EvidenceMatchResult(BaseModel):
    dimensions: dict[str, DimensionMatchResult] = Field(default_factory=dict)
    passes_required: bool = True
    failed_required_dimensions: list[str] = Field(default_factory=list)
    preferred_match_count: int = 0
    preferred_total: int = 0

    @property
    def score(self) -> float:
        """Simple, explainable ranking score: fraction of PREFERRED
        dimensions that came back MATCHED. 1.0 when no dimension was marked
        PREFERRED (nothing configured to rank by, so don't penalize)."""
        if self.preferred_total == 0:
            return 1.0
        return self.preferred_match_count / self.preferred_total


def _no_evidence(dimension: str, note: str) -> DimensionMatchResult:
    return DimensionMatchResult(dimension=dimension, match=DimensionMatch.NO_EVIDENCE, note=note)


def _term_stems(term: ConceptTerm) -> set[str]:
    stems = stemmed_keywords(term.term)
    for syn in term.synonyms:
        stems |= stemmed_keywords(syn)
    return stems


def _match_terms_against_text_values(
    dimension: str,
    profile_terms: list[ConceptTerm],
    evidence_values: list[str],
    *,
    no_profile_note: str,
    no_evidence_note: str,
    mismatch_note: str,
) -> DimensionMatchResult:
    """Compare a list of profile ConceptTerms against a list of free-text
    evidence values (already extracted by the LLM, never raw title/abstract
    text) via the same stem-based overlap RuleBasedRelevanceFilter uses.

    MATCHED/PARTIAL is judged per *profile term*, not per evidence value: if
    the profile names two structural terms and the paper's evidence only
    covers one, that's a genuine PARTIAL, not a MATCHED -- distinguishing
    "the paper fully covers what the user asked about" from "the paper
    covers part of it" is exactly the kind of information this stage exists
    to surface instead of collapsing into one score.
    """
    if not profile_terms:
        return _no_evidence(dimension, no_profile_note)
    if not evidence_values:
        return _no_evidence(dimension, no_evidence_note)

    value_stems = [(value, stemmed_keywords(value)) for value in evidence_values]
    covered_terms: list[str] = []
    matched_values: set[str] = set()
    for term in profile_terms:
        term_stems = _term_stems(term)
        hit = False
        for value, stems in value_stems:
            if term_stems & stems:
                hit = True
                matched_values.add(value)
        if hit:
            covered_terms.append(term.term)

    if not covered_terms:
        return DimensionMatchResult(dimension=dimension, match=DimensionMatch.MISMATCH, note=mismatch_note)
    if len(covered_terms) < len(profile_terms):
        uncovered = [t.term for t in profile_terms if t.term not in covered_terms]
        return DimensionMatchResult(
            dimension=dimension,
            match=DimensionMatch.PARTIAL,
            matched_terms=sorted(matched_values),
            note=f"matches {', '.join(covered_terms)} but not {', '.join(uncovered)}",
        )
    return DimensionMatchResult(dimension=dimension, match=DimensionMatch.MATCHED, matched_terms=sorted(matched_values))


def match_process(profile: ScientificConceptProfile, evidence: PaperEvidence) -> DimensionMatchResult:
    evidence_values = [f"{item.raw_text} {item.normalized_label.replace('_', ' ')}" for item in evidence.processes]
    return _match_terms_against_text_values(
        "task_process",
        profile.scientific_tasks,
        evidence_values,
        no_profile_note="profile has no scientific_tasks to match against",
        no_evidence_note="paper's own text states no process/capability",
        mismatch_note="paper describes process(es), but none overlap the target task(s)",
    )


def match_method(profile: ScientificConceptProfile, evidence: PaperEvidence) -> DimensionMatchResult:
    """Compares `evidence.method.model_class` (what modeling/method class the
    paper's own text names -- coarse-grained, all-atom, structure-based, ...)
    against `profile.methodology_terms` (what the researcher's query said
    they're looking for). This is the one dimension standing in for the old
    single-call assessor's `eligibility` gate, but as a symmetric, purely
    descriptive text-overlap comparison rather than an LLM-produced
    eligible/not_eligible verdict -- a paper mentioning "protein-protein
    interactions" with no coarse-grained (or any) model actually described
    (`model_class` empty) never spuriously passes a `method_class` gate the
    way it could on `task_process` alone; marking BOTH `task_process` and
    `method_class` as `required` closes that gap for a user who cares about
    it (see the paper_evidence_extraction benchmark's mass-spectrometry
    review case, where `processes` picked up review-abstract section
    headings like "Protein-Protein Interactions" despite no model being
    described at all).
    """
    return _match_terms_against_text_values(
        "method_class",
        profile.methodology_terms,
        evidence.method.model_class,
        no_profile_note="profile has no methodology_terms to match against",
        no_evidence_note="paper's own text doesn't name a modeling/method class",
        mismatch_note="paper names a modeling/method class, but it doesn't overlap the target methodology",
    )


def match_studied_systems(profile: ScientificConceptProfile, evidence: PaperEvidence) -> DimensionMatchResult:
    """Compares `profile.structural_terms` against `evidence.system.studied_systems`
    -- a plain stem-overlap topical match against SystemEvidence v3's merged
    free-text field. This can catch e.g. "beta-helix" appearing in both the
    profile and a paper's own studied_systems description, but it is NOT a
    numeric size comparison or a categorical system-type/environment filter
    -- see the module docstring's "known gap" note for what was dropped along
    with the old split fields and why.
    """
    return _match_terms_against_text_values(
        "studied_systems",
        profile.structural_terms,
        evidence.system.studied_systems,
        no_profile_note="profile has no structural_terms to match against",
        no_evidence_note="paper's own text doesn't describe a studied system",
        mismatch_note="paper describes studied system(s), but none overlap the target structural term(s)",
    )


def match_family_fold(profile: ScientificConceptProfile, evidence: PaperEvidence) -> DimensionMatchResult:
    profile_terms = profile.related_families + profile.representative_systems
    evidence_values = evidence.system.family_or_fold + evidence.system.protein_or_system_names
    return _match_terms_against_text_values(
        "family_fold",
        profile_terms,
        evidence_values,
        no_profile_note="profile has no related_families/representative_systems to match against",
        no_evidence_note="paper's own text doesn't name a fold/family or specific system",
        mismatch_note="paper names a fold/family/system, but it doesn't overlap the target family/system(s)",
    )


def match_evidence(
    profile: ScientificConceptProfile,
    evidence: Optional[PaperEvidence],
    criteria: Optional[FilterCriteria] = None,
) -> EvidenceMatchResult:
    """Pure function: (profile, evidence, criteria) -> per-dimension match
    results + a pass/fail + a ranking score. No LLM call, no I/O.

    `evidence=None` (never extracted, e.g. no abstract) is treated exactly
    like an extracted-but-empty PaperEvidence -- every dimension resolves to
    NO_EVIDENCE, which never fails a `required` gate. Missing information is
    never silently treated as a mismatch.
    """
    criteria = criteria or FilterCriteria()
    ev = evidence if evidence is not None else PaperEvidence()

    results: dict[str, DimensionMatchResult] = {
        "task_process": match_process(profile, ev),
        "method_class": match_method(profile, ev),
        "studied_systems": match_studied_systems(profile, ev),
        "family_fold": match_family_fold(profile, ev),
    }

    requirement_by_dimension = criteria.model_dump()
    failed_required = [
        dim
        for dim, result in results.items()
        if requirement_by_dimension.get(dim) == DimensionRequirement.REQUIRED.value
        and result.match == DimensionMatch.MISMATCH
    ]
    preferred_dims = [
        dim for dim, req in requirement_by_dimension.items() if req == DimensionRequirement.PREFERRED.value
    ]
    preferred_matched = sum(1 for dim in preferred_dims if results[dim].match == DimensionMatch.MATCHED)

    return EvidenceMatchResult(
        dimensions=results,
        passes_required=not failed_required,
        failed_required_dimensions=failed_required,
        preferred_match_count=preferred_matched,
        preferred_total=len(preferred_dims),
    )


def rank_papers(
    profile: ScientificConceptProfile,
    papers: list[Paper],
    criteria: Optional[FilterCriteria] = None,
) -> list[tuple[Paper, EvidenceMatchResult]]:
    """Score every paper against `criteria` and sort best-first.

    Returns ALL papers paired with their match result, including ones that
    fail a `required` dimension -- filtering those out is a caller's choice
    (e.g. the CLI), not something this function hides; nothing here silently
    drops a paper the way an LLM-produced not_relevant verdict used to.
    Sorted by (passes_required, score) descending, so passing papers always
    sort ahead of failing ones, ties broken by score.
    """
    criteria = criteria or FilterCriteria()
    scored = [(paper, match_evidence(profile, paper.evidence, criteria)) for paper in papers]
    scored.sort(key=lambda pair: (pair[1].passes_required, pair[1].score), reverse=True)
    return scored
