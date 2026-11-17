"""Stage 3 (cheap relevance filter) handoff for the new RetrievalIntent /
query-role retrieval path.

Inspection of the existing Stage 3 (beam_agent/relevance_filter.py,
RuleBasedRelevanceFilter -- UNCHANGED by this file, not one line touched):
  - No LLM. Pure deterministic stem-based keyword overlap between a paper's
    title/abstract and a match vocabulary.
  - Signature is `filter(profile: ScientificConceptProfile, papers)` -- the
    SAME hard dependency on the legacy Stage-0 profile type already found
    for QueryPlanner. `_profile_match_vocabulary()` builds ONE global
    vocabulary from `profile.raw_query` + `profile.all_terms()` (every
    ConceptTerm's `.term` + `.synonyms`), applied uniformly to every paper
    regardless of which query found it -- it has no concept of
    found_by_queries/query_categories at all.
  - Keep rule: score = title_weight*|title matches| + abstract_weight*
    |abstract matches|, keep if score >= min_score (defaults 2.0/1.0/1.0)
    -- i.e. a SINGLE keyword hit anywhere already passes. Already an
    explicit high-recall design; nothing to fix there.
  - Called by beam_agent/pipeline.py's run_pipeline() (Stage 3 of that
    pipeline). NOT called anywhere in the new chat_with_beam.py /
    execute_query_plan() path before this pass.

Because the profile dependency is architecturally identical to
QueryPlanner's, the fix is the same shape as query_planner_handoff.py:
reuse the underlying, already-profile-agnostic primitives
(beam_agent.text_utils.stem/stemmed_keywords/keywords -- plain string
functions) directly, rather than constructing a legacy profile.

Role-aware scoring (unchanged from the first version of this file): each
paper is scored against the vocabulary of EXACTLY the query string(s) that
found it (`paper.found_by_queries`), not one global profile-wide
vocabulary. A direct_system_evidence paper's own originating query never
contained methodology terms in the first place, so it's never penalized
for lacking them; a methodology_evidence paper's query never demanded the
exact system name, so it's never penalized for that. One uniform,
interpretable rule (>=1 keyword hit) -- no per-role numeric threshold.

--- Methodology surface-form normalization (this pass's addition) ---

Demonstrated false negative: query "...coarse-grained..." (hyphenated,
one token, stems to the joined compound "coar-grai") never matched the
real, relevant paper "The power of coarse graining in biomolecular
simulations" (unhyphenated two words "coarse"+"graining", independently
stemmed to "coar"+"grai" -- two separate strings that can never equal one
joined "coar-grai" string).

This is NOT fixed by weakening stem() globally -- stem()'s hyphen-
preserving compound behavior is exactly what keeps "beta-helix" from
matching a bare "beta" (e.g. "Amyloid beta..." is correctly dropped
against a beta-helix query; removing that distinction would reopen a
false-positive the design deliberately avoids). The fix is scoped
narrowly to methodology wording, reusing EXISTING, already-generic,
already-tested vocabulary rather than inventing a new synonym table:
_METHODOLOGY_TERM_CLUSTERS (imported read-only from concept_resolver.py,
NOT MODIFIED) already lists "coarse-grained" separately from
_STRUCTURAL_TERM_CLUSTERS' "beta-helix" -- that existing separation IS
the generic signal for "this compound is methodology wording, eligible
for lenient surface-form matching" vs. "this compound is a structural
term, keep it strict."

_build_methodology_equivalence() walks every methodology cluster's
canonical + synonym forms once at import time and maps BOTH the
compound-joined stem (the literal-hyphenated-token case) AND each
individual word's own stem (the space-separated-or-independently-
tokenized case) to one canonical key. Every stem computed anywhere in
this module -- for the query vocabulary AND for a paper's title/abstract
-- is passed through this same small lookup before comparison; a stem
that isn't a known methodology variant (e.g. beta-helix's "beta-heli",
or "wnt", "prot") is returned unchanged, so this can only ever affect
methodology wording, never structural terms, by construction (it's a
static table lookup, not a broadened regex or fuzzy comparison).

No suffix rule was needed for "-ed" vs. "-ing" (grained vs. graining):
both already reduce to the same 4-character prefix "grai" under the
existing crude stemmer, once the two words are treated as independent
tokens rather than force-joined into one compound string. That's a
general property of this stemmer for any regularly-suffixed word whose
root exceeds 4 characters -- not something hard-coded for this specific
pair.

--- Review filter + experiment-vs-simulation classifier (this pass) ---

Two NEW, independently-validated stages, added on top of everything
above without touching its logic:

  1. review_filter.py -- a cheap, high-confidence-only check (OpenAlex
     `type`=='review' OR an explicit abstract self-declaration phrase).
     Runs FIRST, before any scoring: a flagged paper is dropped without
     ever spending an embedding call.

  2. sim_classifier.py -- a PCA(60)+LogisticRegression classifier over
     text-embedding-3-small embeddings, trained on ~424 hand-audited
     title/abstract examples, answering "did the authors themselves
     perform a simulation" directly and semantically -- not via literal
     marker substrings. Validated (external 427-row held-out test set)
     to catch real simulation papers the literal _STRUCTURAL_CG_MARKERS
     check below misses (e.g. an abstract describing Martini/CG methods
     without ever writing "coarse-grained"), while also being able to
     reject papers the marker check would have let through.

  Wiring, by design NOT a wholesale replacement of the route-aware gate
  below (see score_paper_against_its_provenance's `classifier_score`
  param): filter_papers() batch-scores every non-review-filtered
  candidate ONCE (one embedding call for up to 200 papers, not one per
  paper) and threads each paper's own score in. Inside the gate, a
  provided classifier_score REPLACES the literal-marker `has_cg_evidence`
  check (both for the structural/flat route AND, newly, for papers
  matched via NO route at all -- previously ungated on this axis) --
  strictly a "does this look like a real simulation" signal, with zero
  opinion on topical relevance (that's still the pre-existing keyword-
  overlap score, checked first and unchanged). `classifier_score=None`
  (the default) falls back to the OLD literal-marker check byte-for-byte
  -- this is what keeps every existing unit test in this file's sibling
  tests/test_relevance_filter_handoff.py passing unmodified, network-free
  -- production's filter_papers() always supplies a real score.
"""

from __future__ import annotations

import re
from typing import Callable, Optional


from cgscout.models import Paper  # noqa: E402

from cgscout.retrieval.types import _CG_MARKERS  # noqa: E402 -- read-only reuse, same markers Hop2Paper.mentions_cg already uses

from cgscout.screening.sim_classifier import SIM_CLASSIFIER_THRESHOLD, default_classify_fn  # noqa: E402

from pydantic import BaseModel, Field  # noqa: E402

# ===== merged from `text.py` (consolidation A5; moved verbatim, its docstring is this comment) ====================================================
# Shared lightweight text helpers: tokenization and stopword filtering.
#
# Deliberately dumb on purpose -- regex tokenization plus a small stopword list,
# no NLP libraries, no LLM. Good enough for query trimming and keyword-overlap
# relevance scoring; this module's keyword-overlap relevance
# scoring builds on it, so query trimming and scoring stay consistent about what counts as a "meaningful word."

STOPWORDS = frozenset(
    {
        "a", "an", "the", "of", "for", "with", "in", "on", "to", "and", "or",
        "using", "based", "study", "studies", "via", "from", "by", "into",
        "is", "are", "was", "were", "be", "been", "being", "that", "this",
        "these", "those", "as", "at", "can", "its", "it", "we", "our", "us",
        "new", "novel", "toward", "towards", "approach", "approaches",
        "analysis", "such", "than", "their", "which", "how", "what",
    }
)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-]+")


def tokenize(text: str) -> list[str]:
    """Lowercased word tokens (letters + internal hyphens only, digits dropped)."""
    return _WORD_RE.findall(text.lower())


def keywords(text: str, *, min_len: int = 3) -> set[str]:
    """Tokenize and drop stopwords / very short tokens."""
    return {t for t in tokenize(text) if t not in STOPWORDS and len(t) >= min_len}


def _stem_word(word: str, *, prefix_len: int = 4) -> str:
    """Truncate a single hyphen-free word to its first `prefix_len` characters."""
    return word if len(word) <= prefix_len else word[:prefix_len]


def stem(token: str, *, prefix_len: int = 4) -> str:
    """Crude, deliberately dumb stemmer for cross-text keyword matching.

    NOT a real stemmer -- no Porter/Snowball morphology rules, just a
    fixed-length prefix per hyphen-separated part. That's enough to unify
    common regular forms (protein/proteins, fold/folding/folded) and, more
    importantly, catches irregular derivations a suffix-stripper would miss
    entirely: "helix" and "helical" share only a 4-letter prefix, not a
    suffix relationship, so no amount of stripping "-ing"/"-ed"/"-s" off
    either one makes them equal.

    The cost is occasional false-positive collisions between unrelated words
    that happen to share a prefix (model/modern both stem to "mode"). That
    tradeoff is intentional: RelevanceFilter is designed to over-retain
    rather than over-reject (see its module docstring) -- a collision here
    just means one extra paper reaches manual/LLM review; a missed match
    means a paper silently vanishes before anyone sees it.
    """
    return "-".join(_stem_word(part, prefix_len=prefix_len) for part in token.split("-"))


def stemmed_keywords(text: str, *, min_len: int = 3) -> set[str]:
    """Like `keywords()`, but each token is reduced via `stem()` first.

    Use this (not `keywords()`) whenever comparing two *different* texts for
    topical overlap -- e.g. a user query against a paper's title/abstract.
    Keep using `keywords()` when working within one text (e.g. trimming a
    query string), where exact surface tokens are what you want to preserve.
    """
    return {stem(t) for t in keywords(text, min_len=min_len)}


# Methodology synonym clusters (canonical term, [synonyms]) -- vocabulary, not protein names; read by the Stage-3 filter below.
METHODOLOGY_TERM_CLUSTERS: list[tuple[str, list[str]]] = [
    ("coarse-grained", ["cg", "coarse grained"]),
    ("all-atom", ["all atom", "atomistic"]),
    ("molecular dynamics", ["md simulation", "md"]),
    ("machine learning", ["ml", "deep learning", "neural network"]),
]

# The Stage-3 code below refers to the table under its private name (it used to be imported under this alias from text.py).
_METHODOLOGY_TERM_CLUSTERS = METHODOLOGY_TERM_CLUSTERS
# ===========================================================================================================================

# ===== merged from `screening/review_filter.py` (moved verbatim; its docstring is this comment) ============================
# Stage 3 high-confidence review/perspective filter -- production version
# of the standalone `cheap_review_filter_v2.py` diagnostic (validated this
# session against a combined ~1076-paper labeled population: 20/444 training
# rows and 84/511 test rows flagged this narrow version alone).
#
# Deliberately narrow -- ONLY the two highest-confidence signal categories,
# per explicit instruction, everything else (venue-name regex, "recent
# advances in"/"we summarize"/"an overview of" phrasing, the eLife peer-
# review-artifact title pattern) left out as too prone to false positives:
#
#   1. OpenAlex's own `type` field says "review" (Paper.oa_type -- see
#      beam_agent/sources/openalex.py's `_SELECT`/`_to_paper()`). OpenAlex
#      normalizes publisher-level sub-types ("Review", "Mini Review",
#      "Advanced Review", ...) into this one coarse category; there is no
#      more granular field to check beyond it. None for any non-OpenAlex
#      source, or if OpenAlex omitted it -- never treated as a match.
#   2. The abstract explicitly self-declares with one of exactly these four
#      phrasings: "in this review", "this review", "here we review",
#      "we review".
#
# Runs BEFORE the classifier in filter.filter_papers() --
# a flagged paper is dropped without ever spending an embedding call.
# ===========================================================================================================================

REVIEW_PHRASE_RE = re.compile(
    r"\bin this review\b|\bthis review\b|\bhere,? we review\b|\bwe review\b",
    re.IGNORECASE,
)


def review_filter_detail(paper: Paper) -> dict:
    type_hit = paper.oa_type == "review"
    m = REVIEW_PHRASE_RE.search(paper.abstract or "")
    return {
        "flagged": type_hit or bool(m),
        "type_hit": type_hit,
        "matched_phrase": m.group(0) if m else "",
    }


# ===== end of the code merged from `screening/review_filter.py` ============================================================


# Same defaults as the existing (unmodified) RuleBasedRelevanceFilter --
# deliberately not re-tuned, to isolate "does provenance-scoped vocabulary
# work at all" from "is this specific threshold right."
TITLE_WEIGHT = 2.0
ABSTRACT_WEIGHT = 1.0
MIN_SCORE = 1.0
MIN_TERM_LEN = 2

# --- structural-route (Hop-2) tightening: "Hop-1 broad, Hop-2 strict" ---
#
# Hop-2 isn't "find broadly related literature" -- it's already asking, per
# discovered system, "does THIS concrete system have CG evidence". A paper
# that only hits on the methodology part of its own originating query
# (Martini 3 -- CG, but never mentions the discovered system) or only on the
# system part (a classic experimental paper about the system, no CG) is not
# what Hop-2 was looking for, even though the existing >=1-keyword-hit rule
# above would keep either. For structural-route papers, Stage 3 requires
# BOTH conditions in title/abstract, or drops -- explicitly abstract-
# grounded (v1 scope, not full-text): a paper whose CG evidence only
# exists in its full text is an accepted, known blind spot, not something
# this stage claims to see.
#
# CG-model-recommendation retrieval-tightening (later session): the flat
# routes (system_specific_cg, process, context) get an analogous but
# SINGLE-condition gate -- CG evidence required, system-name evidence NOT
# required -- see the `elif {"system_specific_cg", "process", "context"} &
# set(...)` branch below. The old direct_system route was deleted (replaced
# by system_specific_cg); modeling_requirement is disabled at the route-
# planning level (route_plan.py), so it never reaches Stage 3.
#
# CG marker vocabulary: _CG_MARKERS (imported above) is the SAME literal
# substring list types.py's own Hop2Paper.mentions_cg already
# uses for an identical "does this title+abstract blob mention CG" check --
# reused, not reinvented. Extended with "cg simulation(s)"/"cg model(s)"
# specifically named in the design decision (not covered by the
# hyphen/space "coarse-grain(ed)" forms _CG_MARKERS already has).
_STRUCTURAL_CG_MARKERS = _CG_MARKERS + (
    "cg simulation", "cg simulations", "cg model", "cg models",
    # Named CG/reduced-representation force fields (this session's fix --
    # diagnosed real false negative: a paper's abstract said only
    # "molecular dynamics simulations", never "coarse-grained", even
    # though its Methods used Martini 3 explicitly). Only ADDED alongside
    # the existing "coarse-grain*" forms, never replacing them -- a paper
    # naming one of these without ever saying "coarse-grained" is still
    # unambiguous CG-methodology evidence, the same way "Martini" already
    # was for the pre-existing (unbroadened) _CG_MARKERS. Deliberately
    # NOT included: bare "force field"/"force-field" -- that term is used
    # by all-atom force fields (AMBER, CHARMM, ...) just as much as CG
    # ones, so adding it here would break the very atomistic-vs-CG
    # distinction this gate exists to enforce. "hps" carries a real
    # acronym-collision risk (e.g. Hermansky-Pudlak Syndrome) -- accepted
    # because a collision here only ADDS a candidate to the pool, which
    # must still separately clear the general topical-relevance keyword
    # score to be retained; it can never on its own retain an otherwise
    # off-topic paper.
    "martini", "calvados", "aicg", "3spn", "hps",
)


def _build_methodology_equivalence() -> dict[str, str]:
    """stem -> canonical methodology-cluster name, for every compound-joined
    AND per-word stem of every _METHODOLOGY_TERM_CLUSTERS form. Computed
    once at import time from the existing, unmodified table."""
    mapping: dict[str, str] = {}
    for canonical, synonyms in _METHODOLOGY_TERM_CLUSTERS:
        for form in [canonical, *synonyms]:
            words = [w for w in re.split(r"[\s\-]+", form.strip()) if w]
            if len(words) >= 2:
                mapping.setdefault(stem("-".join(words)), canonical)  # compound-joined (literal hyphenated token)
            for word in words:
                if len(word) >= MIN_TERM_LEN:
                    mapping.setdefault(stem(word), canonical)  # per-word (space-separated / independent token)
    return mapping


_METHODOLOGY_EQUIVALENCE = _build_methodology_equivalence()


def _normalize_methodology_stem(raw_stem: str) -> str:
    """Pass-through for anything that isn't a known methodology-cluster
    variant -- this can only ever affect methodology wording (see module
    docstring)."""
    return _METHODOLOGY_EQUIVALENCE.get(raw_stem, raw_stem)


class Stage3Verdict(BaseModel):
    """One paper's Stage 3 decision, kept separate from the Paper object
    itself so the reasoning stays inspectable without touching Paper's own
    schema further. No numeric confidence beyond the same interpretable
    keyword-count score the existing filter already uses -- `reason` and
    `matched_role` are what make a drop/keep explicable at a glance."""

    title: str
    keep: bool
    score: float
    matched_keywords: list[str]
    matched_role: str  # paper.query_categories, joined -- "(no provenance)" if none
    reason: str
    # True exactly when this verdict is a reject caused SOLELY by missing
    # abstract-level CG evidence (every other required condition, e.g. the
    # structural gate's system-evidence half, already passed) -- i.e. a
    # candidate genuinely eligible for the full-text Methods fallback in
    # filter_papers(), not just any rejection. False for every other
    # reject reason (below MIN_SCORE, missing system evidence, etc.) and
    # for every keep.
    missing_cg_evidence_only: bool = False

    # --- review filter + classifier audit fields (this pass) ---
    abstract_recovered: bool = False  # True iff the abstract Stage 3 judged was fetched from Europe PMC (OpenAlex had none) -- only with filter_papers(recover_abstracts=True)
    review_flagged: bool = False  # True iff review_filter.py dropped this paper before any scoring
    review_reason: str = ""  # which signal fired ("type", "phrase: '...'", or both) -- "" when not flagged
    classifier_score: Optional[float] = Field(
        default=None,
        description="sim_classifier.py's P(authors performed a simulation), when computed. None means either "
        "the paper was review_flagged (never scored) or it failed the general keyword-overlap score before "
        "classification would even matter.",
    )


def _query_stem_map(query_texts: list[str]) -> dict[str, str]:
    """(possibly methodology-normalized) stem -> readable label, unioned
    across every query that found a paper. When a stem gets normalized to
    a methodology canonical name, that canonical name (already a readable
    string, e.g. "coarse-grained") is used as the label directly."""
    stem_map: dict[str, str] = {}
    for q in query_texts:
        for word in sorted(keywords(q, min_len=MIN_TERM_LEN)):
            raw = stem(word)
            normalized = _normalize_methodology_stem(raw)
            label = normalized if normalized != raw else word
            stem_map.setdefault(normalized, label)
    return stem_map


def _normalized_text_stems(text: str) -> set[str]:
    return {_normalize_methodology_stem(s) for s in stemmed_keywords(text, min_len=MIN_TERM_LEN)}


def _system_evidence_stems(system_name: str) -> set[str]:
    """Every significant stem of a discovered system's own name -- "明确出现
    这个 discovered system" is read as ALL of its stems present (not just
    one), so a multi-word name like "P22 tailspike" needs both "p22" and
    "tailspik" in the text, not just one or the other."""
    return {_normalize_methodology_stem(s) for s in stemmed_keywords(system_name, min_len=MIN_TERM_LEN)}


def score_paper_against_its_provenance(
    paper: Paper, *, classifier_score: Optional[float] = None,
    classifier_threshold: float = SIM_CLASSIFIER_THRESHOLD,
) -> Stage3Verdict:
    """`classifier_score`: sim_classifier.py's P(authors performed a
    simulation) for THIS paper, or None. None (the default) reproduces the
    exact pre-classifier behavior byte-for-byte (literal _STRUCTURAL_CG_
    MARKERS substring check, no general-path classifier gate at all) --
    every existing caller that doesn't pass this (including every test in
    tests/test_relevance_filter_handoff.py) is unaffected. filter_papers()
    -- the only production call site -- always supplies a real score,
    batch-computed once for every non-review-filtered candidate."""
    if not paper.found_by_queries:
        # No provenance at all (e.g. a Paper built by hand, or the old
        # single-query search_and_dedupe() path). Fail OPEN, same
        # high-recall bias as the rest of this stage -- keep rather than
        # guess-drop something there's no basis to judge.
        return Stage3Verdict(
            title=paper.title,
            keep=True,
            score=0.0,
            matched_keywords=[],
            matched_role="(no provenance -- kept by default)",
            reason="paper has no found_by_queries to score against; kept (high-recall default), not guess-dropped",
        )

    stem_map = _query_stem_map(paper.found_by_queries)
    title_stems = _normalized_text_stems(paper.title)
    abstract_stems = _normalized_text_stems(paper.abstract or "")

    matched_title = set(stem_map) & title_stems
    matched_abstract = set(stem_map) & abstract_stems
    score = TITLE_WEIGHT * len(matched_title) + ABSTRACT_WEIGHT * len(matched_abstract)
    matched = sorted({stem_map[s] for s in matched_title | matched_abstract})

    keep = score >= MIN_SCORE
    role_label = "+".join(sorted(set(paper.query_categories))) or "(no category recorded)"

    # Hop-1 broad, Hop-2 strict: a structural-route (Hop-2) paper must ALSO
    # clear a stricter, independent AND-gate -- system evidence AND CG
    # evidence both present in title/abstract -- on top of (not instead of)
    # the general >=1-keyword-hit check above. That general check alone was
    # too permissive for Hop-2 specifically: Hop-2's query is already
    # "{system} {objective} coarse-grained model", so a paper hitting on
    # ONLY the methodology words (e.g. a Martini paper never mentioning the
    # discovered system) or ONLY the system name (e.g. a classic
    # experimental paper with no CG evidence) both satisfied the old rule.
    # Explicitly abstract-grounded (v1 scope): a paper whose CG evidence
    # exists only in its full text is an accepted, known blind spot here,
    # not something this stage claims to see.
    structural_gate_note = ""
    # Existential over ALL systems that independently retrieved this paper
    # (paper.structural_discovered_systems, route_plan.py's
    # RetrievedPaper.hop1_provenances) -- NOT the full set of every Hop-1-
    # discovered system, only the ones whose OWN Hop-2 query actually found
    # THIS paper. Falls back to the single-name field for any caller that
    # only ever set that (backward compat -- e.g. hand-built Paper fixtures
    # in older diagnostics/tests that predate this list). Diagnosed against
    # a real case: the same paper, found by both 'EspP' and 'pertactin'
    # Hop-2 queries, mentions only 'pertactin' in its own title/abstract --
    # checking only whichever system's query happened to be processed first
    # (a Hop-1 support-ranking artifact, not a relevance judgment) produced
    # an order-dependent false drop. No re-ranking, no re-attribution to
    # "whichever system matches best" -- every retrieving system is tried,
    # first pass wins, nothing about WHICH one is preferred.
    candidate_systems = paper.structural_discovered_systems or (
        [paper.structural_discovered_system] if paper.structural_discovered_system else [])
    has_structural_route = "structural" in paper.query_categories and bool(candidate_systems)
    has_flat_route = bool({"system_specific_cg", "process", "context", "need_semantic", "transfer_direct"} & set(paper.query_categories))

    # Route-aware CG/system gate (this session's fix -- diagnosed real false
    # negative). A paper found via MORE THAN ONE route (e.g. both Hop-2/
    # structural AND a flat direct_concept_search -- query_categories is
    # their UNION once transfer.py's _merge_papers combines
    # duplicate refs) used to be judged ONLY by whichever branch an if/elif
    # happened to check first -- in practice always the stricter structural
    # branch whenever "structural" was among its categories, even when the
    # SAME paper's own flat-route provenance would have independently
    # cleared the simpler (CG-evidence-only) gate. A real case: a paper
    # correctly found via a flat transfer_direct query ALSO got attached,
    # via Hop-2, to unrelated discovered systems (a separate Hop-1 mis-
    # attribution, not addressed here) -- the structural branch's unrelated-
    # system-evidence failure silently discarded it, never even reaching
    # the CG-evidence question its flat-route provenance would have passed.
    # Fixed generically, not per-paper: evaluate every route this paper
    # actually has provenance for, and keep if ANY of them independently
    # passes. A route this paper was never found through (has_structural_
    # route / has_flat_route False) contributes nothing -- never a spurious
    # pass; single-route papers see EXACTLY the prior per-branch behavior.
    if has_structural_route or has_flat_route:
        # CG-evidence signal: the trained classifier when a score was
        # supplied (production -- see this function's docstring), else the
        # original literal-marker substring check (classifier_score=None,
        # e.g. every existing test fixture here) -- see module docstring's
        # "Review filter + experiment-vs-simulation classifier" section.
        if classifier_score is not None:
            has_cg_evidence = classifier_score >= classifier_threshold
        else:
            blob = f"{paper.title or ''} {paper.abstract or ''}".lower()
            has_cg_evidence = any(m in blob for m in _STRUCTURAL_CG_MARKERS)

        passing_systems: list[str] = []
        has_system_evidence = False
        if has_structural_route:
            all_stems = title_stems | abstract_stems
            for candidate in candidate_systems:
                system_stems = _system_evidence_stems(candidate)
                if system_stems and system_stems <= all_stems:
                    passing_systems.append(candidate)
            has_system_evidence = bool(passing_systems)

        # structural gate: system evidence AND CG evidence (unchanged rule).
        # flat gate: CG evidence alone (unchanged rule, no system-name
        # requirement -- see the flat-route rationale kept below).
        structural_ok = has_structural_route and has_system_evidence and has_cg_evidence
        flat_ok = has_flat_route and has_cg_evidence

        if not (structural_ok or flat_ok):
            # CG evidence is the SOLE blocker (eligible for filter_papers()'s
            # full-text fallback) iff at least one applicable route needs
            # nothing else: the flat route (no system-evidence requirement
            # at all), or the structural route with system evidence already
            # satisfied. If the ONLY applicable route is structural and it's
            # also missing system evidence, full text can never fix that --
            # a genuine relevance miss, not eligible for the fallback.
            cg_evidence_is_sole_blocker = (
                (has_flat_route or (has_structural_route and has_system_evidence))
                and not has_cg_evidence
            )
            missing = []
            if has_structural_route and not has_system_evidence:
                missing.append(f"system evidence for any of the systems that retrieved it {candidate_systems!r} (structural route)")
            if not has_cg_evidence:
                missing.append("CG/simulation evidence")
            routes = "+".join(r for r, present in (("structural", has_structural_route), ("flat", has_flat_route)) if present)
            return Stage3Verdict(
                title=paper.title, keep=False, score=score, matched_keywords=matched, matched_role=role_label,
                reason=f"route-aware CG/system gate ({routes}): missing {' and '.join(missing)} in title/abstract "
                       f"(abstract-grounded, v1) -- the general keyword check alone would have "
                       f"{'kept' if keep else 'also dropped'} this paper",
                missing_cg_evidence_only=cg_evidence_is_sole_blocker,
                classifier_score=classifier_score,
            )
        notes = []
        if structural_ok:
            notes.append(f"structural gate: system evidence via {passing_systems!r} + CG evidence")
        if flat_ok:
            notes.append("flat-route gate: CG evidence")
        structural_gate_note = f" ({'; '.join(notes)} present)"
    elif classifier_score is not None and classifier_score < classifier_threshold:
        # General path (no structural/flat route provenance at all -- e.g.
        # query_categories like "general_discovery"/"system_specific"/
        # "methodology_evidence"): previously ungated on the CG-evidence
        # axis entirely, kept on >=1 keyword hit alone regardless of
        # whether the paper actually describes a simulation. New: the
        # classifier now ALSO has to clear its threshold here -- this is
        # specifically the "kill more experimental papers" half of the
        # classifier's job (the route-aware branch above already covers
        # the "rescue real simulation papers the marker check missed"
        # half). Only applies when a real score was supplied (production);
        # classifier_score=None (every pre-existing test here) reproduces
        # the exact old ungated behavior.
        return Stage3Verdict(
            title=paper.title, keep=False, score=score, matched_keywords=matched, matched_role=role_label,
            reason=f"matched {len(matched)} keyword(s) ({', '.join(matched)}) against its own originating query, "
                   f"but the classifier predicts this is not a genuine author-performed simulation "
                   f"(score={classifier_score:.3f} < threshold={classifier_threshold})",
            classifier_score=classifier_score,
        )

    if keep:
        plural = "y" if len(paper.found_by_queries) == 1 else "ies"
        reason = f"matched {len(matched)} keyword(s) ({', '.join(matched)}) against its own originating quer{plural}{structural_gate_note}"
    else:
        reason = f"no keyword overlap between title/abstract and its own originating query text{structural_gate_note}"

    return Stage3Verdict(
        title=paper.title, keep=keep, score=score, matched_keywords=matched, matched_role=role_label, reason=reason,
        classifier_score=classifier_score,
    )


# Full-text Methods fallback (this session's fix): a paper whose abstract
# never states its CG methodology (a real, diagnosed false negative -- see
# europepmc.py's module docstring) is not necessarily off-topic, but
# fetching and scanning full text for EVERY paper that fails only this gate
# would be neither fast nor "minimal" -- a real case had 100+ such
# candidates from one query. Bounded instead: only the highest-scoring
# `fulltext_lookup_budget` candidates (ranked by the SAME general topical
# score already computed) get an actual network call, in score order, so
# the papers most likely to be genuinely about the query's system are
# checked first. Every other missing-CG-evidence-only reject keeps its
# original abstract-only verdict unchanged -- an honest "not checked" absence,
# not a claim that full text was examined and found nothing.
_DEFAULT_FULLTEXT_LOOKUP_BUDGET = 20


def _try_fulltext_cg_evidence(paper: Paper, verdict: Stage3Verdict) -> Stage3Verdict:
    from cgscout.sources.europepmc import fetch_methods_text, snippet_around  # local import -- optional network dependency, only touched here

    doi = getattr(paper, "doi", None)
    text = fetch_methods_text(doi)
    if not text:
        return verdict  # unavailable/not OA/fetch failed -- unchanged, original abstract-only verdict stands
    blob = text.lower()
    matched_markers = [m for m in _STRUCTURAL_CG_MARKERS if m in blob]
    if not matched_markers:
        return verdict  # full text fetched, but no CG marker there either -- unchanged
    snippet = snippet_around(text, tuple(matched_markers)) or ""
    return verdict.model_copy(update={
        "keep": True,
        "matched_keywords": sorted(set(verdict.matched_keywords) | {f"fulltext:{m}" for m in matched_markers}),
        "reason": verdict.reason + f" -- OVERRIDDEN: explicit CG evidence found in the paper's full-text "
                                    f"Methods section (not abstract), via Europe PMC. Matched: {matched_markers}. "
                                    f"Snippet: {snippet!r}",
    })


def _recover_missing_abstracts(
    papers: list[Paper], lookup: Optional[Callable[[Paper], Optional[dict]]] = None, workers: int = 8,
) -> tuple[list[Paper], list[bool]]:
    """Europe PMC abstract recovery for papers whose OpenAlex record has no usable abstract (same lookup and
    same MIN_ABSTRACT_CHARS rule as Stage 4's abstract fallback, which this moves in front of Stage 3). Returns the
    papers (enriched copies where an abstract was found) and a per-paper `recovered` flag. A failed or empty
    lookup leaves the paper untouched -- Stage 3 then sees exactly what it saw before. Lookups run in a small
    thread pool; successful answers are cached by stage4 (failures never are)."""
    from concurrent.futures import ThreadPoolExecutor
    from cgscout.assessment.stage4 import MIN_ABSTRACT_CHARS, europepmc_abstract   # local import: optional network dependency, only touched here
    lookup = lookup or europepmc_abstract
    need = [i for i, p in enumerate(papers) if len((p.abstract or "").strip()) < MIN_ABSTRACT_CHARS]
    out, recovered = list(papers), [False] * len(papers)
    if not need:
        return out, recovered

    def one(i: int):
        try:
            res = lookup(papers[i])
            return i, ((res or {}).get("abstract") or "").strip()
        except Exception:  # noqa: BLE001 -- transport failure: keep the paper as retrieved
            return i, ""

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for i, ab in ex.map(one, need):
            if len(ab) >= MIN_ABSTRACT_CHARS:
                out[i] = papers[i].model_copy(update={"abstract": ab}); recovered[i] = True
    return out, recovered


def filter_papers(
    papers: list[Paper], *,
    fulltext_lookup_budget: int = _DEFAULT_FULLTEXT_LOOKUP_BUDGET,
    classifier_threshold: float = SIM_CLASSIFIER_THRESHOLD,
    classify_fn: Optional[Callable[[list[Paper]], list[float]]] = None,
    recover_abstracts: bool = False,
    abstract_lookup: Optional[Callable[[Paper], Optional[dict]]] = None,
) -> tuple[list[Paper], list[Paper], list[Stage3Verdict]]:
    """Stage 3 for the RetrievalIntent path. Returns (retained, rejected,
    verdicts): retained/rejected are Paper objects annotated with
    relevance_score/matched_keywords (same fields, same meaning
    RuleBasedRelevanceFilter already populates for the old pipeline --
    reused, not reinvented), sorted by score descending. `verdicts` carries
    the extra role/reason/review/classifier detail this handoff adds for
    debugging (see Stage3Verdict).

    Processing order, per paper:
      1. review_filter.py's high-confidence review/perspective check --
         if flagged, dropped immediately, no embedding call spent.
      2. Classifier scoring: ONE batched embedding call covering every
         paper that survived step 1 (not one call per paper). Computed
         up front for all of them, regardless of what step 3 below will
         do with it -- efficiency (one batch), not a relevance judgment
         of its own.
      3. score_paper_against_its_provenance()'s general keyword-overlap
         score (unchanged) -- if below MIN_SCORE, dropped; the classifier
         score from step 2 is recorded on the verdict for audit but can
         NEVER override a topical miss (it has no topical-relevance
         signal, only a sim-vs-experiment one).
      4. For papers that cleared step 3: the same function's route-aware
         system-evidence check (structural route only, unchanged) plus
         its CG-evidence check -- now driven by the step-2 classifier
         score instead of the literal _STRUCTURAL_CG_MARKERS substring
         match, for BOTH route-scoped candidates (rescuing real
         simulation papers the marker check would have missed) and,
         newly, papers matched via no route at all (rejecting topically-
         relevant but experimental papers that used to sail through
         ungated on this axis).
      5. If fulltext_lookup_budget > 0: the pre-existing Europe PMC
         full-text Methods fallback, unchanged, still targets exactly the
         `missing_cg_evidence_only` cases step 4 can produce.

    Known limitation, found via a live smoke test against real OpenAlex
    abstracts during this integration (not something the 424/427-row
    train/test split happened to cover): the classifier can be fooled by
    topical closeness. The real 2006 Junker paper (10.1073/pnas.0507923103,
    pure experimental biophysics -- equilibrium denaturation, crystal
    structure -- zero simulation content) scores 0.999 here, because its
    embedding sits very close to genuine Pertactin-passenger-domain
    simulation papers in this embedding space. The OLD literal-marker
    check caught this exact case correctly (no "coarse-grained"/"martini"/
    etc. substring anywhere in that abstract); the classifier does not, by
    construction, since it was never trained against hard negatives this
    topically close. Raising classifier_threshold cannot fix a 0.999 score
    -- this needs a retrain with such cases added as labeled hard
    negatives (sim_classifier_train.py's RELABEL/EXCLUDE sets), not a
    threshold change.

    `classifier_threshold`: the one knob to change the classifier's
    decision boundary -- a plain parameter (default sim_classifier.
    SIM_CLASSIFIER_THRESHOLD, itself a single module-level constant there,
    not hard-coded inline here or anywhere in the gate logic above).

    `classify_fn`: production default (None) resolves to sim_classifier.
    default_classify_fn -- the real, persisted model + a real OpenAI
    embeddings call. Tests inject a network-free fake here (see
    tests/test_relevance_filter_handoff.py) to keep exercising the gate
    logic deterministically; every real caller should leave this unset.

    `fulltext_lookup_budget`: see _try_fulltext_cg_evidence()'s module-level
    comment above -- caps how many Europe PMC lookups one filter_papers()
    call can make. 0 disables the fallback entirely.

    `recover_abstracts` (default False == previous behaviour): before ANY Stage 3 step, papers without a usable
    OpenAlex abstract are looked up in Europe PMC by DOI (the lookup Stage 4 already performs later). The recovered
    abstract is what the review filter, classifier, keyword score and gates judge, and the returned (retained/
    rejected) Paper objects carry it; each verdict's `abstract_recovered` says so. Without it a title-only paper is
    scored on the title alone, which the classifier systematically under-scores (observed on benchmark seeds).
    `abstract_lookup`: injectable lookup (tests); None -> stage4.europepmc_abstract."""
    classify_fn = classify_fn or default_classify_fn
    abstract_flags: list[bool] = [False] * len(papers)
    if recover_abstracts:
        papers, abstract_flags = _recover_missing_abstracts(papers, abstract_lookup)

    review_details = [review_filter_detail(p) for p in papers]
    to_classify = [i for i, rd in enumerate(review_details) if not rd["flagged"]]
    scores: list[Optional[float]] = [None] * len(papers)
    if to_classify:
        computed = classify_fn([papers[i] for i in to_classify])
        for idx, s in zip(to_classify, computed):
            scores[idx] = s

    verdicts: list[Stage3Verdict] = []
    for i, (paper, rd) in enumerate(zip(papers, review_details)):
        if rd["flagged"]:
            signals = []
            if rd["type_hit"]:
                signals.append("OpenAlex type=='review'")
            if rd["matched_phrase"]:
                signals.append(f"abstract self-declares: {rd['matched_phrase']!r}")
            verdicts.append(Stage3Verdict(
                title=paper.title, keep=False, score=0.0, matched_keywords=[],
                matched_role="+".join(sorted(set(paper.query_categories))) or "(no category recorded)",
                reason=f"high-confidence review/perspective filter: {' + '.join(signals)}",
                review_flagged=True, review_reason=" + ".join(signals),
            ))
            continue
        verdicts.append(score_paper_against_its_provenance(
            paper, classifier_score=scores[i], classifier_threshold=classifier_threshold,
        ))

    for i, flag in enumerate(abstract_flags):
        if flag:
            verdicts[i] = verdicts[i].model_copy(update={"abstract_recovered": True})

    if fulltext_lookup_budget > 0:
        candidates = sorted(
            (i for i, v in enumerate(verdicts) if v.missing_cg_evidence_only),
            key=lambda i: verdicts[i].score, reverse=True,
        )[:fulltext_lookup_budget]
        for i in candidates:
            verdicts[i] = _try_fulltext_cg_evidence(papers[i], verdicts[i])

    retained: list[Paper] = []
    rejected: list[Paper] = []
    for paper, verdict in zip(papers, verdicts):
        annotated = paper.model_copy(update={"relevance_score": verdict.score, "matched_keywords": verdict.matched_keywords})
        (retained if verdict.keep else rejected).append(annotated)
    retained.sort(key=lambda p: p.relevance_score or 0.0, reverse=True)
    rejected.sort(key=lambda p: p.relevance_score or 0.0, reverse=True)
    return retained, rejected, verdicts
