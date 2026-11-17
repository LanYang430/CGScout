"""RouteRetrievalPlan: bounded literature retrieval from an EvidenceRoutePlan.

Independent flat routes (system_specific_cg, process, context, optional
need_semantic; the modeling_requirement route name is kept but builds no
queries) each contribute retrieval, deduped and merged by paper_ref with
FULL cross-route provenance preserved. A route with nothing to search never
blocks the others.

The old structural route (TerminologyExpander -> Hop-1 -> Hop-2 inside this
module) is historical code, kept in the git tag `legacy-pre-cgscout`: the live
path never enables it (the structural / related-system work is done by
orchestrator). Its provenance and result types stay in the
schema below.

Reuses, UNMODIFIED:
  - beam_agent.search.search_and_dedupe()        (the flat routes)

Does NOT touch Hop-1/Hop-2 internals, Stage 3/4, evidence matching, or the
live CLI.
"""
from __future__ import annotations

import re
import time
from typing import Callable, Literal, Optional

from pydantic import BaseModel, Field


from cgscout.retrieval.direct.evidence_plan import EvidenceRoutePlan, RouteTerm  # noqa: E402
from cgscout.retrieval.related_systems.discovery import (  # noqa: E402
    ExtractFn,
    SearchFn as StructuralSearchFn,
    _paper_ref,
)
from cgscout.retrieval.types import Hop1Provenance, Hop2Result  # noqa: E402
from cgscout.retrieval.related_systems.terminology import CallFn  # noqa: E402


# --- budgets ------------------------------------------------------------

class RouteRetrievalBudgets(BaseModel):
    """Engineering / query-explosion caps -- NOT scientific thresholds.
    Defaults per the approved design; not tuned here."""
    max_queries_per_route: int = 3
    # Was 5. Raised to 20 (CG-model-recommendation retrieval-tightening
    # decision, this session): diagnosed live against the "pertactin
    # folding" case -- the query "pertactin folding coarse-grained model"
    # genuinely returns the user's own target paper at rank 13, which a
    # top-5 cutoff silently drops before Stage 3 ever sees it; top-20
    # recovers it. Only the active flat routes (system_specific_cg /
    # process / context, see FLAT_ROUTES/build_route_retrieval_plan below)
    # are affected in practice -- modeling_requirement stays disabled.
    max_papers_per_query: int = 20
    # Was 20. Raised to 60 (== max_queries_per_route(3) * the new
    # max_papers_per_query(20)) in the same pass, specifically so THIS cap
    # can never silently re-bind right after the one above was raised --
    # diagnosed live: at max_papers_per_query=20/30 with the old
    # max_papers_per_route=20, the 20-per-route cap became the binding
    # constraint instead, and paper counts stopped changing at all between
    # limit=20 and limit=30 (not "no more relevant papers exist", just this
    # second cap silently taking over). 60 is a safety ceiling matched to
    # the other two knobs, not a retuned scientific threshold.
    max_papers_per_route: int = 60          # cap on NEW unique papers one flat route may contribute
    # Fixed Hop-1-depth-15 budget decision (cross-case retrieval-depth
    # sensitivity experiment, beta-helix/coiled-coil/TIM-barrel, this
    # session): Hop-1's own max_discovered is no longer where downstream
    # cost is controlled. It is now a generous SAFETY CAP only, well above
    # normal yield (observed 4-100 grounded systems across 3 real cases at
    # depth<=25) -- not a selection cutoff. Hop-2's own selection
    # (select_systems_for_hop2, ranked by (-n_support, first_seen_rank),
    # unchanged) is the real cost gate.
    structural_max_discovered_systems: int = 150
    structural_hop2_max_systems: int = 30        # == types.DEFAULT_MAX_SYSTEMS override; the real cost gate
    # Was 60 (see prior regression-fix note in git/session history: the live
    # wiring once silently fell back to discover_related_systems()'s own
    # default of 25). Raised again here, to 150, as a SAFETY CAP alongside
    # structural_max_discovered_systems -- the same cross-case retrieval-
    # depth experiment measured real unique-paper counts at Hop-1 depth=15
    # (beta-helix 123, coiled-coil 138, TIM barrel 109), all above 60. Left
    # at 60 this would silently become a SECOND extraction-limiting
    # bottleneck ahead of ranking, undercutting the depth=15 change above --
    # 150 keeps it a safety cap, not a normal selection cutoff.
    structural_max_papers_to_extract: int = 150
    # Bounded concurrency for Hop-1's per-paper extract_fn calls
    # (discovery.discover_related_systems()'s own
    # max_workers parameter -- ONLY the extract_fn calls run concurrently;
    # grounding/merge/ranking always stays sequential in original order,
    # unaffected). Validated by this session's own standalone benchmark
    # (run_hop1_concurrency_benchmark.py): workers=2 measured -18.5% wall
    # time vs workers=1 at this exact call site, 0 errors, byte-identical
    # output; workers=4 only a further ~5% (diminishing returns) --
    # approved production value is 2, not higher.
    hop1_extraction_max_workers: int = 2
    # Dedicated NEW-unique-paper cap for the structural branch AS A WHOLE --
    # None (default) preserves the previous behavior (structural bounded
    # ONLY by the shared max_total_unique_papers below; existing callers/
    # tests that never set this are unaffected). The live adapter fills
    # this in via resolve_structural_and_safety_budgets() as
    # structural_hop2_max_systems * <Hop-2 per-system search depth>, so the
    # structural branch can, in principle, accept every selected system's
    # FULL Hop-2 result list regardless of which system gets processed
    # first -- diagnosed against a real case: with 20 Hop-1-discovered
    # systems, pertactin (ranked 14th/20) was crowded out of the shared
    # max_total_unique_papers=120 by earlier-ranked systems' own results
    # despite its own query genuinely returning the target paper at rank
    # 12/20 -- not a ranking failure, a pure processing-order budget
    # collision between routes.
    structural_total_budget: Optional[int] = None
    # global cap on NEW unique papers, across all routes. Historically the
    # ONLY cap on the structural branch (see structural_total_budget above
    # for why that was a bug at scale) -- now intended as an outer safety
    # net only: resolve_structural_and_safety_budgets() raises this (never
    # lowers it) to at least sum(every route's own cap), so it becomes
    # mathematically impossible for THIS cap to bind before any individual
    # route's own cap does, as long as every route already respects its
    # own. Still real: it protects against a future bug that bypasses a
    # route's own cap, or a route added later that forgets to set one.
    max_total_unique_papers: int = 120

    # --- flat-lexical / semantic query switches (v4: P1, P2, P3) -- independent so each can be ablated ----------------------
    # P1: a context term that is a single non-entity word ('membrane', 'intracellular', 'cholesterol') -- or one that came from the
    #     legacy system-description marker list -- carries no system identity on its own; it is searched ANCHORED to the system head
    #     instead of alone.  Every other (short or long) context term keeps its own standalone query.  False == v3 behaviour.
    flat_anchor_short_context: bool = True
    # P2: query hygiene -- drop leftover imperative verbs / nominalised actions at the start of a chunk, and dangling function words /
    #     adverbs left behind by a word-count truncation.  False == v3 behaviour.
    flat_query_hygiene: bool = True
    # P3: ONE extra SEMANTIC query built from the planner's own system + objective (specific entities kept verbatim), run as its own
    #     route `need_semantic`.  OFF by default: it costs one extra OpenAlex request per case and enlarges the candidate pool.
    need_semantic_enabled: bool = False
    need_semantic_limit: int = 20            # papers kept from the semantic query (OpenAlex semantic cap is 50)
    need_semantic_include_context: bool = False   # control arm: also append the distinct context terms to the query
    # Stage 3 input fix (independent of the retrieval flags above): papers whose OpenAlex record has no usable abstract are looked up in Europe PMC (by DOI, the same lookup Stage 4 already uses) BEFORE the
    # Stage 3 classifier/gates run, so they are judged on title+abstract instead of on the title alone.  ON by default; False == Stage 3 sees exactly what OpenAlex returned (the previous behaviour).
    stage3_recover_abstracts: bool = True
    # Query Construction v2 (query_construction_v2.py): process route + need_semantic + 3A built from method-free goals and the planner's search_axes
    # (system-specific and context/entity queries unchanged).  OFF == the legacy builders, byte for byte.
    query_construction_v2: bool = False


# --- plan schemas ---------------------------------------------------------

# The four flat routes, in the fixed order run_route_retrieval() executes
# them -- exposed as a constant (not just the literal tuple inline in the
# loop) so resolve_structural_and_safety_budgets() can compute "how much
# could every flat route possibly contribute" from the same single source,
# rather than a second hardcoded "4" living in the live adapter.
#
# system_specific_cg REPLACED the old direct_system route (this session):
# direct_system issued a BARE "{system}" query (e.g. "EspP") -- deleted,
# not disabled, since a bare system-name search was confirmed to pull in
# mostly experimental/epidemiology literature with no CG value. The new
# route issues "{system} coarse-grained model" instead: system-specific CG
# discovery that still fires even when a paper never highlights a specific
# process (folding/binding/...), as long as it uses or builds a CG model
# for THIS system. modeling_requirement stays in the tuple but is still
# not called by build_route_retrieval_plan() (kept disabled, not deleted).
FLAT_ROUTES: tuple[str, ...] = ("system_specific_cg", "process", "context", "modeling_requirement")

FlatRoute = Literal["system_specific_cg", "process", "context", "modeling_requirement"]
# P3's extra route is NOT part of FLAT_ROUTES (so the flat-route loop, the pool-cap arithmetic of the flat routes and every existing
# consumer of FLAT_ROUTES stay byte-identical); it is run by its own block in run_route_retrieval().
NEED_SEMANTIC_ROUTE = "need_semantic"
QueryRoute = Literal["system_specific_cg", "process", "context", "modeling_requirement", "need_semantic"]


def resolve_structural_and_safety_budgets(
    budgets: RouteRetrievalBudgets, *, structural_hop2_search_limit: int,
) -> RouteRetrievalBudgets:
    """Pure function, no side effects. Returns a COPY of `budgets` with:

    1. `structural_total_budget` filled in (if not already explicitly set)
       as `structural_hop2_max_systems * structural_hop2_search_limit` --
       the structural branch's own theoretical yield ceiling (every
       selected system's full Hop-2 result list, in the worst case of zero
       overlap between systems).
    2. `max_total_unique_papers` raised (never lowered) to at least
       `len(FLAT_ROUTES) * max_papers_per_route + structural_total_budget`
       -- the sum of every route's own individual cap. Since the
       accumulator's global counter can only reach as high as the sum of
       what each route is individually permitted to contribute (shared
       papers across routes only ever reduce that sum, never increase it),
       this makes it mathematically impossible for the global cap to bind
       before any individual route's own cap does. It remains a real,
       inert-under-normal-operation safety net -- not removed, not
       disabled, just sized so it stops being an accidental extra
       scientific cutoff.

    `structural_hop2_search_limit` is owned by the live adapter (the only
    place the real Hop-2 per-system search depth is defined) and passed in
    explicitly rather than duplicated as a second constant here, to avoid
    the two drifting out of sync.

    Never lowers either field if the caller already set it explicitly
    higher than the computed floor."""
    b = budgets.model_copy()
    if b.structural_total_budget is None:
        b.structural_total_budget = b.structural_hop2_max_systems * structural_hop2_search_limit
    floor = len(FLAT_ROUTES) * b.max_papers_per_route + b.structural_total_budget
    if b.need_semantic_enabled:
        floor += b.max_papers_per_route          # P3's route gets its own cap, so the global net still cannot bind first
    if b.max_total_unique_papers < floor:
        b.max_total_unique_papers = floor
    return b


class RouteQuery(BaseModel):
    route: QueryRoute
    route_term: str
    route_term_source: Literal["USER_GROUNDED", "LITERATURE_GROUNDED", "PLANNER_DERIVED"]
    query: str
    query_role: str
    source_component: str = "search_and_dedupe"


class RouteRetrievalPlan(BaseModel):
    system_description: str
    scientific_objective: str
    route_queries: list[RouteQuery] = Field(default_factory=list)
    structural_branch_enabled: bool = False
    structural_branch_structural_class: Optional[str] = None
    structural_branch_structural_class_source: Optional[str] = None
    structural_branch_also_known_as: list[str] = Field(default_factory=list)  # filled during execution (TerminologyExpander)
    # audit trail for structural terms NOT chosen as primary (never re-run through Hop-1)
    structural_branch_secondary_terms: list[dict] = Field(default_factory=list)  # [{"term":..., "source":...}]
    budgets: RouteRetrievalBudgets = Field(default_factory=RouteRetrievalBudgets)
    empty_routes: list[str] = Field(default_factory=list)


# --- result schemas --------------------------------------------------------

class RouteProvenanceEntry(BaseModel):
    route: str
    route_term: str
    query: str
    query_role: str
    retrieval_rank: int


class RetrievedPaper(BaseModel):
    paper_ref: str
    title: str
    year: Optional[int] = None
    found_by: list[RouteProvenanceEntry] = Field(default_factory=list)
    hop1_provenance: Optional[Hop1Provenance] = None  # populated only for structural-branch hits
    # ALL discovered systems whose OWN Hop-2 query independently retrieved
    # this paper (not just the first one to claim it) -- `hop1_provenance`
    # above stays the first claimant only, UNCHANGED, for existing
    # consumers; this list is what a caller wanting the FULL set (e.g.
    # Stage 3's existential system-evidence gate) should read. Diagnosed
    # against a real case: when the same paper is found by both 'EspP' and
    # 'pertactin' Hop-2 queries, whichever system's query happened to run
    # first (a Hop-1-ranking artifact, not a relevance judgment) used to be
    # the ONLY one Stage 3 ever saw -- an order-dependent false drop when
    # that first claimant's name doesn't itself appear in the paper's text.
    hop1_provenances: list[Hop1Provenance] = Field(default_factory=list)


class RouteOutcome(BaseModel):
    route: str
    route_term: Optional[str] = None
    queries_tried: list[str] = Field(default_factory=list)
    n_papers_found: int = 0            # papers that survived into the merged result (new OR re-merged)
    n_new_unique_contributed: int = 0  # of those, how many were NEW unique paper_refs this route added
    n_dropped_route_cap: int = 0       # candidates dropped by this route's own max_papers_per_route
    n_dropped_global_cap: int = 0      # candidates dropped by the global max_total_unique_papers
    structural_cleanup_log: list[str] = Field(default_factory=list)  # postprocess.py (cleanup) audit trail (structural route only)
    empty: bool = False
    empty_reason: Optional[Literal[
        "not_applicable_no_grounded_term",
        "queries_returned_zero_papers",
        "search_error",
    ]] = None
    search_errors: list[str] = Field(default_factory=list)


class RouteRetrievalResult(BaseModel):
    plan: RouteRetrievalPlan
    per_route: list[RouteOutcome] = Field(default_factory=list)
    papers: list[RetrievedPaper] = Field(default_factory=list)
    n_unique_papers: int = 0
    n_papers_multi_route: int = 0
    structural_branch_result: Optional[Hop2Result] = None
    routes_with_zero_results: list[str] = Field(default_factory=list)
    retrieval_seconds: float = 0.0


# --- flat-route query builders ---------------------------------------------

_CG_SUFFIX = "coarse-grained model"


def _cap_terms(terms: list[RouteTerm], n: int) -> list[RouteTerm]:
    return terms[:n]


# --- flat lexical query shaping (v2) ----------------------------------------
# OpenAlex lexical `search=` requires (essentially) every term to match, so a query that stacks the whole
# system description, every context item and the full objective sentence matches almost nothing (diagnosed on
# B193: all four flat queries returned 0 papers). Flat queries are therefore SHORT and COMPLEMENTARY: each carries
# one scientific concept chunk (+ the CG suffix), drawn from a different planner field, and chunks that merely
# repeat one another are dropped. Generic rules only -- no per-system / per-benchmark special cases.

_LEX_STOPWORDS = frozenset(
    "a an the of and or in on with to for by using via at from as into through".split())
_MAX_CHUNK_WORDS = 6          # content words allowed in one concept chunk (system head / context term)
_MAX_PROCESS_WORDS = 8        # content words allowed in the process phrase
_MAX_SYSTEM_WORDS = 4         # a system description longer than this is reduced to its head noun phrase
_SHORT_PROCESS_WORDS = 2      # a process phrase this short ("folding") cannot stand alone -> prefix the system head
# where a system description stops being the head noun phrase: a comma/semicolon (followed by space, so 'Na+,K+' is safe),
# a relational preposition, or a relational participle ('associated with', 'forming', 'transporting', 'invaded by' ...)
_SYSTEM_CONNECTOR_RE = re.compile(
    r"\s*[,;]\s+|\s+(?:with|and|in|on|bound\s+to|complexed\s+with|containing|interacting\s+with|plus|including|"
    r"(?:peripherally\s+|tightly\s+)?(?:associated|coupled|attached|bound|linked|anchored|invaded|embedded)\s+(?:with|to|by|in|on)|"
    r"[A-Za-z]{3,}ing)\s+", re.I)
_LEADING_ARTICLE_RE = re.compile(r"^\s*(?:a|an|the)\s+", re.I)
_PROCESS_VERBS = (r"simulat(?:e|es|ing)|stud(?:y|ies|ying)|model(?:l?ing|s)?|investigat(?:e|ing)|characteri[sz](?:e|ing)|"
                  r"determin(?:e|ing)|identify(?:ing)?|understand(?:ing)?|resolv(?:e|ing)|explor(?:e|ing)|examin(?:e|ing)|"
                  r"assess(?:ing)?|evaluat(?:e|ing)|predict(?:ing)?|compar(?:e|ing)|find(?:ing)?|sampl(?:e|ing)")
_PROCESS_LEADING_VERB_RE = re.compile(r"^\s*(?:" + _PROCESS_VERBS + r")\s+(?:whether\s+|how\s+|the\s+)?", re.I)
_PROCESS_TRAILING_METHOD_RE = re.compile(
    r"\s+(?:using|with|by|via|through|based\s+on)\s+(?:[\w-]+\s+){0,3}?(?:coarse[- ]grained|CG)\b.*$", re.I)
# the first clause of a process phrase: stop at a comma/semicolon or a subordinating/elaborating word
_PROCESS_CLAUSE_END_RE = re.compile(r"\s*[,;]\s+|\s+(?:while|including|such\s+as|so\s+that|in\s+order\s+to|with\s+emphasis|to\s+(?:predict|determine|identify|assess|evaluate|understand)|and\s+(?:assess|identify|determine|evaluate)\w*)\s+", re.I)


def _content_words(text: str) -> list[str]:
    return [w for w in re.split(r"\s+", (text or "").strip()) if w and w.lower().strip(".,;:()") not in _LEX_STOPWORDS]


def _squash(text: str) -> str:
    """lowercase alphanumerics only, so 'tetra-nucleosome array' and 'tetranucleosome array' compare equal"""
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _cap_words(text: str, n: int) -> str:
    """keep the first n content words (stopwords between them are kept) -- truncation at a word boundary"""
    out, k = [], 0
    for w in re.split(r"\s+", text.strip()):
        if w.lower().strip(".,;:()") not in _LEX_STOPWORDS:
            k += 1
            if k > n:
                break
        out.append(w)
    while out and out[-1].lower().strip(".,;:()") in _LEX_STOPWORDS:
        out.pop()
    return " ".join(out)


def _system_head(system: str) -> str:
    """The head noun phrase of a (possibly long) system description: text before the first connector
    ('with', 'and', 'in', 'bound to', ',' ...), leading article dropped. Short descriptions are returned unchanged."""
    system = _strip_conversational_wrapper(system or "").strip()
    if len(_content_words(system)) <= _MAX_SYSTEM_WORDS:
        return _LEADING_ARTICLE_RE.sub("", system).strip() or system
    body = _LEADING_ARTICLE_RE.sub("", system)
    head, cut = body, None
    for m in _SYSTEM_CONNECTOR_RE.finditer(body):
        if len(_content_words(body[:m.start()])) >= 2:      # a one-word 'head' is an adjective list ('heterogeneous, partially ...'), not a head noun phrase
            head, cut = body[:m.start()], m
            break
    head = head.strip()
    if cut is None:
        head = body
    return _cap_words(head, _MAX_CHUNK_WORDS) if head else _cap_words(system, _MAX_CHUNK_WORDS)


# --- query hygiene (v4 / P2) -------------------------------------------------------------------------------------------------
# A word-count truncation or a clause split can leave a lexical query with a dangling function word ('... lipid sorting and early'),
# a trailing adverb ('... during mechanically'), or a leftover research verb / nominalised action at its start ('Capturing coupled
# motions ...', 'comparison of empty and loaded capsids').  OpenAlex lexical search requires (stemmed) terms to match, so such words
# only shrink recall.  The rules below are closed-class / morphological -- no vocabulary of any particular system or benchmark.
_TRAIL_FUNC = _LEX_STOPWORDS | frozenset(
    "during while between across among than that which when how whether about toward towards under over after before versus vs "
    "but so if both either its their this these those whose where upon within without against "
    "early late very more most less further also only just".split())
_LEAD_CONJ = frozenset("and or but so while whereas then also plus".split())
_ADVERB_TAIL_RE = re.compile(r"(?:ically|ally|edly|ously|ively|ently|antly|ably|ibly|fully|ingly|ately)$", re.I)
_ACTION_VERBS = (_PROCESS_VERBS + r"|quantif(?:y|ies|ying)|captur(?:e|es|ing)|elucidat(?:e|es|ing)|uncover(?:s|ing)?|reveal(?:s|ing)?|"
                 r"describ(?:e|es|ing)|obtain(?:s|ing)?|measur(?:e|es|ing)|analy[sz](?:e|es|ing)|distinguish(?:es|ing)?|"
                 r"rationali[sz](?:e|es|ing)|represent(?:s|ing)?|reproduc(?:e|es|ing)|prob(?:e|es|ing)|clarif(?:y|ies|ying)")
_ACTION_NOUNS = (r"(?:comparison|analysis|characteri[sz]ation|investigation|simulation|modell?ing|quantification|elucidation)"
                 r"\s+(?:of|between|across|with|for)")
_ACTION_LEAD_RE = re.compile(r"^\s*(?:(?:" + _ACTION_VERBS + r")|" + _ACTION_NOUNS + r")\s+(?:(?:whether|how|the|of|between|across|with|for|to)\s+)*", re.I)


def _is_trailing_adverb(word: str) -> bool:
    w = word.strip(".,;:()")
    return len(w) >= 7 and bool(_ADVERB_TAIL_RE.search(w))


def _clean_chunk(text: str, hygiene: bool, *, leading: bool = True) -> str:
    """P2.  Remove, from one concept chunk: leading conjunctions and leftover action verbs/nominalisations (`leading=True`), and
    trailing function words / adverbs.  Never returns an empty string (falls back to the input).  No-op when `hygiene` is False."""
    if not hygiene or not (text or "").strip():
        return text
    t = text.strip()
    if leading:
        for _ in range(3):
            toks = t.split()
            while toks and toks[0].lower().strip(".,;:()") in _LEAD_CONJ:
                toks.pop(0)
            t2 = _ACTION_LEAD_RE.sub("", " ".join(toks), count=1).strip()
            if t2 == t or not t2:
                t = t2 or " ".join(toks) or t
                break
            t = t2
    toks = t.split()
    while toks and (toks[-1].lower().strip(".,;:()") in _TRAIL_FUNC or _is_trailing_adverb(toks[-1])):
        toks.pop()
    out = " ".join(toks).strip(" ,;")
    return out or text.strip()


# --- short-context anchoring (v4 / P1) ---------------------------------------------------------------------------------------
# A single common word ('membrane', 'intracellular', 'cholesterol') searched alone -- `membrane coarse-grained model` -- no longer
# says anything about the user's system, and the legacy marker extractor (evidence_plan._CONTEXT_LEXICAL_MARKERS) produces
# exactly such words by construction.  They are searched together with the system head instead.  A single token that IS an entity
# (digits, Greek letters, CamelCase / several capitals: 'HP1α', 'PHC3', 'GlnBP') keeps its own query -- it is rare enough to be
# informative alone -- and every context term of two or more content words keeps its own standalone query too.
_GENERIC_ACRONYMS = frozenset("dna rna mrna trna atp gtp adp gdp cg aa md cgmd pdb nmr".split())
_MARKER_PROVENANCE = "system_description_lexical_marker"


def _is_entity_like(term: str) -> bool:
    for tok in re.split(r"\s+", (term or "").strip()):
        core = tok.strip(".,;:()")
        if not core:
            continue
        if re.search(r"[0-9Ͱ-Ͽἀ-῿]", core):                 # digit or Greek letter
            return True
        if re.search(r"[a-z][A-Z]", core):                                      # CamelCase
            return True
        if sum(ch.isupper() for ch in core) >= 2 and core.lower().rstrip("s") not in _GENERIC_ACRONYMS:
            return True
    return False


def _is_low_information_context(term: str, provenance: Optional[dict]) -> bool:
    if (provenance or {}).get("source") == _MARKER_PROVENANCE:
        return True
    return len(_content_words(term)) == 1 and not _is_entity_like(term)


def _anchor_to_head(head: str, term: str) -> str:
    """`{system head} {words of term not already in the head}` (word-level, case/hyphen-insensitive)."""
    have = {_squash(w) for w in head.split()}
    extra = [w for w in term.split() if _squash(w) and _squash(w) not in have]
    return f"{head} {' '.join(extra)}".strip() if extra else head


def _process_core(term: str, hygiene: bool = False) -> str:
    """The process phrase without its imperative verb ('Simulate ...') and without a trailing 'using coarse-grained
    methods' clause (the CG suffix expresses that once); capped at _MAX_CHUNK_WORDS content words.
    `hygiene` (P2): also drop leftover actions / dangling function words and adverbs (see _clean_chunk)."""
    t = _strip_conversational_wrapper(term or "").strip()
    for _ in range(2):
        t2 = _PROCESS_LEADING_VERB_RE.sub("", t, count=1).strip()
        if not t2:
            break
        t = t2
    t = _PROCESS_CLAUSE_END_RE.split(t, maxsplit=1)[0].strip() or t
    t2 = _PROCESS_TRAILING_METHOD_RE.sub("", t).strip(" ,;")
    t = _LEADING_ARTICLE_RE.sub("", t2 or t).strip()
    return _clean_chunk(_cap_words(t, _MAX_PROCESS_WORDS) or t, hygiene)


def _with_cg_suffix(chunk: str) -> str:
    return chunk if re.search(r"coarse[- ]grain", chunk, re.I) else f"{chunk} {_CG_SUFFIX}"


def _redundant(chunk: str, seen: list[str]) -> bool:
    """True if this chunk repeats (contains / is contained in) an already-emitted chunk, ignoring case and hyphenation"""
    a = _squash(chunk)
    return bool(a) and any(a in b or (len(b) >= 4 and b in a) for b in (_squash(x) for x in seen) if b)


_PLAIN_CONNECTORS = frozenset({"with", "and", "in", "on", "plus", "including", ",", ";"})
_MAX_COMBO_WORDS = 12     # safety ceiling for the system+process combination query (the real limit is the chunk budget)


def _system_chunks(system: str) -> tuple[str, list[str]]:
    """Split a system description into (head noun phrase, post-modifier chunks).

    Chunks are cut at the same connectors as _system_head ('with', 'and', 'in', ',', relational participles such as
    'invaded by' / 'forming' ...). A plain preposition/conjunction is dropped from the chunk; a SUBSTANTIVE connector
    ('invaded by', 'forming', 'associated with') is kept because it names the relation ('invaded by a third strand').
    Descriptions of <= _MAX_SYSTEM_WORDS content words are returned whole, without modifiers."""
    system = _strip_conversational_wrapper(system or "").strip()
    body = _LEADING_ARTICLE_RE.sub("", system)
    if len(_content_words(system)) <= _MAX_SYSTEM_WORDS:
        return (body.strip() or system), []
    matches = list(_SYSTEM_CONNECTOR_RE.finditer(body))
    first = next((m for m in matches if len(_content_words(body[:m.start()])) >= 2), None)
    if first is None:
        return _system_head(system), []
    head = _cap_words(body[:first.start()].strip(), _MAX_CHUNK_WORDS)
    rest = [m for m in matches if m.start() >= first.start()]
    chunks: list[str] = []
    for i, m in enumerate(rest):
        end = rest[i + 1].start() if i + 1 < len(rest) else len(body)
        seg = body[m.end():end].strip(" ,;")
        conn = m.group(0).strip(" ,;").lower()
        text = f"{conn} {seg}".strip() if conn and conn not in _PLAIN_CONNECTORS else seg
        text = _LEADING_ARTICLE_RE.sub("", text).strip()
        if _content_words(text):
            chunks.append(_cap_words(text, _MAX_CHUNK_WORDS))
    return head, chunks


def _system_part(system: str, covered: Optional[list[str]]) -> str:
    """The system text used in a flat query: the head noun phrase, plus AT MOST ONE post-modifier chunk (the one nearest the
    head) that no other emitted query already carries. `covered` = the texts of the other emitted chunks (context terms, process
    phrase); None means 'no coverage information' and yields the head alone (the v2 behaviour)."""
    head, chunks = _system_chunks(system)
    if covered is None:
        return head
    extra = next((c for c in chunks if not _redundant(c, covered)), None)
    return f"{head} {extra}" if extra else head


def build_system_specific_cg_queries(
    plan: EvidenceRoutePlan, budgets: RouteRetrievalBudgets, covered: Optional[list[str]] = None
) -> list[RouteQuery]:
    """`{system head} coarse-grained model` -- system-specific CG discovery.

    The system description is reduced to its head noun phrase (see _system_head) so a long description such as
    'a tetra-nucleosome array with HP1a and a truncated Polyhomeotic-like protein' becomes 'tetra-nucleosome array';
    the remaining components reach OpenAlex through their OWN context queries instead of all being required in
    one lexical query. History: replaced the old direct_system route (a bare `{system}` query pulled mostly
    experimental literature); one query per grounded system term; conversational wrapper stripped.

    v3: when `covered` (the other emitted chunks) is given, ONE post-modifier chunk that no other query carries
    (e.g. 'invaded by a third strand') is kept after the head, so scientifically discriminating information is not
    lost; modifiers already searched through their own context query are not repeated here."""
    out = []
    system = _clean_chunk(_system_part(plan.system_description or "", covered), budgets.flat_query_hygiene, leading=False)
    if not system:
        return out
    out.append(RouteQuery(
        route="system_specific_cg", route_term=system, route_term_source="USER_GROUNDED",
        query=_with_cg_suffix(system), query_role="system_specific_cg_evidence",
    ))
    return out


def _with_system_prefix(system: str, term: str) -> str:
    """`{system} {term}` -- unless system is already a substring of term (or
    vice versa), in which case term alone is returned unchanged. Same
    overlap check already used for system_spans dedup in
    the old ConceptResolverSession path's build_evidence_route_plan_from_
    session() (case-insensitive substring either direction), reused here
    rather than reinvented, so e.g. objective="pertactin folding" (which
    already contains the system name verbatim) doesn't produce a query
    with "pertactin" repeated twice."""
    system = system.strip()
    term = term.strip()
    if not system:
        return term
    if system.lower() in term.lower() or term.lower() in system.lower():
        return term
    return f"{system} {term}"


def build_process_queries(plan: EvidenceRoutePlan, budgets: RouteRetrievalBudgets,
                          system_part: Optional[str] = None) -> list[RouteQuery]:
    """`{process} coarse-grained model` (v2: short, system prefix only when the process phrase cannot stand alone),
    PLUS (v3) one system+process combination query when the pair carries joint information.

    The combination `{system part} {process phrase} coarse-grained model` is emitted only when the process phrase is
    descriptive (> _SHORT_PROCESS_WORDS content words; a shorter one already gets the system prefix, which IS the
    combination) and does not already contain the system head (otherwise it would be the same query). `system_part` is the
    head plus at most one uncovered modifier chunk (see _system_part); it never includes context terms. Length is bounded
    by the chunk budget (head + <= 1 modifier + process phrase) with a _MAX_COMBO_WORDS safety ceiling."""
    out = []
    head = _system_head(plan.system_description or "")
    sys_part = system_part or head
    for t in _cap_terms(plan.process_terms, budgets.max_queries_per_route):
        core = _process_core(t.term, budgets.flat_query_hygiene)
        term = _strip_conversational_wrapper(t.term)
        short = len(_content_words(core)) <= _SHORT_PROCESS_WORDS
        text = _with_system_prefix(head, core) if short else core
        out.append(RouteQuery(
            route="process", route_term=term, route_term_source=t.source,
            query=_with_cg_suffix(text), query_role="process_cg_evidence",
        ))
        if not short and sys_part and not _redundant(head, [core]):
            room = max(_MAX_COMBO_WORDS - len(_content_words(sys_part)), 3)
            combo = f"{sys_part} {_clean_chunk(_cap_words(core, room), budgets.flat_query_hygiene)}"
            out.append(RouteQuery(
                route="process", route_term=term, route_term_source=t.source,
                query=_with_cg_suffix(combo), query_role="system_process_cg_evidence",
            ))
    return out[:budgets.max_queries_per_route]


def build_context_queries(plan: EvidenceRoutePlan, budgets: RouteRetrievalBudgets) -> list[RouteQuery]:
    """One short query per DISTINCT context term (v2): `{term} coarse-grained model`, plus `{term} {process}` ONLY when
    the process phrase is generic/short (<= _SHORT_PROCESS_WORDS content words, e.g. 'folding'); a descriptive
    process phrase is searched on its own by the process route (preferred first).

    Context terms are no longer expected to arrive as one concatenated string (the planner bridge now passes each
    search_context item separately), and a term that merely repeats the system head, the process phrase or an
    already-emitted context term (ignoring case/hyphenation) is skipped -- redundant lexical queries cost
    requests and add no new entry point. Grounded-context logic is unchanged (USER_GROUNDED / LITERATURE_GROUNDED
    only, empty when there is no context term); wrapper stripped from every term."""
    if not plan.context_terms:
        return []
    hyg = budgets.flat_query_hygiene
    process_core = _process_core(plan.process_terms[0].term, hyg) if plan.process_terms else None
    head = _system_head(plan.system_description or "")
    seen = [x for x in (head, process_core) if x]

    preferred: list[RouteQuery] = []
    fallback: list[RouteQuery] = []
    for t in plan.context_terms:
        term = _clean_chunk(_strip_conversational_wrapper(t.term), hyg)
        # (v3 wrote the CG test as r"\\bCG\\b", which matches a literal backslash and so never fired; P2 fixes the escape)
        cg_named = re.search(r"coarse[- ]grain|\bCG\b" if hyg else r"coarse[- ]grain|\\bCG\\b", term, re.I)
        if not term.strip() or _redundant(term, seen) or cg_named:
            continue                      # empty / repeats an emitted chunk / names the CG method itself (the suffix already covers it)
        seen.append(term)
        if budgets.flat_anchor_short_context and head and _is_low_information_context(term, t.provenance):
            # P1: a bare generic word ('membrane') says nothing about the system -> search it together with the system head
            fallback.append(RouteQuery(
                route="context", route_term=term, route_term_source=t.source,
                query=_with_cg_suffix(_anchor_to_head(head, term)), query_role="context_anchored_cg_evidence",
            ))
            continue
        if process_core and not _redundant(process_core, [term]) and len(_content_words(process_core)) <= _SHORT_PROCESS_WORDS and \
                len(_content_words(term)) + len(_content_words(process_core)) <= _MAX_CHUNK_WORDS:
            preferred.append(RouteQuery(
                route="context", route_term=term, route_term_source=t.source,
                query=_with_cg_suffix(f"{term} {process_core}"), query_role="context_process_evidence",
            ))
        fallback.append(RouteQuery(
            route="context", route_term=term, route_term_source=t.source,
            query=_with_cg_suffix(term), query_role="context_cg_evidence",
        ))
    ordered = preferred + fallback
    return ordered[:budgets.max_queries_per_route]


# --- need_semantic (v4 / P3) -------------------------------------------------------------------------------------------------
# The 3A / Hop-1 semantic queries are written about a CLASS ("membrane-confined composite cytoskeletal networks"), so the specific
# entities of the request (tau, crosslinkers, HP1a, a subunit isoform ...) never reach a semantic search.  This route issues ONE
# semantic query built straight from the planner's own system + objective (deterministic string assembly, no LLM, independent of
# ExpertPrior), so the specific need is searched as a whole sentence, the way an embedding retriever expects it.  It adds to, and
# does not replace, every other route.
_ACTION_LEAD_BARE_RE = re.compile(r"^\s*(?:(?:" + _ACTION_VERBS + r")|" + _ACTION_NOUNS + r")\s+(?:the\s+)?", re.I)


def _objective_phenomenon(objective: str) -> str:
    """The objective without its leading research verb ('Determine how X ...' -> 'how X ...') and without a trailing 'using
    coarse-grained ...' clause (the sentence already says coarse-grained); first letter lower-cased unless it starts an entity."""
    t = _strip_conversational_wrapper(objective or "").strip().rstrip(". ")
    for _ in range(2):
        t2 = _ACTION_LEAD_BARE_RE.sub("", t, count=1).strip()
        if not t2 or t2 == t:
            break
        t = t2
    t = _PROCESS_TRAILING_METHOD_RE.sub("", t).strip(" ,;")
    first = t.split()[0] if t.split() else ""
    if t and not _is_entity_like(first) and not first.isupper():
        t = t[0].lower() + t[1:]
    return t


def build_need_semantic_queries(plan: EvidenceRoutePlan, budgets: RouteRetrievalBudgets) -> list[RouteQuery]:
    """`Coarse-grained models of {system} to model {objective phenomenon}.` -- one query, or [] when disabled / nothing to say.
    With `need_semantic_include_context` (control arm only) the distinct, non-redundant context terms are appended."""
    if not budgets.need_semantic_enabled:
        return []
    system = _LEADING_ARTICLE_RE.sub("", _strip_conversational_wrapper(plan.system_description or "").strip()).strip().rstrip(".")
    if not system:
        return []
    phen = _objective_phenomenon(plan.scientific_objective or "")
    q = f"Coarse-grained models of {system} to model {phen}." if phen else f"Coarse-grained models of {system}."
    role = "need_semantic_evidence"
    if budgets.need_semantic_include_context:
        seen = [system, phen]
        ctx: list[str] = []
        for t in plan.context_terms:
            term = _clean_chunk(_strip_conversational_wrapper(t.term), True)
            if term.strip() and not _redundant(term, seen) and not re.search(r"coarse[- ]grain|\bCG\b", term, re.I):
                seen.append(term)
                ctx.append(term)
        if ctx:
            q = q[:-1] + ". Relevant aspects: " + "; ".join(ctx[:3]) + "."
            role = "need_semantic_with_context_evidence"
    return [RouteQuery(route=NEED_SEMANTIC_ROUTE, route_term=system, route_term_source="USER_GROUNDED", query=q, query_role=role)]


# --- structural branch: primary-term selection -----------------------------

def select_primary_structural_term(structural_terms: list[RouteTerm]) -> tuple[Optional[RouteTerm], list[RouteTerm]]:
    """Deterministic priority: USER_GROUNDED > LITERATURE_GROUNDED > original
    EvidenceRoutePlan order (as the tiebreak within a tier). Returns
    (primary, secondary_terms_for_audit). structural_terms is guaranteed by
    evidence_plan's own contract to contain only USER_GROUNDED /
    LITERATURE_GROUNDED terms (never PLANNER_DERIVED).
    Hop-1 is run ONCE from the primary term only; TerminologyExpander is
    already responsible for widening structural terminology, so secondary
    terms are preserved for audit, never re-run through Hop-1 separately."""
    if not structural_terms:
        return None, []
    for source in ("USER_GROUNDED", "LITERATURE_GROUNDED"):
        for i, t in enumerate(structural_terms):
            if t.source == source:
                secondary = structural_terms[:i] + structural_terms[i + 1:]
                return t, secondary
    return structural_terms[0], structural_terms[1:]


# --- plan builder ------------------------------------------------------------

def build_route_retrieval_plan(
    evidence_plan: EvidenceRoutePlan, budgets: Optional[RouteRetrievalBudgets] = None
) -> RouteRetrievalPlan:
    budgets = budgets or RouteRetrievalBudgets()

    # CG-model-recommendation retrieval design (this session):
    #   - system_specific_cg  "{system} coarse-grained model" -- replaced
    #     the old direct_system route, which is DELETED (its builder
    #     build_direct_system_queries() and its "direct_system" FlatRoute
    #     literal are gone). Only the field
    #     EvidenceRoutePlan.direct_system_terms remains in
    #     evidence_plan.py (kept so the plan schema is unchanged); nothing
    #     fills or reads it any more.
    #   - process             "{system} {process} coarse-grained model"
    #   - context             unchanged grounded-context logic
    #   - modeling_requirement STILL disabled (no queries are built; the
    #     route name stays in FLAT_ROUTES) -- its query builder was archived
    #     (git tag `legacy-pre-cgscout`). "{planner term} coarse-grained
    #     model" queries were too generic to surface stable candidates.
    # v3: the context queries are decided FIRST so the system query / combination query know which post-modifier chunks of the
    # system description are already searched through their own context query (and so need not be repeated) -- see _system_part.
    context_queries = build_context_queries(evidence_plan, budgets)
    process_core = _process_core(evidence_plan.process_terms[0].term, budgets.flat_query_hygiene) if evidence_plan.process_terms else None
    covered = [q.route_term for q in context_queries] + ([process_core] if process_core else [])
    system_part = _system_part(evidence_plan.system_description or "", covered)

    route_queries: list[RouteQuery] = []
    system_queries = build_system_specific_cg_queries(evidence_plan, budgets, covered)
    qc = None
    if budgets.query_construction_v2:
        from cgscout.retrieval.direct.query_construction_v2 import analyze, process_queries_v2       # lazy: query_construction_v2 imports this module
        qc = analyze(evidence_plan, budgets, kept=[q.query for q in system_queries + context_queries])
        process_queries = process_queries_v2(qc, budgets, n_kept=len(system_queries) + len(context_queries))
    else:
        process_queries = build_process_queries(evidence_plan, budgets, system_part)
    route_queries += system_queries
    route_queries += process_queries
    route_queries += context_queries
    # Only an exact repeat INSIDE the same route is dropped (its provenance entry would be identical). The same string in two
    # different routes is kept on purpose: each route must record its own provenance on the papers it finds, and the second
    # execution is served by the OpenAlex request cache (no HTTP request, no credits).
    seen_in_route: set[tuple[str, str]] = set()
    deduped: list[RouteQuery] = []
    for q in route_queries:
        key = (q.route, _squash(q.query))
        if key not in seen_in_route:
            seen_in_route.add(key)
            deduped.append(q)
    route_queries = deduped
    if qc is not None:
        from cgscout.retrieval.direct.query_construction_v2 import need_semantic_query_v2
        route_queries += need_semantic_query_v2(qc, budgets)                 # P3 (empty unless budgets.need_semantic_enabled)
    else:
        route_queries += build_need_semantic_queries(evidence_plan, budgets)     # P3 (empty unless budgets.need_semantic_enabled)

    primary, secondary = select_primary_structural_term(evidence_plan.structural_terms)

    plan = RouteRetrievalPlan(
        system_description=evidence_plan.system_description,
        scientific_objective=evidence_plan.scientific_objective,
        route_queries=route_queries,
        structural_branch_enabled=primary is not None,
        structural_branch_structural_class=primary.term if primary else None,
        structural_branch_structural_class_source=primary.source if primary else None,
        structural_branch_secondary_terms=[{"term": t.term, "source": t.source} for t in secondary],
        budgets=budgets,
        empty_routes=list(evidence_plan.empty_routes),
    )
    return plan


# --- execution: flat routes --------------------------------------------------

FlatSearchFn = Callable[[str, int], list]  # (query, limit) -> list[Paper]; may raise


def _default_flat_search(query: str, limit: int) -> list:
    from cgscout.retrieval.search import search_and_dedupe
    return search_and_dedupe(query, limit=limit)


class _Accumulator:
    """Global cap semantics: max_total_unique_papers limits only the
    addition of NEW unique paper_refs. An already-seen paper_ref ALWAYS
    merges a new RouteProvenanceEntry, cap or no cap -- cross-route
    provenance/convergence is never erased by the global budget."""

    def __init__(self, budgets: RouteRetrievalBudgets):
        self.budgets = budgets
        self.papers: dict[str, RetrievedPaper] = {}
        self.order: list[str] = []

    def add(self, ref: str, title: str, year: Optional[int], entry: RouteProvenanceEntry,
             hop1_provenance: Optional[Hop1Provenance], route_new_count: int,
             route_cap: Optional[int]) -> tuple[bool, bool]:
        """Returns (accepted, was_new_unique). `route_new_count` is how many
        NEW unique papers this route has already contributed. `route_cap` is
        the max it may still contribute -- pass None for "no per-route cap,
        only the global max_total_unique_papers applies" (the structural
        branch: with one query per system now, its true output volume scales
        with how many systems Hop-1 discovered, not with a route concept
        that was only ever designed for a handful of flat queries; capping
        it at the SAME number as one flat route silently discarded almost
        everything from every system but the very first one processed --
        diagnosed against a real case, see the old structural branch of this
        module, git tag `legacy-pre-cgscout`). Flat routes pass budgets.max_papers_per_route
        explicitly, unchanged from before."""
        if ref in self.papers:
            self.papers[ref].found_by.append(entry)
            if hop1_provenance and not self.papers[ref].hop1_provenance:
                self.papers[ref].hop1_provenance = hop1_provenance
            self._add_hop1_provenance(ref, hop1_provenance)
            return True, False
        # new unique candidate -- gated by the route cap (if any) and the global cap
        if route_cap is not None and route_new_count >= route_cap:
            return False, False
        if len(self.papers) >= self.budgets.max_total_unique_papers:
            return False, False
        rp = RetrievedPaper(paper_ref=ref, title=title, year=year, found_by=[entry],
                             hop1_provenance=hop1_provenance)
        self.papers[ref] = rp
        self.order.append(ref)
        self._add_hop1_provenance(ref, hop1_provenance)
        return True, True

    def _add_hop1_provenance(self, ref: str, hop1_provenance: Optional[Hop1Provenance]) -> None:
        """Appends to hop1_provenances -- the FULL set of discovered systems
        that independently retrieved this paper -- deduped by
        discovered_system name so re-adding the same system's claim is a
        no-op. Independent of (and never overwrites) the first-claimant-only
        `hop1_provenance` field above."""
        if hop1_provenance is None:
            return
        rp = self.papers[ref]
        if any(p.discovered_system == hop1_provenance.discovered_system for p in rp.hop1_provenances):
            return
        rp.hop1_provenances.append(hop1_provenance)


def _run_flat_route(
    route: FlatRoute, queries: list[RouteQuery], search_fn: FlatSearchFn,
    budgets: RouteRetrievalBudgets, acc: _Accumulator,
) -> RouteOutcome:
    outcome = RouteOutcome(route=route)
    if not queries:
        outcome.empty = True
        outcome.empty_reason = "not_applicable_no_grounded_term"
        return outcome

    outcome.route_term = queries[0].route_term
    n_new_unique = 0
    any_papers_returned = False
    for rq in queries:
        outcome.queries_tried.append(rq.query)
        try:
            papers = search_fn(rq.query, budgets.max_papers_per_query)
        except Exception as exc:  # noqa: BLE001 -- one query's failure never blocks the others
            outcome.search_errors.append(f"{rq.query!r}: {exc}")
            continue
        for rank, p in enumerate(papers):
            any_papers_returned = True
            ref = _paper_ref(p)
            entry = RouteProvenanceEntry(
                route=route, route_term=rq.route_term, query=rq.query,
                query_role=rq.query_role, retrieval_rank=rank,
            )
            accepted, was_new = acc.add(ref, p.title or "", getattr(p, "year", None),
                                         entry, None, n_new_unique, budgets.max_papers_per_route)
            if accepted:
                outcome.n_papers_found += 1
                if was_new:
                    n_new_unique += 1
                    outcome.n_new_unique_contributed += 1
            else:
                if n_new_unique >= budgets.max_papers_per_route:
                    outcome.n_dropped_route_cap += 1
                else:
                    outcome.n_dropped_global_cap += 1

    if outcome.search_errors and not any_papers_returned:
        outcome.empty = True
        outcome.empty_reason = "search_error"
    elif not any_papers_returned:
        outcome.empty = True
        outcome.empty_reason = "queries_returned_zero_papers"
    return outcome


# --- TerminologyExpander input: deterministic surface normalization --------
#
# Diagnosed (run_terminology_input_diagnostic.py, 3 reps/input, 100% stable
# each): expand_terminology("beta-helix protein") -> 5 terms including
# "autotransporter passenger domain" / "beta-helix adhesin"; expand_
# terminology("I have a beta-helix protein") -> only 3 terms, family-crossing
# terms absent. The bare fold name ("beta-helix") ALSO only produces 3 terms
# -- so neither of the two inputs actually available in the live pipeline
# (structural_class alone, or system_description with its conversational
# wrapper) reaches the richer output; only the clean noun phrase does.
#
# No existing field on EvidenceRoutePlan/RouteRetrievalPlan carries that
# clean phrase -- ConceptResolverSession.system_span does (it's exactly
# "beta-helix protein" in the diagnosed live case), but route_retrieval_
# plan.py has no session dependency by design, and threading it through
# would mean a new field on two schemas plus a live-adapter change, for a
# fix scoped to one call site. This deterministic stripper is the smaller,
# self-contained alternative: a narrow, closed set of first-person
# conversational/intent PREFIXES (not a biology ontology, not a
# beta-helix-specific mapping) stripped from system_description before it
# reaches TerminologyExpander only -- structural_class, Hop-1, Hop-2, and
# banned_norms grounding all keep using structural_class exactly as before,
# untouched by this normalization.
# A leading article, IF present, must be followed by whitespace -- written
# as `(?:(?:an|a|the|some)\s+)?` (NOT `(?:an|a|the|some)?\s*`) so the bare
# "a" cannot eat the first letter of a content word that happens to start
# with it ("...simulate alpha-synuclein" -> "lpha-synuclein" was exactly
# this bug). "an" still precedes "a" so "an X" strips fully.
_ART = r"(?:(?:an|a|the|some|any|its|their|this|that|our)\s+)?"
_STUDY_VERB = (
    r"(?:study|investigate|explore|understand|examine|analy[sz]e"
    r"|characteri[sz]e|model|simulate|look\s+at|work\s+on|focus\s+on)"
)
# The bare framing head, with OR without a leading pronoun -- A frequently
# grounds an objective span as "want to study its folding" (shortest span,
# no "I"; see A's own prompt), so both "I want to study X" and "want to
# study X" must strip.
_WANT_TO = r"(?:want|wish|would\s+like|would\s+love|need|hope|plan|aim)\s+to\s+"
_CONVERSATIONAL_PREFIX_RE = re.compile(
    # Widened (this session) beyond the original structural-branch use: now
    # ALL flat-route query builders strip this from their term before it
    # reaches OpenAlex. New pieces vs the original: the "simulate"/"look
    # at"/... study verbs, "I('m) interested in", a bare (pronoun-less)
    # "want to study ...", and the "could/can you recommend|suggest|find
    # ... model(s) for/of" assistant-request framing -- still a narrow,
    # closed set of conversational prefixes, no biology ontology, no
    # per-system rules.
    r"^\s*(?:"
    r"i\s+(?:have|had)\s+" + _ART
    + r"|(?:i\s+)?" + _WANT_TO + _STUDY_VERB + r"\s+" + _ART
    + r"|i(?:'m|\s+am)\s+(?:studying|investigating|exploring|working\s+on|interested\s+in|looking\s+at|trying\s+to\s+" + _STUDY_VERB + r")\s+" + _ART
    + r"|we\s+(?:have|had)\s+" + _ART
    + r"|we\s+" + _WANT_TO + _STUDY_VERB + r"\s+" + _ART
    + r"|(?:could|can|would|will)\s+you\s+(?:please\s+)?(?:recommend|suggest|find|help\s+me\s+(?:find|with|identify)|point\s+me\s+(?:to|toward)|tell\s+me\s+about)\s+(?:me\s+)?" + _ART + r"(?:good\s+|suitable\s+|appropriate\s+)?(?:coarse[- ]?grained\s+|cg\s+)?models?\s+(?:for|of|to\s+" + _STUDY_VERB + r")\s+"
    + r"|(?:please\s+)?(?:recommend|suggest|find)\s+(?:me\s+)?" + _ART + r"(?:coarse[- ]?grained\s+|cg\s+)?models?\s+(?:for|of)\s+"
    + r")",
    re.IGNORECASE,
)


def _strip_conversational_wrapper(text: str) -> str:
    """Strip a narrow, closed set of first-person conversational/intent
    prefixes that carry no grounded biological content ('I have a...',
    'I want to study...', 'could you recommend a CG model for...', ...).
    Used by BOTH the structural branch (so TerminologyExpander sees a clean
    noun phrase) AND every flat-route query builder (so the OpenAlex query
    is normalized scientific content, not the user's raw sentence).
    Repeated application handles a stacked prefix; never strips down to an
    empty string -- falls back to the original text if it would.
    Prefix-only; does not attempt to strip trailing clauses."""
    if not text:
        return text
    out = text.strip()
    while True:
        new = _CONVERSATIONAL_PREFIX_RE.sub("", out, count=1).strip()
        if new == out or not new:  # no further match, or would strip to nothing
            break
        out = new
    return out or text


# --- orchestrator -------------------------------------------------------------

def run_route_retrieval(
    evidence_plan: EvidenceRoutePlan,
    *,
    flat_search_fn: Optional[FlatSearchFn] = None,
    budgets: Optional[RouteRetrievalBudgets] = None,
    call: Optional[CallFn] = None,
    hop1_search_fn: Optional[StructuralSearchFn] = None,
    hop1_extract_fn: Optional[ExtractFn] = None,
    hop2_search_fn: Optional[StructuralSearchFn] = None,
    hop1_progress_fn: Optional[Callable[[int, int], None]] = None,
    need_semantic_search_fn: Optional[FlatSearchFn] = None,
) -> RouteRetrievalResult:
    """Run all five routes (plus the optional P3 `need_semantic` route when budgets.need_semantic_enabled). A route with nothing to search, or that errors,
    never blocks the others (each route is independently try/except'd).
    The structural branch (EvidenceRoutePlan.structural_terms non-empty) is no
    longer part of this function: the live path never enables it (the
    structural / related-system work is done by orchestrator), and
    a plan that enables it raises NotImplementedError. The parameters `call`,
    `hop1_search_fn`, `hop1_extract_fn`, `hop2_search_fn` and `hop1_progress_fn`
    are accepted for signature compatibility but not used."""
    t0 = time.time()
    budgets = budgets or RouteRetrievalBudgets()
    flat_search_fn = flat_search_fn or _default_flat_search

    plan = build_route_retrieval_plan(evidence_plan, budgets)
    acc = _Accumulator(budgets)
    per_route: list[RouteOutcome] = []

    for route in FLAT_ROUTES:
        queries = [q for q in plan.route_queries if q.route == route]
        per_route.append(_run_flat_route(route, queries, flat_search_fn, budgets, acc))

    need_queries = [q for q in plan.route_queries if q.route == NEED_SEMANTIC_ROUTE]
    if need_queries:
        if need_semantic_search_fn is None:
            raise ValueError("budgets.need_semantic_enabled is set but no need_semantic_search_fn was provided")
        limit = budgets.need_semantic_limit
        per_route.append(_run_flat_route(NEED_SEMANTIC_ROUTE, need_queries, lambda q, _l: need_semantic_search_fn(q, limit), budgets, acc))

    structural_result: Optional[Hop2Result] = None
    if plan.structural_branch_enabled:
        raise NotImplementedError(
            "the structural (Hop-1/Hop-2) branch of run_route_retrieval() was removed (historical code, git tag "
            "`legacy-pre-cgscout`); the live path never enables it -- planner_retrieval_bridge always clears structural_terms."
        )
    per_route.append(RouteOutcome(
        route="structural", empty=True, empty_reason="not_applicable_no_grounded_term",
    ))

    papers = [acc.papers[ref] for ref in acc.order]
    n_multi = sum(1 for p in papers if len(p.found_by) > 1)
    routes_zero = [o.route for o in per_route if o.empty]

    return RouteRetrievalResult(
        plan=plan, per_route=per_route, papers=papers,
        n_unique_papers=len(papers), n_papers_multi_route=n_multi,
        structural_branch_result=structural_result,
        routes_with_zero_results=routes_zero,
        retrieval_seconds=round(time.time() - t0, 2),
    )
