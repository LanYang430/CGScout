"""Stage 4 (v2): LLM scientific-relevance evaluation of the Stage-3-retained papers.

Pipeline position (BEAM v2):

    Stage 2 retrieval -> Stage 3 cheap filter (keyword gate + simulation classifier)
        -> Stage 4 LLM relevance evaluation + ranking   <- THIS MODULE
        -> final recommendations

Stage 3 only removes the obviously unsuitable (it has no signal about topical relevance); it
cannot decide scientific relevance. This module does, with an explicit graded rubric, an
evidence-based explanation for every deeply-evaluated paper, and a final rank that comes from
that assessment -- never from retrieval route or publication year (those only break ties).

Reuse of the old Stage 4a (beam_agent/evidence_extraction.py): its query-independent extraction
prompt, user-message builder, output field semantics (PaperEvidence method / system / processes)
and its content-hash caching idea are reused unchanged; the LLM transport is
LLMBackend.structured(). The old Stage 4b/5 keyword matching (evidence_matching.py)
is NOT reused -- it needs the old intake's ScientificConceptProfile and compares by stem overlap.

Evidence first: a Stage-3-kept paper with no usable OpenAlex abstract gets ONE cached Europe PMC lookup
(by DOI, else PMID). If that supplies an abstract the paper goes through Stage 4 normally; if not, it is
recorded internally as `missing_evidence` -- never graded (not 0, not 1), never sent to the LLM on its title
alone, and never shown to the end user (benchmark/debug output can still list it).

Two tiers (see Stage4Config):
  tier 1  batch screen of the retained papers (title + abstract, several papers per call). It is a
          REJECT-ONLY gate: a paper is dropped only on positive evidence that it is irrelevant
          (reject, with a confidence the model itself trusts). Lack of evidence is not evidence of
          irrelevance, so `pass`, `insufficient_evidence` (an abstract exists but cannot tell) and
          any low-confidence reject ALL continue to tier 2.
  tier 2  per-paper deep evaluation (cached query-independent fact extraction + graded judgment
          with a verbatim supporting quote) of every paper that was not dropped (capped only as a
          safety valve). If the text still cannot support any grade, tier 2 reports
          insufficient evidence, recorded internally as `missing_evidence` -- never a grade and never a
          relevance rejection.
Tier 1 does not grade; the graded relevance ranking comes only from tier 2.

Failures are never silent grades: a paper whose LLM call fails is status="error" (reported, not
ranked, never treated as "not relevant").
"""
from __future__ import annotations

import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Literal, Optional

from pydantic import BaseModel, Field

from cgscout import cache
from cgscout.models import Paper, ProcessEvidenceItem
from cgscout.llm import LLMBackend

# ===== merged from `assessment/evidence_extraction.py` (moved verbatim; its docstring is this comment) =====================
# Stage 4 per-paper extraction prompt (P8) and its user-message builder.
#
# stage4.py reuses, unchanged, this query-independent extraction prompt
# (`_SYSTEM_PROMPT`, hashed into the Stage 4 extraction cache key by
# `_SYSTEM_PROMPT_HASH`) and the per-paper user message (`_build_user_message`):
# the extractor is given ONLY a paper's title/abstract/metadata -- never the
# user's problem -- so target-concept leakage into the extracted facts is
# structurally impossible and the result is cacheable across queries.
#
# The rest of the old Stage 4a module (the baseline and local-model extractors
# that produced `PaperEvidence`) is historical code, kept in the git tag
# `legacy-pre-cgscout`.
# ===========================================================================================================================

_SYSTEM_PROMPT = """\
You are a scientific literature extraction assistant for BEAM, a tool that helps computational \
biophysicists organize coarse-grained (CG) molecular simulation literature. You are given ONLY a \
paper's title, abstract, and basic metadata -- no search query, no research target, and nothing to \
judge relevance against. Your ONLY job is to extract objective facts this paper's own text states \
about (1) the modeling/simulation method it used, (2) the physical system it studied, and (3) the \
process(es)/capabilities it demonstrated. Do not invent anything not stated -- if the abstract does \
not say something, leave that field empty ([]) or null, never guess.

This extraction is reused across many different future research questions, so it must describe the \
paper itself, not any particular search or target.

METHOD EVIDENCE: model_class is the modeling/method class(es) explicitly named or clearly described \
(e.g. "coarse-grained", "all-atom", "structure-based", "elastic-network", "Go model") -- open \
vocabulary, leave empty if no simulation model/method is actually discussed. If the abstract clearly \
describes a simulation/computational modeling approach being used but does NOT specify its \
resolution/granularity (you cannot tell whether it is coarse-grained, all-atom, or something else), \
do NOT leave model_class empty -- populate it with a generic marker that says so, e.g. "simulation \
(resolution unspecified)" or "molecular dynamics (granularity unspecified)". Leaving model_class \
empty means the paper does not describe any computational method at all -- a materially different, \
stronger claim than "describes one but doesn't say which kind." study_role is what role \
the model played in THIS paper -- "developed" (introduces a new model/method), "applied" (uses an \
existing model to study a system), "benchmarked"/"validated" (tests or compares a model's \
performance/accuracy against something), "reviewed_only" (a review/survey that discusses models \
without running one). Multiple roles can apply. Do NOT judge whether this makes the paper "eligible" \
for anything -- just report what the text says.

SYSTEM EVIDENCE: studied_systems is what system(s) the paper actually studied, simulated, \
benchmarked, developed against, or applied its method to -- kept close to the abstract's own words, \
one entry per distinct system or system-group. Capture identity + structure/architecture + \
qualitative scale together as ONE faithful phrase rather than splitting them apart or normalizing \
into a coarser category -- e.g. "three-helix bundle", "twelve proteins with multiple \
secondary-structure arrangements", "C-terminal segment of the pertactin passenger domain", \
"intrinsically disordered proteins", "protein-ligand complexes". Never infer from general \
biological/domain background what the abstract itself doesn't say -- a protein you happen to know is \
typically membrane-associated is NOT evidence this paper studied it in a membrane, unless the \
abstract itself says so. NEVER include a simulation parameter (duration, temperature, timestep, box \
size) in studied_systems even if a number sits right next to a system description in the same \
sentence -- only include what the text says about the SYSTEM, never about how long or how it was \
simulated ("approximately 9 ms" describes simulation duration, not system size; never let it become \
"9" of anything). A general capability claim about the METHOD (e.g. "this force field can handle \
proteins containing about 1000 residues") is NOT a studied system either, unless the paper's own \
actually-studied examples are that size -- read for what was really used as a benchmark/test/example \
system, not what the method is generically capable of.

family_or_fold is a fold/family name ONLY when the abstract explicitly names one AS a fold/family \
classification (e.g. "TIM-barrel fold", "immunoglobulin fold", "autotransporter passenger domain \
family") -- leave empty otherwise. An inline topology phrase like "three-helix bundle" or \
"three-stranded antiparallel beta-sheet" belongs in studied_systems, not here, even though it \
describes architecture -- do not promote a studied_systems-style description into family_or_fold \
just to have something to put there.

protein_or_system_names is any specific protein/system name explicitly named in the paper's own \
text -- bare names, not full descriptive phrases (those belong in studied_systems). It is completely \
normal for studied_systems and protein_or_system_names to overlap or say the same thing when a name \
is all the abstract gives you.

Leave any system-evidence field empty/null if the paper doesn't specify -- that is the correct, \
honest answer, not a strike against anything.

PROCESS EVIDENCE: processes is a list of {raw_text, normalized_label} pairs, one per process or \
capability the paper's abstract states the model/method was actually shown to study or demonstrate. \
raw_text should stay close to the abstract's own words -- only what is stated, never inferred. \
normalized_label restates the same capability as a short, generic, reusable snake_case label (e.g. \
folding, unfolding, conformational_dynamics, pocket_gating, binding, aggregation, assembly, \
translocation, kinetics, thermodynamics, stability) so the same underlying capability gets the same \
label across different papers -- prefer an existing common term over inventing a near-duplicate, but \
you are not restricted to a fixed list; invent a new concise label if none fit. Do NOT force \
everything into a folding-specific vocabulary -- report whatever process the paper actually studied, \
even if it's something else entirely (e.g. pocket gating, membrane insertion, phase separation).

Do not assess eligibility, evidence strength, confidence, or relevance to any target -- none of that \
is part of your job here.

Respond with a single JSON object matching the given schema, and nothing else."""


def _build_user_message(paper: Paper) -> str:
    return (
        f"Paper to extract evidence from:\n"
        f"Title: {paper.title}\n"
        f"Year: {paper.year or 'unknown'}\n"
        f"Venue: {paper.venue or 'unknown'}\n"
        f"Abstract: {paper.abstract}"
    )


# Computed once at import time (_SYSTEM_PROMPT is static) -- included in every
# Stage 4 extraction cache key (see stage4.py), so an edit to the prompt
# automatically invalidates every previously-cached entry.
_SYSTEM_PROMPT_HASH = cache.content_hash(_SYSTEM_PROMPT)

# stage4.py (below) refers to the extraction prompt, its hash and the user-message builder under these names
# (they used to be imported from the separate module under these aliases):
_EXTRACTION_SYSTEM_PROMPT = _SYSTEM_PROMPT
_EXTRACTION_PROMPT_HASH = _SYSTEM_PROMPT_HASH
_build_extraction_message = _build_user_message


# ===== end of the code merged from `assessment/evidence_extraction.py` =====================================================


SCREEN_PROMPT_VERSION = "s4v2-screen-2"   # v2: reject/pass/insufficient_evidence (v1 graded 0-3)
JUDGE_PROMPT_VERSION = "s4v2-judge-2"   # v2: adds the verdict field (graded | insufficient_evidence)
_NS_SCREEN, _NS_EXTRACT, _NS_JUDGE = "stage4_v2_screen", "stage4_v2_extract", "stage4_v2_judge"
_NS_EPMC = "stage4_epmc_abstract"

Computational = Literal["coarse_grained", "other_computational", "not_computational", "review_or_perspective"]
Confidence = Literal["low", "medium", "high"]
ScreenComputational = Literal["coarse_grained", "other_computational", "not_computational", "review_or_perspective", "unknown"]
ScreenDecision = Literal["reject", "pass", "insufficient_evidence"]
MIN_ABSTRACT_CHARS = 40     # below this there is no usable abstract


# ------------------------------------------------------------------ schemas
class ScreenItem(BaseModel):
    index: int
    decision: ScreenDecision
    computational: ScreenComputational
    confidence: Confidence
    reason: str


class ScreenBatch(BaseModel):
    items: list[ScreenItem]


class _MethodOut(BaseModel):
    model_class: list[str] = Field(default_factory=list)
    study_role: list[str] = Field(default_factory=list)


class _SystemOut(BaseModel):
    studied_systems: list[str] = Field(default_factory=list)
    family_or_fold: list[str] = Field(default_factory=list)
    protein_or_system_names: list[str] = Field(default_factory=list)


class ExtractedFacts(BaseModel):
    """Same three evidence groups as the old PaperEvidence (query-independent), slimmed to what the
    LLM is actually asked to fill (the old code dropped the never-populated grounding_text fields too)."""
    method: _MethodOut
    system: _SystemOut
    processes: list[ProcessEvidenceItem] = Field(default_factory=list)


class Judgment(BaseModel):
    verdict: Literal["graded", "insufficient_evidence"]
    computational: Computational
    methods_used: list[str]
    system_relation: Literal["same_system", "same_family_or_fold", "related_system", "different_system", "not_stated"]
    process_relation: Literal["same_process", "related_process", "different_process", "not_stated"]
    evidence_type: Literal["direct_application", "related_system_transfer", "general_method_reference", "not_applicable"]
    supports_applicability: Literal["yes", "partial", "no"]
    grade: Literal[0, 1, 2, 3]
    explanation: str
    supporting_quote: str


class PaperAssessment(BaseModel):
    key: str
    title: str
    doi: Optional[str] = None
    year: Optional[int] = None
    routes: list[str] = Field(default_factory=list)          # informational only: flat / 3A / 3B
    n_supporting_queries: int = 0
    # assessed = passed tier 1 and graded by tier 2; tier1_rejected = positively irrelevant (grade 0);
    # missing_evidence = could not be judged (no abstract even after the Europe PMC fallback, or tier 2 found the
    # text insufficient): INTERNAL only -- NOT a grade, NOT irrelevance, never shown to end users; not_evaluated_cap = passed
    # tier 1 but cut by the safety-valve cap (NOT irrelevance); error = an LLM call failed.
    status: Literal["assessed", "tier1_pass", "tier1_rejected", "missing_evidence", "not_evaluated_cap", "error"] = "missing_evidence"
    abstract_source: Literal["openalex", "europepmc", "none"] = "openalex"
    screen: Optional[ScreenItem] = None
    facts: Optional[ExtractedFacts] = None
    judgment: Optional[Judgment] = None
    quote_grounded: Optional[bool] = None
    deep_candidate: bool = False
    deep_skipped_by_cap: bool = False
    flags: list[str] = Field(default_factory=list)
    final_grade: Optional[int] = None      # None for missing_evidence / not_evaluated_cap / error
    rank: Optional[int] = None
    error: Optional[str] = None

    @property
    def recommended(self) -> bool:
        return self.final_grade is not None and self.final_grade >= 2


class Stage4Result(BaseModel):
    assessments: list[PaperAssessment]            # ranked best-first; errors last, unranked
    problem_context: str
    workload: dict

    def by_key(self) -> dict[str, PaperAssessment]:
        return {a.key: a for a in self.assessments}

    @property
    def recommended(self) -> list[PaperAssessment]:
        return [a for a in self.assessments if a.recommended]

    @property
    def background(self) -> list[PaperAssessment]:
        return [a for a in self.assessments if a.final_grade == 1]

    @property
    def rejected(self) -> list[PaperAssessment]:
        return [a for a in self.assessments if a.final_grade == 0]

    @property
    def missing_evidence(self) -> list[PaperAssessment]:
        """Internal/debug only: papers that could not be judged for lack of evidence. Never customer-facing."""
        return [a for a in self.assessments if a.status == "missing_evidence"]

    @property
    def not_evaluated(self) -> list[PaperAssessment]:
        return [a for a in self.assessments if a.status == "not_evaluated_cap"]

    @property
    def errors(self) -> list[PaperAssessment]:
        return [a for a in self.assessments if a.status == "error"]


@dataclass
class Stage4Config:
    screen_batch_size: int = 12
    screen_effort: str = "low"
    deep_cap: int = 80                 # safety valve, not a cost control (~$0.001-0.002 per deep paper); Stage 3 keeps <=58 in the 13-case set
    reject_min_confidence: str = "medium"   # a 'reject' the model itself is only 'low'-confident about is treated as 'pass'
    abstract_fallback: bool = True          # Europe PMC lookup, ONLY for papers without a usable OpenAlex abstract
    abstract_lookup: Optional[Callable[["Paper"], Optional[dict]]] = None   # injectable (tests); None -> europepmc_abstract
    abstract_workers: int = 3
    extract_effort: str = "low"
    judge_effort: str = "medium"
    workers: int = 6
    timeout_s: float = 150.0
    # Tier 1 sees title + the first `screen_abstract_chars` characters of the abstract.  It was 1200, which cut off the sentence that names the computational
    # work in long abstracts (B055 @1471, B092 @1695, B185 @1346 -> wrongly rejected as "experimental").  4000 covers every abstract seen so far (max 3052) and is
    # only a safety cap; the cost is ~+4% of the Stage-4 LLM spend (tier 1 is ~12% of it and its abstract part grows ~31%).
    screen_abstract_chars: int = 4000
    retries: int = 1


# ------------------------------------------------------------------ prompts
_RUBRIC = """\
GRADES (use only the title/abstract text you are shown; never rely on keyword overlap alone, and never on what you may know about a paper from elsewhere):
3 = DIRECT: computational / coarse-grained modeling of the user's own system (or a very close variant) that addresses the user's process or objective.
2 = TRANSFERABLE: computational / coarse-grained modeling of a closely related system (one sharing a property named in the transfer hypotheses) or of the same process in another system, with enough in the text to see why it informs the user's problem.
1 = BACKGROUND: weak applicability -- a general method, a review/perspective, a related topic without modeling of the process, or only a partial/tangential match.
0 = NOT RELEVANT: experimental-only with no modeling, an unrelated system and process, or no modeling content.
Hard caps: purely experimental -> 0. Review/perspective -> at most 1. Computational but not coarse-grained (e.g. all-atom MD) -> at most 2. No abstract available -> at most 1.
A paper about a DIFFERENT system is grade 2 only when its text shows a shared property that matters for the user's problem (see the transfer hypotheses); sharing only generic words ("protein", "folding", "simulation") is grade 0-1."""

_SCREEN_SYSTEM = """\
You are the relevance screener (BEAM Stage 4, tier 1) of a literature agent that recommends coarse-grained (CG) and related computational modeling literature to computational biophysicists.

You receive the USER'S PROBLEM and a numbered list of candidate papers (title + abstract only). They came from an automated search followed by a cheap keyword/simulation filter. Your job is ONLY to remove papers that are POSITIVELY shown to be irrelevant. You do not rank and you do not pick the best: a later stage reads every paper you pass in depth. Missing information is NOT evidence of irrelevance.

DECISIONS (use only the title/abstract text you are shown; never what you may remember about a paper from elsewhere):
- reject: the text positively shows the paper cannot help this user, because ONE of
    (a) it has no computational / modeling component at all (purely experimental, clinical, epidemiological, or a pure sequence/annotation survey), or
    (b) its system AND its process are both unrelated to the user's problem, and nothing in the text shares a property named in the transfer hypotheses.
  Name that positive evidence in the reason.
- pass: the paper could plausibly inform the user's problem -- computational / modeling work on the user's system or process; on a related system that shares a property named in the transfer hypotheses; a general modeling method that could apply; a review of the topic; or any case where you are unsure. WHEN IN DOUBT, PASS.
- insufficient_evidence: the text shown is too short or uninformative to tell whether the paper is relevant or not (e.g. a one-line abstract, a bare conference listing). Do NOT reject for lack of information.
Never reject merely because the paper is about a DIFFERENT system than the user's, is not coarse-grained (all-atom MD can inform), does not mention the user's exact process, is a review, or you do not recognise it.

Fields per paper:
- computational: coarse_grained | other_computational | not_computational | review_or_perspective | unknown (what the paper itself does; unknown if the text does not say).
- confidence: low | medium | high -- how sure you are of the decision from the text shown.
- reason: at most 25 words, naming the system/process the paper actually studied (for a reject, the positive evidence of irrelevance).
Return exactly one item per paper, carrying the same index as the input. Respond with a single JSON object matching the schema, and nothing else."""


_JUDGE_SYSTEM = """\
You are the evidence evaluator (BEAM Stage 4, tier 2). You receive the USER'S PROBLEM, one paper's title and abstract, and objective facts already extracted from that paper by an earlier, query-independent extraction step (method class, studied systems, processes). Decide how well THIS paper's evidence supports the user's problem.

""" + _RUBRIC + """

Fields:
- computational: coarse_grained | other_computational | not_computational | review_or_perspective.
- methods_used: the model(s)/simulation method(s) the paper used or developed, as named in its text (empty if none).
- system_relation: same_system | same_family_or_fold | related_system | different_system | not_stated (the paper's studied system vs the user's system).
- process_relation: same_process | related_process | different_process | not_stated.
- evidence_type: direct_application (modeling of the user's own system/process) | related_system_transfer (evidence from a related system or process that informs the user's problem) | general_method_reference | not_applicable.
- supports_applicability: yes = the text shows the method/model working on the user's system, or a closely related system, for the same process; partial = relevant but with a gap (different process, scale or only mentioned); no = not shown.
- verdict: graded | insufficient_evidence. Use insufficient_evidence ONLY when the text shown (e.g. title only, no abstract, or too little text) cannot support ANY grade, not even 0 or 1 -- it means "needs verification" and is NOT a judgement that the paper is irrelevant. Otherwise use graded. When it is insufficient_evidence, set grade 0 (ignored), say in the explanation what is missing, and give the title as the supporting_quote.
- grade: 0-3 per the rubric, consistent with the fields above.
- supporting_quote: a SHORT verbatim quote (at most 30 words) copied EXACTLY from the title or abstract that best supports your grade. Never paraphrase. If nothing in the text supports a grade of 2 or 3, give grade 1 or 0.
- explanation: 1-2 sentences, evidence-based, citing what the paper studied/did.
Respond with a single JSON object matching the schema, and nothing else."""


# ------------------------------------------------------------ context + keys
def build_problem_context(plan, expert_prior=None, user_request: Optional[str] = None) -> str:
    lines = ["USER'S PROBLEM"]
    if user_request:
        lines.append(f"- Original request: {user_request.strip()}")
    lines.append(f"- System: {plan.system}")
    if getattr(plan, "target_component", None):
        lines.append(f"- Target component: {plan.target_component}")
    lines.append(f"- Scientific objective: {plan.scientific_objective}")
    if getattr(plan, "search_context", None):
        lines.append(f"- Stated context / constraints: {'; '.join(plan.search_context)}")
    axes = [a for a in (getattr(expert_prior, "transfer_axes", None) or []) if a.confidence != "low"]
    if axes:
        lines.append("\nTRANSFER HYPOTHESES (from the planner: papers on a DIFFERENT system can still inform the user's problem "
                     "when they share one of these properties)")
        for a in axes:
            feats = "; ".join(a.discriminative_features) if a.discriminative_features else "-"
            lines.append(f"- [{a.basis}] {a.scientific_class}. Distinguishing features: {feats}. Why it matters: {a.objective_relevance}")
    return "\n".join(lines)


def _paper_key(p: Paper) -> str:
    if getattr(p, "doi", None):
        return p.doi.lower()
    oa = getattr(getattr(p, "external_ids", None), "openalex_id", None)
    return (oa or p.title or "")[:120].lower()


def _routes(p: Paper) -> list[str]:
    cats = set(p.query_categories or [])
    out = []
    if cats & {"system_specific_cg", "process", "context"}:
        out.append("flat")
    if "transfer_direct" in cats:
        out.append("3A")
    if "structural" in cats:
        out.append("3B")
    return out


def _abs_hash(p: Paper) -> str:
    return cache.content_hash(p.title or "", p.abstract or "")


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower().replace("’", "'").replace("–", "-").replace("—", "-")).strip()


def quote_grounded(quote: str, paper: Paper) -> bool:
    """Deterministic check that the 'verbatim' supporting quote really is in the title/abstract.
    '...' / ellipsis splits a quote into pieces; every piece (>=10 chars) must appear."""
    text = _norm_ws(f"{paper.title or ''} {paper.abstract or ''}")
    pieces = [_norm_ws(x).strip(" \"'.,;:") for x in re.split(r"\.\.\.|…", quote or "")]
    pieces = [x for x in pieces if len(x) >= 10]
    return bool(pieces) and all(x in text for x in pieces)


# --------------------------------------------------------------- LLM helpers
class _Counter:
    def __init__(self):
        self.calls = {"screen": 0, "extract": 0, "judge": 0}
        self.cache_hits = {"screen": 0, "extract": 0, "judge": 0}
        self.errors = 0
        self.abstract_lookups = 0
        self.abstract_recovered = 0
        self.abstract_lookup_failures = 0


def _call(backend: LLMBackend, cfg: Stage4Config, *, system, user, schema, label, effort, case_id):
    last = None
    for _ in range(cfg.retries + 1):
        try:
            return backend.structured(system=system, user=user, schema=schema, label=label, effort=effort,
                                      purpose="stage4_v2", case_id=case_id, timeout=cfg.timeout_s)
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.0)
    raise last  # type: ignore[misc]


def _screen_message(ctx: str, batch: list[tuple[int, Paper]], cfg: Stage4Config) -> str:
    blocks = [ctx, "\nCANDIDATE PAPERS"]
    for idx, p in batch:
        abstract = (p.abstract or "").strip()
        abstract = abstract[: cfg.screen_abstract_chars] + ("..." if len(abstract) > cfg.screen_abstract_chars else "")
        blocks.append(f"[{idx}] Title: {p.title}\n    Year: {p.year or 'unknown'}\n    Abstract: {abstract or '(not available)'}")
    return "\n".join(blocks)


def has_usable_abstract(p: Paper) -> bool:
    return len((p.abstract or "").strip()) >= MIN_ABSTRACT_CHARS


# ------------------------------------------------- abstract fallback (Europe PMC)
def _strip_markup(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


def europepmc_abstract(p: Paper, *, timeout: float = 20.0) -> Optional[dict]:
    """Look the paper up in Europe PMC by DOI (else PMID) and return {"abstract": str|None, "via": "doi"|"pmid"}.
    Returns None when the paper has neither identifier (nothing to look up). Raises on a transport / HTTP
    failure so the caller can tell 'no abstract exists' from 'could not ask'. Successful answers -- including
    'found, but no abstract' -- are cached, failures never are."""
    doi = (getattr(p, "doi", None) or "").strip().lower()
    pmid = (getattr(getattr(p, "external_ids", None), "pmid", None) or "").strip()
    if doi:
        via, ident, query = "doi", doi, f'DOI:"{doi}"'
    elif pmid:
        via, ident, query = "pmid", pmid, f"EXT_ID:{pmid} AND SRC:MED"
    else:
        return None
    key = cache.content_hash("epmc-abstract-v1", via, ident)
    hit = cache.get(_NS_EPMC, key)
    if hit is not None:
        return hit
    url = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=" + urllib.parse.quote(query)
           + "&resultType=core&format=json&pageSize=1")
    req = urllib.request.Request(url, headers={"User-Agent": "BEAM-stage4-abstract-fallback/0.1", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:        # URLError / HTTPError / timeout propagate
        data = json.loads(r.read(2_000_000).decode("utf-8", "replace"))
    results = (data.get("resultList") or {}).get("result") or []
    rec = results[0] if results else None
    if rec is not None and via == "doi" and (rec.get("doi") or "").strip().lower() != doi:
        rec = None                                                  # not the same record -> do not borrow its abstract
    abstract = _strip_markup(rec.get("abstractText") or "") if rec else ""
    out = {"abstract": abstract or None, "via": via}
    cache.set(_NS_EPMC, key, out)
    return out


def _apply_abstract_fallback(papers: list[Paper], cfg, counter) -> tuple[list[Paper], dict[str, str], dict[str, str]]:
    """OpenAlex abstract -> (only if missing) Europe PMC. Returns the papers (enriched copies where a usable abstract
    was found), the source of each paper's abstract, and per-paper lookup failures. Papers with a usable OpenAlex
    abstract are never looked up."""
    source = {_paper_key(p): ("openalex" if has_usable_abstract(p) else "none") for p in papers}
    failures: dict[str, str] = {}
    need = [p for p in papers if not has_usable_abstract(p)]
    if not need or not cfg.abstract_fallback:
        return papers, source, failures
    lookup = cfg.abstract_lookup or europepmc_abstract

    def one(p: Paper):
        try:
            return p, lookup(p), None
        except Exception as e:  # noqa: BLE001
            return p, None, e

    enriched: dict[str, Paper] = {}
    with ThreadPoolExecutor(max_workers=max(1, cfg.abstract_workers)) as ex:
        for p, res, err in ex.map(one, need):
            k = _paper_key(p)
            counter.abstract_lookups += 1
            if err is not None:
                counter.abstract_lookup_failures += 1
                failures[k] = repr(err)[:160]
            elif res and len((res.get("abstract") or "").strip()) >= MIN_ABSTRACT_CHARS:
                enriched[k] = p.model_copy(update={"abstract": res["abstract"].strip()})
                source[k] = "europepmc"
                counter.abstract_recovered += 1
    return [enriched.get(_paper_key(p), p) for p in papers], source, failures


def _screen_all(papers: list[Paper], ctx: str, ctx_hash: str, backend, cfg, case_id, counter: _Counter) -> dict[str, ScreenItem]:
    results: dict[str, ScreenItem] = {}
    todo: list[Paper] = []
    for p in papers:
        key = cache.content_hash(SCREEN_PROMPT_VERSION, cfg.screen_effort, str(cfg.screen_abstract_chars), ctx_hash, _paper_key(p), _abs_hash(p))
        hit = cache.get(_NS_SCREEN, key)
        if hit is not None:
            results[_paper_key(p)] = ScreenItem.model_validate(hit)
            counter.cache_hits["screen"] += 1
        else:
            todo.append(p)
    batches = [todo[i:i + cfg.screen_batch_size] for i in range(0, len(todo), cfg.screen_batch_size)]

    def run_batch(batch_papers: list[Paper]):
        numbered = list(enumerate(batch_papers, 1))
        out = _call(backend, cfg, system=_SCREEN_SYSTEM, user=_screen_message(ctx, numbered, cfg), schema=ScreenBatch,
                    label="stage4_screen", effort=cfg.screen_effort, case_id=case_id)
        by_idx = {it.index: it for it in out.items}
        return [(p, by_idx.get(i)) for i, p in numbered]

    def safe(batch_papers):
        try:
            counter.calls["screen"] += 1
            res = run_batch(batch_papers)
            missing = [p for p, it in res if it is None]
            if missing and len(batch_papers) > 1:      # the model skipped some items: ask again, smaller
                counter.calls["screen"] += 1
                again = {_paper_key(p): it for p, it in run_batch(missing)}
                res = [(p, it or again.get(_paper_key(p))) for p, it in res]
            return res
        except Exception as e:  # noqa: BLE001
            return [(p, e) for p in batch_papers]

    with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        for res in ex.map(safe, batches):
            for p, it in res:
                if isinstance(it, ScreenItem):
                    results[_paper_key(p)] = it
                    key = cache.content_hash(SCREEN_PROMPT_VERSION, cfg.screen_effort, str(cfg.screen_abstract_chars), ctx_hash, _paper_key(p), _abs_hash(p))
                    cache.set(_NS_SCREEN, key, it.model_dump())
                else:
                    counter.errors += 1
                    results[_paper_key(p)] = ("error", repr(it) if it is not None else "model returned no item for this paper")  # type: ignore[assignment]
    return results


def _extract_facts(p: Paper, backend, cfg, case_id, counter: _Counter) -> ExtractedFacts:
    user = _build_extraction_message(p)
    key = cache.content_hash(_EXTRACTION_PROMPT_HASH, backend.model, cfg.extract_effort, user)
    hit = cache.get(_NS_EXTRACT, key)
    if hit is not None:
        counter.cache_hits["extract"] += 1
        return ExtractedFacts.model_validate(hit)
    counter.calls["extract"] += 1
    facts = _call(backend, cfg, system=_EXTRACTION_SYSTEM_PROMPT, user=user, schema=ExtractedFacts,
                  label="stage4_extract", effort=cfg.extract_effort, case_id=case_id)
    cache.set(_NS_EXTRACT, key, facts.model_dump())
    return facts


def _judge_message(ctx: str, p: Paper, facts: ExtractedFacts) -> str:
    return (f"{ctx}\n\nPAPER\nTitle: {p.title}\nYear: {p.year or 'unknown'}\nVenue: {getattr(p, 'venue', None) or 'unknown'}\n"
            f"Abstract: {(p.abstract or '').strip() or '(not available)'}\n\nEXTRACTED FACTS (objective, query-independent)\n{facts.model_dump_json(indent=1)}")


def _deep_one(p: Paper, ctx: str, ctx_hash: str, backend, cfg, case_id, counter: _Counter):
    key = cache.content_hash(JUDGE_PROMPT_VERSION, cfg.judge_effort, ctx_hash, _paper_key(p), _abs_hash(p))
    hit = cache.get(_NS_JUDGE, key)
    if hit is not None:
        counter.cache_hits["judge"] += 1
        return ExtractedFacts.model_validate(hit["facts"]), Judgment.model_validate(hit["judgment"])
    facts = _extract_facts(p, backend, cfg, case_id, counter)
    counter.calls["judge"] += 1
    judgment = _call(backend, cfg, system=_JUDGE_SYSTEM, user=_judge_message(ctx, p, facts), schema=Judgment,
                     label="stage4_judge", effort=cfg.judge_effort, case_id=case_id)
    cache.set(_NS_JUDGE, key, {"facts": facts.model_dump(), "judgment": judgment.model_dump()})
    return facts, judgment


# ---------------------------------------------------------- grading + ranking
_CONF = {"low": 0, "medium": 1, "high": 2}
_SUFF = {"no": 0, "partial": 1, "yes": 2}
_CG = {"coarse_grained": 2, "other_computational": 1}


def _final_grade(a: PaperAssessment, has_abstract: bool = True) -> int:
    """Deterministic, generic guards that make the tier-2 LLM grade consistent with the rubric. They never
    look at the retrieval route or the year."""
    grade, comp = a.judgment.grade, a.judgment.computational
    if not has_abstract:
        grade = min(grade, 1)
        a.flags.append("no_abstract_capped_at_1")
    if comp == "not_computational":
        grade = 0
    elif comp == "review_or_perspective":
        grade = min(grade, 1)
    elif comp == "other_computational":
        grade = min(grade, 2)
    if a.quote_grounded is False and grade >= 2:
        grade -= 1
        a.flags.append("quote_not_found_in_text_demoted_1")
    return grade


def _sort_key(a: PaperAssessment):
    """Rank by the tier-2 assessment. Tier 1 no longer grades, so it contributes no ranking signal."""
    comp = a.judgment.computational if a.judgment else "not_computational"
    suff = _SUFF[a.judgment.supports_applicability] if a.judgment else 0
    # route (via n_supporting_queries) and year appear ONLY here, as the last tie-breakers, after every relevance signal
    return (-a.final_grade, -suff, -_CG.get(comp, 0), -a.n_supporting_queries, -(a.year or 0), a.title)


# ---------------------------------------------------------------- entry point
def run_stage4_v2(plan, expert_prior, retained: list[Paper], backend: LLMBackend, *,
                  user_request: Optional[str] = None, case_id: Optional[str] = None,
                  config: Optional[Stage4Config] = None) -> Stage4Result:
    cfg = config or Stage4Config()
    t0 = time.time()
    ctx = build_problem_context(plan, expert_prior, user_request)
    ctx_hash = cache.content_hash(ctx)
    counter = _Counter()

    seen, papers = set(), []
    for p in retained:                      # one assessment per distinct paper
        k = _paper_key(p)
        if k not in seen:
            seen.add(k)
            papers.append(p)

    # evidence: OpenAlex abstract -> (only if missing) Europe PMC -> else missing_evidence (internal) ------------
    papers, abstract_source, lookup_failures = _apply_abstract_fallback(papers, cfg, counter)
    by_key = {_paper_key(p): p for p in papers}

    assessments: dict[str, PaperAssessment] = {}
    for p in papers:
        k = _paper_key(p)
        assessments[k] = PaperAssessment(
            key=k, title=p.title or "", doi=getattr(p, "doi", None), year=p.year,
            routes=_routes(p), n_supporting_queries=len(p.found_by_queries or []), abstract_source=abstract_source[k])
        if not has_usable_abstract(p):
            # no abstract anywhere we looked: not graded, not sent to the LLM on a title alone, not user-facing
            assessments[k].status = "missing_evidence"
            assessments[k].flags.append("missing_evidence_no_abstract")
            if k in lookup_failures:
                assessments[k].flags.append("abstract_lookup_failed")
                assessments[k].error = lookup_failures[k]
    evaluable = [p for p in papers if has_usable_abstract(p)]

    # tier 1 -----------------------------------------------------------------
    screened = _screen_all(evaluable, ctx, ctx_hash, backend, cfg, case_id, counter)
    min_conf = _CONF[cfg.reject_min_confidence]
    n_lowconf_reject_passed = 0
    for k, res in screened.items():
        a = assessments[k]
        if not isinstance(res, ScreenItem):
            a.status, a.error = "error", res[1]
            continue
        a.screen = res
        decision = res.decision
        if decision == "reject" and _CONF[res.confidence] < min_conf:
            decision = "pass"                       # no positive evidence the model itself trusts -> do not kill
            a.flags.append("tier1_low_confidence_reject_passed")
            n_lowconf_reject_passed += 1
        if decision == "reject":
            a.status = "tier1_rejected"
        else:
            # pass, insufficient_evidence (an abstract exists but cannot tell): lack of evidence is not evidence of
            # irrelevance, so both go on to tier 2. Provisional status; becomes 'assessed' / 'missing_evidence'
            # (tier 2 still cannot judge) / 'not_evaluated_cap' below.
            a.status = "tier1_pass"

    # tier 2 -----------------------------------------------------------------
    candidates = [a for a in assessments.values() if a.status == "tier1_pass"]    # pass AND insufficient_evidence AND low-confidence reject
    # only matters when the safety-valve cap binds: most-confident passes first, never flat-route-first
    candidates.sort(key=lambda a: (-_CONF[a.screen.confidence], -a.n_supporting_queries, a.title))
    chosen, skipped = candidates[: cfg.deep_cap], candidates[cfg.deep_cap:]
    for a in chosen:
        a.deep_candidate = True
    for a in skipped:
        a.deep_skipped_by_cap = True
        a.status = "not_evaluated_cap"

    def deep(a: PaperAssessment):
        try:
            return a, _deep_one(by_key[a.key], ctx, ctx_hash, backend, cfg, case_id, counter), None
        except Exception as e:  # noqa: BLE001
            return a, None, e

    with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        for a, out, err in ex.map(deep, chosen):
            if err is not None:
                a.status, a.error = "error", repr(err)
                counter.errors += 1
            else:
                a.facts, a.judgment = out
                a.quote_grounded = quote_grounded(a.judgment.supporting_quote, by_key[a.key])
                a.status = "missing_evidence" if a.judgment.verdict == "insufficient_evidence" else "assessed"

    # grading + ranking ------------------------------------------------------
    for a in assessments.values():
        if a.status == "assessed":
            a.final_grade = _final_grade(a, has_usable_abstract(by_key[a.key]))
        elif a.status == "tier1_rejected":
            a.final_grade = 0                       # positive evidence of irrelevance
        # missing_evidence / not_evaluated_cap / error keep final_grade=None: not graded, not 'not relevant'
    graded = sorted((a for a in assessments.values() if a.final_grade is not None), key=_sort_key)
    for i, a in enumerate(graded, 1):
        a.rank = i
    ungraded = sorted((a for a in assessments.values() if a.final_grade is None and a.status != "error"), key=lambda a: (a.status, -a.n_supporting_queries, a.title))
    errs = [a for a in assessments.values() if a.status == "error"]
    n_by = lambda st: sum(1 for a in assessments.values() if a.status == st)

    workload = {
        "n_retained": len(papers), "n_screened": len(screened),
        "n_screen_llm_papers": len(evaluable),
        "n_abstract_openalex": sum(1 for v in abstract_source.values() if v == "openalex"),
        "n_abstract_lookup_attempted": counter.abstract_lookups, "n_abstract_recovered_europepmc": counter.abstract_recovered,
        "n_abstract_lookup_failed": counter.abstract_lookup_failures,
        "n_tier1_pass": sum(1 for a in assessments.values() if a.screen and a.screen.decision == "pass"),
        "n_tier1_rejected": n_by("tier1_rejected"),
        "n_tier1_insufficient": sum(1 for a in assessments.values() if a.screen and a.screen.decision == "insufficient_evidence"),
        "n_tier1_insufficient_continued_to_tier2": sum(1 for a in assessments.values() if a.screen and a.screen.decision == "insufficient_evidence" and a.judgment is not None),
        "n_missing_evidence": n_by("missing_evidence"),    # internal: no abstract after the fallback, or tier 2 could not judge
        "n_tier1_low_conf_reject_passed": n_lowconf_reject_passed,
        "n_deep_candidates": len(candidates),
        "n_deep_evaluated": len(chosen), "deep_cap": cfg.deep_cap, "deep_cap_binding": len(skipped) > 0,
        "n_deep_skipped_by_cap": len(skipped),
        "llm_calls": dict(counter.calls), "cache_hits": dict(counter.cache_hits), "errors": counter.errors,
        "n_recommended": sum(1 for a in graded if a.final_grade >= 2), "n_background": sum(1 for a in graded if a.final_grade == 1),
        "n_rejected": sum(1 for a in graded if a.final_grade == 0), "n_quote_demotions": sum(1 for a in graded if "quote_not_found_in_text_demoted_1" in a.flags),
        "seconds": round(time.time() - t0, 1),
    }
    return Stage4Result(assessments=graded + ungraded + errs, problem_context=ctx, workload=workload)
