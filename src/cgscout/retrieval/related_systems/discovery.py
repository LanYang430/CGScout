"""Hop-1 of Literature-grounded RelatedSystemDiscovery.

Objective-free. Given ONE grounded system description + its terminology
expansions:
  1. build OBJECTIVE-FREE structural discovery queries
  2. retrieve structural-biology literature
  3. per paper (ONE paper -> ONE LLM call, no batching), extract the
     SPECIFIC biological systems THAT TEXT says have this structural class
  4. deterministically verify every returned name + quote is a VERBATIM
     substring of that real paper; drop anything that is not
  5. merge across papers, dedupe names, rank by simple auditable signals
     (support count, then retrieval order), keep at most K

Every surviving system name traces verbatim to >= 1 real retrieved paper.
Zero names from LLM memory.

Hop 2 (discovered systems x objective x methodology) and any live wiring
are NOT here.

Pure orchestration + deterministic grounding/merge. `search_fn` and
`extract_fn` are injected (stubs in the tests, the backend-injected
extraction.extract_related_systems_rich in production).
"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from pydantic import BaseModel, Field

from cgscout.retrieval.related_systems.terminology import _STRUCTURAL_MORPHEMES, strip_size_qualifiers  # noqa: E402

_WS = re.compile(r"\s+")
_LEADING_ARTICLE = re.compile(r"^(the|a|an|its|their)\s+", re.IGNORECASE)
_DASHES = re.compile(r"[‐-―−]")            # ‐‑‒–—―  and  −
_GREEK = {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta",
          "Α": "alpha", "Β": "beta", "Γ": "gamma", "Δ": "delta"}


def _norm(s: str) -> str:
    return _WS.sub(" ", (s or "").strip().lower())


def _dnorm(s: str) -> str:
    """`_norm` with every unicode dash folded to '-'. For substring/containment
    checks only: the string stays otherwise verbatim (parens, digits, other
    punctuation preserved), so "HET-s(218-289)" and "HET-s(218–289)" compare
    equal without loosening grounding in any other way."""
    return _norm(_DASHES.sub("-", s or ""))


_PUNCT = re.compile(r"[^a-z0-9\s]+")


def _canon(s: str) -> str:
    """Aggressive canonical form for CLASS-VARIANT rejection only (never for
    verbatim grounding): fold Greek letters to names, every unicode dash and
    all other punctuation to a space, collapse whitespace. So "β‐Solenoid",
    "beta-solenoid", "(beta/alpha)8" and "beta solenoid" compare cleanly."""
    t = _norm(s)
    for g, a in _GREEK.items():
        t = t.replace(g.lower(), a)
    t = _DASHES.sub(" ", t).replace("-", " ")
    t = _PUNCT.sub(" ", t)
    return _WS.sub(" ", t).strip()


# a name that is JUST the structural class + one of these is not a system
_CLASS_HEAD_WORDS = {
    "protein", "proteins", "fold", "folds", "domain", "domains", "repeat",
    "repeats", "structure", "structures", "family", "families", "superfamily",
    "motif", "motifs", "region", "regions", "architecture", "architectures",
    "topology", "conformation", "conformations", "arrangement", "scaffold",
    "scaffolds", "framework", "frameworks", "enzyme", "enzymes",
}
_CONNECTIVES = {"and", "or", "other", "others", "the", "a", "an", "of", "type",
                "types", "like", "containing", "with", "in", "class", "based",
                "such", "as", "eg"}


def _strip_edge_words(tokens: list[str], words: set[str]) -> list[str]:
    lo, hi = 0, len(tokens)
    while lo < hi and tokens[lo] in words:
        lo += 1
    while hi > lo and tokens[hi - 1] in words:
        hi -= 1
    return tokens[lo:hi]


def _class_variant_reason(name: str, banned_canon: set[str], concrete: bool = False) -> Optional[str]:
    """Drop-reason if `name` is a spelling / plural / Greek / hyphenation
    variant of the structural class (or an expansion term), or is built ONLY
    from structural-class vocabulary (the tokens of the class + its expansion
    terms, structural morphemes, bare digits, connectives, generic head
    words) with no specific identifier of its own.

    `concrete=True` (the extractor typed the mention a specific, individually
    named protein/system): the class's OWN words no longer count as "class
    vocabulary" for the last rule, because a concrete member can legitimately
    share a word with the class label ("MscS" inside "MscS-related channels",
    "dynamin" inside "dynamin-family GTPases"). Spelling variants of the class
    and "class + generic head word" are still rejected."""
    c = _canon(name)
    if not c:
        return "empty name"
    if c in banned_canon:
        return "name is a spelling/plural/Greek variant of the structural class"
    class_tokens = set() if concrete else {tok for phrase in banned_canon for tok in phrase.split()}
    core = _strip_edge_words(c.split(), _CLASS_HEAD_WORDS | _CONNECTIVES)
    if core and " ".join(core) in banned_canon:
        return "name is the structural class plus a generic head word"
    if core and all(
        t in _CONNECTIVES or t in _CLASS_HEAD_WORDS or t in class_tokens
        or t.isdigit() or any(m in t for m in _STRUCTURAL_MORPHEMES)
        for t in core
    ):
        return "name is only structural-class vocabulary, not a specific named system"
    return None


# --- deterministic structural-class-modifier stripping ------------------
# Peel a structural-class modifier off a candidate name ("Pertactin beta-helix"
# -> "Pertactin", "P22 tailspike beta-helix domain" -> "P22 tailspike") and
# keep the concrete system name. NO LLM, NO protein-name table. Conservative:
# only leading/trailing all-structural runs, plus an embedded run that EXACTLY
# equals the matched structural term or a full expansion term, are touched.
# trim only leading/trailing whitespace + loose separators -- NEVER brackets,
# parens or quotes (they carry meaning: "HET-s(218-289)", "p53 (tetramer)")
_STRIP_PUNCT_EDGES = " \t\r\n.,;:/–—-"


def _token_canon_words(tok: str) -> list[str]:
    return _canon(tok).split()


def _strip_class_modifier(name: str, structural_class: str,
                          banned_terms: set[str], stn: str, concrete: bool = False) -> str:
    """`concrete=True`: the mention is a specific named entity, so only unmistakable structural WORDING (morphemes such
    as 'beta-helix', and an exact class/alias phrase) is peeled -- the class label's ordinary words are not, otherwise a
    member whose name is also the class's head word ("MscS", "dynamin") would be stripped to nothing."""
    toks = name.split()
    if not toks:
        return ""
    cw = [_token_canon_words(t) for t in toks]
    class_words = set() if concrete else (set(_canon(structural_class).split()) | {
        w for b in banned_terms for w in _canon(b).split()})

    def _hard(w: str) -> bool:
        """A word that is unmistakably structural-class vocabulary: a class
        token or a structural morpheme. Generic head words ('protein',
        'domain'-as-plain-head), connectives, and bare digits are NOT hard on
        their own -- a trailing "27-30" / "1" is usually part of the entity
        name (PrP 27-30, HET-s(218-289)), not a class modifier."""
        return w in class_words or any(m in w for m in _STRUCTURAL_MORPHEMES)

    def _soft(w: str) -> bool:
        return w.isdigit() or w in _CLASS_HEAD_WORDS or w in _CONNECTIVES

    def _tok_strippable(k: int) -> bool:
        # every canon-word of the token is structural-class vocabulary
        return bool(cw[k]) and all(_hard(w) or _soft(w) for w in cw[k])

    def _tok_hard(k: int) -> bool:
        # a strippable token that carries a real morpheme / class token / index
        # digit (so "beta-helix", "(betaalpha)8", "barrels" -- but NOT a token
        # like "HET-s(218-289)" whose digits sit among a concrete identifier)
        return _tok_strippable(k) and any(_hard(w) for w in cw[k])

    keep = [True] * len(toks)

    # (1) drop a run of whole tokens whose canon EXACTLY equals stn or a full
    #     banned term -- unambiguous, so safe to remove anywhere in the name
    flat = [(k, w) for k in range(len(toks)) for w in cw[k]]
    targets = {_canon(stn)} | {_canon(b) for b in banned_terms}
    for tgt in sorted({t for t in targets if t}, key=lambda s: -len(s.split())):
        tw = tgt.split()
        for a in range(len(flat) - len(tw) + 1):
            if [w for _, w in flat[a:a + len(tw)]] == tw:
                ks = sorted({flat[b][0] for b in range(a, a + len(tw))})
                if [w for k in ks for w in cw[k]] == tw:      # whole-token span
                    for k in ks:
                        keep[k] = False

    # (2) peel a leading / trailing run of class-modifier tokens. A token is
    #     peeled if it is hard-structural, or already removed by (1); a token
    #     that is ONLY a generic head word / connective ("proteins", "of") is
    #     peeled ONLY when it is contiguous with peeled material toward the
    #     interior -- so "P22 tailspike beta-helix domain" -> "P22 tailspike"
    #     but "Type V secretory proteins" is left intact.
    def _peel(order: list[int]) -> None:
        pending: list[int] = []           # soft-only tokens seen so far in this edge run
        for k in order:
            if not keep[k]:               # already removed -> commit the soft run beside it
                for p in pending:
                    keep[p] = False
                pending = []
                continue
            if _tok_hard(k):
                keep[k] = False
                for p in pending:
                    keep[p] = False
                pending = []
                continue
            if _tok_strippable(k):        # generic head / connective -- hold until we know
                pending.append(k)
                continue
            break                         # hit a concrete token -- pending stays KEPT

    n = len(toks)
    _peel(list(range(n - 1, -1, -1)))     # trailing first
    _peel(list(range(n)))                 # then leading

    cleaned = " ".join(toks[k] for k in range(len(toks)) if keep[k])
    return cleaned.strip(_STRIP_PUNCT_EDGES).strip()


def _quote_ties_to_class(qn: str, stn: str, banned_canon: set[str]) -> bool:
    """The evidence quote must still tie the system to the structural class."""
    if stn and stn in qn:
        return True
    qc = _canon(qn)
    if any(b and b in qc for b in banned_canon):
        return True
    return any(m in qc for m in _STRUCTURAL_MORPHEMES)


# --- generic / non-system name rejection ---------------------------------
_GENERIC_HEADS = {
    "protein", "proteins", "enzyme", "enzymes", "domain", "domains", "system",
    "systems", "complex", "complexes", "receptor", "receptors", "toxin", "toxins",
    "adhesin", "adhesins", "peptide", "peptides", "molecule", "molecules",
    "fold", "structure", "architecture", "class", "family", "motif",
    "virulence factor", "virulence factors",
}
_GENERIC_LEADERS = ("the ", "a ", "an ", "this ", "these ", "those ", "several ",
                    "many ", "some ", "various ", "other ", "such ", "its ", "their ")


def _is_generic_or_class_name(name: str, banned_norms: set[str]) -> Optional[str]:
    """Return a drop-reason if `name` is not a specific named system, else None."""
    n = _norm(name)
    if not n:
        return "empty name"
    if n in banned_norms:
        return "name is the structural class / an expansion term itself"
    # NOTE: a name is NOT dropped merely for CONTAINING the structural term --
    # _strip_class_modifier peels the modifier first, and _class_variant_reason
    # catches anything that is still only class vocabulary once stripped.
    if n in _GENERIC_HEADS:
        return "generic category word, not a named system"
    if n.startswith(_GENERIC_LEADERS):
        rest = _norm(_LEADING_ARTICLE.sub("", name))
        # after dropping the leader, must contain a capitalised token OR a
        # digit-bearing identifier in the ORIGINAL string to count as named
        if not re.search(r"[A-Z]", name) and not re.search(r"\d", name):
            return "generic unnamed phrase (no proper noun / identifier)"
        if rest in _GENERIC_HEADS:
            return "generic category word, not a named system"
    return None


# --- models -------------------------------------------------------------
class SupportingPaper(BaseModel):
    paper_ref: str
    title: str
    year: Optional[int] = None
    found_by_query: str
    retrieval_rank: int
    evidence_quote: str                 # verbatim from the paper -- NEVER modified
    structural_term_matched: str
    system_name: str = ""               # the cleaned name grounded against THIS quote
    name_as_extracted: str = ""         # what the extractor returned, pre-cleanup
    name_in_quote: bool = False         # was the raw extracted name already in the quote
    # Optional, additive (extraction.py). None for an extractor
    # whose mention has no such attributes (extraction.SystemMention),
    # since the getattr() defaults in _ground_mention() below just leave these
    # None. When present: `entity_type`
    # is the extractor's own classification of what kind of thing `name` is
    # ("specific_protein_or_system" / "protein_family_or_class" /
    # "organism_or_species" / "generic_descriptor" / "construct_or_variant");
    # `identity_context` is a verbatim (grounding-checked, same as
    # evidence_quote) phrase from the SAME paper naming the broader class
    # this entity belongs to (e.g. "autotransporter" for "Pet"), or None if
    # the text never states one. Both are read, never written, by
    # identity.py -- this file has no opinion on what they
    # mean, it only preserves and verbatim-verifies them.
    entity_type: Optional[str] = None
    identity_context: Optional[str] = None


class DiscoveredSystem(BaseModel):
    name: str
    norm_name: str
    supporting_papers: list[SupportingPaper] = Field(default_factory=list)
    n_support: int = 0
    first_seen_rank: int = 10**9


class DroppedCandidate(BaseModel):
    name: str                           # as extracted
    cleaned_name: str = ""              # after deterministic class-modifier strip
    evidence_quote: str
    paper_ref: str
    reason: str


class DiscoveryResult(BaseModel):
    structural_class: str
    also_known_as: list[str] = Field(default_factory=list)
    discovery_queries: list[str] = Field(default_factory=list)
    hop1_papers: list[dict] = Field(default_factory=list)
    n_papers_retrieved: int = 0
    n_papers_extracted: int = 0
    discovered_systems: list[DiscoveredSystem] = Field(default_factory=list)
    dropped: list[DroppedCandidate] = Field(default_factory=list)
    raw_candidate_names: list[str] = Field(default_factory=list)   # every mention.name, pre-filter
    ungrounded_survivor_count: int = 0
    extraction_seconds_total: float = 0.0
    extraction_errors: list[str] = Field(default_factory=list)
    search_errors: list[str] = Field(default_factory=list)


# --- query building ----------------------------------------------------
def build_discovery_queries(system_core: str, expansion_terms: list[str], *, cap: int = 16) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for t in [strip_size_qualifiers(system_core)] + [strip_size_qualifiers(x) for x in expansion_terms]:
        k = _norm(t)
        if k and k not in seen:
            seen.add(k)
            terms.append(t.strip())
    queries: list[str] = []
    qseen: set[str] = set()
    for t in terms:
        for q in (t, f"{t} structure"):        # OBJECTIVE-FREE -- no folding / CG / objective
            k = _norm(q)
            if k and k not in qseen:
                qseen.add(k)
                queries.append(q)
    return queries[:cap]


# --- grounding one extraction mention --------------------------------
def _paper_text(p) -> str:
    return f"{p.title or ''}  {getattr(p, 'abstract', '') or ''}"


def _paper_ref(p) -> str:
    if getattr(p, "doi", None):
        return p.doi
    oa = getattr(getattr(p, "external_ids", None), "openalex_id", None)
    return oa or (p.title or "")[:80]


def _ground_mention(mention, paper, query: str, rank: int, structural_class: str,
                    banned_norms: set[str], banned_canon: set[str]):
    """Returns (SupportingPaper, cleaned_name) or (None, (drop_reason, cleaned_name)).

    The candidate name is deterministically cleaned of a structural-class
    modifier ("Pertactin beta-helix" -> "Pertactin") and then RE-grounded:
    the cleaned name must be verbatim in the evidence quote, and the quote
    must still tie the system to the structural class. The evidence quote and
    the paper provenance are never modified. No LLM, no name table."""
    tn = _norm(_paper_text(paper))
    raw_quote = (mention.evidence_quote or "").strip()
    qn = _norm(raw_quote)
    stn = _norm(mention.structural_term_matched)
    raw_name = (mention.name or "").strip()

    if not qn:
        return None, ("empty evidence_quote", "")
    if qn not in tn:
        return None, ("evidence_quote is not a verbatim substring of the paper", "")
    if not raw_name:
        return None, ("empty name", "")
    if _norm(raw_name) not in tn:
        return None, ("name is not a verbatim substring of the paper", "")
    if stn and stn not in qn:
        return None, ("structural_term_matched not inside the evidence_quote", "")

    entity_type = getattr(mention, "entity_type", None)
    concrete = entity_type == "specific_protein_or_system"          # the extractor judged it a specific named entity
    clean = _strip_class_modifier(raw_name, structural_class, banned_norms,
                                  mention.structural_term_matched or "", concrete=concrete)
    cn = _norm(clean)
    if not cn:
        return None, ("name is only structural-class wording once the class modifier is stripped", clean)
    gen = _is_generic_or_class_name(clean, banned_norms)
    if gen:
        return None, (gen, clean)
    cv = _class_variant_reason(clean, banned_canon, concrete=concrete)
    if cv:
        return None, (cv, clean)
    if _dnorm(clean) not in _dnorm(raw_quote):
        return None, ("cleaned system name is not verbatim in the evidence quote", clean)
    # Relationship to the class: when the extractor supplied an axis-aligned class_membership_quote and it is verbatim in
    # the paper, THAT is the evidence (it was produced against the axis's own criterion), so the old structural-vocabulary
    # tie test -- fold/shape oriented -- is not applied on top. Mentions without one (the old pipeline) keep the old test.
    membership = (getattr(mention, "class_membership_quote", None) or "").strip()
    membership_ok = bool(membership) and _norm(membership) in tn
    if not membership_ok and not _quote_ties_to_class(qn, stn, banned_canon):
        return None, ("evidence quote no longer ties the system to the structural class", clean)

    # Optional pass-through (see SupportingPaper's own docstring comment):
    # `mention` won't have these attributes at all for the plain
    # extraction.SystemMention -- getattr(..., None) just leaves both
    # None there. `identity_context`, when present, gets
    # the SAME verbatim-substring discipline every other field here already
    # gets -- an ungrounded claim is dropped (None), never trusted as-is.
    identity_context = getattr(mention, "identity_context", None)
    if identity_context and _norm(identity_context) not in tn:
        identity_context = None

    return SupportingPaper(
        paper_ref=_paper_ref(paper),
        title=paper.title or "",
        year=getattr(paper, "year", None),
        found_by_query=query,
        retrieval_rank=rank,
        evidence_quote=raw_quote,
        structural_term_matched=(mention.structural_term_matched or "").strip(),
        system_name=clean,
        name_as_extracted=raw_name,
        name_in_quote=(_dnorm(raw_name) in _dnorm(raw_quote)),
        entity_type=entity_type,
        identity_context=identity_context,
    ), clean


# --- orchestrator ---------------------------------------------------
SearchFn = Callable[[str], list]                       # query -> list[Paper]  (may raise)
ExtractFn = Callable[..., object]                       # (title, abstract, structural_class, also_known_as) -> SystemExtraction


def discover_related_systems(
    system_description: str,
    expansion_terms: list[str],
    *,
    search_fn: SearchFn,
    extract_fn: ExtractFn,
    max_papers_to_extract: int = 25,
    max_discovered: int = 8,
    progress_fn: Optional[Callable[[int, int], None]] = None,
    max_workers: int = 1,
) -> DiscoveryResult:
    """`progress_fn`, if given, is called as `progress_fn(done, total)` once
    per paper in `to_extract`, immediately after that paper's extraction
    attempt resolves (success OR failure -- a failed/erroring paper still
    counts as "done", so progress can never stall on it). `total` is
    `len(to_extract)`, known only once retrieval+dedup above has finished --
    exactly why this is a callback rather than something the caller could
    compute up front. Pure observability: does not read or affect any
    retrieval/grounding/merge/ranking decision below. Default (None)
    reproduces the exact previous behavior.

    `max_workers`, if > 1, runs the extract_fn calls themselves with
    bounded concurrency (concurrent.futures.ThreadPoolExecutor) -- ONLY
    extract_fn calls run concurrently. The grounding/merge step (building
    `systems` below) ALWAYS runs afterward, strictly in the ORIGINAL
    to_extract order -- first_seen_rank/supporting_papers order/ranking/
    provenance never depend on completion order, concurrent or not. With
    concurrency on, progress_fn still fires once per paper, but in
    COMPLETION order rather than to_extract order (a real-time progress
    count -- "N of M done" -- doesn't need to match index order; the merge
    step below is what has the ordering guarantee). Validated by this
    session's own standalone benchmark (run_hop1_concurrency_benchmark.py)
    at this exact call site: workers=2 measured -18.5% wall time vs
    workers=1, 0 errors, byte-identical output across all conditions
    tested. Default (1) reproduces the EXACT previous behavior -- the
    original inline sequential for-loop, not merely
    ThreadPoolExecutor(max_workers=1), so a caller that never opts in sees
    zero surface change at all."""
    structural_class = strip_size_qualifiers(system_description)
    aka = [t.strip() for t in expansion_terms if t.strip()]
    banned_norms = {_norm(structural_class), _norm(system_description)} | {_norm(t) for t in aka}
    banned_norms.discard("")
    banned_canon = {_canon(x) for x in banned_norms}
    banned_canon.discard("")

    queries = build_discovery_queries(system_description, expansion_terms)

    # -- retrieve + dedupe, remembering first (query, rank) that found each paper --
    by_ref: dict[str, dict] = {}
    search_errors: list[str] = []
    rank_counter = 0
    for q in queries:
        try:
            papers = search_fn(q)
        except Exception as exc:  # noqa: BLE001
            search_errors.append(f"{q!r}: {exc}")
            continue
        for p in papers:
            ref = _paper_ref(p)
            if ref not in by_ref:
                by_ref[ref] = {"paper": p, "query": q, "rank": rank_counter}
                rank_counter += 1

    ordered = sorted(by_ref.values(), key=lambda d: d["rank"])
    to_extract = ordered[:max_papers_to_extract]

    result = DiscoveryResult(
        structural_class=structural_class,
        also_known_as=aka,
        discovery_queries=queries,
        n_papers_retrieved=len(ordered),
        n_papers_extracted=len(to_extract),
        search_errors=search_errors,
        hop1_papers=[
            {"paper_ref": _paper_ref(d["paper"]), "title": d["paper"].title,
             "year": getattr(d["paper"], "year", None), "found_by_query": d["query"],
             "has_abstract": bool(getattr(d["paper"], "abstract", None))}
            for d in ordered
        ],
    )

    # -- per-paper extraction (ONE call each, no batching) + grounding --
    systems: dict[str, DiscoveredSystem] = {}
    t0 = time.time()

    def _safe_extract(d: dict) -> tuple[object, Optional[Exception]]:
        p = d["paper"]
        try:
            return extract_fn(p.title or "", getattr(p, "abstract", "") or "",
                              structural_class, aka), None
        except Exception as exc:  # noqa: BLE001 -- a failed paper contributes nothing
            return None, exc

    # index-addressed regardless of completion order -- the merge loop below
    # always walks it in ORIGINAL to_extract order, concurrent or not.
    extraction_results: list[Optional[tuple[object, Optional[Exception]]]] = [None] * len(to_extract)
    if max_workers <= 1:
        for i, d in enumerate(to_extract):
            extraction_results[i] = _safe_extract(d)
            if progress_fn:
                progress_fn(i + 1, len(to_extract))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_index = {pool.submit(_safe_extract, d): i for i, d in enumerate(to_extract)}
            done = 0
            for future in as_completed(future_to_index):
                extraction_results[future_to_index[future]] = future.result()
                done += 1
                if progress_fn:
                    progress_fn(done, len(to_extract))

    for i, d in enumerate(to_extract):
        p, q, rank = d["paper"], d["query"], d["rank"]
        extraction, exc = extraction_results[i]
        if exc is not None:
            result.extraction_errors.append(f"{_paper_ref(p)!r}: {exc}")
            continue
        for mention in extraction.systems:
            result.raw_candidate_names.append(mention.name)
            sp, payload = _ground_mention(mention, p, q, rank, structural_class,
                                          banned_norms, banned_canon)
            if sp is None:
                reason, cleaned_name = payload
                result.dropped.append(DroppedCandidate(
                    name=mention.name, cleaned_name=cleaned_name,
                    evidence_quote=mention.evidence_quote,
                    paper_ref=_paper_ref(p), reason=reason))
                continue
            name_display = payload
            key = _LEADING_ARTICLE.sub("", _norm(name_display))
            ds = systems.get(key)
            if ds is None:
                ds = DiscoveredSystem(name=name_display, norm_name=key, first_seen_rank=rank)
                systems[key] = ds
            if not any(s.paper_ref == sp.paper_ref for s in ds.supporting_papers):
                ds.supporting_papers.append(sp)
            ds.first_seen_rank = min(ds.first_seen_rank, rank)
    result.extraction_seconds_total = round(time.time() - t0, 1)

    for ds in systems.values():
        ds.n_support = len(ds.supporting_papers)
    ranked = sorted(systems.values(), key=lambda s: (-s.n_support, s.first_seen_rank))
    result.discovered_systems = ranked[:max_discovered]

    # -- final audit: for EVERY surviving (system, paper) the quote must be
    #    verbatim in that paper AND the cleaned system name verbatim in the quote --
    text_by_ref = {_paper_ref(d["paper"]): _norm(_paper_text(d["paper"])) for d in ordered}
    ungrounded = 0
    for ds in result.discovered_systems:
        for sp in ds.supporting_papers:
            if _norm(sp.evidence_quote) not in text_by_ref.get(sp.paper_ref, ""):
                ungrounded += 1
            elif _dnorm(sp.system_name or ds.name) not in _dnorm(sp.evidence_quote):
                ungrounded += 1
    result.ungrounded_survivor_count = ungrounded
    return result
