"""hop2_converged (scientific-planner-v2 branch) -- a CONVERGED Hop-2 for the transfer-aware related-system branch.

Hop-1 may diverge (it discovers concrete related systems); Hop-2 must converge: "we know the system's name, now find ITS
coarse-grained literature as simply as possible".  The earlier Hop-2 concatenated the system, the scientific class, the whole
objective sentence and structural descriptors into one lexical query, which (OpenAlex lexical search requires every term) made
the query so specific that correct papers were not matched -- several Hop-2 queries returned 0 papers.

Per axis:
  * at most HOP2_MAX_SYSTEMS_PER_AXIS grounded systems enter Hop-2 (ranked as Hop-1 already ranked them: support, then rank);
  * a light, STRUCTURED search-anchor normalization (one batched LLM call; fields search_anchor / core_process /
    stable_descriptor -- the model never writes the query) followed by deterministic validation of every field;
  * per system, normally two primary queries and, only if needed, one fallback:
        A  "{search_anchor} {core_process} coarse-grained model"      (core_process: one phrase from the objective)
        B  "{search_anchor} {stable_descriptor} coarse-grained model" (one identity descriptor)
        C  "{search_anchor} coarse-grained model"                      (fallback: only if A and B return almost nothing)
No class-level expansion, no permutation of descriptors/processes, no class phrase, no objective sentence, no taxon qualifier.
No system name is special-cased anywhere in this module.
"""
from __future__ import annotations

import json
import re
from typing import Callable, Literal, Optional

from pydantic import BaseModel, Field

from cgscout import cache
from cgscout.llm import LLMBackend

METHOD_PHRASE = "coarse-grained model"
HOP2_MAX_SYSTEMS_PER_AXIS = 2          # grounded related systems that may enter Hop-2 per transfer axis
HOP2_MAX_TO_NORMALIZE = 6              # candidates sent to the single normalization call (merge duplicates before cutting to 2)
HOP2_FALLBACK_MAX_RESULTS = 2          # fallback C runs only if A and B together returned at most this many distinct papers
MAX_ANCHOR_WORDS, MAX_PROCESS_WORDS, MAX_DESCRIPTOR_WORDS = 5, 4, 3
_NS = "hop2_anchor_normalization"
_PROMPT_VERSION = "hop2-anchor-v2"

_GENERIC = {"protein", "proteins", "complex", "complexes", "system", "systems", "assembly", "assemblies", "domain", "domains",
            "family", "families", "class", "structure", "structures", "enzyme", "enzymes", "model", "models", "polymer",
            "the", "a", "an", "of", "and", "in", "for", "with", "type"}
# molecular STATE / FORM words: "X" and "X polymer" / "X oligomer" are the same system for the purpose of choosing which
# systems enter Hop-2 (used for the merge key only; the anchor text itself is never rewritten here)
_STATE_WORDS = {"monomer", "monomers", "dimer", "dimers", "trimer", "trimers", "tetramer", "tetramers", "oligomer", "oligomers",
                "polymer", "polymers", "multimer", "homodimer", "heterodimer", "complex", "complexes", "assembly", "assemblies"}
_WORD = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> list[str]:
    return _WORD.findall((s or "").lower().replace("–", " ").replace("—", " "))


def _stem(w: str) -> str:
    return w[:5]


# --------------------------------------------------------------------------- schemas
class _AnchorItem(BaseModel):
    index: int
    search_anchor: str
    core_process: str
    stable_descriptor: str
    descriptor_scope: str = ""     # whole_system_identity | part_or_partner | state_or_condition | none  (self-classification, see prompt)


class _AnchorBatch(BaseModel):
    items: list[_AnchorItem]


class SystemInput(BaseModel):
    index: int
    label: str
    entity_type: Optional[str] = None
    identity_context: Optional[str] = None
    aliases: list[str] = Field(default_factory=list)
    evidence_quotes: list[str] = Field(default_factory=list)
    source_titles: list[str] = Field(default_factory=list)
    n_support: int = 0
    first_seen_rank: int = 10**9


class NormalizedSystem(BaseModel):
    index: int
    label: str
    search_anchor: str
    core_process: str = ""
    stable_descriptor: str = ""
    flags: list[str] = Field(default_factory=list)       # which fields the deterministic validation rejected / fell back on
    n_support: int = 0
    first_seen_rank: int = 10**9


class PlannedQuery(BaseModel):
    role: Literal["A", "B", "C"]
    query: str


class Hop2SystemPlan(BaseModel):
    """What was planned and executed for one system (audit record)."""
    label: str
    search_anchor: str
    core_process: str
    stable_descriptor: str
    flags: list[str]
    queries: list[dict] = Field(default_factory=list)    # {role, query, n_results, error}
    fallback_used: bool = False
    n_papers: int = 0


# --------------------------------------------------------------------------- normalization
_SYSTEM_PROMPT = """\
You prepare SEARCH ANCHORS for a literature search about related biological systems. You receive the SCIENTIFIC OBJECTIVE of \
the user's project and a numbered list of related systems that were found in the literature, each with the label that was \
found, what kind of thing it is, an optional context phrase, and the verbatim evidence sentences and paper titles it came from.

For each system output three short fields. Use only words that appear in that system's label / context / evidence / titles \
(for search_anchor and stable_descriptor) or in the objective (for core_process). Do not use outside knowledge and do not write \
a query.
- search_anchor: the MINIMAL canonical name that identifies the biological system itself, at most 5 words. Keep the system \
noun that identifies it: a bare organism name is NOT a system, so combine it with the system it names (taken from the context \
or evidence). Drop words that only describe a construct, state or experimental form (for example "complex", "polymer", \
"truncation", "cocrystal", "pre-initiation") when the system name stays clear without them. If the label is one organism's \
instance of a more general named system and the evidence shows that, you may give the more general system-level name.
- core_process: ONE short phrase (at most 4 words) naming the core scientific process of the objective, copied or lightly \
inflected from the objective's own words. Add no new mechanism, model requirement or conclusion.
- stable_descriptor: ONE short phrase (at most 3 words) that describes the WHOLE system's stable identity or class \
characteristic -- what kind of thing the system as a whole is, or a defining feature that every instance of that system has. It \
must NOT describe only a part of the system: not a subunit or subcomponent, binding partner, ligand or cofactor, local \
structure or region of one instance, a particular state or conformation, a construct, or an assay/experiment-specific feature. \
It must also NOT be a fold, architecture or other structural-class term, not a reason the system is related, and not an \
organism/taxon qualifier. Use an empty string if no such phrase exists.
- descriptor_scope: classify the stable_descriptor you gave as exactly one of: "whole_system_identity" (it describes the whole \
system's identity/class, as defined above); "part_or_partner" (it names a part of the system, a partner, ligand or local \
structure); "state_or_condition" (a state, construct, assay or condition); or "none" (stable_descriptor is empty). Be strict: \
if the descriptor is a component of the system rather than a description of the system itself, it is "part_or_partner".
Return exactly one item per system, carrying the same index. Respond with a single JSON object matching the schema, and nothing else."""


def _user_message(objective: str, systems: list[SystemInput]) -> str:
    blocks = [f"SCIENTIFIC OBJECTIVE: {objective}", "", "RELATED SYSTEMS"]
    for s in systems:
        blocks.append(f"[{s.index}] label: {s.label}\n    kind: {s.entity_type or 'unknown'}\n    context: {s.identity_context or '-'}")
        for q in s.evidence_quotes[:2]:
            blocks.append(f"    evidence: {q}")
        for t in s.source_titles[:2]:
            blocks.append(f"    paper: {t}")
    return "\n".join(blocks)


def _corpus(s: SystemInput) -> set[str]:
    text = " ".join([s.label, s.identity_context or "", *s.aliases, *s.evidence_quotes, *s.source_titles])
    return set(_tokens(text))


def _fallback_anchor(s: SystemInput) -> str:
    """Deterministic anchor when the model's is unusable: the label, plus the context phrase when the label alone is a bare
    one- or two-word token (e.g. an organism) and the combination stays short."""
    lab = s.label.strip()
    ctx = re.sub(r"^(a|an|the)\s+", "", (s.identity_context or "").strip(), flags=re.I)
    if ctx and len(_tokens(lab)) <= 2 and len(_tokens(f"{lab} {ctx}")) <= MAX_ANCHOR_WORDS + 1:
        return f"{lab} {ctx}"
    return lab


def _validate(item: _AnchorItem, s: SystemInput, objective: str) -> tuple[str, str, str, list[str]]:
    flags: list[str] = []
    corpus, obj_stems = _corpus(s), {_stem(w) for w in _tokens(objective)}
    anchor = " ".join((item.search_anchor or "").split())
    at = _tokens(anchor)
    content = [t for t in at if t not in _GENERIC]
    if not at or len(at) > MAX_ANCHOR_WORDS:
        flags.append("anchor_length"); anchor = ""
    elif not content:
        flags.append("anchor_generic"); anchor = ""
    elif any(t not in corpus for t in at):
        flags.append("anchor_not_grounded_in_hop1_evidence"); anchor = ""
    if not anchor:
        anchor = _fallback_anchor(s); flags.append("anchor_fallback_to_label")
    process = " ".join((item.core_process or "").split())
    pt = _tokens(process)
    if process and (len(pt) > MAX_PROCESS_WORDS or any(_stem(t) not in obj_stems for t in pt if t not in _GENERIC)):
        flags.append("process_not_from_objective"); process = ""
    desc = " ".join((item.stable_descriptor or "").split())
    dt = _tokens(desc)
    if desc:
        if (item.descriptor_scope or "").strip().lower() != "whole_system_identity":
            flags.append("descriptor_not_system_identity"); desc = ""
        elif len(dt) > MAX_DESCRIPTOR_WORDS:
            flags.append("descriptor_length"); desc = ""
        elif not [t for t in dt if t not in _GENERIC]:
            flags.append("descriptor_generic"); desc = ""
        elif any(t not in corpus for t in dt):
            flags.append("descriptor_not_grounded"); desc = ""
        elif set(dt) <= set(_tokens(anchor)):
            flags.append("descriptor_redundant_with_anchor"); desc = ""
    return anchor, process, desc, flags


def normalize_search_anchors(systems: list[SystemInput], objective: str, backend: LLMBackend,
                             case_id: Optional[str] = None) -> list[NormalizedSystem]:
    """One batched, cached, structured call; every returned field is then validated deterministically. A model failure
    degrades to label-only anchors (flagged), never to an exception."""
    if not systems:
        return []
    key = cache.content_hash(_PROMPT_VERSION, backend.name, backend.model, _SYSTEM_PROMPT, _user_message(objective, systems))
    hit = cache.get(_NS, key)
    batch: Optional[_AnchorBatch] = None
    if hit is not None:
        try:
            batch = _AnchorBatch.model_validate(hit)
        except Exception:  # noqa: BLE001
            batch = None
    err = None
    if batch is None:
        try:
            batch = backend.structured(system=_SYSTEM_PROMPT, user=_user_message(objective, systems), schema=_AnchorBatch,
                                       label="hop2_search_anchor", effort="low", purpose="hop2_search_anchor",
                                       case_id=case_id, timeout=90)
            cache.set(_NS, key, batch.model_dump())
        except Exception as exc:  # noqa: BLE001
            batch, err = None, repr(exc)[:120]
    by_idx = {it.index: it for it in (batch.items if batch else [])}
    out = []
    for s in systems:
        it = by_idx.get(s.index)
        if it is None:
            out.append(NormalizedSystem(index=s.index, label=s.label, search_anchor=_fallback_anchor(s), n_support=s.n_support,
                                        first_seen_rank=s.first_seen_rank, flags=["normalization_unavailable" + (f":{err}" if err else "")]))
            continue
        anchor, process, desc, flags = _validate(it, s, objective)
        out.append(NormalizedSystem(index=s.index, label=s.label, search_anchor=anchor, core_process=process, stable_descriptor=desc,
                                    flags=flags, n_support=s.n_support, first_seen_rank=s.first_seen_rank))
    return out


# --------------------------------------------------------------------------- selection + queries
def merge_and_select(normalized: list[NormalizedSystem], max_systems: int = HOP2_MAX_SYSTEMS_PER_AXIS) -> list[NormalizedSystem]:
    """Systems whose normalized anchors coincide (e.g. an entity and its polymer/complex form) become ONE system; then the
    top `max_systems` by (support, first Hop-1 rank) -- the same auditable order Hop-1 already used."""
    merged: dict[str, NormalizedSystem] = {}
    for n in normalized:
        toks = _tokens(n.search_anchor)
        k = " ".join([t for t in toks if t not in _STATE_WORDS] or toks)
        if k in merged:
            m = merged[k]
            m.n_support += n.n_support
            m.first_seen_rank = min(m.first_seen_rank, n.first_seen_rank)
            m.core_process = m.core_process or n.core_process
            m.stable_descriptor = m.stable_descriptor or n.stable_descriptor
        else:
            merged[k] = n.model_copy()
    return sorted(merged.values(), key=lambda m: (-m.n_support, m.first_seen_rank))[:max_systems]


def primary_queries(n: NormalizedSystem) -> list[PlannedQuery]:
    qs, seen = [], set()
    for role, mid in (("A", n.core_process), ("B", n.stable_descriptor)):
        if not mid:
            continue
        q = f"{n.search_anchor} {mid} {METHOD_PHRASE}".strip()
        if q.lower() not in seen:
            seen.add(q.lower()); qs.append(PlannedQuery(role=role, query=q))
    return qs


def fallback_query(n: NormalizedSystem) -> PlannedQuery:
    return PlannedQuery(role="C", query=f"{n.search_anchor} {METHOD_PHRASE}".strip())


def run_converged_hop2(selected: list[NormalizedSystem], search_fn: Callable[[str], list], paper_ref_fn: Callable) -> tuple[list[Hop2SystemPlan], dict[str, dict]]:
    """Executes the plan.  Returns the per-system audit plans and, per system label, {paper_ref: [queries that found it]}.
    Primary queries: A+B; when either cannot be formed (e.g. no valid stable_descriptor), plain C takes its place as the second
    primary (so at most two primaries per system).  When both A and B ran, fallback C runs only if they returned at most
    HOP2_FALLBACK_MAX_RESULTS distinct papers."""
    plans, found = [], {}
    for n in selected:
        plan = Hop2SystemPlan(label=n.label, search_anchor=n.search_anchor, core_process=n.core_process,
                              stable_descriptor=n.stable_descriptor, flags=n.flags)
        refs: dict[str, list[str]] = {}

        def _run(pq: PlannedQuery) -> int:
            try:
                hits = search_fn(pq.query)
                err = None
            except Exception as exc:  # noqa: BLE001 -- (budget exhaustion is a BaseException and still propagates)
                hits, err = [], repr(exc)[:120]
            for p in hits:
                refs.setdefault(paper_ref_fn(p), []).append(pq.query)
            plan.queries.append({"role": pq.role, "query": pq.query, "n_results": len(hits), "error": err})
            return len(hits)

        prim = primary_queries(n)
        c_is_primary = len(prim) < 2                      # no descriptor (B) or no process (A): plain C is the second primary
        if c_is_primary:
            prim = prim + [fallback_query(n)]
        for pq in prim:
            _run(pq)
        distinct_ab = len(refs)
        if c_is_primary:
            pass                                          # C already ran as a primary query; nothing further
        elif distinct_ab <= HOP2_FALLBACK_MAX_RESULTS:
            fb = fallback_query(n)
            if fb.query.lower() not in {q["query"].lower() for q in plan.queries}:
                plan.fallback_used = True
                _run(fb)
        plan.n_papers = len(refs)
        plans.append(plan)
        found[n.label] = refs
    return plans, found
