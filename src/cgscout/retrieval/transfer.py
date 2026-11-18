"""transfer (formerly orchestrator.py; consolidation 2b, 3) -- the transfer-axis executors that stay out of the dispatch: the 3A executor (direct_concept_search), the search closures
used by Hop-1 / Hop-2 (_make_search_fn: the recorder classifies searches by these function names), the paper-merge helper, and the three assembly steps of 3B
(hop1_seed_terms, hop2_candidates, tag_hop2_papers).  The dispatch over the axes, the 3B workflow itself (cgscout.agent._discover_related_systems), the merge of the direct and
transfer streams and Stage 3 are in cgscout.agent.  What follows is the original description of the module, kept as it was (it still says "this file" for code that used to
share a file with the dispatch and with the 3B workflow).

Original description: executes an
ExpertPriorOutput (transfer_axes) for a resolved
PlannerOutput, merges the transfer evidence stream with the existing
direct-evidence stream, and runs the existing Stage 3 gate ONCE over the
union. NEW, additive module.

Owns zero scientific-judgment logic of its own -- transfer_axes come from
ExpertPriorPlanner (expert_prior_planner.py); this module only executes
what they say, deterministically:

  - transfer_axes: only confidence in {high, medium} are ever retrieved.
    A low-confidence axis is recorded (TransferAxisExecutionLog with
    executed=False) but never searched and never shown -- per this
    session's explicit correction (Stage 3 can only judge "is this a CG
    paper", not "is this transfer scientifically valid", so a scientifically
    weak transfer hypothesis must be stopped before retrieval, not after).
  - execution_mode dispatch reads axis.execution_mode AS GIVEN by the
    model for that specific axis -- there is no basis -> execution_mode
    table anywhere in this file.

direct_concept_search fires 3 fixed, deterministic template queries per
axis (see concept_3a.render_direct_concept_queries(),
no LLM call) via cgscout.retrieval.search.search_and_dedupe() directly
(function-level reuse, bypassing route_plan.py's routing layer
-- there is nothing case-specific this needs from that layer), and unions
their results.

discovery calls discovery.discover_related_
systems() (Hop-1) and hop2 (Hop-2) DIRECTLY
-- the raw retrieval primitives -- rather than going through route_
retrieval_plan.py's structural-branch orchestration. That orchestration's
EvidenceRoutePlan.structural_terms / RouteTerm.source contract is
specifically "only USER_GROUNDED / LITERATURE_GROUNDED, no LLM inference
about what fold a system belongs to" (evidence_plan.py's own
docstring) -- exactly what a transfer_axes hypothesis intentionally IS.
Routing it through that contract would both mislabel provenance and risk
breaking that grounding rule. Calling the two Hop
functions directly reuses the actual, already-validated retrieval
machinery without stretching a different module's contract to cover a
case it was not designed for -- exactly the old lexical structural
morpheme TRIGGER is what is not reused; the Hop-1/Hop-2 EXECUTION machinery
is reused unmodified, seeded by axis.concept instead.

Every LLM step of this branch (terminology expansion, Hop-1 per-paper
extraction, ...) uses the ONE injected `backend` (llm.LLMBackend); a
failure surfaces as that one axis's TransferAxisExecutionLog.error, never a
crash of the whole run.

Fourth-pass fix (same session, diagnosed live via a full provenance audit
on Pertactin -> Pet/Tsh/BpaB/"Bordetella pertussis"): a bare discovered
label is either too generic to search alone (an acronym collision risk --
"Pet" vs polyethylene terephthalate) or, worse, not actually a specific
system at all ("Bordetella pertussis" is an organism, not a protein).
Fixed upstream of this file, not in it: Hop-1's own per-paper extraction
call (still ONE call per paper, no new LLM call) now also classifies each
mention's entity_type and states its grounded identity_context, if the
text gives one -- see extraction.py and related_system_
postprocess.py (identity section). This file's own job is unchanged: it just consumes
identity.query_anchor as before.

Reuses, UNMODIFIED:
  - from_planner.run_retrieval_from_plan()  (direct stream)
  - discovery.discover_related_systems() (Hop-1 orchestration)
  - hop2 (Hop-2: structured search-anchor normalization, <=2 systems per axis, queries A/B (+fallback C);
    replaces the earlier types query concatenation in THIS branch -- see hop2.py)
  - cleanup.cleanup_discovered_systems()
  - terminology.expand_terminology()  (its `call` is
    llm.make_call_fn(backend, response_schema=TerminologyExpansion, ...)
    -- the injected backend, so the whole run uses one backend)
  - extraction.extract_related_systems_rich  (Hop-1's per-paper LLM call, richer schema, also backend-injected -- see that module)
  - filter.filter_papers()  (Stage 3, run ONCE over the merged pool)
  - search.search_and_dedupe()

Query generation (this session's structured-signature redesign): axis.
concept (a single free-text field) is GONE -- TransferAxis now carries
scientific_class + discriminative_features + objective_relevance
separately (expert_prior_planner.py). objective_relevance is reasoning/
audit text only, never literally injected into a query -- the RESOLVED
plan.scientific_objective is what participates in query text, exactly the
same objective the direct route already uses. Every query this module
builds keeps scientific_class as a mandatory prefix; a bare discriminative
feature or a bare related-system label is never searched on its own -- see
postprocess.py (identity section) for how a Hop-1-discovered label is qualified
before ever becoming a Hop-2 anchor.

Second-pass fix (same session, diagnosed live on the phospho-IDP case):
discriminative_features is full scientific-reasoning prose (can run 15+
words per entry, by design -- that field's job is reasoning quality, not
retrievability). Concatenating scientific_class with a whole discriminative_
features sentence produced conjunctive queries so specific OpenAlex
returned zero results, silently erasing the transfer axis's contribution.
Query text is now built ONLY from axis.retrieval_anchors (short,
literature-searchable noun phrases the model compresses from its own
scientific_class/discriminative_features/objective_relevance -- a
SEPARATE field, never a truncation of discriminative_features performed
here). discriminative_features itself is never read for query text
anywhere in this file any more; it is carried through only for display/
audit (see run_expert_prior_live_cases.py).

Third-pass fix (same session, diagnosed live on Pertactin -> EspP):
Hop-1's job is discovery ("why is this related system worth finding"),
Hop-2's job is retrieval ("now that we know its name, search its CG
literature as simply as possible") -- these had been blurring together.
Concretely: (1) `identity.query_anchor` used to be able to lose the
literal discovered label entirely (see postprocess.py's (identity section) own
docstring for the "another SPATE (EspP)" false-positive) -- fixed there,
not here; this file's consumption of `identity.query_anchor` is
unchanged, it just receives a correct value now. (2) Hop-2's own
`scientific_objective` argument used to be the RAW ScientificPlanner
sentence (e.g. "Folding of the Pertactin passenger domain") baked
verbatim into every Hop-2 query -- an earlier version ran it through
`objective_normalization.normalize_objective()` (historical code, git tag
`legacy-pre-cgscout`) so the query became "{related_system}
{normalized_objective} {methodology}" (e.g. "EspP folding coarse-grained
model"); Hop-2 is now built by hop2.py (structured search-anchor
normalization). Hop-1's own query construction (scientific_class +
retrieval_anchors, above) is NOT touched by this pass -- scientific_class
stays central to discovery; it is Hop-2 specifically that should default
to just the discovered system's own name once discovery is done.
Supporting-paper quotes were never query material and still aren't --
`postprocess.py (identity section)`'s own fix already guarantees that.
"""
from __future__ import annotations

from typing import Optional

from cgscout.planning.expert_prior_planner import TransferAxis  # noqa: E402
from cgscout.retrieval.direct.route_plan import RouteRetrievalBudgets  # noqa: E402
from cgscout.retrieval.related_systems.discovery import DiscoveredSystem, _paper_ref  # noqa: E402
from cgscout.retrieval.related_systems.hop2 import SystemInput  # noqa: E402 -- converged Hop-2: structured anchor normalization + at most A/B (+C) queries per system
from cgscout.retrieval.related_systems.postprocess import RelatedSystemIdentity  # noqa: E402
from cgscout.models import Paper  # noqa: E402
from cgscout.retrieval.search import search_and_dedupe  # noqa: E402


def _make_search_fn(
    paper_cache: dict[str, Paper], limit: int, sources=("openalex",),
    *, mode: str = "lexical", semantic_query_override: Optional[str] = None,
):
    """`mode`/`semantic_query_override` are this session's route-based
    lexical/semantic split (see module docstring's Hop-1/Hop-2 callers).
    When `semantic_query_override` is set and mode="semantic", the `query`
    argument the CALLER passes in (e.g. discovery.py's own
    per-term lexical-style query, built inside the UNCHANGED
    build_discovery_queries()) is ignored, and this closure always searches
    with the single precomputed override sentence instead -- relying on
    search_and_dedupe()'s (query, limit, sources, mode)-keyed cache to
    collapse the resulting repeated identical calls into exactly one real
    HTTP round-trip, not a new memoization layer here."""
    def fn(query: str) -> list[Paper]:
        actual_query = semantic_query_override if (mode == "semantic" and semantic_query_override) else query
        papers = search_and_dedupe(actual_query, limit=limit, sources=sources, mode=mode)
        # search_and_dedupe() never raises -- a source failure (including
        # OpenAlex's OpenAlexRateLimitExceeded) is swallowed into
        # .last_errors and an empty/partial result is returned, so a
        # rate-limit exhaustion would otherwise look identical to a
        # genuine "this query has no results" here. Surface it explicitly
        # rather than let a whole axis's zero yield masquerade as a
        # retrieval/scientific finding -- this is the ONLY place that
        # inspects .last_errors; search_and_dedupe's own contract is
        # unchanged.
        errs = getattr(search_and_dedupe, "last_errors", [])
        if errs and any("rate_limit_exceeded" in e for e in errs):
            raise RuntimeError(f"rate_limit_exceeded (query={actual_query!r}): {errs}")
        for p in papers:
            ref = _paper_ref(p)
            existing = paper_cache.get(ref)
            if existing is None:
                paper_cache[ref] = p.model_copy(update={"retrieval_modes": [mode]})
            elif mode not in (existing.retrieval_modes or []):
                paper_cache[ref] = existing.model_copy(update={
                    "retrieval_modes": list(dict.fromkeys((existing.retrieval_modes or []) + [mode])),
                })
        return papers
    return fn


def _run_direct_concept_search(
    axis: TransferAxis, *, queries: list[str],
    paper_cache: dict[str, Paper], budgets: RouteRetrievalBudgets,
) -> tuple[list[Paper], list[str]]:
    """direct_concept_search axes describe a transferable PROCESS/
    CAPABILITY/ENVIRONMENT, never one named entity, by construction -- so
    this route is semantic. `queries` is precomputed by the caller
    (agent.retrieve, via concept_3a.
    render_direct_concept_queries() -- 3 fixed deterministic templates,
    no LLM call) -- this function just executes them and unions the
    results; it builds no query of its own.

    Earlier attempts on this file: (a) a SINGLE fixed Python "Find X,
    relevant to Y" string template -- diagnosed live as not just
    ungrammatical but semantically wrong (silently substituted concepts,
    duplicated clauses, imposed a relationship the two fragments didn't
    actually have); (b) a plan-level canonical_search_intent field on
    ScientificPlanner -- reverted as unnecessary architecture for a
    single-route need; (c) a batched LLM cleanup call -- worked, but was
    shown (2026-09-27/28 forensic trace) to be its own source of run-to-run
    retrieval variance. The current design differs from (a) in the same
    way it differs from betting on one LLM output: 3 differently-structured
    templates are searched and their results unioned here (the `queries`
    loop below), not one template treated as ground truth -- see
    concept_3a.py's module docstring for the
    validation run (12-case pilot benchmark, 0 regressions, 3 recoveries)
    and the one known residual risk it also surfaced."""
    mode = "semantic"
    queries = [q.strip() for q in (queries or []) if q and q.strip()]
    if not queries:
        # Empty means the caller had nothing to render for this axis (cap
        # reached, or empty scientific_class/objective) -- fail closed
        # rather than fabricate a query with no real content.
        return [], []

    combined: dict[str, Paper] = {}
    for query in queries:
        papers = search_and_dedupe(query, limit=budgets.max_papers_per_query, sources=("openalex",), mode=mode)
        errs = getattr(search_and_dedupe, "last_errors", [])
        if errs and any("rate_limit_exceeded" in e for e in errs):
            raise RuntimeError(f"rate_limit_exceeded (query={query!r}): {errs}")
        for p in papers:
            ref = _paper_ref(p)
            cached = paper_cache.get(ref)
            if cached is None:
                paper_cache[ref] = p.model_copy(update={"retrieval_modes": [mode]})
            elif mode not in (cached.retrieval_modes or []):
                paper_cache[ref] = cached.model_copy(update={
                    "retrieval_modes": list(dict.fromkeys((cached.retrieval_modes or []) + [mode])),
                })
            if ref in combined:
                existing = combined[ref]
                combined[ref] = existing.model_copy(update={
                    "found_by_queries": list(dict.fromkeys(existing.found_by_queries + [query])),
                })
            else:
                combined[ref] = p.model_copy(update={
                    "query_categories": list(dict.fromkeys((p.query_categories or []) + ["transfer_direct", f"transfer:{axis.basis}"])),
                    "found_by_queries": list(dict.fromkeys((p.found_by_queries or []) + [query])),
                    "retrieval_modes": list(dict.fromkeys((p.retrieval_modes or []) + [mode])),
                })
            if len(combined) >= budgets.max_papers_per_route:
                break
        if len(combined) >= budgets.max_papers_per_route:
            break
    return list(combined.values()), queries


def hop1_seed_terms(axis: TransferAxis, scientific_class: str, scientific_objective: str, term_aka: list[str]) -> list[str]:
    """The Hop-1 `expansion_terms`: the class combined with each retrieval anchor, the class combined with the objective, THEN the P3 aliases.  PURE, but ORDER-SENSITIVE:
    discover_related_systems receives the list whole and in this order, so it is part of every P5 user message and therefore of every P5 cache key
    (tests/test_consolidation_p5.py).  Moved from the former _run_related_system_discovery (consolidation 3); the statements are unchanged."""
    # Hop-1 seed: scientific_class alone is system_description; every
    # combination below rides in as an "expansion term" of build_discovery_
    # queries() (discovery.py, UNCHANGED) -- which builds
    # [term, term+" structure"] per term -- so scientific_class prefixes
    # every single Hop-1 query by construction, never a bare anchor word.
    # Built from retrieval_anchors (short, searchable phrases), NEVER from
    # discriminative_features sentences -- see module docstring.
    combo_terms = [f"{scientific_class} {anchor}".strip() for anchor in axis.retrieval_anchors]
    combo_terms.append(f"{scientific_class} {scientific_objective}".strip())
    expansion_terms = combo_terms + term_aka
    return expansion_terms


def hop2_candidates(identities: list[RelatedSystemIdentity], systems: list[DiscoveredSystem], *, cap: int) -> list[SystemInput]:
    """The candidates for Hop-2's single normalization call (P6): grounded identities only, paired with the discovered system of the same label, in the stable order
    (-n_support, first_seen_rank), cut to `cap`, re-indexed from 0.  PURE, but CACHE-KEY SENSITIVE: the order, the cut and the indices are part of the P6 user message.
    Two systems whose lower-cased names are equal share one entry of `ds_by_label` (the LAST one wins), so both take its n_support / first_seen_rank.
    `systems` is the cleaned list (cleanup_discovered_systems has already merged duplicates into the survivors, in place).  Moved from the former
    _run_related_system_discovery (consolidation 3); the statements are unchanged apart from `systems` and `cap`."""
    # Hop-2 is CONVERGED (hop2.py): Hop-1 may diverge, Hop-2 must not. Only grounded identities enter; their
    # `query_anchor` (which for distinctive_structure axes carries the whole class phrase) is NOT used as query text.
    ds_by_label = {ds.name.strip().lower(): ds for ds in systems}
    candidates = []
    for ident in identities:
        ds = ds_by_label.get(ident.label.strip().lower())
        if ident.query_anchor and ds is not None:
            candidates.append(SystemInput(
                index=len(candidates), label=ident.label, entity_type=ident.entity_type, identity_context=ident.identity_context,
                aliases=ident.aliases, evidence_quotes=ident.source_grounding, source_titles=ident.source_paper_titles,
                n_support=ds.n_support, first_seen_rank=ds.first_seen_rank))
    candidates = sorted(candidates, key=lambda c: (-c.n_support, c.first_seen_rank))[:cap]
    candidates = [c.model_copy(update={"index": k}) for k, c in enumerate(candidates)]
    return candidates


def tag_hop2_papers(found: dict, paper_cache: dict[str, Paper], axis: TransferAxis) -> list[Paper]:
    """Tag the papers Hop-2 found with their system label, route categories and queries.  COPY-ON-WRITE, but it READS MUTABLE STATE: every copy is derived from the
    CURRENT entry of `paper_cache` (the dict shared by all axes, which the search closures have been updating), never from an earlier copy, with shallow `model_copy`
    semantics (nested objects stay shared with the cache entry); a paper found by two systems yields two copies, each carrying its own label; a ref the cache does not
    have is skipped silently.  The output order is `found` order x refs order, which later becomes the order of transfer_papers.  Moved from the former
    _run_related_system_discovery (consolidation 3); the statements are unchanged."""
    tagged: list[Paper] = []
    for label, refs in found.items():
        for ref, queries in refs.items():
            p = paper_cache.get(ref)
            if p is None:
                continue
            existing_systems = list(p.structural_discovered_systems or [])
            if label not in existing_systems:
                existing_systems.append(label)
            q_categories = list(dict.fromkeys((p.query_categories or []) + ["structural", f"transfer:{axis.basis}"]))
            q_queries = list(dict.fromkeys((p.found_by_queries or []) + list(queries)))
            tagged.append(p.model_copy(update={
                "query_categories": q_categories,
                "found_by_queries": q_queries,
                "structural_discovered_system": p.structural_discovered_system or label,
                "structural_discovered_systems": existing_systems,
            }))
    return tagged


def _merge_papers(existing: dict[str, Paper], new_papers: list[Paper]) -> None:
    for p in new_papers:
        ref = _paper_ref(p)
        if ref in existing:
            prev = existing[ref]
            existing[ref] = prev.model_copy(update={
                "query_categories": list(dict.fromkeys(prev.query_categories + p.query_categories)),
                "found_by_queries": list(dict.fromkeys(prev.found_by_queries + p.found_by_queries)),
                "structural_discovered_systems": list(dict.fromkeys(
                    (prev.structural_discovered_systems or []) + (p.structural_discovered_systems or []))),
                "retrieval_modes": list(dict.fromkeys(
                    (prev.retrieval_modes or []) + (p.retrieval_modes or []))),
            })
        else:
            existing[ref] = p


