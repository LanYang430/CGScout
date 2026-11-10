"""Hop-1 conservative cleanup: high-confidence duplicate/fragment handling
ONLY. Called from orchestrator.py, strictly between
discover_related_systems() (Hop-1 discovery + grounding, UNTOUCHED) and the
Hop-2 query building (UNTOUCHED) -- so
Hop-1's own recall behavior is provably unaffected by anything in this
module; this is a post-hoc cleanup layer over its RESULT, not a change to
how that result is produced.

Deliberately does NOT judge whether a candidate is a "concrete system" --
that boundary is unstable (diagnosed this session: Chlamydia/pylori/CRD/TPS/
short+long repeats(SRs/LRs)/Polymers are all left untouched on purpose, even
though some look borderline). Only three narrow, high-confidence
transformations:

  1. suffix duplicate merge -- a candidate whose normalized name ends, at a
     whole-word boundary, with an already-surviving system's normalized
     name (e.g. "of the Escherichia coli Autotransporter EspP" ends with
     "EspP") merges into it. Pure string containment, no semantic judgment
     about what the extra words mean.
  2. exact fragment blocklist -- a candidate whose (normalized) name is
     EXACTLY one of a small, hand-curated, closed set of pure relational/
     connective words that carry no naming content by themselves. Seeded
     ONLY with real observed cases ("related", "double-stranded") -- expand
     this set only by adding another CONFIRMED observed case, never by
     guessing plausible-sounding analogues ("single-stranded" is a
     predictable sibling but has NOT been observed, so it deliberately
     stays out).
  3. objective-aware interaction-duplicate merge -- a "{known system}...
     {interaction/complex/binding word}"-shaped candidate (e.g. "EspP-BamA
     interaction") folds back into the known system it names, but ONLY when
     the objective is classified (by a small closed keyword list, no LLM)
     as a monomer/structural-process objective (folding, unfolding,
     conformational stability, ...). For an interaction/binding-type
     objective -- or an objective this module cannot confidently classify
     -- the candidate is left standalone: interaction may BE the thing
     being studied, and this module must never guess wrong toward deleting
     evidence for it.

Every merge/drop is recorded in the returned log entry list -- never
silent; provenance (supporting_papers / evidence_quote / paper_ref) is
never discarded by a merge, only consolidated onto the surviving name.
"""
from __future__ import annotations

import re


from cgscout.retrieval.related_systems.discovery import DiscoveredSystem, _dnorm, _norm  # noqa: E402 -- reused, not reimplemented

__all__ = ["cleanup_discovered_systems"]


# --- 2. exact fragment blocklist -------------------------------------------
# Closed set. Every entry is a REAL case observed this session surviving
# Hop-1 grounding despite carrying zero naming content on its own (a bare
# connective/relational adjective torn out of the sentence immediately
# around the structural-class mention). See module docstring: expand only
# by adding another CONFIRMED case.
_FRAGMENT_BLOCKLIST = {_dnorm(x) for x in ("related", "double-stranded")}


# --- 1. suffix duplicate merge ---------------------------------------------
_MIN_MERGE_TARGET_ALNUM_LEN = 4  # a candidate shorter than this can never be a merge TARGET


def _alnum_len(s: str) -> int:
    return len(re.sub(r"[^a-z0-9]", "", s.lower()))


def _ends_with_word_boundary(a: str, b: str) -> bool:
    """True if normalized `a` ends with normalized `b` at a whole-word
    boundary (not merely as a raw substring) -- "...autotransporter espp"
    ends with "espp" this way; a hypothetical "prespp" would not."""
    if not a or not b or not a.endswith(b):
        return False
    if len(a) == len(b):
        return True
    return not a[-(len(b) + 1)].isalnum()


def _contains_word_boundary(haystack: str, needle: str) -> bool:
    if not needle:
        return False
    pattern = rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])"
    return re.search(pattern, haystack) is not None


def _merge_supporting_papers(target: DiscoveredSystem, source: DiscoveredSystem) -> None:
    """Appends `source`'s supporting papers onto `target`'s -- never
    prepended, so target.supporting_papers[0] (used elsewhere as "the"
    primary evidence for this system) always stays the target's own
    original evidence, never the absorbed duplicate's. Dedupes by
    paper_ref so the same paper is never listed twice."""
    existing_refs = {sp.paper_ref for sp in target.supporting_papers}
    for sp in source.supporting_papers:
        if sp.paper_ref not in existing_refs:
            target.supporting_papers.append(sp)
            existing_refs.add(sp.paper_ref)
    target.first_seen_rank = min(target.first_seen_rank, source.first_seen_rank)
    target.n_support = len(target.supporting_papers)


def _merge_suffix_duplicates(systems: list[DiscoveredSystem], log: list[str]) -> list[DiscoveredSystem]:
    # Process SHORTEST names first, so a short/simple name is already an
    # established candidate TARGET by the time a longer candidate is
    # checked against it (a longer candidate needs the short one to
    # already be present in `survivors` to find it).
    ordered = sorted(systems, key=lambda d: _alnum_len(d.norm_name))
    survivors: list[DiscoveredSystem] = []
    for d in ordered:
        target = None
        for s in survivors:
            if s.norm_name == d.norm_name:
                continue  # identical normalized names are already merged upstream (Hop-1's own dict-by-key merge)
            if _alnum_len(s.norm_name) < _MIN_MERGE_TARGET_ALNUM_LEN:
                continue
            if _ends_with_word_boundary(_dnorm(d.norm_name), _dnorm(s.norm_name)):
                target = s
                break
        if target is not None:
            _merge_supporting_papers(target, d)
            log.append(
                f"duplicate_merge: {d.name!r} -> {target.name!r} "
                f"(normalized name ends with existing system name at a word boundary)"
            )
        else:
            survivors.append(d)
    return survivors


# --- 3. objective-aware interaction-duplicate merge -------------------------
_INTERACTION_WORDS = ("interaction", "interactions", "complex", "complexes", "binding", "interface")

_MONOMER_STRUCTURAL_OBJECTIVE_KEYWORDS = (
    "folding", "unfolding", "refolding", "misfolding", "fold",
    "conformational stability", "conformation", "stability", "denaturation",
    "structural stability",
)
_INTERACTION_OBJECTIVE_KEYWORDS = (
    "binding", "interaction", "protein-protein interaction", "complex formation",
    "translocation", "docking", "recognition", "assembly",
)


def _objective_allows_interaction_merge(clean_objective: str) -> bool:
    """Conservative, closed-keyword-only classification -- no LLM, no new
    classifier. If the objective mentions ANY interaction-type keyword, an
    interaction candidate is always left standalone (interaction may BE the
    thing being studied). Merge is allowed ONLY when the objective clearly
    names a monomer/structural-process keyword AND mentions none of the
    interaction-type keywords. An empty, unrecognized, or ambiguous
    objective defaults to NOT merging -- the safer direction when uncertain."""
    obj = _norm(clean_objective)
    if not obj:
        return False
    if any(kw in obj for kw in _INTERACTION_OBJECTIVE_KEYWORDS):
        return False
    return any(kw in obj for kw in _MONOMER_STRUCTURAL_OBJECTIVE_KEYWORDS)


def _merge_interaction_duplicates(
    systems: list[DiscoveredSystem], clean_objective: str, log: list[str],
) -> list[DiscoveredSystem]:
    if not _objective_allows_interaction_merge(clean_objective):
        return systems
    known = [d for d in systems if _alnum_len(d.norm_name) >= _MIN_MERGE_TARGET_ALNUM_LEN]
    survivors: list[DiscoveredSystem] = []
    for d in systems:
        dn = _dnorm(d.norm_name)
        has_interaction_word = any(_contains_word_boundary(dn, w) for w in _INTERACTION_WORDS)
        target = None
        if has_interaction_word:
            # Existing rank order (systems is already ranked coming in)
            # decides which known name wins if a candidate happens to name
            # more than one -- first match in that order, always logged so
            # it stays inspectable rather than silently arbitrary.
            for k in known:
                if k.norm_name == d.norm_name:
                    continue
                if _contains_word_boundary(dn, _dnorm(k.norm_name)):
                    target = k
                    break
        if target is not None:
            _merge_supporting_papers(target, d)
            log.append(
                f"interaction_merge: {d.name!r} -> {target.name!r} "
                f"(contains known system name + interaction-type word; "
                f"objective classified as monomer/structural-process)"
            )
        else:
            survivors.append(d)
    return survivors


# --- orchestrator -----------------------------------------------------------

def cleanup_discovered_systems(
    discovered_systems: list[DiscoveredSystem], *, clean_objective: str,
) -> tuple[list[DiscoveredSystem], list[str]]:
    """Applies, in order: (1) exact fragment blocklist, (2) suffix duplicate
    merge, (3) objective-aware interaction-duplicate merge -- then re-ranks
    survivors by the SAME (-n_support, first_seen_rank) key
    discover_related_systems() already uses (reused, not a new ranking
    rule: merging can raise a survivor's n_support, e.g. EspP 1->2, so
    re-applying the existing tie-break after merging is necessary for the
    result to stay correctly ordered, not an added scientific judgment).

    Order matters: the fragment filter runs FIRST so a blocklisted word
    (e.g. "related") can never itself become a duplicate-merge TARGET
    before being dropped.

    Returns (cleaned_systems, audit_log) -- audit_log has one line per
    merge/drop, never silent."""
    log: list[str] = []
    systems = list(discovered_systems)

    kept = []
    for d in systems:
        if _dnorm(d.name) in _FRAGMENT_BLOCKLIST or _dnorm(d.norm_name) in _FRAGMENT_BLOCKLIST:
            log.append(f"fragment_drop: {d.name!r} (exact match in closed fragment blocklist)")
            continue
        kept.append(d)
    systems = kept

    systems = _merge_suffix_duplicates(systems, log)
    systems = _merge_interaction_duplicates(systems, clean_objective, log)

    systems = sorted(systems, key=lambda s: (-s.n_support, s.first_seen_rank))
    return systems, log
