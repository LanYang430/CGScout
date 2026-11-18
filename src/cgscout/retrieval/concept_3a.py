"""direct_concept_search query construction. NEW, additive module.

Scoped minimally, per explicit instruction: this round only touches
direct_concept_search's own query construction, nothing else (Hop-1,
Hop-2, the direct route, canonical-method verification, ExpertPriorPlanner's
own schema are all untouched -- see transfer.py).

History on this one file (kept here rather than in git alone, since the
earlier attempts are exactly what this round had to reckon with):
  1. A fixed Python "Find X, relevant to Y" single string template --
     diagnosed live as not just ungrammatical but semantically wrong
     (silently substituted concepts, duplicated clauses, imposed a
     relationship the two fragments didn't actually have). Deleted.
  2. A single batched LLM cleanup call (axis.scientific_class + the
     objective, rewritten into one natural sentence by GPT-5.6 Luna,
     cached per (scientific_class, objective) pair). This is what THIS
     file held immediately before the current round.
  3. THIS round: back to deterministic, no-LLM rendering -- but not a
     repeat of (1). Where (1) tried to guarantee ONE "correct" sentence,
     this renders 3 differently-STRUCTURED templates per axis and unions
     their search results (transfer.py's
     _run_direct_concept_search already loops over a queries list, so this
     is a union at the retrieval layer, not a single bet). Motivation: a
     forensic trace on one case (the 2026-09-27/28 thermoresponsive-IDP
     session) showed the LLM cleanup step (2) itself varies run to run --
     same axis, same objective, different final sentence, sometimes
     flipping a paper from found to not-found at the real
     max_papers_per_query=20 cutoff -- while OpenAlex's own semantic
     ranking is deterministic for identical query text (verified directly:
     two replays of the same string return the same rank). Removing (2)'s
     LLM call removes that one variance source outright.
     Validated (not just argued) before landing: replayed real production
     axes from 8 of the 12-case pilot benchmark (the other 4 either
     triggered a ScientificPlanner clarification or had their only axis
     routed to discovery, so had no direct_concept_search
     axis to compare) against this exact rendering, same OpenAlex
     mode=semantic depth=20 as production -- 15 (axis, target) pairs: 0
     regressions (baseline hit -> template miss), 3 recoveries (baseline
     miss -> template hit), the rest unchanged or unchanged-improved rank.
     The scripts and the log of the raw run (2026-09-29 session) are not
     part of this repository -- not reproduced here since it depends
     on live OpenAlex/LLM calls, but the aggregate is the empirical
     grounding for this design, not a hunch.
     KNOWN RESIDUAL RISK, found while spot-checking the SAME validation
     run's rendered query text (not hypothetical): (2)'s LLM cleanup step
     was implicitly doing light cleanup on messy upstream `objective` text
     (e.g. it silently dropped a visible self-correction artifact --
     "...(UCST? No -- lower-critical-solution-temperature, LCST-like)..."
     -- that ScientificPlanner had leaked into scientific_objective for
     one case in that run). This deterministic renderer does NOT clean
     that up; it substitutes the field verbatim. It did not cause a
     measured regression in the validation run (that case's baseline
     already missed too), but it is a real, demonstrated gap -- fixing it
     belongs to making ScientificPlanner/ExpertPriorPlanner's own output
     more canonical (a separate, deferred piece of work), not to this
     file papering over upstream noise.
"""
from __future__ import annotations

import re

# Two tiers: (1) ANY leading gerund ("...ing ") is stripped unconditionally
# -- ScientificPlanner's scientific_objective is, by its own prompt design,
# always phrased as a stated research action, so a leading "-ing" word is
# reliably that action verb, not a noun modifier. (2) A bare-infinitive
# form (no reliable suffix to detect generically) falls back to an
# explicit, deliberately generous list. Found empirically incomplete twice
# already during this round's own production-verification runs ("Represent
# glycan flexibility..." wasn't caught the first time -- see module
# docstring); this list is a pragmatic heuristic, not a closed set --
# expect it to need occasional additions as new objective phrasings show up.
_LEADING_GERUND_RE = re.compile(r"^\s*([A-Za-z]+ing)\s+")
_LEADING_BARE_VERB_RE = re.compile(
    r"^\s*(Model|Simulate|Predict|Understand|Determine|Investigate|Identify|Assess|"
    r"Study|Explore|Test|Analyze|Characterize|Establish|Evaluate|Quantify|Capture|"
    r"Represent|Describe|Reproduce|Resolve|Address|Explain|Reveal|Demonstrate|"
    r"Account|Enable|Couple|Recognize|Recapitulate|Distinguish|Compare|Probe|"
    r"Elucidate|Interrogate|Map|Recover|Extract|Infer|Derive|Compute|Calculate|"
    r"Design|Construct|Build|Develop|Validate|Verify|Show|Test)\s+",
    re.IGNORECASE,
)


def extract_phenomenon(objective: str) -> str:
    """Deterministic stand-in for a 'phenomenon' controlled field: strips
    ONE leading gerund/verb (e.g. "Modeling ", "Determine ", "Represent ")
    from a scientific_objective sentence so it reads naturally inside a
    template slot ("...to model {phenomenon}", "...modeling of
    {phenomenon}").

    Deliberately does NOT also strip a following "whether" -- "to model
    whether X" / "modeling of whether X" are both grammatical on their
    own; stripping "whether" too turns "Determine whether X alter Y" into
    the bare clause "X alter Y", which reads as a broken noun phrase once
    substituted into a template slot (found live in this round's own
    validation run, case 4_enzyme_channeling -- see module docstring)."""
    s = (objective or "").strip()
    new_s, n = _LEADING_GERUND_RE.subn("", s, count=1)
    if n == 0:
        new_s, n = _LEADING_BARE_VERB_RE.subn("", s, count=1)
    s = new_s.rstrip(". ")
    if s:
        s = s[0].lower() + s[1:]
    return s


def render_direct_concept_queries(scientific_class: str, objective: str) -> list[str]:
    """Returns 3 fixed, deterministic search sentences for one
    direct_concept_search axis (scientific_class, scientific_objective),
    or [] if either input is empty (fail closed, same contract the
    previous LLM-cleanup version had for "no query produced"). No LLM
    call, no cache, no case_id -- purely a function of its two inputs, so
    identical inputs always render identical queries."""
    scientific_class = (scientific_class or "").strip()
    objective = (objective or "").strip()
    if not scientific_class or not objective:
        return []
    phenomenon = extract_phenomenon(objective)
    if not phenomenon:
        return []
    return [
        f"Coarse-grained models for {scientific_class} to model {phenomenon}.",
        f"Modeling {phenomenon} in {scientific_class} using coarse-grained approaches.",
        f"{scientific_class}: coarse-grained modeling of {phenomenon}.",
    ]
