"""Hop-1 semantic search-intent rephrasing. NEW, additive module.

Replaces this round's earlier deterministic "Find X, relevant to Y"
string-template renderer (now deleted -- see history), which mechanically
concatenated axis.scientific_class and the objective through a fixed
connector. Diagnosed live: the result was not just ungrammatical but
semantically WRONG -- "folded domain" silently became "protein clients",
"condensate" got duplicated, and "relevant to" imposed a relationship the
two fragments don't actually have. No fixed template can generalize across
arbitrary (scientific_class, objective) pairs, because two independently-
generated fragments don't compose into fluent, meaning-preserving prose by
string concatenation alone -- an actual rephrasing step is needed.

Why THIS is a separate LLM call rather than a shared one: an earlier
version of this round tried routing direct_concept_search's queries
through a NEW plan-level field on ScientificPlanner and considered reusing
that same field here too -- rejected for Hop-1 even before the field
itself was reverted, because Hop-1's job is discovering OTHER, not-yet-
named systems that share a STRUCTURAL CLASS -- a genuinely different
scientific statement (e.g. "related to autotransporter passenger domain,
in the context of pertactin folding") than a sentence describing the
user's OWN system (e.g. "pertactin folding"), which would, if reused
verbatim, search for MORE pertactin literature instead of related systems.
This cannot be mechanically derived from such a sentence by further string
surgery for the same reason the old template failed -- so this narrow,
Hop-1-specific call is genuinely needed. direct_concept_search's own
equivalent call now lives in concept_3a.py, entirely
separate from this one (each scoped to its own route).

Design, per this round's explicit constraints:
  - ONE call per DISTINCT (scientific_class, objective) pair, not one call
    per query -- discovery.py's own internal loop (build_
    discovery_queries(), UNCHANGED) fires many lexical-style candidate
    queries per Hop-1 axis, but orchestrator.py calls this
    module exactly once per axis (same override-closure trick the deleted
    template used), and the persistent cache below collapses any repeat of
    the exact same pair -- across axes, across cases, across process runs
    -- into zero additional calls.
  - No exposed temperature knob exists on this codebase's LLMBackend
    protocol (see llm.py -- GPT-5.6 Luna is reasoning-effort based,
    not sampling-temperature based); reproducibility is instead achieved
    by the persistent cache below (same inputs -> same cached sentence on
    every call after the first), which is the property that actually
    matters here (stable queries across repeated runs), not literal
    token-level LLM determinism.
  - Strict JSON schema (Hop1SemanticIntent: one string field).
  - Input is ONLY the structured (scientific_class, objective) pair -- no
    target paper, no candidate list, no new scientific assumptions.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from cgscout import cache
from cgscout.llm import LLMBackend

_CACHE_NAMESPACE = "hop1_semantic_intent"
_PROMPT_VERSION = "v1"  # bump to invalidate every cached entry if the prompt below changes meaning

_SYSTEM_PROMPT = """\
You turn two short structured fields from a literature-search planning \
system into ONE natural, grammatically fluent English sentence describing \
a related-system literature search. The sentence is sent directly to a \
semantic-search engine (embedding-based, not keyword matching) -- it must \
read like something a scientist would actually write, not two fragments \
glued together with a fixed connector.

You will be given:
- STRUCTURAL CLASS: a structural or mechanistic class of protein/ \
biomolecular system (e.g. "autotransporter passenger domain", "beta-helix \
fold"). The search should find OTHER systems that share this class -- \
systems not yet named, being discovered BECAUSE they share this \
structural or mechanistic property.
- OBJECTIVE: the scientific process/question the ORIGINAL system (not \
named here, and not to be named in your output) is being studied for \
(e.g. "Folding of the pertactin passenger domain").

Write one sentence that asks a literature search engine to find OTHER \
proteins or biomolecular systems sharing the given structural class, in a \
way that is relevant to the given objective. Integrate the two into ONE \
coherent scientific statement -- do not just glue them together with a \
fixed connector, and do not lose either piece of information.

Hard rules:
- Do not invent new biological facts, mechanisms, or systems beyond what \
the two fields given imply.
- Do not rename or substitute concepts (e.g. do not turn "folded domain" \
into "protein client", do not turn "condensate" into something else) -- \
preserve the given terminology's actual meaning.
- Do not name the ORIGINAL system anywhere in the sentence -- this search \
is specifically for OTHER, related systems, not the original one.
- No conversational wrapper, no "please", no meta-commentary about being \
a search query.
- One sentence, concise, literature-search register -- the kind of \
sentence that would appear in a review paper's topic sentence.

Respond with a single JSON object matching the given schema, and nothing \
else."""


class Hop1SemanticIntent(BaseModel):
    natural_language_intent: str


def _user_message(scientific_class: str, objective: str) -> str:
    lines = [f'STRUCTURAL CLASS: "{scientific_class}"']
    if objective:
        lines.append(
            f'OBJECTIVE (of the original system -- context only, do not '
            f'name the original system): "{objective}"'
        )
    return "\n".join(lines)


def rephrase_hop1_semantic_intent(
    scientific_class: str, objective: str, *, backend: LLMBackend, case_id: Optional[str] = None,
) -> str:
    scientific_class = (scientific_class or "").strip()
    objective = (objective or "").strip()

    key = cache.content_hash(_PROMPT_VERSION, scientific_class, objective)
    cached = cache.get(_CACHE_NAMESPACE, key)  # no ttl_seconds -- same (class, objective) pair means the same intended meaning, not a time-varying query result
    if cached is not None:
        return cached["natural_language_intent"]

    result = backend.structured(
        system=_SYSTEM_PROMPT,
        user=_user_message(scientific_class, objective),
        schema=Hop1SemanticIntent,
        label="hop1_semantic_intent_rephrase",
        purpose="hop1_semantic_query_rendering",
        case_id=case_id,
    )
    cache.set(_CACHE_NAMESPACE, key, result.model_dump())
    return result.natural_language_intent
