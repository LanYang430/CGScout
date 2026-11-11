"""ExpertPriorPlanner (scientific-planner-v2 branch) -- the second, focused
Luna structured call. NEW, additive module.

Runs ONLY after ScientificPlanner has reached status="ready" (clarification
already resolved) -- see session.PlannerSession / cli.py.
Produces ONE judgment: the transfer hypothesis.  (A former second judgment, `canonical_methods`, was removed on 2026-10-28: BEAM no longer
proposes or independently verifies "commonly considered" CG methods; method papers reach the user only through the normal retrieval routes.)

  transfer hypothesis (transfer_needed / transfer_axes): which
     PROPERTIES of this system/objective would make CG-model evidence from
     ANOTHER system scientifically transferable -- not "which other system
     looks similar". Replaces the old lexical-morpheme-triggered structural
     branch. Every axis also carries execution_mode, the MODEL's own
     judgment of how that axis should be searched (direct_concept_search vs
     discovery) -- this module and its downstream bridge
     never hardcode a basis -> execution_mode table; dispatch always reads
     whatever the model returned for that specific axis.

Each transfer_axes entry is a STRUCTURED signature (scientific_class +
discriminative_features + objective_relevance), not a single free-text
`concept` string -- that single-field design was tried first and diagnosed
live (this session) to silently collapse two different jobs into one
string: "what scientific class constrains the search space" and "which
specific feature is discriminative". Whichever job the model happened to
carry over into that one string on a given call was pure wording luck --
confirmed by a live, reproducible asymmetry (EspP's concept happened to
retain "autotransporter passenger domain" and could rediscover Pertactin;
Pertactin's concept dropped it and searched pure fold vocabulary instead,
so it could not rediscover EspP, in either of two independent reruns).
Splitting the fields removes the wording-luck dependency: scientific_class
is always carried into every downstream query (see transfer_retrieval_
bridge.py), so retrieval symmetry no longer depends on which way the model
happened to phrase one string.

`retrieval_anchors` (added one pass later, this session) is a further
split -- diagnosed live: even after the scientific_class/discriminative_
features/objective_relevance split above, orchestrator.py
was concatenating scientific_class with a full discriminative_features
SENTENCE into one query, and a 15-20 word conjunctive query routinely
returned zero OpenAlex results (a live, reproducible regression on the
phospho-IDP case). discriminative_features' job is unchanged -- full
scientific reasoning, as long a sentence as needed, never shortened for
retrieval's sake. retrieval_anchors' job is purely the search-term
compression of that same content -- short noun phrases, faithful to what
the axis already says, never a new concept -- so reasoning quality and
query retrievability stop being the same field fighting over one length
budget.

Generic by construction, same discipline as scientific_planner.py: the
prompt names no protein, no fold family, no specific system anywhere.

Abstain is a first-class, expected outcome, not a fallback: transfer_needed
may be false with an empty transfer_axes list (no property here is
discriminative enough to be worth searching for), and even when
transfer_needed is true, individual bases the model is not confident about
are simply omitted -- never force-filled "for completeness".

Input surface is deliberately narrower than the full PlannerOutput:
system / target_component / scientific_objective / search_context only.
PlannerOutput.search_axes is NOT passed in (explicit design decision, this
session) -- it is ScientificPlanner's own free-text hint field, and feeding
it here risked anchoring this call's transfer-axis judgment on wording a
different call already produced for a different purpose.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from cgscout.llm import LLMBackend


class TransferAxis(BaseModel):
    basis: Literal[
        "distinctive_structure",
        "family_homology",
        "physical_process",
        "environment_context",
        "chemistry_representation",
    ]
    scientific_class: str
    discriminative_features: list[str] = Field(default_factory=list)
    objective_relevance: str
    retrieval_anchors: list[str] = Field(default_factory=list)
    rationale: str
    confidence: Literal["high", "medium", "low"]
    execution_mode: Literal["direct_concept_search", "related_system_discovery"]


class ExpertPriorOutput(BaseModel):
    transfer_needed: bool
    transfer_axes: list[TransferAxis] = Field(default_factory=list)


EXPERT_PRIOR_SYSTEM_PROMPT = """\
You are the expert-prior planner for a literature-discovery agent focused on \
coarse-grained (CG) protein/biomolecular simulation models. You are given \
an already-resolved system + scientific objective (any clarification has \
already happened -- do not ask questions, do not request more information). \
Produce the transfer hypothesis described below.

=== transfer hypothesis (transfer_needed / transfer_axes) ===

The question is NOT "what other system looks similar to this one?". It is:

    For this scientific task, which PROPERTIES of the system/objective are \
most likely to determine whether CG-model evidence from ANOTHER system \
is scientifically transferable to this one?

A model is worth transferring when the source and target share the \
physical features that matter to the modeling objective -- not merely \
because two systems resemble each other in some generic way. Each \
transfer_axes entry must be built from four SEPARATE fields, not one \
blended phrase:

- `scientific_class`: the scientific class this target belongs to that \
genuinely CONSTRAINS the search space for related evidence -- e.g. a named \
structural/functional class, not a bare fold word and not the system's own \
proper name. This is the field that keeps retrieval anchored: it must be \
specific enough that literature search on it alone already returns a \
meaningfully narrower population than "protein" -- e.g. "immunoglobulin-like \
domain" narrows the search space; "beta-sandwich" alone, detached \
from what KIND of thing has that fold, does not carry the same constraint \
and must not be used as the scientific_class by itself when a narrower \
class is available.
  For an axis whose execution_mode is "related_system_discovery", \
scientific_class must ALSO be general enough to cover OTHER named systems \
that could transfer to the target: state the shared property one level of \
abstraction ABOVE the target itself. Do not carry over qualifiers that are \
specific to the target -- its species, organism, taxonomic domain, \
construct, or cell type -- unless that qualifier is itself the property \
that makes evidence transferable. If the class would still only describe \
the target's own taxon or its immediate relatives, it is too narrow: \
systems from other organisms or domains of life that share the property \
would be excluded from discovery by construction.
- `discriminative_features`: a list of the specific physical/structural/\
mechanistic features that make evidence transfer scientifically justified \
for THIS case, one of these bases per axis:
  - distinctive_structure: an unusual, specific fold/topology/architecture \
(NOT "contains alpha helices" or "contains beta sheets" -- those carry no \
selective power and must never be used alone as a feature).
  - family_homology: family/homolog membership, but ONLY when membership \
in that family genuinely implies a similar physical modeling problem -- \
not membership for its own sake.
  - physical_process: a specific physical process or mechanism (e.g. a \
folding regime that is mechanistically distinct from generic folding, an \
association/insertion process, an ensemble-generating process) that is \
itself the discriminating feature, not just "folding" or "binding" stated \
generically.
  - environment_context: a physical environment/context (confinement, a \
crowded/cellular environment, a bound partner, ...) that materially \
changes what CG representation choices matter.
  - chemistry_representation: a specific representation requirement (a \
post-translational modification, an electrostatics/charge-pattern need, a \
sequence-specificity requirement, ...) that a CG model must explicitly \
handle.
- `objective_relevance`: one sentence on why these features matter FOR THE \
USER'S STATED OBJECTIVE specifically (this is a reasoning/audit field, \
read by a human and provided as context for downstream literature \
relevance assessment -- it is not itself literature-searchable text, so \
do not write it as a query fragment).
- `retrieval_anchors`: a SEPARATE list of short, literature-searchable \
scientific noun phrases distilled from scientific_class/discriminative_\
features/objective_relevance above -- the phrases a scientist would \
actually type into Google Scholar or OpenAlex, not the reasoning prose \
itself. This is a distinct job from discriminative_features: that field's \
job is to state the full scientific justification (as complete a \
sentence as needed); this field's job is to compress that justification \
into search terms. Rules:
  - Each anchor must be faithful to content already stated in this axis's \
own scientific_class/discriminative_features/objective_relevance -- never \
introduce a new scientific concept, a new named CG model/method, or a \
recommendation here.
  - Short noun phrases (roughly 2-5 words), not full sentences and not a \
restatement of the rationale.
  - Give as many as are genuinely useful, but keep them distinct from each \
other -- do not pad with near-duplicates.
  - Leave this empty only if you cannot compress the axis into any useful \
search phrase (should be rare when scientific_class/discriminative_\
features are themselves substantive).

For each axis genuinely worth naming, also give: `rationale` (why this \
property, for this case, is likely to determine transferability), \
`confidence` (high/medium/low -- how confident you are this is a genuinely \
discriminative, case-specific property, not a generic one), and \
`execution_mode`:

- "related_system_discovery": use when answering this axis really \
requires FIRST discovering which OTHER NAMED systems share this property \
(typically -- but not exclusively -- a structural/topological/family-type \
property), grounded via literature, before methodology evidence on those \
systems is searchable.
- "direct_concept_search": use when scientific_class + the user's scientific \
objective is already directly literature-searchable together with "coarse-grained \
model" -- no prior step of naming another system is needed (typically -- \
but not exclusively -- a process/environment/representation-type \
property).
Choose whichever mode actually fits THIS axis's reasoning, case by case -- \
these are illustrative tendencies, not a fixed rule tied to the basis \
category.

Hard rules:
- transfer_needed=false with an empty transfer_axes list is a common, \
correct answer -- most requests have no genuinely discriminative property \
worth transfer-searching for. Do not manufacture an axis to look thorough.
- Even when transfer_needed=true, only include a basis you are actually \
confident is discriminative for THIS case -- omit any basis that does not \
apply rather than filling all five "for completeness". A basis that is not \
genuinely discriminative here (e.g. a common fold, a process named only in \
generic terms, no stated environment/representation need) must not appear \
at all, not appear with low confidence as a placeholder.
- A property so generic that it would apply to almost any protein in this \
broad category is NOT discriminative -- do not use it.
- Prefer abstaining over generating a scientifically weak transfer axis.

Respond with a single JSON object matching the given schema, and nothing else."""


def _build_user_message(
    system: str,
    target_component: Optional[str],
    scientific_objective: str,
    search_context: list[str],
) -> str:
    lines = [f'SYSTEM (already resolved): "{system}"']
    if target_component:
        lines.append(f'TARGET COMPONENT: "{target_component}"')
    lines.append(f'SCIENTIFIC OBJECTIVE (already resolved): "{scientific_objective}"')
    if search_context:
        lines.append(f"STATED CONTEXT: {search_context!r}")
    return "\n".join(lines)


class ExpertPriorPlanner:
    def __init__(self, backend: LLMBackend, *, effort: Optional[str] = None):
        self._backend = backend
        self._effort = effort

    def propose(
        self, *, system: str, target_component: Optional[str],
        scientific_objective: str, search_context: list[str],
        case_id: Optional[str] = None,
    ) -> ExpertPriorOutput:
        return self._backend.structured(
            system=EXPERT_PRIOR_SYSTEM_PROMPT,
            user=_build_user_message(system, target_component, scientific_objective, search_context),
            schema=ExpertPriorOutput,
            label="expert_prior_planner",
            effort=self._effort,
            purpose="transfer_expansion",
            case_id=case_id,
        )
