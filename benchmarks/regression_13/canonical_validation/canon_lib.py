"""Offline canonical-schema experiment library (NOT production code; nothing in
beam_agent imports this).

Layer 1 (deterministic, no LLM): normalize_system() -> entity / part / aliases /
scale_hint / residual clause.
Layer 2 (LLM, one call per record): fills only fields needing judgement.
  Variant A: process / condition / context(list[str]) / structural_features /
             modeling_goal (<=30 words, must preserve every stated constraint)
  Variant B: A, but modeling_goal is a short main ask (<=15 words) and every
             explicit user constraint/contrast goes into `constraints`.
Revisions vs the first offline round (driven by its data): context is a flat
list (its `kind` drifted and no downstream consumer reads it), no
system_type (unstable, redundant with entity), structural_features kinds
collapsed to fold/family/disorder_state.
"""
import re
from typing import Literal, Optional
from pydantic import BaseModel

# ---------------------------------------------------------------- layer 1
_ARTICLE_RE = re.compile(r"^(a|an|the)\s+", re.I)
_QUANT_RE = re.compile(r"^(thousands|hundreds|millions|dozens|many|numerous|several|multiple)(\s+of)?\s+", re.I)
_CLAUSE_RE = re.compile(r"\s+(that|which|undergoing|inside|within|in|when|under|at|during|as|while|upon|with|containing)\s+", re.I)
_PAREN_RE = re.compile(r"\(([^)]*)\)")
_SING = {"proteins": "protein", "domains": "domain", "enzymes": "enzyme", "peptides": "peptide",
         "complexes": "complex", "glycoproteins": "glycoprotein", "assemblies": "assembly"}
_GENERIC = {"protein", "proteins", "folded", "unfolded", "single", "small", "large", "domain", "region"}
_PART_NOUN = r"(?:domain|region|subunit|terminus|loop|segment)"
_PART_OF_RE = re.compile(rf"^(?P<part>[\w\-]+(?:\s[\w\-]+)*?\s?{_PART_NOUN})\s+of\s+(?P<ent>.+)$", re.I)
_PART_TRAIL_RE = re.compile(rf"^(?P<ent>.+?)\s+(?P<mod>[\w\-]+)\s+(?P<noun>{_PART_NOUN})$", re.I)


def normalize_system(raw: str) -> dict:
    s = (raw or "").strip().rstrip(".")
    s = _ARTICLE_RE.sub("", s)
    scale_hint = None
    m = _QUANT_RE.match(s)
    if m:
        scale_hint = m.group(0).strip()
        s = s[m.end():]
    residual = None
    m = _CLAUSE_RE.search(s)
    if m and m.start() > 0:
        residual = s[m.start():].strip()
        s = s[:m.start()].strip()
    aliases = [a.strip() for a in _PAREN_RE.findall(s)]
    s = _PAREN_RE.sub("", s).strip()
    s = re.sub(r"\bprotein (domain|region)\b", r"\1", s, flags=re.I)
    toks = s.split()
    if toks and toks[-1].lower() in _SING:
        toks[-1] = _SING[toks[-1].lower()]
    s = " ".join(toks)
    part, entity = None, s
    m = _PART_OF_RE.match(s)
    if m:
        part, entity = m.group("part").strip(), m.group("ent").strip()
    else:
        m = _PART_TRAIL_RE.match(s)
        if m and m.group("mod").lower() not in _GENERIC and \
                any(t.lower() not in _GENERIC for t in m.group("ent").split()):
            entity = m.group("ent").strip()
            part = f"{m.group('mod')} {m.group('noun')}"
    entity = _ARTICLE_RE.sub("", entity)
    system_text = f"{entity} {part}".strip() if part else entity
    return {"entity": entity, "part": part, "aliases": aliases, "scale_hint": scale_hint,
            "residual": residual, "system_text": system_text}


def key_of(norm: dict):
    return (norm["entity"].casefold(), (norm["part"] or "").casefold() or None)


# ---------------------------------------------------------------- layer 2
class ProcessItem(BaseModel):
    head: Literal["folding", "phase_separation", "aggregation", "binding_recognition",
                  "assembly_organization", "conformational_change", "transport_diffusion", "other"]
    qualifier: Optional[str] = None


class ConditionItem(BaseModel):
    variable: Literal["temperature", "pH", "ionic_strength", "crowding", "force",
                      "pressure", "ligand", "concentration", "other"]
    change: str


class StructFeature(BaseModel):
    kind: Literal["fold", "family", "disorder_state"]
    value: str
    source: Literal["user_stated", "knowledge_inferred"]


class CanonicalA(BaseModel):
    process: list[ProcessItem]
    condition: list[ConditionItem]
    context: list[str]
    structural_features: list[StructFeature]
    modeling_goal: str


class CanonicalB(BaseModel):
    process: list[ProcessItem]
    condition: list[ConditionItem]
    context: list[str]
    structural_features: list[StructFeature]
    modeling_goal: str
    constraints: list[str]


_COMMON = """\
You canonicalize a scientific literature-search plan into a small structured \
schema. The plan was already extracted by an earlier step (system / \
target_component / scientific_objective / search_context). The entity name \
and its sub-part have ALREADY been extracted deterministically and are given \
as ENTITY / PART -- do not output them. A RESIDUAL CLAUSE may also be given: \
text that was attached to the original `system` field but is not part of the \
entity's name; route its content into the correct field below.

Fields (each has one narrow job; when content could fit two fields use \
priority condition > process > context):

- process: ONLY the physical/biological process(es) taking place in the \
system itself. `head` must be one of the allowed values (folding covers \
folding/unfolding/refolding; `other` only if nothing fits); `qualifier` is \
short free text. A relationship between the system and its surroundings is \
NOT a process.
- condition: ONLY a direct trigger/modulator that changes whether or how \
much the process happens (temperature, pH, ionic strength, crowding, force, \
...), with `change` saying how. If it merely describes the system's setting, \
use context instead. When unsure, prefer context.
- context: the system's environment / scale / organization / bound partner / \
modification / named construct, as short phrases (never sentences, never a \
restatement of the process or objective). Default bucket.
- structural_features: structural class information about the ENTITY. \
`kind` in {fold, family, disorder_state}. source=user_stated when the given \
text says it; source=knowledge_inferred ONLY when the ENTITY is a specific, \
named, well-characterized system whose fold/family is established domain \
knowledge. NEVER infer for a generic, unnamed, disordered, or unfamiliar \
entity -- leave empty rather than guess. Short canonical names.

Do not invent facts about the user's request in process/condition/context. \
Only structural_features may use outside domain knowledge. Do NOT introduce \
terminology the input does not already use (for example do not name \
specific model families, or substitute a technical term of your own for the \
user's wording).
"""

PROMPT_A = _COMMON + """\
- modeling_goal: ONE noun phrase, up to 30 words, stating what the user \
wants out of this. Remove only a leading verb ("Model", "Determine", \
"Capture"). It MUST preserve every constraint, contrast, exclusion and \
qualification the given objective states, in the user's own words -- for \
example "not just whether X", "beyond Y", "while keeping/retaining Z", \
"sequence-specific", "predict affinities". Dropping such a clause is an error.

Respond with a single JSON object matching the given schema, and nothing else."""

PROMPT_B = _COMMON + """\
- modeling_goal: ONE short noun phrase (<=15 words) for the main thing the \
user wants out of this. Remove only a leading verb.
- constraints: list of short phrases, one per explicit user constraint, \
contrast, exclusion or qualification found anywhere in the given objective / \
context (e.g. "beyond whether they phase separate", "retain a \
simulation-based physical model", "sequence-specific"), in the user's own \
words. Empty list only if the user stated none. Dropping a stated \
constraint is an error.

Respond with a single JSON object matching the given schema, and nothing else."""

_CONTRAST_RULE = """\
CONTRAST RULE (applies to `process`): when the user says "not just X", \
"beyond X", "rather than X" or similar, X is a BASELINE the user already \
takes for granted -- do NOT list X as the main process. List the process the \
user actually wants (what the contrast points toward) in `process`, and \
record the contrast itself in `constraints`.
"""
PROMPT_C = PROMPT_B.replace("Do not invent facts", _CONTRAST_RULE + "\nDo not invent facts", 1)

VARIANTS = {"A": (CanonicalA, PROMPT_A), "B": (CanonicalB, PROMPT_B), "C": (CanonicalB, PROMPT_C)}


def build_user_message(rec: dict, norm: dict) -> str:
    return (
        f"ENTITY: {norm['entity']!r}\n"
        f"PART: {norm['part']!r}\n"
        f"RESIDUAL CLAUSE (from original system field): {norm['residual']!r}\n"
        f"SCALE HINT: {norm['scale_hint']!r}\n"
        f"target_component: {rec.get('target_component')!r}\n"
        f"scientific_objective: {rec['scientific_objective']!r}\n"
        f"search_context: {rec.get('search_context')!r}"
    )


# ------------------------------------------------------------- rendering
def _toks(s: str) -> set:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def render_for_expert_prior(norm: dict, canon: dict, rec: dict) -> dict:
    """Deterministic rendering of canonical fields into the 4-field shape the
    downstream consumers take today. The objective is rendered as ONE NATURAL
    SENTENCE -- '<goal> of <subject>, <constraints>' -- not as labeled
    fragments: the old consumers (flat route, 3A templates) paste it into
    sentences verbatim, and the labeled form measurably hurt retrieval
    (case 6, depth-50 probe: 8/11 vs 11/11)."""
    system = norm["system_text"]
    if norm["aliases"]:
        system = f"{system} ({norm['aliases'][0]})"
    goal = canon.get("modeling_goal") or ", ".join(
        p.get("qualifier") or p["head"].replace("_", " ") for p in canon["process"])
    subj = norm["system_text"]
    base = goal if (not goal or subj.lower() in goal.lower()) else f"{goal} of {subj}"
    parts = [base or rec["scientific_objective"]]
    conds = [f"{c['variable']} {c['change']}" for c in canon["condition"]]
    cond_txt = ", ".join(conds)
    if cond_txt and not _toks(cond_txt) <= _toks(" ".join(parts)):
        parts.append("under " + cond_txt)
    parts += canon.get("constraints") or []
    ctx = list(canon["context"])
    if norm["scale_hint"]:
        ctx.append(f"{norm['scale_hint']} copies")
    return {"system": system, "target_component": rec.get("target_component"),
            "scientific_objective": ", ".join(x for x in parts if x), "search_context": ctx}


def apply_option_a(p: dict) -> dict:
    """Conservative option A: ONLY entity/part normalization of `system`
    (deterministic). scientific_objective is kept verbatim. A residual clause
    stripped off `system` (and a dropped scale hint) is appended to
    search_context unless the objective/context already carries its words, so
    no stated information is removed from what downstream sees."""
    n = normalize_system(p["system"])
    system = n["system_text"] + (f" ({n['aliases'][0]})" if n["aliases"] else "")
    ctx = list(p.get("search_context") or [])
    vis = " ".join([p["scientific_objective"]] + ctx)
    if n["residual"] and not _toks(n["residual"]) <= _toks(vis):
        ctx.append(n["residual"])
    if n["scale_hint"] and not _toks(n["scale_hint"]) <= _toks(vis):
        ctx.append(f"{n['scale_hint']} copies")
    out = dict(p)
    out["system"], out["search_context"] = system, ctx
    return out
