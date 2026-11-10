"""Retrieval terminology expansion (P3): the LLM step that lists the alternative
terms the literature uses for the same structural class, plus the deterministic
helpers around it. Pure Python except for the injected `call`.

1. expand_terminology(system_term, call=...) -- the P3 LLM call: prompt
   (TERMINOLOGY_EXPANDER_SYSTEM_PROMPT), schema (TerminologyExpansion), user
   message (terminology_expander_user_message). `call` is any callable
   (system_prompt, user_message, schema_dict) -> raw JSON content string; the
   live path builds it with llm_backend.make_call_fn.

2. strip_size_qualifiers(term): drop leading size / residue-count / mass
   qualifiers ("200-aa beta-helix protein" -> "beta-helix protein") so a
   system query is built from the structural core, not the user's arbitrary
   size. This is text hygiene, NOT a synonym table.

3. filter_instance_terms(terms, ...): drop any TerminologyExpander output
   that looks like a SPECIFIC protein / gene / organism / PDB identifier
   rather than a structural or architectural-class description. This is a
   SHAPE filter -- it removes "looks like an instance" -- it never maps a
   description to a set of names, so it does not reintroduce the old
   hard-coded association table.
"""
from __future__ import annotations

import re
from typing import Callable

from pydantic import BaseModel, Field

from cgscout.structured import _schema  # noqa: E402

# leading "200-aa" / "200 aa" / "200-residue" / "~200 aa" / "50 kDa" / "12 nm"
_SIZE_PREFIX = re.compile(
    r"^\s*[~<>]?\s*\d[\d,\.]*\s*[-–]?\s*"
    r"(aa|a\.a\.|amino[- ]acid|residues?|res|kda|kd|nm|Å|angstroms?|bp|nt|chains?|mers?)\b\s*",
    re.IGNORECASE,
)

# structural / architectural morphemes -- a term with none of these AND a
# name-like shape is treated as an instance and dropped.
_STRUCTURAL_MORPHEMES = (
    "helix", "helic", "solenoid", "barrel", "sheet", "strand", "fold", "coil",
    "sandwich", "propeller", "roll", "prism", "bundle", "hairpin", "loop",
    "domain", "motif", "repeat", "turn", "ladder", "spiral", "supercoil",
    "topology", "architecture", "class", "family", "superfamily", "passenger",
    "adhesin", "lectin", "transporter", "autotransporter", "amyloid", "fibril",
    "cross-beta", "globular", "disordered", "membrane", "beta", "alpha",
    "parallel", "antiparallel", "left-handed", "right-handed", "single-stranded",
    "double-stranded", "triple",
)

# obvious instance / identifier shapes
_GENE_NAME = re.compile(r"\b[A-Za-z]{1,6}\d{1,4}[A-Za-z]?\b")          # P22, EspP1, T4
_CAMEL_ACRONYM = re.compile(r"\b[A-Z][a-z]{1,3}[A-Z][A-Za-z]*\b")      # EspP, TolC-like
_ALLCAPS = re.compile(r"\b[A-Z]{2,6}(?:-[A-Z0-9]{1,4})?\b")           # SARS-CoV, GFP, NALP
_PDB = re.compile(r"\b\d[A-Za-z0-9]{3}\b")                             # 1XYZ


def strip_size_qualifiers(term: str) -> str:
    prev = None
    cur = term.strip()
    while cur != prev:
        prev = cur
        cur = _SIZE_PREFIX.sub("", cur).strip()
    return cur or term.strip()


_TRAILING_INSTANCE_TOKEN = re.compile(r"[\s-]([A-Z]|[IVX]{1,4}|\d{1,3}[a-z]?)$")   # "... C", "... III", "... 2a"


def _looks_like_instance(term: str) -> bool:
    t = term.strip()
    if not t:
        return True
    low = t.lower()
    # a specific isozyme / serotype suffix ("pectate lyase C", "toxin III")
    if _TRAILING_INSTANCE_TOKEN.search(t):
        return True
    has_struct = any(m in low for m in _STRUCTURAL_MORPHEMES)
    if has_struct:
        return False  # a structural morpheme present -> treat as a description/class
    return bool(
        _GENE_NAME.search(t) or _CAMEL_ACRONYM.search(t) or _ALLCAPS.search(t) or _PDB.search(t)
        # a bare 1-2 word Title-Case phrase with no structural morpheme (e.g. "Pertactin")
        or (len(t.split()) <= 2 and t[:1].isupper() and t.lower() != t)
    )


def filter_instance_terms(
    terms: list[str], *, ban_substrings: tuple[str, ...] = ()
) -> tuple[list[str], list[str]]:
    """Returns (kept, dropped). `ban_substrings` (lowercased) additionally
    drops a term that merely repeats the objective/methodology wording."""
    kept: list[str] = []
    dropped: list[str] = []
    seen: set[str] = set()
    bans = tuple(b.lower() for b in ban_substrings if b)
    for raw in terms:
        t = " ".join(raw.split())
        key = t.lower()
        if not t or key in seen:
            continue
        seen.add(key)
        if _looks_like_instance(t):
            dropped.append(t)
        elif any(b and b in key for b in bans):
            dropped.append(t)
        else:
            kept.append(t)
    return kept, dropped


# ---------------------------------------------------------------------------
# TerminologyExpander -- READY-side retrieval expansion. NOT part of intake
# readiness; it never changes ASK/READY. Given ONE grounded system
# description, it lists the alternative terms the LITERATURE uses for the
# SAME kind of structure/architecture/class, so retrieval does not miss
# papers that describe it differently (e.g. a "beta-helix" paper indexed as
# "beta-solenoid" / "autotransporter passenger domain"). It must NEVER emit
# a specific protein/gene/organism/PDB name -- that would recreate the old
# hard-coded association table. Deterministic instance-shape filtering
# (terminology_expansion.filter_instance_terms) is a second line of defence.
# ---------------------------------------------------------------------------

MAX_EXPANSION_TERMS = 6


class TerminologyExpansion(BaseModel):
    original_term: str
    expansion_terms: list[str] = Field(default_factory=list)


TERMINOLOGY_EXPANDER_SYSTEM_PROMPT = """\
You are given ONE structural / system description a researcher used for the \
protein system they study. List the alternative terms the SCIENTIFIC \
LITERATURE commonly uses for the SAME kind of structure, fold, or \
architectural class -- so a literature search does not miss relevant papers \
that describe it with different wording.

Give ONLY:
- alternative names for the fold / architecture itself (for "beta-helix": \
"beta-solenoid", "parallel beta-helix", "beta-helical fold", \
"single-stranded left-handed beta-helix", "pectin lyase-like fold")
- structural or functional CLASS names the literature uses for proteins \
built on this architecture (for "beta-helix": "autotransporter passenger \
domain", "beta-helix adhesin")

You must NEVER give:
- a specific protein, gene, domain, organism, strain, or PDB identifier \
(NOT "pertactin", "P22 tailspike", "pectate lyase C", "EspP", "hemolysin") \
-- only descriptions and architectural CLASS names
- a scientific process, method, or property (NOT "folding", "aggregation", \
"stability", "coarse-grained", "simulation")
- the researcher's exact original wording repeated back
- a size / residue count / organism qualifier

Between 0 and 6 terms. If the description is already generic, or you are \
not confident of real literature synonyms, return FEWER (even zero) -- do \
NOT pad with guesses or invented compounds.

Respond with a single JSON object matching the given schema, and nothing else."""


def terminology_expander_user_message(term: str) -> str:
    return f'STRUCTURAL / SYSTEM DESCRIPTION: "{term}"'


CallFn = Callable[[str, str, dict], str]


def expand_terminology(system_term: str, *, call: CallFn) -> TerminologyExpansion:
    content = call(TERMINOLOGY_EXPANDER_SYSTEM_PROMPT, terminology_expander_user_message(system_term),
                   _schema(TerminologyExpansion))
    return TerminologyExpansion.model_validate_json(content)
