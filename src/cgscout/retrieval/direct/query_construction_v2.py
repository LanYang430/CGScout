"""Query Construction v2 (flat lexical process route, need_semantic and 3A semantic queries).  Switch: RouteRetrievalBudgets.query_construction_v2.

Why: the legacy builders paste the planner's `scientific_objective` sentence into templates ('Coarse-grained models of X to model Use coarse-grained
simulations to model ...'), cut it mechanically after 8 content words, keep only its first clause and never read the planner's `search_axes`, which are
already complete scientific concepts.  This module is deterministic (no LLM, no network) and generic (no per-system or per-benchmark rules):

  * method frames ('Use/Apply ... <method> to ...', '... using coarse-grained methods', 'Coarse-grained modeling of ...') are removed, the
    method is expressed once by the query template;
  * the objective is split into complete goals (clause level; a qualifier introduced by 'including ...' or 'while <verb>ing ...' becomes its own goal);
  * search_axes are cleaned (evidence-category labels such as 'system-specific CG model precedent' and list-like axes are dropped, method words are
    stripped) and ranked by how many of their words are entities/tasks of the user's own system + objective;
  * the production system-specific and context (entity) queries are kept EXACTLY as before; only the process route is replaced;
  * the semantic queries are one natural sentence (need_semantic) and, per 3A class, [one complete-need query] + distinct emphases;
  * the flat budget stays at the existing ceiling (6 lexical queries); 3A never exceeds the existing 3 queries per class.

Used by route_plan.build_route_retrieval_plan (lexical + need_semantic) and orchestrator.run_expert_prior_retrieval (3A).
ExpertPriorPlanner's route decisions are untouched: it still chooses which classes get 3A / 3B; only the text of the 3A queries changes."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import cgscout.retrieval.direct.route_plan as RP

_cw = RP._content_words

# --- method frames ---------------------------------------------------------------------------------------------------------------------------
_METHOD_NP = r"(?:(?:coarse[- ]grained|CG|all[- ]atom|atomistic|multiscale|mesoscale)\s+)?(?:simulations?|models?|modell?ing|methods?|approach(?:es)?|techniques?|calculations?)"
# '(Use|Apply|Employ|...) <up to 2 words> <method phrase> to ...' at the START of an objective
_FRAME_LEAD = re.compile(
    rf"^\s*(?:use|using|apply|applying|employ|employing|utili[sz]e|utili[sz]ing|adopt|adopting|leverage|leveraging)\s+(?:(?:a|an|the)\s+)?(?:[\w-]+\s+){{0,2}}?{_METHOD_NP}\s+(?:to|for|in\s+order\s+to)\s+", re.I)
# '... using/with/by <method phrase>' at the END
_FRAME_TAIL = re.compile(rf"[\s,;]+(?:using|with|by|via|through|based\s+on)\s+(?:(?:a|an|the)\s+)?(?:[\w-]+\s+){{0,3}}?{_METHOD_NP}\b.*$", re.I)
_ADJ = r"(?:coarse[- ]grained|CG|mesoscopic|mesoscale|multiscale|multi-scale|hierarchical|all[- ]atom|atomistic)"
# 'topology-aware coarse-grained modeling of X', 'mesoscopic or CG models for X' (wrapper in front of the concept)
_WRAP = re.compile(rf"^\s*(?:[\w-]+-(?:aware|specific|based|resolved)\s+)?(?:{_ADJ}(?:,?\s+(?:or|and)\s+|\s+))+(?:models?|modell?ing|simulations?|treatments?|approach(?:es)?|methods?)\s+(?:of|for)\s+", re.I)
_SPEC_WRAP = re.compile(r"^\s*[\w-]+-specific\s+(?:CG|coarse-grained)\s+(?:models?|modell?ing)(?:\s+(?:of|for)\s+)?", re.I)
_CG_TAIL = re.compile(r"\s+(?:in|for|of|with|using|within)\s+(?:the\s+)?(?:coarse[- ]grained|CG|mesoscopic|multiscale)\b.*$", re.I)
_TRAIL_METHOD = re.compile(r"\s+(?:methods?|approaches|models?)$", re.I)
# evidence-category labels (what KIND of literature), not scientific concepts
_CATEGORY = re.compile(
    r"\b(?:system|process|mechanism|nucleic-acid|dna-origami)[\w-]*-specific\b.*\b(?:CG|coarse-grained)\b|(?:system|process)[- ]specific|validation|transferab|precedent|"
    r"structurally[- ]related|transfer|literature|evidence|\bbenchmark|\breview|\bsurvey|framing|cover both|\bbroad", re.I)
# clause structure of the objective
_QUAL = re.compile(r",?\s+(?:including|especially|such as|particularly)\s+", re.I)
_SUBORD = re.compile(r"\s+(?:while|whereas)\s+(?=(?:[a-z]+ing)\b)", re.I)
_CONSTRAINT_VERB = re.compile(r"^\s*(?:relating|retaining|preserving|maintaining|keeping|accounting\s+for|allowing|enabling|ensuring|while\s+\w+ing)\s+", re.I)
_EFFECT = re.compile(r"^\s*(?:the\s+)?(?:effects?|roles?|impacts?|influence)\s+of\s+(?:the\s+)?", re.I)
_BOUNDARY = re.compile(r"\s+(?:during|between|across|among|from|under|through|via|while|including|within|in|on|at|into|as|when|where|that|which|for|with)\s+", re.I)
_SPLIT = re.compile(r"\s*(?:;|,)\s*(?:and\s+)?|\s+and\s+")
_SUB_STOP = frozenset({"and", "the", "for", "with", "via", "into", "from", "using", "that", "than", "during", "while", "between", "across"})


# --- small text helpers -----------------------------------------------------------------------------------------------------------------------
def _low_first(s: str) -> str:
    return s if not s or RP._is_entity_like(s.split()[0]) or s.split()[0].isupper() else s[0].lower() + s[1:]


def _sub(s: str) -> set[str]:
    """sub-word tokens, hyphen/dash insensitive, lightly stemmed (for overlap / coverage tests only)"""
    out: set[str] = set()
    for w in re.findall(r"[a-zα-ωA-ZΑ-Ω0-9]+", (s or "").lower()):
        if len(w) < 3 or w in _SUB_STOP:
            continue
        if len(w) > 5 and w.endswith("ly"):
            w = w[:-2]
        out.add(w[:-1] if len(w) > 4 and w.endswith("s") else w)
    return out


def _jac(a: str, b: str) -> float:
    A, B = {RP._squash(w) for w in a.split()} - {""}, {RP._squash(w) for w in b.split()} - {""}
    return len(A & B) / max(len(A | B), 1)


def _diverse(items: list[str], k: int, thr: float) -> list[str]:
    out: list[str] = []
    for x in items:
        if all(_jac(x, y) < thr and RP._squash(x) != RP._squash(y) for y in out):
            out.append(x)
        if len(out) >= k:
            break
    return out


def _covered(concept: str, queries: list[str]) -> bool:
    """>= 80% of the concept's sub-words are already carried by the given queries taken together"""
    a = _sub(concept)
    u: set[str] = set().union(*[_sub(q) for q in queries]) if queries else set()
    return bool(a) and len(a & u) / len(a) >= 0.8


def _with_cg(x: str) -> str:
    return x if re.search(r"coarse[- ]grain|\bCG\b", x, re.I) else f"{x} coarse-grained model"


def _lexical_form(c: str) -> str:
    return re.sub(r"^(?:how|whether)\s+", "", c, flags=re.I)             # a lexical query does not carry the question word


def _shorten(phrase: str, max_words: int = 8) -> str:
    """shorten ONLY at a phrase boundary (preposition / clause word); with no safe boundary the whole phrase is kept (never cut inside a phrase)"""
    if len(_cw(phrase)) <= max_words:
        return phrase
    best = None
    for m in _BOUNDARY.finditer(phrase):
        head = phrase[:m.start()].strip(" ,;")
        if 3 <= len(_cw(head)) <= max_words:
            best = head
    return best or phrase


def _strip_cg(c: str) -> str:
    c = _CG_TAIL.sub("", c)                                              # 'X in coarse-grained protein models' -> 'X'
    t = re.sub(r"\b(?:coarse[- ]grained|CG|mesoscopic|mesoscale|multiscale)\b[,]?\s*", "", c, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip(" ,;")
    return t if len(_cw(t)) >= 2 else c


# --- objective -> goals ---------------------------------------------------------------------------------------------------------------------
def strip_frames(text: str) -> str:
    """remove the instruction frame ('Use X to', 'Determine', 'Model') and the method frame ('using coarse-grained methods'); 'how'/'whether' are kept"""
    t = RP._strip_conversational_wrapper(text or "").strip().rstrip(". ")
    t = _FRAME_LEAD.sub("", t, count=1).strip()
    keep = None
    for _ in range(2):
        m = RP._ACTION_LEAD_RE.match(t)
        if not m or not t[m.end():].strip():
            break
        w = re.search(r"\b(how|whether)\b", m.group(0), re.I)
        keep = keep or (w.group(1).lower() if w else None)
        t = t[m.end():].strip()
    t = _FRAME_TAIL.sub("", t).strip(" ,;")
    return f"{keep} {t}" if keep else t


def split_goals(objective: str) -> list[str]:
    """complete, method-free goal phrases.  Split only where the next segment starts with an action verb (so 'A, B and C' noun lists stay whole);
    'including ...' qualifiers and 'while <verb>ing ...' clauses become goals of their own instead of being discarded."""
    t = _WRAP.sub("", strip_frames(objective), count=1).strip()
    t = _SUBORD.sub("; ", t)
    out, last = [], 0
    for m in _SPLIT.finditer(t):
        if ";" in m.group(0) or RP._ACTION_LEAD_RE.match(t[m.end():]):
            out.append(t[last:m.start()])
            last = m.end()
    out.append(t[last:])
    goals: list[str] = []
    pending: list[str] = []
    for g in out:
        pieces = _QUAL.split(g.strip(" ,;."), maxsplit=1)
        g = pieces[0].strip(" ,;.")
        qual = _EFFECT.sub("", pieces[1]).strip(" ,;.") if len(pieces) > 1 else ""
        if _cw(qual):
            pending.append(qual)
        for _ in range(2):
            g2 = RP._ACTION_LEAD_RE.sub("", g, count=1).strip()
            if not g2 or g2 == g:
                break
            g = g2
        if goals or pending:
            g = _CONSTRAINT_VERB.sub("", g, count=1)
        g = RP._clean_chunk(g, True)
        if len(_cw(g)) >= 2:
            goals.append(_low_first(g))
    for q in pending:
        q = RP._clean_chunk(q, True)
        if _cw(q) and not RP._redundant(q, goals):
            goals.append(_low_first(q))
    return goals or ([_low_first(t)] if t else [])


def axis_concepts(search_axes: list[str]) -> list[str]:
    """complete scientific concepts from the planner's search_axes: category labels and list-like axes dropped, method wrappers/words stripped"""
    out: list[str] = []
    for a in search_axes or []:
        a = (a or "").strip()
        if not a or _CATEGORY.search(a):
            continue
        if "," in a or (re.search(r"\b(or|and)\b.*\b(methods?|models?|approaches)\b", a, re.I) and len(_cw(a)) > 7):
            continue                                                   # a list of model types, not one concept
        c = _SPEC_WRAP.sub("", _WRAP.sub("", a, count=1), count=1).strip(" ,;")
        c = _strip_cg(_TRAIL_METHOD.sub("", c).strip())
        if 2 <= len(_cw(c)) <= 9:
            out.append(_low_first(c))
    return out


# --- analysis shared by flat, need_semantic and 3A -------------------------------------------------------------------------------------
@dataclass
class QCAnalysis:
    head: str
    goals: list[str]
    axes: list[str]
    kept: list[str]                       # production system-specific + context queries (unchanged)
    concepts: list[tuple[str, str]]       # selected (concept, 'axis'|'goal'), best first (<= 2)
    needs: list[str] = field(default_factory=list)   # 3A needs: [complete need] + distinct emphases (<= 3)

    @property
    def tasks(self) -> list[str]:
        return [c for c, _ in self.concepts]


def analyze(plan, budgets, kept: Optional[list[str]] = None) -> QCAnalysis:
    """`plan` = EvidenceRoutePlan.  `kept` = the system-specific + context queries of the same plan (computed here the legacy way when not given)."""
    head = RP._system_head(plan.system_description or "")
    goals = split_goals(plan.scientific_objective or "")
    axes = axis_concepts(getattr(plan, "search_axes", None) or [])
    if kept is None:
        kept = [q.query for q in legacy_system_and_context_queries(plan, budgets)]
    ref = _sub(plan.system_description) | _sub(plan.scientific_objective) | _sub(" ".join(t.term for t in plan.context_terms))
    cands: list[tuple[float, str, str]] = []
    for i, c in enumerate(axes):
        a = _sub(c)
        cands.append((len(a & ref) / max(len(a), 1) + 0.1 / (1 + i), c, "axis"))
    for g in goals:
        if len(_cw(g)) >= 2:
            cands.append((0.9 - 0.03 * max(len(_cw(g)) - 5, 0), _shorten(g, 8), "goal"))
    cands.sort(key=lambda x: -x[0])
    picked: list[tuple[str, str]] = []
    for _sc, c, src in cands:
        if _covered(c, kept):                                           # the kept system/context queries already carry it
            continue
        if all(_jac(c, p) < 0.5 for p, _ in picked):
            picked.append((c, src))
        if len(picked) >= 2:
            break
    an = QCAnalysis(head=head, goals=goals, axes=axes, kept=kept, concepts=picked)
    tasks = an.tasks
    rest = [g for g in goals if not _covered(g, tasks[:1])]
    complete = "; ".join(([tasks[0]] if tasks else []) + rest[:3])
    emph = tasks[:2] + [g for g in goals if not _covered(g, tasks[:2])]
    an.needs = _diverse([x for x in [complete] + emph if x], 3, 0.97)       # the complete need is nested in the emphases by design; only exact duplicates dropped
    return an


def legacy_system_and_context_queries(plan, budgets) -> list:
    """The production system-specific and context queries, computed exactly as RP.build_route_retrieval_plan does (these stay unchanged in v2)."""
    context_queries = RP.build_context_queries(plan, budgets)
    process_core = RP._process_core(plan.process_terms[0].term, budgets.flat_query_hygiene) if plan.process_terms else None
    covered = [q.route_term for q in context_queries] + ([process_core] if process_core else [])
    return RP.build_system_specific_cg_queries(plan, budgets, covered) + context_queries


# --- builders ----------------------------------------------------------------------------------------------------------------------------------
_MAX_FLAT_TOTAL = 6                       # the existing ceiling of flat lexical queries (system 1 + process 3 + context 3 is the theoretical max of 7; 6 in practice)


def process_queries_v2(an: QCAnalysis, budgets, n_kept: int) -> list:
    """process route: up to 2 complete concepts (search_axes first, objective goals second) + at most one short system+task combination"""
    out = []
    for c, src in an.concepts:
        out.append(RP.RouteQuery(route="process", route_term=c, route_term_source=("PLANNER_DERIVED" if src == "axis" else "USER_GROUNDED"),
                                 query=_with_cg(_lexical_form(c)), query_role="process_axis_concept_cg_evidence" if src == "axis" else "process_goal_cg_evidence"))
    tasks = an.tasks
    if tasks and n_kept < 3 and len(_cw(tasks[0])) <= 4 and len(_cw(an.head)) <= 4:
        have = {RP._squash(x) for x in an.head.split()}
        extra = [w for w in _lexical_form(tasks[0]).split() if RP._squash(w) not in have]
        if extra:
            out.append(RP.RouteQuery(route="process", route_term=tasks[0], route_term_source="PLANNER_DERIVED",
                                     query=_with_cg(f"{an.head} {' '.join(extra)}"), query_role="system_process_cg_evidence"))
    room = max(min(budgets.max_queries_per_route, _MAX_FLAT_TOTAL - n_kept), 0)
    keep: list = []
    seen = list(an.kept)
    for q in out:
        if all(_jac(q.query, s) < 0.9 for s in seen):
            keep.append(q)
            seen.append(q.query)
    return keep[:room]


def need_semantic_query_v2(an: QCAnalysis, budgets) -> list:
    """one natural sentence: 'Coarse-grained modeling of <system>: <task>; <goals not already covered>.'"""
    if not budgets.need_semantic_enabled:
        return []
    system = re.sub(r"^(?:a|an|the)\s+", "", an.head, flags=re.I).strip().rstrip(".")
    if not system:
        return []
    tasks = an.tasks
    parts = tasks[:1] + [g for g in an.goals if not _covered(g, tasks[:1])]
    q = f"Coarse-grained modeling of {system}: " + "; ".join(parts[:4]) + "." if parts else f"Coarse-grained modeling of {system}."
    return [RP.RouteQuery(route=RP.NEED_SEMANTIC_ROUTE, route_term=system, route_term_source="USER_GROUNDED", query=q, query_role="need_semantic_evidence")]


def render_direct_concept_queries_v2(scientific_class: str, an: QCAnalysis) -> list[str]:
    """3A: per transfer class [complete need] + distinct emphases (<= 3 queries, never more than the legacy three templates)."""
    scientific_class = (scientific_class or "").strip()
    if not scientific_class or not an.needs:
        return []
    return [f"Coarse-grained models for {scientific_class} to model {n}." for n in an.needs[:3]]
