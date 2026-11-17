"""Consolidation safety net, Phase 0 (zero API, zero network, no production code touched).

A deterministic "world" for the retrieval flow of `run_expert_prior_retrieval` (the per-axis loop, direct_concept_search, the 3B
related-system-discovery chain P3 -> P4 -> Hop-1/P5 -> cleanup -> identity -> P6 -> Hop-2 -> tag write-back, the merge and Stage 3).
Only the network/LLM-bound collaborators are replaced by fakes; everything that is orchestration (the code a refactor moves) runs for real:
`_run_related_system_discovery`, `_make_search_fn`, `_run_direct_concept_search`, `_merge_papers`, `cleanup_discovered_systems`,
`build_related_system_identities`, `normalize_search_anchors` (against a scripted backend), `merge_and_select`, `run_converged_hop2`.

Every scenario produces a TRACE: the ordered events at the collaborator boundaries (arguments included), the cache get/set events (namespace +
full key), and the serialized result.  `tests/fixtures/consolidation/baseline_traces.json` holds the traces produced by the BASELINE code
(c8be98c, tag consol-0-baseline).  After each refactoring phase the same scenarios must reproduce those traces EXACTLY (a trace difference
is a stop-and-report event, never something to "fix" by editing the fixture).

Where the flow lives, how it is called, which names are patched and which module renames are approved is declared in ONE place: a `Layout` in
tests/consolidation_layout.py.  This file names no production module for the flow.  Every patch goes through `Patches.set(<logical name>)`, which
refuses a target the module does not have (rule R-2: assigning to a missing module attribute silently does nothing), counts the calls (a patch
that is never called is a failure, rule R-1), and the HTTP boundary is guarded so that a collaborator the layout forgot to patch cannot reach the network.

Two scenario families: SCENARIOS (collaborator-boundary fakes; fixture baseline_traces.json, never edited) and SCENARIOS_P5 (the REAL Hop-1 discovery and
the REAL P5 extractor run, only the LLM is scripted; fixture baseline_p5_traces.json).  NOTE for readers of the first family: its fake `discover_related_systems`
hands the extractor only `expansion_terms[:3]`; that is a property of the fake, NOT of production -- production passes the whole list to the P5 message
and cache key, which is what the second family pins.

Regenerate a fixture ONLY on the baseline, and never over an existing file:  python tests/consolidation_world.py --write | --write-p5
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

import consolidation_layout as LY
from cgscout import cache
from cgscout.models import Paper
from cgscout.planning.expert_prior_planner import ExpertPriorOutput, TransferAxis
from cgscout.planning.scientific_planner import PlannerOutput
from cgscout.retrieval.direct.route_plan import RouteRetrievalBudgets
from cgscout.retrieval.related_systems.discovery import DiscoveredSystem, DiscoveryResult, SupportingPaper
from cgscout.retrieval.related_systems.extraction import RichSystemExtraction, RichSystemExtractionLLM, RichSystemMentionLLM
from cgscout.retrieval.related_systems.hop2 import _AnchorBatch, _AnchorItem
from cgscout.retrieval.related_systems.terminology import TerminologyExpansion

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "consolidation" / "baseline_traces.json"
FIXTURE_P5 = Path(__file__).resolve().parent / "fixtures" / "consolidation" / "baseline_p5_traces.json"


class BudgetExhausted(BaseException):
    """Stands in for OpenAlexRateLimitExceeded-style budget exhaustion: a BaseException that no `except Exception` may swallow."""


# ------------------------------------------------------------------------------------------------------------------------------------------------
class Patches:
    def __init__(self, layout):
        self.layout = layout
        self._undo, self.calls = [], {}

    def set(self, logical, make):
        """make(orig) -> replacement, installed on EVERY (module, attribute) the layout lists for the logical collaborator `logical`; calls are counted
        under the logical (= baseline attribute) name.  R-2: each attribute must exist on its module."""
        assert logical in self.layout.patch_targets, f"layout {self.layout.name!r} declares no patch target for {logical!r}"
        repl = counted = None
        for module, attr in self.layout.patch_targets[logical]:
            m = importlib.import_module(module)
            assert hasattr(m, attr), f"R-2: {m.__name__} has no attribute {attr!r}; assigning to it would silently do nothing"
            orig = getattr(m, attr)
            if repl is None:
                repl = make(orig)

                def counted(*a, _r=repl, **k):
                    self.calls[logical] = self.calls.get(logical, 0) + 1
                    return _r(*a, **k)
                counted.__wrapped__ = repl
                for extra in ("last_errors",):
                    if hasattr(repl, extra):
                        setattr(counted, extra, getattr(repl, extra))
            self._undo.append((m, attr, orig))
            setattr(m, attr, counted)
        return counted

    def restore(self):
        for m, attr, orig in reversed(self._undo):
            setattr(m, attr, orig)
        self._undo.clear()


class Trace:
    """Events at the collaborator boundaries.  Production runs Hop-1's P5 extractions in a thread pool (`hop1_extraction_max_workers`, default 2), so the
    events those threads emit interleave differently from run to run; the ORDER BETWEEN CONCURRENT PAPERS is not a contract (discovery.py merges in
    index order).  Events from a non-main thread carry `worker: true`, and `canonical()` orders each contiguous block of them deterministically: one unit
    per paper (a unit starts at its `cache.get`), units sorted by that cache key.  Everything on the main thread keeps its exact order."""

    def __init__(self):
        self.events, self._tids = [], []

    def add(self, kind, **payload):
        worker = threading.current_thread() is not threading.main_thread()
        self.events.append({"kind": kind, **payload, **({"worker": True} if worker else {})})
        self._tids.append(threading.get_ident() if worker else None)

    def canonical(self):
        ev, tids, out, i = self.events, self._tids, [], 0
        while i < len(ev):
            if tids[i] is None:
                out.append(ev[i])
                i += 1
                continue
            j = i
            while j < len(ev) and tids[j] is not None:
                j += 1
            per = {}
            for k in range(i, j):
                per.setdefault(tids[k], []).append(ev[k])
            units = []
            for seq in per.values():                   # one worker thread handles several papers one after the other: split ITS events at each cache.get
                cur = None
                for e in seq:
                    if e["kind"] == "cache.get" or cur is None:
                        cur = [e]
                        units.append(cur)
                    else:
                        cur.append(e)
            units.sort(key=lambda u: u[0].get("key", ""))
            out.extend(e for u in units for e in u)
            i = j
        return out


def _frames():
    f = sys._getframe(2)
    while f is not None:
        yield f
        f = f.f_back


def _recorder():
    ev = str(Path(__file__).resolve().parents[1] / "benchmark" / "evaluation")
    if ev not in sys.path:
        sys.path.insert(0, ev)
    import recorder
    return recorder


def _h(s) -> str:
    return hashlib.sha1(str(s).encode()).hexdigest()[:8]


def _sj(o):
    return json.loads(json.dumps(o, sort_keys=True, default=str))


# ------------------------------------------------------------------------------------------------------------------------------------------------
class ScriptBackend:
    """The scripted LLM.  First family: only P6 (hop2_search_anchor) is not faked at its function boundary, so it is the only schema answered.
    Second family (`detailed`): the REAL P5 extractor runs, so RichSystemExtractionLLM is answered too and every call is recorded in full (system-prompt
    fingerprint, purpose, the whole user message)."""
    name, model = "mock", "mock-1"

    def __init__(self, trace, *, fail_p6=False, core_process="folding", detailed=False, p5_names=()):
        self.trace, self.fail_p6, self.core_process = trace, fail_p6, core_process
        self.detailed, self.p5_names = detailed, tuple(p5_names)
        self.calls = 0

    def _p5(self, user):
        abstract = user.split("\n\nABSTRACT: ", 1)[1].split("\n\nRELATIONSHIP CRITERION:", 1)[0]
        sentences = re.split(r"(?<=\.)\s+", abstract)
        out = []
        for name in self.p5_names:
            sent = next((x for x in sentences if name in x), None)
            if sent:
                out.append(RichSystemMentionLLM(name=name, evidence_quote=sent, entity_type="specific_protein_or_system", identity_context="a scripted system",
                                                class_membership_quote=sent))
        return RichSystemExtractionLLM(systems=out)

    def structured(self, *, system, user, schema, label, effort=None, purpose=None, case_id=None, timeout=None):
        self.calls += 1
        ev = dict(label=label, schema=schema.__name__, case_id=case_id, effort=effort, timeout=timeout, user_sha=_h(user), user_head=user[:60])
        if self.detailed:
            ev.update(purpose=purpose, system_sha=_h(system), user=user)
        self.trace.add("llm", **ev)
        if self.detailed and schema is RichSystemExtractionLLM:
            return self._p5(user)
        if schema is not _AnchorBatch:
            raise AssertionError(f"unexpected LLM schema {schema.__name__} (this world fakes every other LLM step at its function boundary)")
        if self.fail_p6:
            raise RuntimeError("p6 boom")
        items = []
        for line in user.splitlines():
            if line.startswith("[") and "] label: " in line:
                idx, label_ = line[1:].split("] label: ", 1)
                items.append(_AnchorItem(index=int(idx), search_anchor=label_.strip(), core_process=self.core_process, stable_descriptor="", descriptor_scope="none"))
        return _AnchorBatch(items=items)


# ------------------------------------------------------------------------------------------------------------------------------------------------
def axis(mode="related_system_discovery", conf="high", cls="alpha helix repeat", basis="distinctive_structure", anchors=("repeat fold", "stacked strands")):
    return TransferAxis(basis=basis, scientific_class=cls, discriminative_features=["a stacked repeat"], objective_relevance="shares the folding constraint",
                        retrieval_anchors=list(anchors), rationale="r", confidence=conf, execution_mode=mode)


def system(name, support, rank, *, entity_type="specific_protein_or_system", ctx="a secreted toxin"):
    ref = f"10.9/{_h(name)}"
    sp = SupportingPaper(paper_ref=ref, title=f"{name} structure", year=2020, found_by_query="q", retrieval_rank=rank,
                         evidence_quote=f"{name} adopts an alpha helix repeat fold", structural_term_matched="alpha helix repeat",
                         system_name=name, name_as_extracted=name, name_in_quote=True, entity_type=entity_type, identity_context=ctx)
    return DiscoveredSystem(name=name, norm_name=name.lower(), supporting_papers=[sp], n_support=support, first_seen_rank=rank)


def _results(query, n):
    shared = Paper(title="Shared coarse-grained study", doi="10.1/shared", abstract="A shared abstract about coarse-grained models.", year=2021)
    return [shared] + [Paper(title=f"Hit {i} for {query}", doi=f"10.1/{_h(query)}{i}", abstract=f"Abstract {i} of {query}", year=2020 + i) for i in range(n)]


# ------------------------------------------------------------------------------------------------------------------------------------------------
# scenario = dict(prior=[axes], direct=[Paper titles], p3, p4, hop1, systems, hop2, ...) -- see run_scenario
def _systems_basic():
    return [system("Aquila protein", 3, 1), system("Borealis protein", 2, 2), system("Caelum protein", 2, 5)]


def _systems_many():
    out = [system(f"Sys{chr(65 + i)} protein", 1 + (i % 3), i + 1) for i in range(9)]
    out[0] = system("SysA protein", 3, 1)
    return out + [system("sysA protein", 3, 20)]          # same lower-cased label as "SysA protein": ds_by_label keeps the LAST one for both


SCENARIOS = {
    "S01_3b_happy": dict(prior=[axis()], aliases=["alpha-solenoid", "repeat protein"], systems=_systems_basic),
    "S02_3b_truncation_ties": dict(prior=[axis()], aliases=["alpha-solenoid"], systems=_systems_many),
    "S03_p3_fails_rest_ok": dict(prior=[axis()], p3="fail", systems=_systems_basic),
    "S04_p3_fails_hop1_raises": dict(prior=[axis()], p3="fail", hop1="raise"),
    "S05_no_systems_keeps_p3_error": dict(prior=[axis()], p3="fail", systems=lambda: []),
    "S06_no_candidates_identities_insufficient": dict(prior=[axis()], p3="fail", systems=_systems_basic, insufficient_identities=True),
    "S07_hop2_raises_replaces_p3_error": dict(prior=[axis()], p3="fail", systems=_systems_basic, hop2="raise"),
    "S08_p4_raises_axis_level": dict(prior=[axis()], p4="fail", systems=_systems_basic),
    "S09_budget_exhaustion_propagates": dict(prior=[axis()], systems=_systems_basic, search_raises_after=2, expect_raised=True),
    "S10_tag_writeback_missing_ref": dict(prior=[axis()], systems=_systems_basic, inject_missing_ref=True),
    "S11_p6_unavailable_fallback": dict(prior=[axis()], systems=_systems_basic, fail_p6=True),
    "S12_multi_axis_loop": dict(
        prior=[axis(conf="low"), axis("direct_concept_search", cls="solenoid fold"), axis("direct_concept_search", cls="repeat topology"),
               axis("direct_concept_search", cls="curved scaffold"), axis("direct_concept_search", cls="fourth class"), axis(),
               axis("direct_concept_search", conf="medium", cls="late class")],
        direct=["Direct A", "Direct B", "Shared coarse-grained study"], systems=_systems_basic, explicit_budgets=True),
    "S13_unrecognized_mode_and_empty": dict(prior=["unrecognized"], direct=["Direct A"]),
    "S14_no_axes_direct_only": dict(prior=[], direct=["Direct A", "Direct B"]),
    "S15_warm_cache_second_run": dict(prior=[axis()], aliases=["alpha-solenoid"], systems=_systems_basic, runs=2),
    # the REAL search stack (search_and_dedupe + its cache) with only the HTTP boundary (search_openalex) faked and the benchmark Recorder installed:
    # baseline of the recorder acceptance (logical records L, external calls X, external calls outside any record window U); cold run, then warm run
    "R01_real_search_stack_recorder": dict(prior=[axis("direct_concept_search", cls="solenoid fold"), axis()], aliases=["alpha-solenoid"],
                                           systems=_systems_basic, runs=2, real_search=True),
}


# ---- second family: the REAL Hop-1 (discover_related_systems) and the REAL P5 extractor; only the LLM answer and the search results are scripted ----
P5_NAMES = ("Aquila protein", "Borealis protein", "Caelum protein")
ALIASES_A = ["alpha-solenoid", "repeat protein", "armadillo-like fold", "tandem repeat motif"]      # 4 aliases: they sit AFTER the 3 combination terms


def _results_p5(query, n):
    """Hop-1 results whose abstracts name a system in a sentence that ties to the class "alpha helix repeat" (so the real grounding admits it)."""
    shared = Paper(title="Shared coarse-grained study", doi="10.1/shared", year=2021,
                   abstract="Aquila protein adopts an alpha helix repeat fold. A shared abstract about coarse-grained models.")
    return [shared] + [Paper(title=f"Hit {i} for {query}", doi=f"10.1/{_h(query)}{i}", year=2020 + i,
                             abstract=f"{P5_NAMES[(i + 1) % 3]} adopts an alpha helix repeat fold in vitro. Extra context {i}.") for i in range(n)]


SCENARIOS_P5 = {
    "P5R01_p3_aliases_reach_p5": dict(prior=[axis()], aliases=ALIASES_A, real_p5=True),
    "P5R02_other_aliases": dict(prior=[axis()], aliases=["helical repeat", "solenoid protein"], real_p5=True),
    "P5R03_p3_fails_no_aliases": dict(prior=[axis()], p3="fail", real_p5=True),
    "P5R04_warm_second_run": dict(prior=[axis()], aliases=ALIASES_A, real_p5=True, runs=2),
    "P5R05_basis_physical_process": dict(prior=[axis(basis="physical_process")], aliases=ALIASES_A, real_p5=True),
}
ALL_SCENARIOS = {**SCENARIOS, **SCENARIOS_P5}


def _paper_row(p):
    return {"title": p.title, "doi": p.doi, "found_by_queries": list(p.found_by_queries), "query_categories": list(p.query_categories),
            "retrieval_modes": list(p.retrieval_modes), "sd": p.structural_discovered_system, "sds": list(p.structural_discovered_systems)}


def run_scenario(name: str, layout=None, overrides=None) -> dict:
    """Run one scenario through `layout` (default: the active layout) and return its trace.  `overrides` replaces scenario keys (used by tests that vary
    one input, e.g. the P3 aliases, and compare the effect live)."""
    layout = layout or LY.active_layout()
    sc = {**ALL_SCENARIOS[name], **(overrides or {})}
    tr, pt = Trace(), Patches(layout)
    backend = ScriptBackend(tr, fail_p6=sc.get("fail_p6", False), detailed=bool(sc.get("real_p5")), p5_names=P5_NAMES)
    plan = PlannerOutput(system="Aquila", scientific_objective="folding mechanism", clarification_needed=False, search_context=[], search_axes=[])
    prior_axes = [TransferAxis.model_construct(**{**axis().model_dump(), "execution_mode": "weird_mode"}) if a == "unrecognized" else a for a in sc["prior"]]
    prior = ExpertPriorOutput(transfer_needed=bool(prior_axes), transfer_axes=prior_axes)
    budgets = RouteRetrievalBudgets(flat_anchor_short_context=True, flat_query_hygiene=True, need_semantic_enabled=True, stage3_recover_abstracts=True,
                                    query_construction_v2=False) if sc.get("explicit_budgets") else None
    n_search = {"n": 0}

    def fake_expand(system_term, *, call):
        tr.add("P3", system_term=system_term, call_is_callable=callable(call))
        if sc.get("p3") == "fail":
            raise RuntimeError("p3 boom")
        return TerminologyExpansion(original_term=system_term, expansion_terms=list(sc.get("aliases", ["alias one", "alias two"])))

    def fake_rephrase(scientific_class, objective, *, backend, case_id=None):
        tr.add("P4", scientific_class=scientific_class, objective=objective, backend=backend.name, case_id=case_id)
        if sc.get("p4") == "fail":
            raise RuntimeError("p4 boom")
        return f"semantic sentence about {scientific_class}"

    def fake_search(query, limit=20, *, sources=("openalex",), mode="lexical"):
        n_search["n"] += 1
        tr.add("search", query=query, limit=limit, sources=list(sources), mode=mode)
        if sc.get("search_raises_after") is not None and n_search["n"] > sc["search_raises_after"]:
            raise BudgetExhausted("budget exhausted")
        return (_results_p5 if sc.get("real_p5") else _results)(query, 3)
    fake_search.last_errors = []

    def fake_extract(title, abstract, structural_class, also_known_as=None, *, backend, basis=None, case_id=None):
        tr.add("P5_extract", title=title, abstract_sha=_h(abstract), structural_class=structural_class, aka=list(also_known_as or []), basis=basis,
               case_id=case_id, backend=backend.name)
        return RichSystemExtraction(systems=[])

    def fake_discover(scientific_class, expansion_terms, *, search_fn, extract_fn, max_papers_to_extract=25, max_discovered=8, progress_fn=None, max_workers=1):
        tr.add("P5_discover", scientific_class=scientific_class, expansion_terms=list(expansion_terms), max_discovered=max_discovered,
               max_papers_to_extract=max_papers_to_extract, max_workers=max_workers)
        if sc.get("hop1") == "raise":
            raise RuntimeError("hop1 boom")
        hits = search_fn(expansion_terms[0] if expansion_terms else "seed")        # the closure ignores the term and sends the P4 sentence
        tr.add("hop1_search_result", titles=[p.title for p in hits])
        if hits:
            extract_fn(hits[0].title, hits[0].abstract or "", scientific_class, list(expansion_terms)[:3])
        systems = sc["systems"]() if "systems" in sc else []
        return DiscoveryResult(structural_class=scientific_class, discovered_systems=systems)

    def spy(attr, summarize):
        def make(orig):
            def f(*a, **k):
                tr.add(attr, **summarize(*a, **k))
                return orig(*a, **k)
            return f
        return make

    def sys_rows(systems):
        return [{"name": s.name, "n_support": s.n_support, "rank": s.first_seen_rank, "norm": s.norm_name, "papers": len(s.supporting_papers)} for s in systems]

    def make_identities(orig):
        def f(systems, *, scientific_class, basis=None):
            tr.add("identities_in", systems=sys_rows(systems), scientific_class=scientific_class, basis=basis)
            out = orig(systems, scientific_class=scientific_class, basis=basis)
            if sc.get("insufficient_identities"):
                out = [i.model_copy(update={"query_anchor": None, "grounding": "insufficient"}) for i in out]
            tr.add("identities_out", identities=[(i.label, i.grounding, i.query_anchor, i.n_support, i.first_seen_rank) for i in out])
            return out
        return f

    def make_hop2(orig):
        def f(selected, search_fn, paper_ref_fn):
            tr.add("hop2_in", selected=[(n.label, n.search_anchor, n.core_process, n.stable_descriptor, n.flags, n.n_support, n.first_seen_rank) for n in selected])
            if sc.get("hop2") == "raise":
                raise RuntimeError("hop2 boom")
            plans, found = orig(selected, search_fn, paper_ref_fn)
            if sc.get("inject_missing_ref") and found:
                first = next(iter(found))
                found[first]["10.404/not-in-cache"] = ["ghost query"]
            tr.add("hop2_out", found={k: {r: list(q) for r, q in v.items()} for k, v in found.items()})
            return plans, found
        return f

    def direct_papers():
        return [Paper(title=t, doi=("10.1/shared" if t.startswith("Shared") else f"10.2/{_h(t)}"), abstract=f"abs {t}", year=2019) for t in sc.get("direct", [])]

    def fake_route(plan_, budgets=None):
        tr.add("direct_route", budgets=None if budgets is None else _h(json.dumps(budgets.model_dump(), sort_keys=True, default=str)))
        return SimpleNamespace(), SimpleNamespace(), {}

    def fake_stage3_papers(route_result, direct_cache):
        return direct_papers()

    def fake_prov(route_result):
        return {f"10.2/{_h(t)}": SimpleNamespace(found_by=[SimpleNamespace(query=f"dq-{t}", route="flat"), SimpleNamespace(query=f"dq2-{t}", route="process")])
                for t in sc.get("direct", []) if not t.startswith("Shared")}

    def fake_filter(papers, *, classifier_threshold, recover_abstracts):
        tr.add("stage3", refs=[p.doi or p.title for p in papers], threshold=classifier_threshold, recover=recover_abstracts)
        rejected = papers[-1:] if len(papers) > 1 else []
        return papers[: len(papers) - len(rejected)], rejected, []

    gets = []
    real_get, real_set = cache.get, cache.set

    def traced_get(ns, key, **kw):
        v = real_get(ns, key, **kw)
        tr.add("cache.get", ns=ns, key=key, hit=v is not None)
        return v

    def traced_set(ns, key, value, **kw):
        tr.add("cache.set", ns=ns, key=key)
        return real_set(ns, key, value, **kw)

    pt.set("expand_terminology", lambda o: fake_expand)
    pt.set("rephrase_hop1_semantic_intent", lambda o: fake_rephrase)
    xs, state, rec = [], {"http": 0}, None
    if sc.get("real_search"):
        def fake_openalex(query, limit=20, *, mode="lexical", **kw):
            state["http"] += 1
            inside = any(f.f_code.co_name == "wrapper" and f.f_globals.get("__name__") == "recorder" for f in _frames())
            xs.append({"query": query, "limit": limit, "mode": mode, "inside_record": inside})
            return _results(query, 3)
        pt.set("search_openalex", lambda o: fake_openalex)
        rec = _recorder().Recorder(lambda: {"http_requests": state["http"], "cache_hits": 0}, lambda p: False, {}).install()
    else:
        pt.set("search_and_dedupe", lambda o: fake_search)

        def http_guard(query, limit=20, *, mode="lexical", **kw):
            # a binding of search_and_dedupe the layout did not patch ends up here instead of on the network; the event makes the trace differ
            tr.add("UNPATCHED_HTTP_BOUNDARY_REACHED", query=query, limit=limit, mode=mode)
            raise AssertionError("an unpatched search reached the HTTP boundary (the layout missed a binding of search_and_dedupe)")
        pt.set("search_openalex", lambda o: http_guard)
    if not sc.get("real_p5"):
        pt.set("extract_related_systems_rich", lambda o: fake_extract)
        pt.set("discover_related_systems", lambda o: fake_discover)
    pt.set("run_retrieval_from_plan", lambda o: fake_route)
    pt.set("papers_for_stage3", lambda o: fake_stage3_papers)
    pt.set("provenance_index", lambda o: fake_prov)
    pt.set("filter_papers", lambda o: fake_filter)
    pt.set("cleanup_discovered_systems", spy("cleanup_in", lambda systems, *, clean_objective: {"systems": sys_rows(systems), "objective": clean_objective}))
    pt.set("build_related_system_identities", make_identities)
    pt.set("normalize_search_anchors", spy("P6_in", lambda systems, objective, backend, case_id=None: {
        "systems": [(s.index, s.label, s.n_support, s.first_seen_rank, s.aliases, s.entity_type, s.identity_context) for s in systems],
        "objective": objective, "backend": backend.name, "case_id": case_id}))
    pt.set("merge_and_select", spy("merge_and_select", lambda normalized, max_systems=2: {"n": len(normalized), "max": max_systems}))
    pt.set("run_converged_hop2", make_hop2)
    cache.get, cache.set = traced_get, traced_set

    out = {"runs": []}
    try:
        for _ in range(sc.get("runs", 1)):
            run = {"events_from": len(tr.events), "x_from": len(xs), "l_from": len(rec.records) if rec else 0}
            try:
                res = LY.call_entry(layout, plan, prior, backend=backend, budgets=budgets, case_id="T")
                run["raised"] = None
                run["result"] = _sj({
                    "transfer_needed": res.transfer_needed,
                    "axis_logs": [{"mode": l.axis.execution_mode, "conf": l.axis.confidence, "executed": l.executed, "skip": l.skip_reason, "direct_queries": l.direct_queries,
                                   "hop1_queries": l.hop1_queries, "identities": [i.model_dump() for i in l.related_system_identities], "hop2_queries": l.hop2_queries,
                                   "hop2_plan": l.hop2_plan, "n_papers_found": l.n_papers_found, "error": l.error} for l in res.axis_logs],
                    "direct_papers": [_paper_row(p) for p in res.direct_papers], "transfer_papers": [_paper_row(p) for p in res.transfer_papers],
                    "retained": [_paper_row(p) for p in res.retained], "rejected": [_paper_row(p) for p in res.rejected], "verdicts": len(res.stage3_verdicts)})
            except LY.ContractViolation:
                raise                    # a layout/adapter setup error, not a behaviour of the flow
            except BaseException as e:   # noqa: BLE001 -- recorded; BudgetExhausted must have propagated out of the flow
                run["raised"] = f"{type(e).__name__}: {e}"
                run["result"] = None
            run["events_to"] = len(tr.events)
            run["x_to"], run["l_to"] = len(xs), (len(rec.records) if rec else 0)
            out["runs"].append(run)
    finally:
        cache.get, cache.set = real_get, real_set
        if rec is not None:
            out["recorder_installed_in"] = sorted(n for n in _recorder().MODULES if getattr(getattr(sys.modules.get(n), "search_and_dedupe", None), "_is_recorder", False))
            out["recorder"] = {
                "L": [{k: r[k] for k in ("n", "kind", "mode", "limit", "query", "served_from", "n_results", "error")} for r in rec.records],
                "X": xs, "U": [x for x in xs if not x["inside_record"]]}
            rec.uninstall()
        pt.restore()
    out["events"] = _sj(tr.canonical())
    out["llm_calls"] = backend.calls
    out["patch_calls"] = dict(sorted(pt.calls.items()))
    if sc.get("expect_raised"):
        assert out["runs"][0]["raised"] and out["runs"][0]["raised"].startswith("BudgetExhausted"), "BaseException must propagate out of the flow"
    return out


def all_traces(scenarios=None, layout=None) -> dict:
    return {name: run_scenario(name, layout) for name in (scenarios or SCENARIOS)}


def _write(fixture: Path, scenarios) -> None:
    if fixture.exists():
        sys.exit(f"{fixture.name} exists: a baseline fixture is written once, from tag consol-0-baseline, and never edited")
    fixture.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    for n in scenarios:
        os.environ["CGSCOUT_CACHE_DIR"] = tempfile.mkdtemp()      # a cold cache per scenario, like the autouse fixture in conftest.py
        data[n] = run_scenario(n)
    fixture.write_text(json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    print("wrote", fixture, len(data), "scenarios")


if __name__ == "__main__":
    if "--write" in sys.argv:
        _write(FIXTURE, SCENARIOS)
    elif "--write-p5" in sys.argv:
        _write(FIXTURE_P5, SCENARIOS_P5)
    else:
        sys.exit("usage: python tests/consolidation_world.py --write | --write-p5   (baseline only; refuses to overwrite)")
