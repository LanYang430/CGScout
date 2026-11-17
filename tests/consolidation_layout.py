"""Consolidation safety net -- WHERE the retrieval flow lives, HOW it is called, WHAT is patched, and WHICH module-path renames are approved.

The trace world (consolidation_world.py) must keep reproducing `tests/fixtures/consolidation/baseline_traces.json` (written once from tag
consol-0-baseline, never edited) while the production code moves under it.  Everything that can legitimately change in a move is declared here, in ONE
`Layout`; nothing else in the world names a production module.

  entry            (module, attribute) of the callable that runs retrieval, plus the ADAPTER that calls it.  An adapter hides signature differences
                   (old: `run_expert_prior_retrieval(plan, prior, backend=, budgets=, case_id=)`; a unified entry takes its budgets inside a config object
                   and also runs Stage 4, which the world must switch off) and must return the RETRIEVAL result object (RESULT_ATTRS) -- never a wrapper.
  patch_targets    logical collaborator name -> every (module, attribute) that binds it.  The logical names are the BASELINE attribute names; they are
                   also the keys of `patch_calls` in the trace, so a move changes this table and no trace key.  A flow function looks a patched name up in
                   ITS module's globals; if the new flow binds the name in a module this table does not list, the patch silently does nothing -- which the
                   world turns into a failure (patched collaborator never called / unpatched HTTP boundary reached / unexpected LLM schema).
  recorder_binders modules in which `search_and_dedupe` is bound at module level (the recorder must wrap exactly these; guards verify it).
  module_aliases   APPROVED rename map  new module path -> baseline module path.  The only normalization the comparison applies (MODULE_FIELDS below):
                   it can make a renamed module compare equal to its old name, it can never hide a missing, extra or unmapped module.
  approval         who approved this layout and for which phase (text; the baseline layout is the tagged code itself).

Adding a layout is part of the commit that moves the code (never before): the layout, the production move and (if it changed) the recorder's MODULES go
in together, then `pytest tests/test_consolidation_*.py` with CGSCOUT_CONSOLIDATION_LAYOUT=<name> must pass without touching any fixture.
tests/test_consolidation_migration.py proves, offline and with no production change, that this machinery really survives a move (a simulated one).
"""
from __future__ import annotations

import copy
import importlib
import os
from dataclasses import dataclass, field

BASE_FLOW = "cgscout.retrieval.orchestrator"
SEARCH_MODULE = "cgscout.retrieval.search"

# every collaborator the world replaces, under its BASELINE attribute name
COLLABORATORS = (
    "expand_terminology", "rephrase_hop1_semantic_intent", "search_and_dedupe", "search_openalex", "extract_related_systems_rich",
    "discover_related_systems", "run_retrieval_from_plan", "papers_for_stage3", "provenance_index", "filter_papers",
    "cleanup_discovered_systems", "build_related_system_identities", "normalize_search_anchors", "merge_and_select", "run_converged_hop2",
)

# what an adapter must hand back (attribute names the trace serializes)
RESULT_ATTRS = ("transfer_needed", "axis_logs", "direct_papers", "transfer_papers", "retained", "rejected", "stage3_verdicts")
AXIS_LOG_ATTRS = ("axis", "executed", "skip_reason", "direct_queries", "hop1_queries", "related_system_identities", "hop2_queries", "hop2_plan",
                  "n_papers_found", "error")

# the trace fields that carry module paths (the only place `module_aliases` is applied)
MODULE_FIELDS = ("recorder_installed_in",)
# the facets that must be equal to the baseline
FACETS = ("events", "runs", "llm_calls", "patch_calls", "recorder", "recorder_installed_in")


@dataclass(frozen=True)
class Layout:
    name: str
    entry: tuple                                   # (module, attribute)
    adapter: str                                   # key of ADAPTERS
    patch_targets: dict
    recorder_binders: tuple
    module_aliases: dict = field(default_factory=dict)
    approval: str = ""


def baseline_targets(flow: str = BASE_FLOW) -> dict:
    t = {name: ((flow, name),) for name in COLLABORATORS}
    t["search_openalex"] = ((SEARCH_MODULE, "search_openalex"),)        # the HTTP boundary never moves
    return t


# ---------------------------------------------------------------- adapters
def legacy_retrieval(layout: Layout, plan, prior, *, backend, budgets, case_id):
    """Baseline entry: run_expert_prior_retrieval(plan, prior, backend=, budgets=, case_id=).  `budgets=None` is passed through as None (the function's
    own `budgets or RouteRetrievalBudgets()` applies the default), exactly like cli.run_retrieval does."""
    fn = getattr(importlib.import_module(layout.entry[0]), layout.entry[1])
    return fn(plan, prior, backend=backend, budgets=budgets, case_id=case_id)


def agent_execute(layout: Layout, plan, prior, *, backend, budgets, case_id):
    """Unified entry (Phase 2a): cgscout.agent.execute.  (1) `budgets` goes into the agent configuration; None stays None = the default configuration, NEVER the paper
    profile; (2) Stage 4 is switched OFF by a gate that says no (the world fakes Stage 3 only; Stage 4 has its own entry baselines in
    tests/test_consolidation_entries.py); (3) backend/case_id pass unchanged; (4) the adapter returns the RETRIEVAL result of the AgentResult, not the wrapper."""
    mod = importlib.import_module(layout.entry[0])
    out = getattr(mod, layout.entry[1])(plan, prior, backend=backend, config=mod.AgentConfig(budgets=budgets), case_id=case_id, gate=lambda res: False)
    return out.retrieval


ADAPTERS = {"legacy_retrieval": legacy_retrieval, "agent_execute": agent_execute}


class ContractViolation(AssertionError):
    """The adapter did not return the retrieval result.  The world re-raises it (it is a setup error, not a behaviour of the flow to be traced)."""


def check_result_contract(res) -> None:
    missing = [a for a in RESULT_ATTRS if not hasattr(res, a)]
    if missing:
        raise ContractViolation(f"adapter returned an object without {missing}: it must return the retrieval result, not a wrapper")
    for log in res.axis_logs:
        miss = [a for a in AXIS_LOG_ATTRS if not hasattr(log, a)]
        if miss:
            raise ContractViolation(f"axis log lacks {miss}")


def call_entry(layout: Layout, plan, prior, *, backend, budgets, case_id):
    res = ADAPTERS[layout.adapter](layout, plan, prior, backend=backend, budgets=budgets, case_id=case_id)
    check_result_contract(res)
    return res


# ---------------------------------------------------------------- layouts
BASELINE = Layout(
    name="baseline",
    entry=(BASE_FLOW, "run_expert_prior_retrieval"), adapter="legacy_retrieval",
    patch_targets=baseline_targets(),
    recorder_binders=("cgscout.retrieval.orchestrator", "cgscout.retrieval.search_adapters"),
    module_aliases={}, approval="baseline = tag consol-0-baseline (c8be98c); nothing to approve")

# Phase 3 (3B L2): the 3B workflow is cgscout.agent._discover_related_systems; the former retrieval/transfer._run_related_system_discovery is gone.  Where each patched
# collaborator is LOOKED UP now (rule R-1):
#   agent         P3 expand_terminology, P4 rephrase_hop1_semantic_intent, P5 extract_related_systems_rich (in the closure), discover_related_systems, cleanup_discovered_systems,
#                 build_related_system_identities, normalize_search_anchors (P6), merge_and_select, run_converged_hop2, and filter_papers (_stage3_screen)
#   transfer      search_and_dedupe (the 3A executor and the Hop-1 / Hop-2 closures made by _make_search_fn)
#   from_planner  run_retrieval_from_plan, papers_for_stage3, provenance_index (retrieve_direct), and search_and_dedupe (the capturing wrappers)
# agent binds NO search_and_dedupe.  Recorder binders, aliases and entry are those of 2b (the closures that search are still in transfer, function names unchanged).
TRANSFER, PLANNER_BRIDGE, AGENT = "cgscout.retrieval.transfer", "cgscout.retrieval.direct.from_planner", "cgscout.agent"


def phase3_targets() -> dict:
    t = {name: ((AGENT, name),) for name in (
        "expand_terminology", "rephrase_hop1_semantic_intent", "extract_related_systems_rich", "discover_related_systems", "cleanup_discovered_systems",
        "build_related_system_identities", "normalize_search_anchors", "merge_and_select", "run_converged_hop2", "filter_papers")}
    t["search_and_dedupe"] = ((TRANSFER, "search_and_dedupe"),)
    t.update({name: ((PLANNER_BRIDGE, name),) for name in ("run_retrieval_from_plan", "papers_for_stage3", "provenance_index")})
    t["search_openalex"] = ((SEARCH_MODULE, "search_openalex"),)                # the HTTP boundary never moves
    assert set(t) == set(COLLABORATORS)
    return t


PHASE3 = Layout(
    name="phase3", entry=(AGENT, "execute"), adapter="agent_execute", patch_targets=phase3_targets(),
    recorder_binders=(TRANSFER, PLANNER_BRIDGE),
    module_aliases={TRANSFER: BASE_FLOW, PLANNER_BRIDGE: "cgscout.retrieval.search_adapters"},
    approval="Phase 3: 3B L2 skeleton in agent.py (owner approved Phase 1-4 on 2026-11-16; CONSOLIDATION_PLAN_v2 sections 4, 5.3, 8 + Addendum; own commit, own acceptance); "
             "earlier layouts are history: their patch targets are no longer looked up where they were")

# `baseline` stays as the reference object for the aliases; the earlier layouts were valid only while their modules / functions existed.
LAYOUTS = {"phase3": PHASE3}
CURRENT = "phase3"


def active_layout() -> Layout:
    name = os.environ.get("CGSCOUT_CONSOLIDATION_LAYOUT", CURRENT)
    assert name in LAYOUTS, f"unknown layout {name!r}; known: {sorted(LAYOUTS)} (a layout is registered in the commit that moves the code)"
    return LAYOUTS[name]


# ---------------------------------------------------------------- normalization and comparison
def normalize(trace: dict, layout: Layout) -> dict:
    """Apply the APPROVED module renames to the module-path fields, nothing else.  A module with no approved alias stays as it is (and therefore differs)."""
    t = copy.deepcopy(trace)
    for f in MODULE_FIELDS:
        if f in t:
            t[f] = sorted({layout.module_aliases.get(m, m) for m in t[f]})
    return t


def diff_against_baseline(got: dict, base: dict, layout: Layout) -> dict:
    """{} when the trace equals the baseline after the approved normalization; else {facet: why}.  A patched collaborator the baseline called but the new
    flow never called is reported on its own (R-1/R-2: the patch targeted the wrong module or the flow no longer references the name)."""
    got = normalize(got, layout)
    out = {f: "differs from the baseline" for f in FACETS if got.get(f) != base.get(f)}
    never = sorted(k for k, v in base.get("patch_calls", {}).items() if v and not got.get("patch_calls", {}).get(k))
    if never:
        out["patch_never_called"] = f"{never}: the baseline called these patched collaborators, the migrated flow did not (wrong patch target or the name is no longer looked up there)"
    return out
