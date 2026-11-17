# Architecture

CGScout turns a modeling question into a ranked, explained list of papers. Prompts are numbered P1-P9 in the order they run; the nine texts are in [../prompts/](../prompts/INDEX.md).

## Control flow

The order of the steps is fixed and lives in one module, `agent.py`. It is a workflow-driven agent: the LLM does the scientific planning and the evidence judgement, the code decides what
runs when, and results are not guaranteed to repeat run to run.

```
agent.run(text)                       one user turn:  P1 planner -> P2 expert prior -> execute   (a Clarification is returned when P1 needs an answer)
agent.execute(plan, prior)            retrieve -> gate -> assess                                   (the CLI, the evaluation harness and the tests all go through it)
  agent.retrieve                      direct stream -> per-axis loop (3A | 3B) -> merge -> Stage 3
  agent.assess                        Stage 4
```

`gate` is how the two callers differ on purpose: the command line's default gate runs Stage 4 only when something was retained; the evaluation harness passes a gate that raises on budget
exhaustion and skips Stage 4 for a degraded run (failed HTTP requests or an axis error) but runs it on every healthy run, also when nothing was retained.

## Stage, function, prompt

The middle column names the function that does the work (`module.function`, resolved by `tests/test_architecture_doc.py`).

| stage | function | LLM prompt |
|---|---|---|
| plan | `session.PlannerSession.submit_user_message` (one `scientific_planner.ScientificPlanner.plan` call; asks a clarifying question only when the ambiguity changes what to search) | P1 |
| expert prior | `agent.propose_prior` (`expert_prior_planner.ExpertPriorPlanner.propose`: related systems/classes worth searching, each with confidence and execution mode) | P2 |
| Direct stream | `from_planner.retrieve_direct` (deterministic flat queries from the plan: system-specific CG, process, context, optional semantic; `route_plan.run_route_retrieval`) | - |
| 3A, concept queries | `agent._plan_axis_queries` (fixed template queries per high/medium-confidence axis, `concept_3a.render_direct_concept_queries`), executed by `transfer._run_direct_concept_search` | - |
| 3B, Hop-1 | `agent._discover_related_systems`: `terminology.expand_terminology`, `transfer.hop1_seed_terms`, `semantic_intent.rephrase_hop1_semantic_intent`, `discovery.discover_related_systems` (per paper `extraction.extract_related_systems_rich`), `postprocess.cleanup_discovered_systems`, `postprocess.build_related_system_identities` | P3 terminology, P4 semantic sentence, P5 per-paper extraction |
| 3B, Hop-2 | same workflow: `transfer.hop2_candidates`, `hop2.normalize_search_anchors`, `hop2.merge_and_select`, `hop2.run_converged_hop2`, then `transfer.tag_hop2_papers` | P6 search-anchor normalization |
| merge | `agent.retrieve` with `transfer._merge_papers`: union of Direct + 3A + 3B with per-paper route provenance | - |
| Stage 3 | `agent._stage3_screen` -> `filter.filter_papers`: keyword/CG-evidence gate, review filter, embedding classifier (`sim_classifier`), Europe PMC abstract recovery | - (embeddings only) |
| Stage 4 | `agent.assess` -> `stage4.run_stage4_v2`: tier-1 screen, evidence extraction, judge, rank | P7 screen, P8 extraction, P9 judge |
| display order (no Stage 4) | `agent.rank_for_display` | - |

The 3B workflow is the one place with a failure contract that others depend on (its docstring in `agent.py` has the full list): P3 has its own `try` (a failure means no aliases, the axis goes on and a later
success keeps the error); P4 is in no `try` (a failure ends that axis with `error = str(exc)`); a Hop-1 failure returns `hop1: ...`; "no system found" keeps the P3 error; a Hop-2 failure returns `hop2: ...` and
*replaces* the P3 error; budget exhaustion (a `BaseException`) passes every handler. A non-empty `error` on any axis marks the whole run degraded in the evaluation harness.

Files: `agent.py` (the flow, the result types, the 3B workflow, Stage 3 as a private step), `retrieval/transfer.py` (the 3A executor, the search closures of Hop-1/Hop-2, the paper-merge helper and the three
assembly steps of 3B), `retrieval/direct/*` (the direct stream: `evidence_plan.py`, `route_plan.py`, `from_planner.py` with the capturing search wrappers, `query_construction_v2.py` experimental and off),
`retrieval/related_systems/*` (`terminology`, `semantic_intent`, `extraction`, `discovery`, `postprocess` = cleanup + identity, `hop2`), `retrieval/concept_3a.py`, `retrieval/search.py` (`search_and_dedupe`, the Hop depths),
`screening/{filter,sim_classifier}.py` (filter.py also holds the keyword helpers and the review/perspective check), `assessment/stage4.py` (also holds the P8 extraction prompt and message builder).

Sources: `sources/openalex.py` (+ `openalex_usage.py`: budget, request cache, ledger), `sources/europepmc.py`, `sources/semantic_scholar.py` (optional, off by default).
Shared: `llm.py` (backend protocol, OpenAI backend, usage log), `cache.py`, `models.py` (`Paper`), `config.py` (`PAPER`, `INTERACTIVE`; also the `CGSCOUT_*` environment lookup).

## Invariants worth knowing

- One run uses exactly one LLM backend, resolved once (`llm.get_backend()`) and passed down explicitly; no module resolves a backend on its own.
- Caches are content-keyed (prompt + schema + input [+ backend/model]) so an edit to a prompt invalidates exactly the entries it affects. Searches are cached for 24 h.
- Hop-1 discovery grounds every discovered system in a verbatim quote of a retrieved paper; no system name comes from the model's memory.
- Stage 4 never grades a paper without an abstract (`missing_evidence`), and a failed LLM call is an error, never a "not relevant".
- Run configuration is explicit data: `config.py` (`PAPER`, `INTERACTIVE`; also the `CGSCOUT_*` environment lookup).

## Known issues, recorded but not changed in the cleanup

The code was reorganized without changing behaviour, so these stay as they were: the P3 expansion list is overridden by P4's sentence in Hop-1 queries; `transfer_needed` does not gate the routes;
`structural_hop2_max_systems` (30) is not the number of systems Hop-2 actually takes (2 per axis); P4/P7/P9 cache keys do not contain the prompt/model;
the `modeling_requirement` route name is kept in `FLAT_ROUTES` although no queries are built for it; the library default has `need_semantic_enabled=False` while the paper profile turns it on.

Removed in the Phase 3 cleanup because nothing read their output (verified by a reader trace and by an old-vs-new differential of the direct route): the analysers that filled
`EvidenceRoutePlan.direct_system_terms`, `modeling_requirement_terms` and the lexically derived `structural_terms`. The fields stay in the models (schemas unchanged), as do the explicit-input
handling of `plan_evidence_routes` (grounded structural feature, literature mentions), `select_primary_structural_term` and the `NotImplementedError` guard in `run_route_retrieval`.
So `empty_routes` / `route_justification` now report the modeling-requirement route as always empty, and the structural route as empty unless a structural term is passed in explicitly (no code reads those two fields).
