# Architecture

CGScout turns a modeling question into a ranked, explained list of papers. Prompts are numbered P1–P9 in the order they run.

## Stages and where they live

| stage | what it does | module(s) | LLM prompt |
|---|---|---|---|
| planning | decompose the request: system, target component, objective, search context/axes; ask a clarifying question only when the ambiguity changes what to search | `planning/scientific_planner.py`, `planning/session.py` | P1 |
| expert prior | propose related systems/classes ("transfer axes") worth searching, each with confidence and execution mode | `planning/expert_prior_planner.py` | P2 |
| Direct route | deterministic flat queries (system-specific CG, process, context; optional semantic query) from the plan; `evidence_plan.py` fills the process and context terms | `retrieval/direct/from_planner.py`, `evidence_plan.py`, `route_plan.py`, `query_construction_v2.py` (experimental, off) | — |
| 3A (concept queries) | fixed template queries per high/medium-confidence axis | `retrieval/concept_3a.py`, `retrieval/orchestrator.py` | — |
| 3B Hop-1 | find concrete related systems in the literature | `retrieval/related_systems/{terminology,semantic_intent,extraction,discovery,identity,cleanup}.py` | P3 terminology, P4 semantic intent, P5 per-paper extraction |
| 3B Hop-2 | search each discovered system's own CG literature | `retrieval/related_systems/hop2.py` | P6 search-anchor normalization |
| merge | union of Direct + 3A + 3B with per-paper route provenance | `retrieval/orchestrator.py` | — |
| Stage 3 | keyword/CG-evidence gate, review filter, embedding classifier, Europe PMC abstract recovery | `screening/{filter,review_filter,sim_classifier}.py`, `sources/europepmc.py` | — (embeddings only) |
| Stage 4 | tier-1 screen → evidence extraction → judge → rank | `assessment/stage4.py`, `assessment/evidence_extraction.py` | P7 screen, P8 extraction, P9 judge |

Sources: `sources/openalex.py` (+ `openalex_usage.py`: budget, request cache, ledger), `sources/europepmc.py`, `sources/semantic_scholar.py` (optional, off by default).
Shared: `llm.py` (backend protocol, OpenAI backend, usage log), `cache.py`, `models.py` (`Paper`), `text.py`, `structured.py`.

## Invariants worth knowing

- One run uses exactly one LLM backend, resolved once (`llm.get_backend()`) and passed down explicitly; no module resolves a backend on its own.
- Caches are content-keyed (prompt + schema + input [+ backend/model]) so an edit to a prompt invalidates exactly the entries it affects. Searches are cached for 24 h.
- Hop-1 discovery grounds every discovered system in a verbatim quote of a retrieved paper; no system name comes from the model's memory.
- Stage 4 never grades a paper without an abstract (`missing_evidence`), and a failed LLM call is an error, never a "not relevant".
- Run configuration is explicit data: `profiles.py` (`PAPER`, `INTERACTIVE`).

## Known issues, recorded but not changed in the cleanup

The code was reorganized without changing behaviour, so these stay as they were: the P3 expansion list is overridden by P4's sentence in Hop-1 queries; `transfer_needed` does not gate the routes;
`structural_hop2_max_systems` (30) is not the number of systems Hop-2 actually takes (2 per axis); P4/P7/P9 cache keys do not contain the prompt/model;
the `modeling_requirement` route name is kept in `FLAT_ROUTES` although no queries are built for it; the library default has `need_semantic_enabled=False` while the paper profile turns it on.

Removed in the Phase 3 cleanup because nothing read their output (verified by a reader trace and by an old-vs-new differential of the direct route): the analysers that filled
`EvidenceRoutePlan.direct_system_terms`, `modeling_requirement_terms` and the lexically derived `structural_terms`. The fields stay in the models (schemas unchanged), as do the explicit-input
handling of `plan_evidence_routes` (grounded structural feature, literature mentions), `select_primary_structural_term` and the `NotImplementedError` guard in `run_route_retrieval`.
So `empty_routes` / `route_justification` now report the modeling-requirement route as always empty, and the structural route as empty unless a structural term is passed in explicitly (no code reads those two fields).
