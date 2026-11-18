# Phase 3 cleanup record

What was removed or moved after the Phase 2 restructure, why that was safe, and how to undo each step. Prompts, schemas, cache keys, budgets, configuration and the retrieval / screening /
scoring / ranking algorithms were not touched (prompt sha256, pydantic schemas, defaults and the P1/P2 module hashes are identical to the Phase 2 state, the commit whose subject starts `Phase 2 / S7:`).

| commit (subject starts with) | step |
|---|---|
| `Phase 3 / S8a:` | S8a: 7 unreferenced definitions and 3 unused imports removed (every unused parameter was kept: signatures are call interfaces) |
| `Phase 3 / S8b-1:` | S8b-1: `_extract_direct_system_terms` removed |
| `Phase 3 / S8b-2:` | S8b-2: `_infer_modeling_requirements` removed |
| `Phase 3 / S8b-3:` | S8b-3: `extract_lexical_structural_feature` (+ Case 3 of `_extract_structural_terms`) removed, together with its 18 tests |
| `Phase 3 / S8e:` | S8e: offline tests for `cgscout` (`tests/test_cli.py`), `profiles.py` docstring and architecture notes corrected; `cli.py` unchanged |
| `Phase 3 / merge 1:` | merge 1: `assessment/evidence_extraction.py` -> `assessment/stage4.py` |
| `Phase 3 / merge 2:` | merge 2: `screening/review_filter.py` -> `screening/filter.py` |
| `Phase 3 / merge 3:` | merge 3: `structured.py` -> `retrieval/related_systems/terminology.py` |
| `Phase 3 / merge 4:` | merge 4: `retrieval/dedupe.py` -> `retrieval/search.py` |

Commits are identified by their subject line, not by a hash: a commit's hash changes whenever the history is rewritten (for example when it is cleaned for publication), its subject does not.
Show one with `git log --oneline --grep='^Phase 3 / S8b-2:'`, or undo it with

```bash
git revert "$(git rev-list --reverse --grep='^Phase 3 / S8b-2:' HEAD | head -n 1)"
```

(`--reverse | head -n 1` takes the oldest match, so a later commit that quotes the subject can never be picked by mistake).

Not done on purpose: `query_construction_v2` and Semantic Scholar are kept; `semantic_intent` (P4) is not merged into `terminology` (P3).

## S8b: diagnostic fields whose values changed

The three removed analysers filled fields of `EvidenceRoutePlan` that no active code reads. A reader trace (every read of these fields while the real code ran over 8,625 plan runs and the
6-case real replay) found only these readers: `plan_evidence_routes` itself (to build `empty_routes`), `build_route_retrieval_plan` (copies `empty_routes` into `RouteRetrievalPlan.empty_routes`,
which nothing reads, and passes `structural_terms` to `select_primary_structural_term`) and the `structural_branch_enabled` guard in `run_route_retrieval`. In the live path `structural_terms` is always
empty at that point (the bridge `from_planner.evidence_plan_from_planner_output` clears it), so the guard never fires.

| field (`EvidenceRoutePlan`) | before | after |
|---|---|---|
| `direct_system_terms` | one `USER_GROUNDED` term (a copy of the system text) when the text was non-blank (8,540 of 8,625 runs) | always `[]` |
| `modeling_requirement_terms` | up to three `PLANNER_DERIVED` hypotheses for "binding / interface", "folding", "conformational change" objectives (3,560 of 8,625 runs) | always `[]` |
| `structural_terms` of a **direct** `plan_evidence_routes` call | a lexically derived term such as "beta-helix" (4,915 of 8,625 runs) | `[]` unless a structural term is passed in explicitly (`grounded_structural_feature`, `literature_structural_mentions`: unchanged) |
| `empty_routes` / `route_justification` (and the copy in `RouteRetrievalPlan.empty_routes`) | recorded the structural and modeling-requirement routes as empty only when their analyser found nothing | always record `modeling_requirement` as empty (appended last, as before) and `structural` as empty unless an explicit structural term was passed; no `modeling_requirement` justification key |

Kept: all model fields (schemas unchanged), `select_primary_structural_term`, the `NotImplementedError` guard, the bridge line that clears `structural_terms`, every parameter of
`plan_evidence_routes`, the explicit-input handling of `_extract_structural_terms`.

Verified run by run (exact values) and, for everything else, by an old-vs-new differential of the whole direct route (124 real planner outputs + 1,601 synthetic ones x 5 budget settings; every query string,
route, role, budget and the accumulator result identical).

**If any active code is ever found to depend on one of these fields**, restore the matching analyser by reverting its commit (each command resolves the commit by its subject, as described above):

```bash
# modeling requirements
git revert "$(git rev-list --reverse --grep='^Phase 3 / S8b-2:' HEAD | head -n 1)"
# lexical structural extraction, with its 18 tests
git revert "$(git rev-list --reverse --grep='^Phase 3 / S8b-3:' HEAD | head -n 1)"
# direct-system term
git revert "$(git rev-list --reverse --grep='^Phase 3 / S8b-1:' HEAD | head -n 1)"
```

## Module merges (pure moves)

The body of each merged module sits verbatim in its target, under a comment that carries the old module docstring; only imports were adjusted.

| old module | now in | note |
|---|---|---|
| `assessment/evidence_extraction.py` (P8 prompt, builder, prompt hash) | `assessment/stage4.py` | three alias assignments keep the names stage4 used for the imported objects |
| `screening/review_filter.py` | `screening/filter.py` | |
| `structured.py` (`_schema`) | `retrieval/related_systems/terminology.py` | |
| `retrieval/dedupe.py` | `retrieval/search.py` | `tests/test_dedupe.py` imports `dedupe_papers` from `cgscout.retrieval.search` |

Source modules (without `__init__.py`): 37 -> 33.

## Checks used for every step

full test suite; 6-case offline golden replay (byte compare, no LLM attempt); static fingerprints of prompts, long strings, constants and every pydantic schema; definition-level AST comparison
(every old definition must exist with an identical AST, in any module); for merges, the merged body must appear verbatim in its target; the direct-route differential; sha256 of the nine system prompts,
the prompt versions, the budget / Stage-4 / profile defaults, the cache namespaces and the P1/P2 module hashes.
