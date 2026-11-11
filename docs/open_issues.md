# Open issues (recorded, not fixed)

Each item must be resolved, with the retrieval ability kept, **before the final 200-query run** (freeze the plans only after it is fixed).

## OPEN-1 `target_component` is an unresolved ambiguity but is used as a resolved component

Status: recorded 2026-11-11, owner decision: record as an independent bug, minimal-fix proposal to be prepared later; no code or P1 schema change yet.

**What happens.** P1 sets `target_component` only when the request is ambiguous about which part of a multi-component system is meant, and the prompt tells it to "name the ambiguity concretely" (e.g. "X vs Y vs full-length"). So the field holds an **unresolved choice**, not a component. The downstream code treats it as the chosen component:

- `retrieval/direct/from_planner.py:52-53`: `system_description = f"{system} ({target_component})"`. The ambiguity phrase becomes part of the system text from which the direct-route queries are built.
- `planning/expert_prior_planner.py` (user message of P2): `TARGET COMPONENT: "<phrase>"`.
- `assessment/stage4.py:373-374` (problem context of P7/P9): `- Target component: <phrase>`.
- `cli.py`: printed and passed on.

**Evidence.**
- Stored ID21 run (`benchmark_new37_beam/results/CUR/ID21.json`): `target_component = "individual nucleolar condensate or subcompartment versus the nucleolar assembly as a whole"`; three of the four flat queries contain it, e.g. `nucleolar condensates (individual nucleolar condensate or subcompartment coarse-grained model` (words cut off, parenthesis not closed).
- 2 of 123 stored plans set it (ID21, P01); in the P1 comparison it was set for P01 by the current prompt (`cytoskeletal filament vs mechanosensitive scaffold protein vs the coupled filament–scaffold assembly`) and for ID21 and P01 by the original prompt.
- In the benchmark, P1's question is auto-answered with a fixed no-information sentence, so the ambiguity is never resolved there and the phrase stays in the plan.

**Constraints for the fix.** Keep the P1 output schema (8 fields) and the retrieval ability; do not rewrite P1's clarification rules to work around it; check the three consumers above; verify with the stored ID21 / P01 plans (query strings before / after) and the differential harness before any change is accepted.
