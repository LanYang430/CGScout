# Prompt audit: P2 (`EXPERT_PRIOR_SYSTEM_PROMPT`)

Status: **edits applied and committed (2026-11-11); live P2 comparison done; owner decision on E2 pending.** Next prompt: P3 (not started).

Owner's rule for this prompt: keep the debugged original as far as possible (sentence structure, concrete examples, long explanations); change only project / case-specific wording, clear typos, and descriptions that contradict the code; no prompt optimisation, no new rules.

## What was changed (branch `cleanup/phase2-cgscout-package`)

| edit | content | commit |
|---|---|---|
| E1 | project name removed ("... planner for a literature-discovery agent focused on coarse-grained ...") | `f6eede1` |
| E3a | "three SEPARATE fields" -> "four" (`retrieval_anchors` was added later) | `f6eede1` |
| E3b-e | stray continuation whitespace (two 5-space runs), `physical/structural/ mechanistic`, two `discriminative_ features`, grammar "A property so generic that it would apply ..." | `f6eede1` |
| E4 | `direct_concept_search`: "scientific_class + a retrieval_anchor" -> "scientific_class + the user's scientific objective" (the 3A templates use scientific_class and the objective phenomenon, not the anchors) | `f6eede1` |
| E5 | `objective_relevance`: "... read by a human and by downstream logic that decides whether to search at all" -> "... and provided as context for downstream literature relevance assessment" (only reader: `stage4.build_problem_context` -> P7 screen, P9 judge) | `f6eede1` |
| E2 | example pair "autotransporter passenger domain" / "beta-helix" -> "immunoglobulin-like domain" / "beta-sandwich" (good/bad contrast sentence unchanged) | `5f2fcc3` (own commit, revertable alone) |

Gate for both commits: 438 tests pass; 6-case offline golden replay byte-identical (0 LLM attempts); manifest differs only in the P2 prompt (chars, sha256) and the `expert_prior_planner` module hash; AST identical outside the prompt string. P2 sha256 prefix: original `28218d41e6`, after `f6eede1` `061b025269`, current `6c03b0d4eb`. P1, P3-P9 and all 54 schemas unchanged. Retrieval, scoring, cache, schema and the user-message template were not touched.

Facts checked before choosing the E2 terms: "immunoglobulin / Ig-like / titin / I27 / beta-sandwich / sandwich" have 0 hits in the 200-query CSV (query, paper title, Source X, Target Y, v1 query), the unified workbook and the 123 stored plans. The old terms (autotransporter, pertactin, EspP, beta-helix, solenoid) also have 0 hits in the 200-query CSV; B034E (`benchmark_b034_espp_beam`), I11 (`benchmark_initial11`) and ID1 (`benchmark_new37_beam`) are development / extra sets, not part of the 200 (the 200's B034 is a DNA nanopore).

## Evidence: live P2 comparison (P2 only; no retrieval, no Stage 3/4)

Design: 20 inputs, fixed (frozen P1 output reused; the P1 plan of B122 was generated once and shared by both arms): B034E, I11, ID1, B071, B055, M01, ID28, ID12, M04, B204, ID15, C9, FRESH-019, M07, B190, ID21, P01, P14, P19, B122. B250 (fibronectin) was dropped: P1 did not reach READY in 3 attempts, so it would not be a valid benchmark input. Model gpt-5.6-luna, effort medium, no response cache (`llm.py`), same call path as `harness.propose`. Old = tree at `52e8554`, new = HEAD. Runs: every case 2 x old and 2 x new; ID1, M01, ID12, ID21, P19 (flagged after round 1), B034E, I11 (key cases) 4 x each; A-only arm (tree at `f6eede1`, E2 not applied) 2 x on ID1, ID21, B071, B055, C9, ID28, B204. 132 calls, about $0.18. Raw data, inputs, prompt texts, tools: `archived-audit-data/P2/`.

**What stayed inside the run-to-run noise** (54 old vs 54 new runs; noise = old-vs-old pairs):

| metric | old | new |
|---|---|---|
| axes per run | 1.96 | 2.07 |
| transfer_needed = false | 7/54 | 6/54 (same cases: M07, P14; P19 flips in both arms) |
| fold-word-only scientific_class | 0 | 0 |
| confidence low / medium | 0 / 9 | 0 / 3 |
| anchors per axis, features per axis | 4.75, 2.33 | 4.78, 2.32 |
| pair stats: abs. difference in axis count (old-old / new-old / new-new) | 0.36 / 0.43 / 0.55 | |
| pair stats: basis-set Jaccard | 0.78 / 0.75 / 0.73 | |
| pair stats: class-token Jaccard | 0.45 / 0.44 / 0.46 | |

**Scientific-class constraint preserved**: no fold-only or generic class in either arm; B034E (EspP) and I11 (pertactin) kept an "autotransporter passenger domain"-type class in 4/4 new runs each (the historical asymmetry, where the class was dropped, did not come back); no echo of the new example terms ("immunoglobulin", "Ig-like", "sandwich") in any new output, including the generic-protein case B122. Mean class length 4.07 -> 3.79 words; the extra short (<= 2 words) classes (6 -> 13) are mainly M01 ("charge-patterned proteins", "metamorphic proteins": same kind of phrases appear in the old arm) and "secretin-family GPCRs"; none is generic.

**Two shifts that are NOT run-to-run noise** (both reproduce inside each case and are attributed to E2 by the A-only arm):

1. **Plural -> singular class names.** Share of scientific_class strings ending in a plural noun: old 78% (83/106), new 41% (46/112); the stored older plans give 82% (31/38 on the same cases). Six cases flipped completely (ID1 13/15 -> 0/13, ID21 12/12 -> 0/11, B071 3/4 -> 0/4, B055 4/4 -> 0/4, C9 3/3 -> 0/2, ...), the rest are unchanged. On the seven attribution cases: old 94% (44/47), A-only (E1, E3, E4, E5 applied, E2 not) 91% (31/34), new 11% (5/44). The scientific content of the classes is the same ("multiphase biomolecular condensates" -> "multiphase biomolecular condensate"). The effect on retrieval was not measured: the 3A templates insert the string verbatim, and query-text changes can move a paper across the rank-20 cutoff in either direction.
2. **Fewer related_system_discovery axes in two cases.** ID1: 2 discovery axes in 4/4 old runs and 2/2 A-only runs, 1 in 4/4 new runs (the "bacterial autotransporter ... biogenesis/adhesin" axis is no longer a discovery axis; it becomes a direct axis or disappears). ID21: discovery axis present in 4/4 old and 2/2 A-only runs, in 1/4 new runs (the "multiphase biomolecular condensate" axis moves from distinctive_structure/discovery to environment_context/direct). Excluding ID1, ID21 (and M01) the discovery share is 25% vs 24%. Reading: the ID1 change is the removal of the autotransporter prior (the intended effect of E2); the ID21 change has no such explanation and is not understood. Whether either is better or worse for retrieval is unknown.

E1, E3, E4, E5 together (A-only arm) kept the old behaviour in the cases tested (plural share 91% vs 94%, discovery counts unchanged), so E2 is the only edit with a measurable behavioural effect.

## Known limitations
- Small sample (20 inputs; 2-4 runs per case per arm); no statistical test; the old arm is the pre-edit prompt run now, not the stored plans (those come from several earlier prompt versions: 2.5 axes per plan, 97.6% transfer_needed true, versus 1.96-2.07 axes and 87-89% transfer_needed true now).
- The A-only arm covers 7 of the 20 cases; E1/E3/E4/E5 were not tested one by one.
- The mechanism of the E2 effect is not established (the old example is singular too).
- No retrieval was run: the retrieval effect of the singular names and of the discovery-axis changes is inferred, not measured.
- B250 not tested (P1 not READY); B122's P1 plan is a single sample.

## Recorded, not changed
- Abstention wording ("false with empty axes is common ... prefer abstaining") vs behaviour (transfer_needed true in 87-89% of the current runs and 97.6% of stored plans): changing it would change the number of transfer queries; kept.
- `confidence` is never `low` (the prompt forbids low placeholders), so the downstream "skip low" gate never fires; kept. `rationale` has no reader but is generated before `confidence` and `execution_mode`; kept.
- Input description of TARGET COMPONENT / STATED CONTEXT in the prompt waits for OPEN-1 (`docs/open_issues.md`).
- OPEN-2 (to record in `docs/open_issues.md` when decided): the 3A templates ignore `retrieval_anchors` for `direct_concept_search` axes (233 of 312 stored axes); feeding them to 3A is a retrieval change and needs its own A/B.
