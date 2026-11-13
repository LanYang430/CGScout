# Prompt audit: P3 (`TERMINOLOGY_EXPANDER_SYSTEM_PROMPT`)

Status: **CLOSED (2026-11-13). Edit applied and committed (commit `40dd17c`): the tested "G3" text.** Regression gate passed (below). **Not every pre-set evaluation criterion passed:** the count-based flag (b) was tripped, and one systematic over-expansion (sensor histidine kinases) is open; the owner accepted this trade-off. Next prompt: P4 (not started).

Owner's rules for this prompt: keep the three teaching roles of the original (alternative names of the same structure / class names of proteins built on it / forbidden specific instances); keep concrete, specific examples (no replacement by abstract rules); replace the case-specific examples (beta-helix / autotransporter / pertactin / EspP) that come from a development case; choose new examples by the input types P3 really receives and by scientific accuracy, **never by seed hit rates**; do not widen the task; E1 keeps the protein-focused wording.

## Where it runs and what its output really does
- `retrieval/related_systems/terminology.py`; called once per `related_system_discovery` axis from `orchestrator._run_related_system_discovery`; input = the axis' P2 `scientific_class`; no response cache in the library (the benchmark runner memoises per class in `terminology_memo.json`, keyed by the **class string only, not by the prompt**: after this edit use a new, empty run directory, as `docs/reproduce.md` already requires; memos in older directories hold old-prompt outputs).
- P3 terms do **not** enter the Hop-1 search (one P4 sentence is used for every term, `semantic_query_override`); they reach the P5 per-paper message ("also written in the literature as", mixed with the class+anchor and class+objective strings), the banned-name / class-variant drop in `discovery.py`, `structural_term_matched` and the quote-ties-to-class test.

## What was changed (`40dd17c`; only the prompt string; 1432 -> 1523 chars, sha256 prefix `7d17860494` -> `a27f4651ad`)
| spot | before | after |
|---|---|---|
| input source (sentence 1) | `... description a researcher used for the protein system they study.` | `... description for a class of protein systems.` |
| alternative names | `(for "beta-helix": "beta-solenoid", "parallel beta-helix", "beta-helical fold", "single-stranded left-handed beta-helix", "pectin lyase-like fold")` | `(for "alpha-helix": "α-helix", "alpha-helical segment"; for "beta-sheet": "β-sheet", "beta-pleated sheet")` |
| class names | `built on this architecture (for "beta-helix": "autotransporter passenger domain", "beta-helix adhesin")` | `built on this architecture, only when that relationship is well established in the scientific literature (for "globin fold": "globin family", "truncated hemoglobin family"; for "jelly-roll fold": "legume lectin family", "galectin family")` |
| NEVER instances | `(NOT "pertactin", "P22 tailspike", "pectate lyase C", "EspP", "hemolysin")` | `(NOT "myoglobin", "hemoglobin", "green fluorescent protein", "streptavidin", "concanavalin A")` |
| NEVER: wording | `- the researcher's exact original wording repeated back` | `- the given description's exact wording repeated back` |

The NEVER rules themselves, the 0-6 term limit, the abstention sentence, the JSON instruction and the schema are unchanged. The evaluated prompt equals the tested G3 text byte for byte (`prompt_audit/P3/tools/p3_edit.py` asserts it).

Gate (`prompt_audit/P3/tools/p3_gate.sh g3 HEAD`): 438 tests passed, 2 skipped; 6-case offline golden replay byte-identical, 0 LLM attempts (P3 is replayed from the memo, so the replay cannot see the prompt); manifest differs only in `prompts/P3/chars` and `prompts/P3/sha256`; AST of `terminology.py` identical outside the prompt string; one file changed. No other prompt, schema, cache key or downstream code was touched.

## Evidence (all P3-only; no retrieval, no seed / Hit@k data; about 5.4 USD in total)
**Input census (86 distinct stored P3 inputs = classes of discovery axes):** membrane protein / membrane-associated system 27 (31%), complex / filament / capsid assembly 18 (21%), disordered / peptide 8, protein family 8, other 8, fold / architecture 7 (8%; only 4 protein folds, the others are RNA / nucleic-acid motifs; 14 = 16% on a loose reading), nucleic acid / chromatin 6, condensate 4. The original single beta-helix example therefore covers a minority type.

**Overlap checks (working 200 = `Run1 copy.xlsx`):** the original examples overlap the development query I11 and B034 (EspP passenger domain folding, the I11 sister query); TIM barrel was rejected as a new example (B235 triosephosphate isomerase is in the 200); pLGIC / Cys-loop (B070, B072, B073), aquaporin (B157) and ABC transporter (B161) were excluded; all final example terms have 0 hits in the question, seed-title and system text of the 200, in the 86 inputs and in the stored P3 outputs.

**Round 1 (31 inputs x 3):** the model does not need the beta-helix example: beta-helix -> beta-solenoid / parallel beta-helix / beta-helical fold in 3/3 runs in every arm, including the arm with no examples and the arm with alpha-helix / beta-sheet only.

**Round 2 (86 inputs + 5 handedness probes, 5 repeats, arms O / N / G / A / B, blind pooled two-pass LLM judge, protocol fixed first):**
| arm | O (original) | N (no examples) | G (alpha-helix / beta-sheet) | A (MFS) | B (cullin-RING) |
|---|---|---|---|---|---|
| problem terms %, judge 1 / 2 | 8.5 / 8.5 | 7.2 / 8.5 | 5.9 / 5.1 | 7.7 / 7.1 | 6.3 / 5.7 |
| terms containing a protein / complex proper name % | 6.6 / 6.0 | 5.7 / 5.2 | 3.5 / 3.1 | 7.7 / 7.6 | 5.7 / 5.5 |
Paired bootstrap over inputs vs O (problem rate, pp): G -2.6 / -3.4 (CI excludes 0 in both passes), B -2.2 / -2.8, A -0.8 / -1.5, N -1.3 / 0.0. **Handedness:** the original example "single-stranded left-handed beta-helix" made the original prompt write "left-handed" for right-handed inputs (beta-helical adhesins 3 terms, autotransporter passenger domains 1 term, all judged wrong) and in 4/5 runs for the generic "beta-helix"; the other arms did not (0 errors). P5 downstream check on 7 cached Hop-1 paper sets (only 5 produced systems): arm differences lay inside P3 / P5 run-to-run noise (same-terms overlap 0.62, another sample of the original prompt 0.38, arms vs O 0.21-0.43).

**Round 3 (G vs G re-run vs G2 = G + new E3 "globin fold / jelly-roll fold", 101 inputs incl. 10 architecture probes):** G2 raised the problem rate by +2.1 / +2.9 pp vs the G re-run (CI excludes 0; G vs its own re-run: +0.4 / +1.0, CI includes 0) and the family-name terms per run from 0.079 to 0.181 (x2.3).

**Round 4 (G3 = G2 + the condition "only when that relationship is well established in the scientific literature"):**
| | G | G re-run | G2 | **G3 (applied)** |
|---|---|---|---|---|
| problem terms %, judge 1 / 2 | 5.6 / 4.9 | 5.2 / 3.8 | 7.3 / 6.8 | **5.6 / 4.5** |
| related-but-different %, judge 1 / 2 | 5.2 / 4.7 | 4.7 / 3.6 | 6.7 / 6.4 | 5.1 / 4.3 |
| class-name share %, judge 1 / 2 | 54.7 / 54.5 | 55.3 / 55.7 | 58.1 / 57.7 | 55.4 / 56.0 |
| family-name terms per run (share judged problem) | 0.072 (3.2%) | 0.079 (2.9%) | 0.181 (3.8%) | **0.151 (1.5%)** |
| terms per run / zero-return runs (of 430) | 4.58 / 0 | 4.63 / 0 | 4.61 / 0 | 4.39 / 4 |
| terms with a proper name % (judge 1 / 2) | 3.3 / 2.8 | 4.5 / 3.9 | 6.2 / 6.1 | 6.0 / 5.1 |
G3 vs the G re-run: problem +0.4 [-0.9, +1.7] / +0.6 [-1.1, +2.6] pp, related-but-different +0.4 / +0.8 (CIs include 0, same order as G's own noise); G3 vs G2: -1.7 [-3.2, -0.5] / -2.3 [-3.9, -0.9]. Example words (globin, lectin, galectin, jelly-roll) leaked into 0 of 430 real-input runs; no left-handed term for any right-handed input; architecture probes: G3 6.6% problem terms (G re-run 6.2%, G2 8.2%).

**Pre-set flags for G3 vs the G re-run (`prompt_audit/P3/live4/PROTOCOL4.md`):** (a) problem / related-but-different > +1.5 pp with CI excluding 0: not tripped; **(b) family-name terms per run > 1.5x: TRIPPED (1.91x)**; (c) example-word leakage: not tripped; (d) architecture probes > +5 pp: not tripped; (e) left-handed for right-handed input: not tripped; (f) under-output: not tripped. **So not all pre-set criteria passed.** Reading accepted by the owner: the extra family-name terms are mostly real families (Hsp90, TRP channel, MraY/WecA, Snf7, Piezo, P-type ATPase) and are judged problematic less often than before (1.5%), and the direct accuracy measures are within noise of G; the count flag was nevertheless a pre-set criterion and is not waived here.

**Judge reliability:** judge = gpt-5.6-luna (same model family as the generator), two passes (agreement on accept-vs-not 91-94%); my blind labels on 79 stratified pairs agreed with judge 1 on accept-vs-not for only 63%: the judge is stricter than a human reader, so absolute problem rates are overstated; only between-arm differences (same blind judge, pooled terms) are used.

## Known limitations and watch items
- **W-2 (systematic over-expansion, open):** for the input "dimeric multi-pass membrane phosphotransferases" (MraY/WecA-like polyprenyl-phosphate phosphotransferases) the new E3 makes P3 list a different family, "sensor histidine kinases / two-component sensor kinases": 3 of 5 runs with such a term in G2 (11 terms) and in G3 (11 terms), versus 0/5 for the original prompt, N, A, B and the G re-run, and 1/5 in the first G run. The judge labels these related-but-different (sensor kinases include soluble members and are not membrane phosphotransferases of this class). The condition added in G3 did not remove it. Accepted by the owner; to watch in formal runs for inputs of this kind, because these terms reach P5 as "also known as" aliases and as banned names.
- Terms containing a protein / complex proper name (mostly family-level, e.g. Snf7-family, cohesin) are more frequent in G3 (6.0% / 5.1%) than in the G re-run (4.5% / 3.9%) and the first G (3.3% / 2.8%); the rise is in complexes (0.9-2.1% -> 6.3%) and disordered proteins (4.8-4.9% -> 8.0%); clear specific-instance violations stayed near 0.
- Process words and size-prefixed names occur rarely in every arm (e.g. "90-kDa heat shock protein", "30-nm chromatin fiber"); not discriminating between arms.
- Small samples (86 real inputs, 5 repeats per arm; per-type groups of 4-27 inputs); the P5 check used 7 cases and 15 cached papers each; no Hop-2 / Stage 3 / Stage 4; no statistical correction for multiple comparisons.
- 28 of the 86 inputs come from questions that are in the working 200 and 7 are development-lineage inputs (5 inputs are both; annotated in `prompt_audit/P3/live2/inputs.json`; analysis subsets: dev lineage 7, other working-200-linked 23, neither 56); the arms were not chosen by any seed result.

## Recorded, not changed (found while reading P3; none touched here)
- The stated purpose of P3 (widen the Hop-1 search) is not realised in the query path (known issue in `docs/architecture.md`); its terms act through P5's message, the banned-name list and the grounding checks.
- `filter_instance_terms` (the "second line of defence" named in a code comment) and `MAX_EXPANSION_TERMS = 6` are not used by the production path; the 0-6 limit, the instance rule and the size rule rest on the model; in the 89 stored outputs examined there were no process words and no echo, 1 size-prefixed term.
- The schema field `original_term` is required but never read, and the prompt does not say how to fill it (the model often paraphrases the input; one stored case was empty).
- "Return fewer, even zero" is almost never used (2-6 terms in all 89 stored outputs; 5 zero-return runs of 430 in the original prompt).
- The module comment above the prompt (`terminology.py`, "TerminologyExpander -- READY-side retrieval expansion ...") still cites the beta-helix / autotransporter example, and `_STRUCTURAL_MORPHEMES` contains development-case words (autotransporter, passenger, adhesin, lectin); both are comments / code, outside this prompt edit.
- The P4 sentence overrides every P3 term in Hop-1 queries (architecture.md); P3 and P4 are audited separately; merge 5 (P4 -> P3) was not done.

## Data and tools
All raw outputs, protocols, judge labels, scripts and the gate are under `archived-audit-data/P3/` (census: `P3_input_types.csv`; round 1: `live/`; round 2: `live2/` incl. `PROTOCOL2.md`, `RESULTS2_*.txt`; round 3: `live3/`; round 4: `live4/`; the applied text: `candidates/P3_G3_candidate.txt`; edit and gate: `tools/`).
