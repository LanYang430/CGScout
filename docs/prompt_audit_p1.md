# Prompt audit: P1 (`PLANNER_SYSTEM_PROMPT`)

Status: **closed for now (accepted by the owner, 2026-11-11)**. Next prompt: P2.

## What was changed (branch `cleanup/phase2-cgscout-package`)

| edit | content | state |
|---|---|---|
| E1a/E1b/E1c | no project name in the prompt; conversation role "assistant"; the history prefix sent to P1 is `Assistant (asked):` (was `BEAM (asked):`, `session.py`) | kept |
| E2a/E2b | clarification rules reconciled: a request whose system / component / sub-problem cannot be determined is a scope ambiguity (record it in `scientific_ambiguity` or `target_component`, `clarification_needed` true, focused question); no asking for non-essential detail; the Hard rule does not forbid clarification when essential information is missing | kept |
| E3 | `search_context`: only context stated or directly entailed by the user's wording | **reverted** (`723a9d7`), original paragraph restored byte-for-byte |
| E4a/E4b/E4c | `search_axes`: no promise that retrieval covers a noted framing; the field is descriptive (it has no consumer in the PAPER configuration) | kept |
| E5a | example "domain A vs domain B vs full-length protein" instead of an autotransporter-specific one | kept |
| E5b | the "multiple IDRs of the same protein" example | **not changed** (tied to the diagnosed eIF4B over-clarification) |

Commits: `3ed4782` (E1, E2, E3, E4, E5a), `723a9d7` (E3 reverted). P1 sha256 prefix: original `1226716342`, E1–E5 `5e102cff90`, current `6409f46ebf`. P2–P9 and all 54 pydantic schemas are byte-identical to before.

## Evidence: the old-vs-new comparison (P1 only; no retrieval, no Stage 3/4)

Design: 20 questions (10 hand-written probes, the eIF4B regression query from `tests/test_scientific_planner_clarification.py`, 9 queries from stored plans), first turn of the real `PlannerSession` path, model `gpt-5.6-luna`, effort medium. Original prompt once; E1–E5 version once; E3-reverted version once; then the 7 cases with extra search_context or changed clarification (S-ID21, S-ID14, S-eIF4B, S-B039, H-AMP, S-M04, H-TMhelix) three times per arm; S-P03 and S-M09 twice more for the original and the current version. About 140 calls, about $0.05 in total. Raw data: `archived-audit-data/P1/`.

Results that matter:
- **Over-clarification**: the 13 clear or over-clarification-prone questions (villin, WW, titin, HIV capsid, nucleosome, riboswitch, TM helix, antimicrobial peptide, eIF4B, N09, M04, M09, B205) were never asked about, by any arm; the live eIF4B regression test passes.
- **Real missing information**: "my protein" and "my complex" were asked about by all arms (one run each).
- **E3 (reverted)**: it made the model copy the system / objective into `search_context` (S-ID21 0 → 2–3 items, S-B039 0 → 3–4, repeat-stable), added generic context queries (7-case sample: 30 → 36 direct-route query strings) and made the field unstable (identical pairs of runs: 2/21 vs 14/21). The items first suspected as "inferred" (aqueous environment, GPU, long timescales) were in fact stated by the users, so there was no evidence for the edit.
- **After the revert** (current version): S-ID21, S-ID14 and H-AMP context equal the original; item total over the 20 questions 23 (original 26); direct-route query strings 87 (original 88); identical pairs 10/21.
- **Clarification with E2 kept**: S-ID21 (asked in 2 of 3 original runs) and S-ID14 (2 of 3) were not asked in 3 of 3 runs of the current version; S-B039, S-P01, S-P03 are still asked. The change cannot be attributed to E2 alone (E1, E4, E5a are also present). Scientific reading of the two cases: ID21 ("are nucleolar condensates homogeneous liquids or network-like?") is not a scope-changing ambiguity (stored pool of 87 papers: 70 generic condensate papers, 15 nucleolar, no split by whole nucleolus vs subcompartment); B039 (DNA vs RNA strand displacement) is a weak, borderline ambiguity (stored pool of 118: 21 strand-displacement titles, 15 DNA, 5 hybrid, 1 RNA; a generic nucleic-acid search already returns RNA and hybrid papers).

## Known limitations of this evidence
- Small sample: 20 questions, 7 repeated 3 times, 2 repeated twice; no statistical test; run pairs are not independent.
- **Residual echo not attributed**: with E3 reverted, the current version still copies system / objective into `search_context` more often than the original in three cases (S-eIF4B 3/3 runs vs 0/3, S-B039 2/3 vs 0/3, S-M04 second item 3/3 vs 1/3). The cause (E1, E2, E4 or E5a) was not established; a single-factor ablation was proposed and **deliberately not run**.
- `search_context` is noisy even in the original prompt: it switches between an empty list and three restated items from run to run (S-M09: 3, 3, 0 items; S-P03: 3, 3, 3 for the original vs 0, 0, 3 for the current version).
- `scientific_objective` and `search_axes` wording differ in nearly every run, also between runs of the same prompt (identical objective in 5/21 pairs of the original), so only the structured decisions (asked or not, item counts) were compared.
- The 7-case runs 2 and 3 of the original prompt were overwritten by mistake; their key results (asked flags, context lists) were recorded in the session only, the full plans are not preserved.
- No retrieval was run: the effect of extra or missing context queries on recall is inferred from query strings, not measured. The pools used for the E2 analysis come from older stored runs (B039: an earlier code version) and DNA / RNA / hybrid were classified by title keywords.
- Benchmark clarification is non-interactive: `freeze_plans.py` answers every question with a fixed sentence ("I have no further details beyond my question as written; please proceed with your best reasonable interpretation."), so asking cannot add information there.

## Appendix: results of the original prompt's lost runs (recorded from the session output)

Runs 2 and 3 of the original prompt on the seven repeated cases (run 1 is preserved in `old_run1.json`). `asked` = `clarification_needed`.

| case | run 2 | run 3 |
|---|---|---|
| S-ID21 | asked no; context `[]` | asked yes (`target_component` = "the whole nucleolar condensate versus a specific nucleolar p…"); context `[]` |
| S-ID14 | asked yes; context `[bacterial outer membrane, explicit lipopolysaccharides, native membrane proteins]` | asked no; same context |
| S-eIF4B | no; `[]` | no; `[]` |
| S-B039 | yes; `[]` | yes; `[]` |
| H-AMP | no; `[bacterial membranes]` | no; `[bacterial membranes]` |
| S-M04 | no; `[aqueous environment (water)]` | no; `[aqueous environment, monomeric peptide]` |
| H-TMhelix | no; `[membrane-embedded transmembrane helix, lipid bilayer]` | no; `[membrane-embedded environment, lipid bilayer]` |
