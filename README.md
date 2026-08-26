# beam_agent

The AI-assisted side of BEAM's pre-simulation workflow: given a scientific question
about a coarse-grained (CG) modeling problem, find and vet the literature that bears
on it. Originally scoped as "literature retrieval only" (and named `beam_literature`
accordingly); renamed to `beam_agent` because the intended scope is broader --
LLM-based evidence extraction, scientific decision support (deterministic
System/Process/Method evidence matching against a user's own criteria), and
eventually integration with BEAM's existing post-simulation analysis tools. Stages
0-3 and 4b/5 are rule-based, no LLM. Stage 4a makes exactly one LLM call per paper
(to a local model by default); every other stage is LLM-free.

## Pipeline

```
user query (natural-language scientific question)
  -> ConceptResolver.resolve()   Stage 0: structured concept profile          [done]
  -> QueryPlanner.plan()         Stage 1: system-specific + general-discovery
                                  search queries                              [done]
  -> search each query x each source (Semantic Scholar + OpenAlex)            [done]
  -> dedupe_papers()             merge duplicates across queries+sources      [done]
  -> RelevanceFilter.filter()    Stage 3: cheap keyword-overlap filter        [done]
  -> PaperEvidenceExtractor.extract_many()  Stage 4a: single-pass,
                                  query-independent structured evidence
                                  extraction from title+abstract (default
                                  backend: local model via Ollama; a no-LLM
                                  stub is also available). Cacheable per
                                  paper -- no ScientificConceptProfile is
                                  ever part of this call.                     [done]
  -> match_evidence() / rank_papers()  Stage 4b/5: deterministic comparison
                                  of the cached PaperEvidence against the
                                  query's profile, dimension by dimension,
                                  plus user-configurable required/preferred/
                                  optional filtering and ranking. No LLM.      [done]
  -> Crossref DOI/metadata validation                                        [later]
  -> full-text retrieval (OA PDF / Europe PMC / preprint, fallback chain)     [later]
  -> LLM evidence extraction, applied to full text                           [later]
```

Every stage is a `Protocol` (`ConceptResolver`, `QueryPlanner`, `RelevanceFilter`,
`PaperEvidenceExtractor`) with at least one implementation, specifically so a
different implementation of any single stage -- including a different LLM backend
for Stage 4a -- can be dropped in later without touching the orchestration
(`pipeline.py`) or the CLI that calls it. Stage 4b/5 (`evidence_matching.py`) is
plain functions, not a Protocol -- there is no LLM involved to swap out.

**Why Stage 0/1 are split from search:** query expansion used to happen directly on
raw query text (strip stopwords, drop size qualifiers, substitute a synonym). That
conflated "understanding the science" with "generating search strings." Now
`ConceptResolver` extracts a structured `ScientificConceptProfile` (structural terms,
biological families, representative systems, scientific tasks, methodology, size
constraints) first, and `QueryPlanner` consumes *that* to generate two categories of
query: `system_specific` (tied to the user's literal system) and `general_discovery`
(deliberately system-agnostic -- a useful CG model may never have been tested on the
user's exact system, so a query strategy that only ever searches for the exact system
would systematically miss it).

**Stage 4a -- single-pass evidence extraction.** The cheap Stage 3 filter is a
keyword-overlap heuristic biased toward recall -- good at catching "obviously
irrelevant," useless at real topical organization (it can't tell you what a paper's
model actually *does*). Stage 4a extracts objective, query-independent facts from
title + abstract + metadata (not full text yet) into three independent groups -- see
`evidence_extraction.py`'s module docstring for the full design rationale, and
`benchmarks/stage4_design_experiments/` for why this replaced an earlier design where
the LLM also judged eligibility/task-match and produced a decision:

1. **Method evidence** (`MethodEvidence`) -- what modeling/method class the paper's
   own text names (`model_class`: coarse-grained, all-atom, structure-based,
   elastic-network, Go model, ... open vocabulary) and what role it played in *this*
   paper (`study_role`: developed, applied, benchmarked, validated, reviewed_only).
   Purely descriptive -- there is no `eligible`/`not_eligible` judgment anywhere in
   this schema.
2. **System evidence** (`SystemEvidence`) -- what physical system the paper studied,
   simulated, benchmarked, developed against, or applied its method to:
   `studied_systems` (open text, one entry per distinct system/system-group, kept
   close to the abstract's own words -- identity + structure/architecture +
   qualitative scale together as one faithful phrase, e.g. "three-helix bundle",
   "twelve proteins with multiple secondary-structure arrangements"), `family_or_fold`
   (only when the abstract explicitly names a fold/family classification), and
   `protein_or_system_names` (bare names). Deliberately NOT split into a
   `system_type`/`structural_class`/`approx_size_residues`/`environment` schema --
   an earlier version was, and real extraction runs turned up concrete
   misassignments (a "three-helix bundle" topology count read as `approx_size_residues:
   3`; "approximately 9 ms" of simulation time read as `approx_size_residues: 9`; a
   genuine beta-helix compressed into the coarse bucket "globular") that the merged,
   open-text field removes the *slots* for rather than just prompting more carefully
   against. Every field stays empty/null if the abstract doesn't say -- never
   inferred, never copied from anywhere else (there is nothing to copy from: this
   call never sees a search target, and general biological background knowledge
   about a named system is not a substitute for what the abstract itself states).
3. **Process evidence** (`processes`, a list of `ProcessEvidenceItem`) -- one
   `{raw_text, normalized_label}` pair per process/capability the paper actually
   demonstrated (folding, unfolding, conformational dynamics, pocket gating, binding,
   aggregation, assembly, translocation, kinetics, thermodynamics, stability, ...).
   `normalized_label` is open-vocabulary, deliberately not a fixed folding-specific
   enum, so a future query about e.g. binding-pocket dynamics needs no code change.

No `ScientificConceptProfile` is ever part of this call -- structurally, not just by
prompt instruction, so target-concept leakage into `studied_systems`/`family_or_fold`
is impossible. That also makes the result safe to cache and reuse across different
queries against the same paper (no cache implementation yet; `Paper.evidence` is
where a cache would read/write). `BaselineEvidenceExtractor` (no LLM calls, every
field empty) is the CLI default -- pass `--extractor ollama` to use the real one; a
connection failure, timeout, or schema-violating response degrades to an empty
`PaperEvidence` with an explanatory note rather than crashing the run. Papers with no
abstract skip the LLM call entirely.

**Stage 4b/5 -- deterministic matching, filtering, ranking.** `evidence_matching.py`
compares a paper's `PaperEvidence` against the query's `ScientificConceptProfile`,
dimension by dimension (`task_process`, `method_class`, `studied_systems`, `family_fold`
-- see the module docstring's "known gap" note on what a numeric size or
categorical system-type/environment comparison would need that this merged,
open-text schema doesn't provide by itself), using the same stem-based overlap machinery Stage 3 uses. Each
dimension comes back `matched`/`partial`/`no_evidence`/`mismatch`; results are never
collapsed into a single relevant/not_relevant verdict here. A `FilterCriteria` object
(fully user-configurable, e.g. "folding required, topology optional" or "folding AND
beta-helix both required") marks each dimension `required`/`preferred`/`optional`/
`irrelevant`; `rank_papers()` sorts by (passes every `required` dimension, fraction of
`preferred` dimensions matched) without ever silently dropping a paper -- a future
agent can translate natural-language preferences into `FilterCriteria`, but BEAM
itself never hands back an unexplainable final CG-model recommendation. Missing
evidence (`no_evidence`) never fails a `required` gate; only an explicit `mismatch`
does.

No API keys are required for search. Both Semantic Scholar's Graph API and OpenAlex's
Works API are free and open without registration; an optional Semantic Scholar API key
(`S2_API_KEY` env var) raises your rate limit but is not required to run this. Note:
the unauthenticated Semantic Scholar tier is prone to transient 429 rate limits under
shared-network conditions -- the code retries with backoff and degrades gracefully
(OpenAlex results still come back, with a warning) rather than failing the whole run.

## Install

```bash
cd beam_agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For real Stage 4a extraction (`--extractor ollama`), also install
[Ollama](https://ollama.com) and pull a model:

```bash
ollama pull qwen3:8b   # ~5GB, the default; comfortable on 16GB unified memory
```

## Use

**Full pipeline** (resolve + plan + search + dedupe + filter + extract + rank -- the
recommended entry point):

```bash
python -m beam_agent.pipeline_cli \
    "200-aa beta-helix protein folding coarse-grained model" \
    --limit-per-query 15 \
    --show 15 \
    --extractor ollama \
    --snapshot \
    --out results.json
```

This prints the resolved concept profile, the planned queries (by category),
raw/deduped/filtered/extracted counts, and the top-ranked retained papers (per
dimension match, under the default `FilterCriteria`); `--out` writes the full result
to JSON; `--snapshot` additionally writes a full reproducibility snapshot outside the
repo (see "Reproducibility snapshots" below). Omit `--extractor ollama` (or pass
`--extractor baseline`, the default) to skip LLM calls entirely -- every
abstract-bearing paper then comes back with an empty `PaperEvidence`.

**Raw two-source search only** (no concept resolution, no query planning, no
filtering -- useful for comparing against the full pipeline, or as a minimal building
block):

```bash
python -m beam_agent.cli "200-aa beta-helix protein folding coarse-grained model" \
    --limit 20 \
    --out results.json
```

Or from Python:

```python
from beam_agent.evidence_extraction import OllamaPaperEvidenceExtractor
from beam_agent.evidence_matching import FilterCriteria, DimensionRequirement, rank_papers
from beam_agent.pipeline import run_pipeline

result = run_pipeline(
    "beta-helix protein folding coarse-grained model",
    limit_per_query=15,
    evidence_extractor=OllamaPaperEvidenceExtractor(model="qwen3:8b"),  # omit for the baseline stub
)
print(result.concept_profile.structural_terms)
print(result.query_plan.queries)  # PlannedQuery: query, category, rationale
for p in result.retained_papers:
    print(p.relevance_score, p.year, p.title, p.doi, p.sources, p.matched_keywords)
for p in result.evidence_papers:
    print(p.evidence.method.model_class, [proc.normalized_label for proc in p.evidence.processes], p.title)

# Stage 4b/5: deterministic matching + ranking, rerunnable with different criteria
# without re-running retrieval or the LLM call.
criteria = FilterCriteria(task_process=DimensionRequirement.REQUIRED)
for paper, match in rank_papers(result.concept_profile, result.evidence_papers, criteria):
    print(match.passes_required, match.score, paper.title)
```

## Reproducibility snapshots

Opt-in only (`--snapshot`, or call `save_run_snapshot(result)` yourself); `run_pipeline()`
itself performs no file I/O. Snapshots are written *outside* the git repo -- a sibling
directory, `../beam_agent_runs/`, so they can never be accidentally committed or pushed
-- under a timestamped run id like `2026-08-23_001`. Each run directory contains:

- `run_metadata.json` -- query, timestamp, sources, limits, pipeline version, every
  component's config, counts at each stage
- `concept_profile.json` -- the resolved `ScientificConceptProfile`
- `queries.json` -- the full `QueryPlan` (categories + rationale)
- `raw_results.json` -- every candidate before dedup (duplicates included)
- `deduplicated.json` -- candidates after cross-source/cross-query dedup
- `retained.json` / `rejected.json` -- Stage 3 cheap-filter output
- `paper_evidence.json` -- Stage 4a output: retained papers annotated with their
  extracted `PaperEvidence`, plus coarse extraction counts (not a relevant/
  not_relevant bucket -- this stage produces no such verdict)
- `report.md` -- human-readable summary of all of the above

Override the output location with `--snapshot-dir`, the `BEAM_AGENT_RUNS_DIR`
environment variable, or the `runs_dir` argument to `save_run_snapshot()`.

## Output schema

Each entry is a `Paper` (see `beam_agent/models.py`):

- `title`, `authors`, `year`, `doi`
- `abstract` (may be `None` if not returned by either source)
- `venue`
- `sources` — which of `["semantic_scholar", "openalex"]` returned this paper
  (a paper found by both is one entry, not two)
- `external_ids` — raw per-source IDs (`semantic_scholar_id`, `openalex_id`) kept for
  provenance / later Crossref lookups
- `is_open_access`, `oa_pdf_url` — best-effort OA info, `None` if unknown
- `url` — a landing page link (source's own page, not necessarily OA)
- `retrieved_at` — UTC timestamp of the search, for reproducibility/snapshotting later
- `relevance_score`, `matched_keywords` — set by Stage 3's `RelevanceFilter`
- `evidence` — the full `PaperEvidence` set by Stage 4a's `PaperEvidenceExtractor`:
  `method` (`MethodEvidence`: `model_class`, `study_role`), `system`
  (`SystemEvidence`: `studied_systems`, `family_or_fold`, `protein_or_system_names`),
  `processes` (list of `ProcessEvidenceItem`: `raw_text` + `normalized_label`),
  `missing_information`, `reasoning_summary`. `None` means never extracted; distinct
  from an extracted-but-empty `PaperEvidence`. No decision/confidence/relevance field
  anywhere in this schema -- see `evidence_matching.py` for the deterministic,
  query-specific comparison layer that consumes it.

## Module map

- `text_utils.py` — shared tokenize/stopword/stem helpers (no NLP libraries; see its
  docstrings for why `stem()` is a crude fixed-length prefix match, not a real
  stemmer, and why that's an intentional recall-over-precision tradeoff)
- `concept_resolver.py` — Stage 0: `ConceptResolver` protocol + `BaselineConceptResolver`,
  the `ScientificConceptProfile`/`ConceptTerm`/`SizeConstraint` schema
- `query_planner.py` — Stage 1: `QueryPlanner` protocol + `BaselineQueryPlanner`,
  the `QueryPlan`/`PlannedQuery` schema
- `sources/semantic_scholar.py`, `sources/openalex.py` — per-source API clients
- `dedupe.py` — cross-source/cross-query dedup (DOI first, title+year fallback)
- `relevance_filter.py` — Stage 3: `RelevanceFilter` protocol + `RuleBasedRelevanceFilter`
- `evidence_extraction.py` — Stage 4a: `PaperEvidenceExtractor` protocol,
  `BaselineEvidenceExtractor` (no LLM), `OllamaPaperEvidenceExtractor` (local model
  via Ollama, structured JSON-schema output, single call per paper, no profile). The
  `PaperEvidence`/`MethodEvidence`/`SystemEvidence`/`ProcessEvidenceItem` schema
  lives in `models.py` (re-exported here) since `Paper` needs to reference it
  directly.
- `evidence_matching.py` — Stage 4b/5: `FilterCriteria`/`DimensionRequirement`
  (user-configurable per-dimension weighting), `match_evidence()` (pure function:
  profile + evidence + criteria -> per-dimension `DimensionMatchResult` +
  pass/fail + score), `rank_papers()` (sorts a paper list by that score). No LLM.
- `pipeline.py` — orchestrates Stage 0 through 4a; `run_pipeline()` is the main entry
  point for programmatic use, and performs no file I/O. Stage 4b/5 is intentionally
  separate -- call `evidence_matching.rank_papers()` on the result afterward, as many
  times as needed with different `FilterCriteria`, without re-running retrieval or
  the LLM call.
- `snapshot.py` — optional reproducibility snapshots (see above); never called
  automatically
- `search.py`, `cli.py` — the original Phase 1 raw two-source search (still useful
  standalone; `pipeline.py` does not depend on it)
- `pipeline_cli.py` — CLI for the full pipeline (`--extractor baseline|ollama`,
  `--extractor-model`, `--extractor-host`)

## What's intentionally *not* here yet

- No Crossref metadata validation
- No Europe PMC / full-text retrieval
- Stage 4a extraction is title/abstract/metadata only -- no full-text reading yet;
  `PaperEvidence.missing_information` is exactly the list of gaps a full-text pass
  should target
- No persistent per-paper cache yet for `PaperEvidence` -- the schema and extractor
  are designed to be cacheable (query-independent, keyed by paper identity), but
  nothing currently reads/writes such a cache; every `run_pipeline()` call
  re-extracts every retained paper
- No API-based (Claude/OpenAI/Gemini) Stage 4a backend yet -- `OllamaPaperEvidenceExtractor`
  is local-only for now; adding a BYOK backend is a new class behind the same
  `PaperEvidenceExtractor` Protocol, not an architecture change
- No batching of multiple papers into one LLM call, and no Batches API usage -- Stage 4a
  makes one call per paper, prioritizing simplicity/debuggability over cost for this
  first implementation
- No dedicated numeric size or categorical system-type/environment matching --
  `SystemEvidence` v3 merged those into one open-text `studied_systems` field (see
  its schema docstring), so `evidence_matching.py` can only do stem-overlap topical
  matching against it, not a real size comparison or a system-type/environment
  filter. Recovering either would need a further pass over `studied_systems` text
  (or a dedicated, carefully-grounded field), not attempted yet.
- No natural-language-to-`FilterCriteria` translation yet -- a user (or an agent
  acting for them) constructs `FilterCriteria` directly for now
- `ConceptResolver`'s baseline never populates `related_families` or
  `representative_systems` from raw text alone -- it uses a small, explicitly-tagged
  (`source="baseline-curated"`) hand-curated association table keyed off matched
  structural terms, not a general protein database (see `concept_resolver.py`)
- No integration with BEAM's post-simulation analysis code

Those are later phases per the BEAM plan; keeping each stage swappable behind its
Protocol makes it easy to unit test and to change any single stage's implementation
(a source, the expander, a Stage 4 backend) without touching anything else.
