# beam_agent

The AI-assisted side of BEAM's pre-simulation workflow: given a scientific question
about a coarse-grained (CG) modeling problem, find and vet the literature that bears
on it. Originally scoped as "literature retrieval only" (and named `beam_literature`
accordingly); renamed to `beam_agent` because the intended scope is broader --
LLM-based evidence extraction, scientific decision support (the BEAM evidence schema
+ deterministic comparison layer), and eventually integration with BEAM's existing
post-simulation analysis tools. **No LLM calls exist in this codebase yet** -- what's
built so far is the fully rule-based front half of the pipeline.

## Pipeline

```
user query (natural-language scientific question)
  -> QueryExpander            rule-based, several complementary phrasings   [done]
  -> search each query x each source (Semantic Scholar + OpenAlex)          [done]
  -> dedupe_papers()          merge duplicates across queries+sources       [done]
  -> RelevanceFilter          drop obvious-irrelevant candidates            [done]
  -> retained papers, ranked by relevance score                             [done]
  -> Crossref DOI/metadata validation                                       [later]
  -> full-text retrieval (OA PDF / Europe PMC / preprint, fallback chain)   [later]
  -> LLM evidence extraction                                                [later]
  -> BEAM evidence schema + deterministic comparison layer                  [later]
```

`QueryExpander` and `RelevanceFilter` are both `Protocol`s specifically so an
LLM-backed implementation of either can be dropped in later without touching the
orchestration (`pipeline.py`) or the CLI that calls it.

No API keys are required. Both Semantic Scholar's Graph API and OpenAlex's Works API
are free and open without registration; an optional Semantic Scholar API key
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

## Use

**Full pipeline** (query expansion + search + dedupe + relevance filter -- the
recommended entry point):

```bash
python -m beam_agent.pipeline_cli \
    "200-aa beta-helix protein folding coarse-grained model" \
    --limit-per-query 15 \
    --show 15 \
    --out results.json
```

This prints the generated queries, raw/deduped/filtered counts, and the top retained
papers, then (with `--out`) writes the full result -- including rejected papers, for
inspection -- to JSON.

**Raw two-source search only** (no expansion, no filtering -- useful for comparing
against the full pipeline, or as a minimal building block):

```bash
python -m beam_agent.cli "200-aa beta-helix protein folding coarse-grained model" \
    --limit 20 \
    --out results.json
```

Or from Python:

```python
from beam_agent.pipeline import run_pipeline

result = run_pipeline("beta-helix protein folding coarse-grained model", limit_per_query=15)
print(result.expanded_queries)
for p in result.retained_papers:
    print(p.relevance_score, p.year, p.title, p.doi, p.sources, p.matched_keywords)
```

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
- `relevance_score`, `matched_keywords` — set by `RelevanceFilter`; `None`/`[]` if the
  paper was never scored (e.g. output of `search_and_dedupe()` directly, bypassing the
  pipeline)

## Module map

- `text_utils.py` — shared tokenize/stopword/stem helpers (no NLP libraries; see its
  docstrings for why `stem()` is a crude fixed-length prefix match, not a real
  stemmer, and why that's an intentional recall-over-precision tradeoff)
- `query_expansion.py` — `QueryExpander` protocol + `RuleBasedQueryExpander`
- `sources/semantic_scholar.py`, `sources/openalex.py` — per-source API clients
- `dedupe.py` — cross-source/cross-query dedup (DOI first, title+year fallback)
- `relevance_filter.py` — `RelevanceFilter` protocol + `RuleBasedRelevanceFilter`
- `pipeline.py` — orchestrates all of the above; `run_pipeline()` is the main entry
  point for programmatic use
- `search.py`, `cli.py` — the original Phase 1 raw two-source search (still useful
  standalone; `pipeline.py` does not depend on it)
- `pipeline_cli.py` — CLI for the full pipeline

## What's intentionally *not* here yet

- No Crossref metadata validation
- No Europe PMC / full-text retrieval
- No LLM calls of any kind -- query expansion and relevance filtering are both purely
  rule-based (regex + a small stopword/synonym table + prefix stemming)
- No BEAM evidence schema (protein/system type, folding evidence, structural fidelity,
  dynamics, sampling efficiency, cost, validation, limitations) or comparison layer
- No integration with BEAM's post-simulation analysis code

Those are later phases per the BEAM plan; keeping this phase LLM-free and
dependency-light makes it easy to unit test and to swap any single stage (a source, the
expander, the filter) without touching anything else.
