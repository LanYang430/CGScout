# beam_literature

Phase 1 of BEAM's pre-simulation literature module: **search only, no LLM yet.**

Given a query (e.g. `"200-aa beta-helix protein folding coarse-grained model"`), this
searches Semantic Scholar and OpenAlex in parallel, normalizes both result sets into a
common `Paper` schema, deduplicates across sources (by DOI first, then by normalized
title+year), and writes a single structured JSON file.

This is deliberately the *narrowest* useful slice of the pipeline described in the BEAM
Fall 2026 plan:

```
User query
  -> Search Agent (Semantic Scholar + OpenAlex)
  -> dedupe + normalize
  -> structured JSON   <-- this repo stops here for now
  -> [later] Crossref validation
  -> [later] full-text retrieval
  -> [later] LLM evidence extraction
  -> [later] BEAM evidence schema + deterministic comparison
```

No API keys are required. Both Semantic Scholar's Graph API and OpenAlex's Works API
are free and open without registration; an optional Semantic Scholar API key
(`S2_API_KEY` env var) raises your rate limit but is not required to run this.

## Install

```bash
cd beam_literature
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Use

```bash
python -m beam_literature.cli "200-aa beta-helix protein folding coarse-grained model" \
    --limit 20 \
    --out results.json
```

Or from Python:

```python
from beam_literature.search import search_and_dedupe

papers = search_and_dedupe("beta-helix protein folding coarse-grained model", limit=20)
for p in papers:
    print(p.year, p.title, p.doi, p.sources)
```

## Output schema

Each entry in the output JSON is a `Paper` (see `beam_literature/models.py`):

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

## What's intentionally *not* here yet

- No Crossref metadata validation
- No Europe PMC / full-text retrieval
- No LLM calls of any kind
- No ranking beyond each source's own relevance order (ties broken by "found in both
  sources first")

Those are later phases per the BEAM plan; keeping this phase LLM-free and dependency-light
makes it easy to unit test and to swap search sources without touching anything downstream.
