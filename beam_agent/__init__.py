"""beam_agent: CGScout's (BEAM's) pre-simulation literature + evidence layer.

Shared building blocks of the retrieval -> Stage 3 -> Stage 4 path: models,
cache, multi-source search (OpenAlex, Europe PMC, optional Semantic Scholar),
dedupe and text utilities. The earlier v1 pipeline modules in this package
(concept resolution, query planning, keyword matching, snapshots) are historical
code, kept in the git tag `legacy-pre-cgscout`.
"""

# Kept in sync with pyproject.toml's [project].version by hand -- this package
# isn't pip-installed (no setup.py/importlib.metadata entry to read instead),
# so a plain constant is the simplest source of truth. Referenced by
# snapshot.py's run_metadata.json for reproducibility.
__version__ = "0.1.0"

# The v1 pipeline names (run_pipeline, ConceptResolver, QueryPlanner, ...) are NOT re-exported here any more: importing this package used to load
# nine v1 modules (pipeline, query_planner, concept_resolver, evidence_matching, relevance_filter, snapshot, ...) that the CGScout / BEAM
# path never uses.  Import what you need from its own submodule, e.g. `from beam_agent.pipeline import run_pipeline`.
__all__ = ["__version__"]
