"""beam_agent: BEAM's pre-simulation literature + evidence layer.

Implements the full pipeline through single-pass evidence extraction plus
deterministic matching/ranking: concept resolution, query planning,
multi-source search, dedupe, cheap keyword-overlap filtering, structured
query-independent evidence extraction via a local model through Ollama
(default) or the honest no-LLM baseline stub, and deterministic
System/Process/Method matching against a user's profile and configurable
per-dimension criteria. See README.md for the full pipeline this is meant to
grow into (Crossref validation, full-text retrieval).
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
