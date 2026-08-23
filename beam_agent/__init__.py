"""beam_agent: BEAM's pre-simulation literature + evidence layer.

Currently implements the rule-based front half of the pipeline -- query
expansion, multi-source search, dedupe, relevance filtering -- with no LLM
calls anywhere in this package yet. See README.md for the full pipeline this
is meant to grow into (Crossref validation, full-text retrieval, LLM evidence
extraction, the BEAM evidence schema + comparison layer).
"""

# Kept in sync with pyproject.toml's [project].version by hand -- this package
# isn't pip-installed (no setup.py/importlib.metadata entry to read instead),
# so a plain constant is the simplest source of truth. Referenced by
# snapshot.py's run_metadata.json for reproducibility.
__version__ = "0.1.0"

from beam_agent.models import Paper
from beam_agent.pipeline import PipelineResult, run_pipeline
from beam_agent.query_expansion import QueryExpander, RuleBasedQueryExpander
from beam_agent.relevance_filter import RelevanceFilter, RuleBasedRelevanceFilter
from beam_agent.search import search_and_dedupe
from beam_agent.snapshot import DEFAULT_RUNS_DIR, save_run_snapshot

__all__ = [
    "__version__",
    "Paper",
    "search_and_dedupe",
    "run_pipeline",
    "PipelineResult",
    "QueryExpander",
    "RuleBasedQueryExpander",
    "RelevanceFilter",
    "RuleBasedRelevanceFilter",
    "save_run_snapshot",
    "DEFAULT_RUNS_DIR",
]
