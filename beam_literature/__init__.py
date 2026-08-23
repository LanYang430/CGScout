"""beam_literature: BEAM's pre-simulation literature + evidence layer.

Currently implements the rule-based front half of the pipeline -- query
expansion, multi-source search, dedupe, relevance filtering -- with no LLM
calls anywhere in this package yet. See README.md for the full pipeline this
is meant to grow into (Crossref validation, full-text retrieval, LLM evidence
extraction, the BEAM evidence schema + comparison layer).
"""

from beam_literature.models import Paper
from beam_literature.pipeline import PipelineResult, run_pipeline
from beam_literature.query_expansion import QueryExpander, RuleBasedQueryExpander
from beam_literature.relevance_filter import RelevanceFilter, RuleBasedRelevanceFilter
from beam_literature.search import search_and_dedupe

__all__ = [
    "Paper",
    "search_and_dedupe",
    "run_pipeline",
    "PipelineResult",
    "QueryExpander",
    "RuleBasedQueryExpander",
    "RelevanceFilter",
    "RuleBasedRelevanceFilter",
]
