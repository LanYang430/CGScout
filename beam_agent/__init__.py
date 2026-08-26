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

from beam_agent.concept_resolver import (
    BaselineConceptResolver,
    ConceptResolver,
    ConceptTerm,
    ScientificConceptProfile,
    SizeConstraint,
)
from beam_agent.evidence_extraction import (
    BaselineEvidenceExtractor,
    OllamaPaperEvidenceExtractor,
    PaperEvidenceExtractor,
)
from beam_agent.evidence_matching import (
    DimensionMatch,
    DimensionMatchResult,
    DimensionRequirement,
    EvidenceMatchResult,
    FilterCriteria,
    match_evidence,
    rank_papers,
)
from beam_agent.models import (
    ExternalIds,
    MethodEvidence,
    Paper,
    PaperEvidence,
    ProcessEvidenceItem,
    SystemEvidence,
)
from beam_agent.pipeline import PipelineResult, run_pipeline
from beam_agent.query_planner import BaselineQueryPlanner, PlannedQuery, QueryPlan, QueryPlanner
from beam_agent.relevance_filter import RelevanceFilter, RuleBasedRelevanceFilter
from beam_agent.search import search_and_dedupe
from beam_agent.snapshot import DEFAULT_RUNS_DIR, save_run_snapshot

__all__ = [
    "__version__",
    "Paper",
    "ExternalIds",
    "search_and_dedupe",
    "run_pipeline",
    "PipelineResult",
    "ConceptResolver",
    "BaselineConceptResolver",
    "ConceptTerm",
    "SizeConstraint",
    "ScientificConceptProfile",
    "QueryPlanner",
    "BaselineQueryPlanner",
    "PlannedQuery",
    "QueryPlan",
    "RelevanceFilter",
    "RuleBasedRelevanceFilter",
    "PaperEvidenceExtractor",
    "BaselineEvidenceExtractor",
    "OllamaPaperEvidenceExtractor",
    "MethodEvidence",
    "SystemEvidence",
    "ProcessEvidenceItem",
    "PaperEvidence",
    "DimensionRequirement",
    "FilterCriteria",
    "DimensionMatch",
    "DimensionMatchResult",
    "EvidenceMatchResult",
    "match_evidence",
    "rank_papers",
    "save_run_snapshot",
    "DEFAULT_RUNS_DIR",
]
