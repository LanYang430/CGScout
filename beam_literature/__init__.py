"""beam_literature: Phase 1 search layer for BEAM's pre-simulation literature module.

Search only, no LLM. See README.md for the full pipeline this fits into.
"""

from beam_literature.models import Paper
from beam_literature.search import search_and_dedupe

__all__ = ["Paper", "search_and_dedupe"]
