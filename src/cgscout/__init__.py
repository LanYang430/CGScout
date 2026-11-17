"""cgscout: a literature-retrieval agent that recommends coarse-grained (CG) modeling papers.

Pipeline: planning (P1 scientific planner, P2 expert prior) -> retrieval (direct / 3A / 3B related-system routes, P3-P6) -> Stage 3 screening
-> Stage 4 assessment (P7-P9).  Subpackages: planning, retrieval, sources, screening, assessment; shared modules: llm, cache, models, config.
Importing this package loads none of them; import what you need from its own submodule, e.g. `from cgscout.agent import run, execute`.
The earlier v1 pipeline and the old session path are historical code, kept in the git tag `legacy-pre-cgscout`.
"""

# Kept in sync with pyproject.toml's [project].version by hand: a plain constant is the simplest source of truth.
__version__ = "0.1.0"

__all__ = ["__version__"]
