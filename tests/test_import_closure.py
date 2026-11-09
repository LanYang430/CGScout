"""Guards the phase-1 import decoupling: the CGScout (BEAM) path must not drag the v1 pipeline or the old ConceptResolverSession stack into memory.

Why it matters: before this change, `import beam_agent` loaded nine v1 modules, `search.py` loaded `pipeline`/`query_planner`, `relevance_filter_handoff`
loaded the v1 `concept_resolver`, and `route_retrieval_live_adapter` loaded `mellea_adapter` (which hard-imports the `mellea` package) -- none of which the
retrieval -> Stage 3 -> Stage 4 path uses.  The closure checks run in a fresh interpreter, because other tests import these modules in-process.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CR = ROOT / "benchmarks" / "concept_resolver_reasoning"
MI = CR / "mellea_integration"

V1_MODULES = [
    "beam_agent.pipeline", "beam_agent.query_planner", "beam_agent.concept_resolver",
    "beam_agent.evidence_matching", "beam_agent.relevance_filter", "beam_agent.snapshot",
]
SESSION_STACK = ["concept_resolver_session", "intake_readiness_gate", "mellea_adapter"]

_PRELUDE = f"""
import sys, json
sys.dont_write_bytecode = True
for p in ({str(ROOT)!r}, {str(CR)!r}, {str(MI)!r}, {str(ROOT / "benchmarks" / "regression_13")!r}):
    sys.path.insert(0, p)
"""


def _loaded(code: str, watch: list[str]) -> dict:
    """Run `code` in a fresh interpreter and report which of `watch` ended up in sys.modules (and whether the run itself succeeded)."""
    src = _PRELUDE + code + f"\nprint('@@' + json.dumps({{m: (m in sys.modules and sys.modules[m] is not None) for m in {watch!r}}}))\n"
    proc = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, cwd=str(ROOT),
                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.rsplit("@@", 1)[1])


def test_importing_the_package_loads_no_v1_module():
    loaded = _loaded("import beam_agent", V1_MODULES)
    assert not [m for m, v in loaded.items() if v], loaded


def test_search_module_does_not_load_the_v1_pipeline():
    loaded = _loaded("import beam_agent.search", V1_MODULES)
    assert not [m for m, v in loaded.items() if v], loaded


def test_cgscout_path_loads_no_v1_or_session_stack_and_needs_no_mellea():
    # `sys.modules['mellea'] = None` makes `import mellea` raise ImportError: the path must work on an install without that package.
    code = "sys.modules['mellea'] = None\nimport transfer_retrieval_bridge, stage4_v2, regression_lib\n"
    loaded = _loaded(code, V1_MODULES + SESSION_STACK + ["mellea"])
    assert not [m for m, v in loaded.items() if v], loaded


def test_legacy_session_path_keeps_its_module_level_names():
    """The old scripts (and benchmark recorders) access these as attributes of the module; they must still exist."""
    import route_retrieval_live_adapter as adapter  # noqa: E402 -- sys.path is set up by the module itself, like every other test of this directory
    for name in ("plan_evidence_routes", "run_route_retrieval", "search_and_dedupe", "STRUCTURAL_HOP1_SEARCH_LIMIT",
                 "STRUCTURAL_HOP2_SEARCH_LIMIT", "STRUCTURAL_SEARCH_SOURCES", "_make_capturing_search_fns", "make_need_semantic_search_fn",
                 "papers_for_stage3", "provenance_index", "run_live_route_retrieval", "build_evidence_route_plan_from_session"):
        assert hasattr(adapter, name), name


def test_methodology_table_has_a_single_definition():
    from beam_agent import concept_resolver, text_utils
    import relevance_filter_handoff as handoff
    assert concept_resolver._METHODOLOGY_TERM_CLUSTERS is text_utils.METHODOLOGY_TERM_CLUSTERS
    assert handoff._METHODOLOGY_TERM_CLUSTERS is text_utils.METHODOLOGY_TERM_CLUSTERS


def test_hop1_helpers_have_a_single_definition():
    pytest.importorskip("mellea")   # mellea_adapter hard-imports it (legacy Ollama/Mellea path)
    import mellea_adapter
    import transfer_entity_extraction as tee
    assert mellea_adapter._find_structural_term_matched is tee._find_structural_term_matched
    assert mellea_adapter._MAX_ABSTRACT_CHARS_FOR_EXTRACTION == tee._MAX_ABSTRACT_CHARS_FOR_EXTRACTION == 4000
    assert mellea_adapter._EXTRACTION_TIMEOUT_SECONDS == tee._EXTRACTION_TIMEOUT_SECONDS == 20
