"""Guards the package's import structure: the CGScout path must not drag legacy code, the optional local-model stack, or any path patch into memory.

Background: before the cleanup, `import beam_agent` loaded nine v1 modules, `search.py` loaded `pipeline`/`query_planner`, the Stage 3 filter loaded the v1
`concept_resolver`, and the live search adapter loaded `mellea_adapter` (which hard-imports the `mellea` package) -- none of which the
retrieval -> Stage 3 -> Stage 4 path uses.  That code is now historical (git tag `legacy-pre-cgscout`); these tests keep it from creeping back.
The closure checks run in a fresh interpreter, because other tests import these modules in-process.
"""
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PKG = SRC / "cgscout"

# Names that must never be loaded by the CGScout path: the old package, the v1 / old-session modules, and the local-model stack that was dropped from the project.
LEGACY = ["beam_agent", "concept_resolver_session", "intake_readiness_gate", "mellea", "ollama", "mellea_adapter"]


def _loaded(code: str, watch: list[str]) -> dict:
    """Run `code` in a fresh interpreter (with src/ importable) and report which names of `watch` -- or submodules of them -- ended up in sys.modules."""
    src = ("import sys, json\nsys.dont_write_bytecode = True\n" + code +
           f"\nprint('@@' + json.dumps({{w: sorted(m for m in sys.modules if sys.modules[m] is not None and (m == w or m.startswith(w + '.'))) for w in {watch!r}}}))\n")
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": os.pathsep.join([str(SRC)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))}
    proc = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, cwd=str(ROOT), env=env)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.rsplit("@@", 1)[1])


def test_importing_the_package_loads_no_submodule_and_no_legacy_code():
    loaded = _loaded("import cgscout", LEGACY + ["cgscout"])
    assert loaded["cgscout"] == ["cgscout"], loaded          # the bare package imports none of its submodules
    assert not [w for w in LEGACY if loaded[w]], loaded


def test_search_module_loads_no_legacy_code():
    loaded = _loaded("import cgscout.retrieval.search", LEGACY)
    assert not [w for w in LEGACY if loaded[w]], loaded


def test_cgscout_path_loads_no_legacy_code_and_needs_no_local_model_packages():
    # `sys.modules[name] = None` makes `import name` raise ImportError: the path must work on an install without these packages.
    code = ("sys.modules['mellea'] = None\nsys.modules['ollama'] = None\n"
            "import cgscout.retrieval.orchestrator, cgscout.assessment.stage4, cgscout.cli\n")
    loaded = _loaded(code, LEGACY)
    assert not [w for w in LEGACY if loaded[w]], loaded


def test_search_wrappers_keep_the_module_level_names_the_harness_uses():
    """The two bridges and the benchmark recorders access these as attributes of the modules (the recorder wraps `search_and_dedupe` in from_planner and
    classifies requests by the closure names inside `_make_capturing_search_fns`); they must keep existing under the same names.
    (Formerly retrieval/search_adapters.py: the wrappers and adapters now live in direct/from_planner.py, the Hop-1/Hop-2 depths in retrieval/search.py.)"""
    import cgscout.retrieval.direct.from_planner as planner_bridge
    import cgscout.retrieval.search as search
    for name in ("search_and_dedupe", "_make_capturing_search_fns", "make_need_semantic_search_fn", "papers_for_stage3", "provenance_index"):
        assert hasattr(planner_bridge, name), name
    for name in ("STRUCTURAL_HOP1_SEARCH_LIMIT", "STRUCTURAL_HOP2_SEARCH_LIMIT", "STRUCTURAL_SEARCH_SOURCES"):
        assert hasattr(search, name), name


def test_methodology_table_has_a_single_definition():
    """The table (formerly text.METHODOLOGY_TERM_CLUSTERS) lives in screening/filter.py; its private alias is the same object, not a second copy."""
    import cgscout.screening.filter as handoff
    assert handoff._METHODOLOGY_TERM_CLUSTERS is handoff.METHODOLOGY_TERM_CLUSTERS


def test_hop1_extraction_constants_are_unchanged():
    import cgscout.retrieval.related_systems.extraction as tee
    assert tee._MAX_ABSTRACT_CHARS_FOR_EXTRACTION == 4000
    assert tee._EXTRACTION_TIMEOUT_SECONDS == 20


def _source_files():
    return sorted(PKG.rglob("*.py"))


def test_source_has_no_mellea_or_ollama_import_and_no_mellea_adapter_name():
    """AST scan of src/cgscout: no import of mellea / ollama, no `mellea_adapter` identifier or string."""
    offenders = []
    for path in _source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            if any(m.split(".")[0] in ("mellea", "ollama") for m in mods):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} imports {mods}")
            if isinstance(node, ast.Name) and "mellea_adapter" in node.id:
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} name {node.id}")
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and "mellea_adapter" in node.value:
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} string mentions mellea_adapter")
    assert not offenders, offenders


def test_source_does_not_patch_sys_path():
    """The package is imported as `cgscout.*` -- no module may rely on (or add to) sys.path."""
    offenders = []
    for path in _source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute) and node.attr == "path" and isinstance(node.value, ast.Name) and node.value.id == "sys"):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not offenders, offenders
