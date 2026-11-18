"""cgscout.retrieval.types is gone (zero API, zero network).

Phase 5 / C2e-1: the module held the result types of the removed Hop-2 chain (Hop1Provenance, Hop2Query, Hop2Paper, Hop2SystemResult, Hop2Result) -- no longer part of
any schema, since C2c and C2d -- and the tuple `_CG_MARKERS`, which moved verbatim into screening/filter.py (pinned by tests/test_cg_markers.py, which passed
unchanged before and after)."""
import ast
import importlib
import importlib.util
import pkgutil
from pathlib import Path

import cgscout
import cgscout.screening.filter as F

SRC = Path(cgscout.__file__).resolve().parent
GONE = ("Hop1Provenance", "Hop2Query", "Hop2Paper", "Hop2SystemResult", "Hop2Result")


def test_the_module_does_not_exist_and_no_source_file_refers_to_it():
    assert importlib.util.find_spec("cgscout.retrieval.types") is None
    assert not (SRC / "retrieval" / "types.py").exists()
    for path in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert "retrieval.types" not in (node.module or "") and not (node.module == "cgscout.retrieval" and any(a.name == "types" for a in node.names)), path
            elif isinstance(node, ast.Import):
                assert not any("retrieval.types" in a.name for a in node.names), path


def test_no_module_of_the_package_defines_the_removed_result_types():
    for info in pkgutil.walk_packages(cgscout.__path__, "cgscout."):
        module = importlib.import_module(info.name)
        assert not [n for n in GONE if hasattr(module, n)], info.name


def test_the_cg_markers_are_defined_in_the_filter_module_itself():
    assert F._CG_MARKERS == ("coarse-grained", "coarse grained", "coarsegrained", "coarse-grain ", "coarse grain ")
    assert F._STRUCTURAL_CG_MARKERS[:len(F._CG_MARKERS)] == F._CG_MARKERS
    defined_here = {t.id for node in ast.parse(Path(F.__file__).read_text()).body if isinstance(node, ast.Assign) for t in node.targets if isinstance(t, ast.Name)}
    assert {"_CG_MARKERS", "_STRUCTURAL_CG_MARKERS"} <= defined_here
