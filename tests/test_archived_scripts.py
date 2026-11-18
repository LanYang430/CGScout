"""The two archived benchmark-generation scripts (routing_smoke.py, routing_smoke_v2.py) cannot run on the current version; they stay in the repository as the record of how
the benchmark was built, and must say so.  If the modules they import ever come back, these tests fail and the archive note has to be revisited."""
import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "benchmark" / "generation"
SCRIPTS = ["routing_smoke.py", "routing_smoke_v2.py"]
NOTE = "ARCHIVED / HISTORICAL SCRIPT"


@pytest.mark.parametrize("name", SCRIPTS)
def test_the_script_says_it_is_archived_and_where_to_find_the_code_it_was_written_for(name):
    doc = ast.get_docstring(ast.parse((GEN / name).read_text(encoding="utf-8")), clean=False)
    assert doc.startswith(NOTE), "the archive note must be the first thing in the module docstring"
    assert "legacy-pre-cgscout" in doc and f"benchmarks/beam_bench_v1/{name}" in doc


def test_the_archive_note_is_still_true():
    """The note says these imports fail; if one of them starts to work the note is stale."""
    assert importlib.util.find_spec("cgscout.retrieval.orchestrator") is None
    assert importlib.util.find_spec("cgscout.retrieval.transfer") is not None          # the module it was renamed to
    assert not list((ROOT / "benchmark").rglob("regression_lib.py"))
    assert (ROOT / "benchmark" / "evaluation" / "harness.py").is_file()                   # what replaced it


def test_reproduce_md_explains_the_archive_and_how_to_trace_it_and_the_outputs_are_kept():
    text = (ROOT / "docs" / "reproduce.md").read_text(encoding="utf-8")
    for needle in ("routing_smoke.py", "routing_smoke_v2.py", "archived", "legacy-pre-cgscout", "ModuleNotFoundError"):
        assert needle in text, needle
    outputs = sorted((GEN / "data").glob("routing_smoke*.json"))
    assert len(outputs) == 8, [p.name for p in outputs]


def test_the_docs_do_not_point_into_the_authors_working_directories():
    """The audit records name their raw data as 'the archived audit data, not part of this repository'; no document names a directory of the author's machine."""
    internal = ("CSSE_fellow" + "_BEAM", "prompt_audit" + "/", "benchmark_new37" + "_beam", "beam_agent" + "_runs")
    for doc in sorted((ROOT / "docs").glob("*.md")) + [ROOT / "README.md"]:
        text = doc.read_text(encoding="utf-8")
        for needle in internal:
            assert needle not in text, f"{doc.name} mentions {needle!r}"
