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


NINE = ["routing_smoke.py", "routing_smoke_v2.py", "generate.py", "seeds.py", "propose14.py", "overlap_audit.py", "l3_find_more.py", "l3_verify.py", "probe.py"]
GONE = ("regression_lib", "sim_classifier_train")


def _imported_modules(path):
    mods = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            mods |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods.add(node.module)
    return mods


def test_the_archived_list_is_exactly_the_generation_scripts_that_import_something_that_no_longer_exists():
    """Nine scripts of benchmark/generation cannot run (docs/reproduce.md lists them as archived / historical).  If one is repaired, or a new broken one appears, this fails."""
    broken = set()
    for py in sorted(GEN.glob("*.py")):
        for m in _imported_modules(py):
            top = m.split(".")[0]
            if top in GONE or (top == "cgscout" and importlib.util.find_spec(m) is None):
                broken.add(py.name)
    assert broken == set(NINE), sorted(broken ^ set(NINE))
    assert all(importlib.util.find_spec(m) is None for m in GONE)                      # still missing, so the list is still true
    assert not list((ROOT / "benchmark").rglob("sim_classifier_train.py"))


def test_reproduce_md_lists_all_nine_archived_scripts_in_its_table():
    text = (ROOT / "docs" / "reproduce.md").read_text(encoding="utf-8")
    table = text.split("## Benchmark-generation scripts are a historical record", 1)[1]
    rows = [line for line in table.splitlines() if line.startswith("| `")]
    assert sorted(line.split("`")[1] for line in rows) == sorted(NINE), rows
    assert "archived / historical" in table and "nothing was re-run" in table


def test_reproduce_md_says_exactly_what_the_legacy_tag_keeps_and_what_it_cannot_reproduce():
    text = (ROOT / "docs" / "reproduce.md").read_text(encoding="utf-8")
    for needle in ("30 reviewed Python files", "Only those 30 Python files are kept from `benchmarks/` in the published history", "benchmarks/regression_13/",
                   "sim_classifier_train.py", "not part of the published history", "cannot reproduce the historical experiments or their outputs",
                   "Publisher abstracts were removed from every version of the published history before publication", "no experiment was re-run"):
        assert needle in text, needle
    assert "in no version at all" not in text and "no abstract exists" not in text      # nothing may suggest that other copies of the repository are free of them
