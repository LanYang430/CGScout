"""The Europe PMC abstract lookup lives in sources/europepmc.py (moved there verbatim from assessment/stage4.py, Phase 5 / C3).  Offline.

What its behaviour is, is pinned by tests/test_stage4_v2.py and tests/test_europepmc_abstract_paths.py, which were written BEFORE the move and are unchanged by it.
These tests are about the move itself: one definition, the names Stage 4 still exports, which module Stage 3 reads, the layering, and the names that are gone."""
import ast
import inspect
import json
import urllib.request
from pathlib import Path

import cgscout.assessment.stage4 as s4
import cgscout.sources.europepmc as epmc
from cgscout.models import Paper
from cgscout.screening.filter import _recover_missing_abstracts

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "cgscout"


class _Resp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode()

    def read(self, n=-1):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _paper(doi="10.1/m"):
    return Paper(title="Assembly of a target system", abstract=None, doi=doi, sources=["openalex"])


# B1 ---------------------------------------------------------------------------------------------------------------- one definition, Stage 4 re-exports it
def test_stage4_exports_the_very_objects_defined_in_sources_europepmc():
    assert s4.europepmc_abstract is epmc.europepmc_abstract
    assert epmc.europepmc_abstract.__module__ == "cgscout.sources.europepmc"
    assert s4.MIN_ABSTRACT_CHARS == epmc.MIN_ABSTRACT_CHARS == 40
    assert not hasattr(s4, "_strip_markup") and not hasattr(s4, "_NS_EPMC")          # the private helpers moved, they were not copied


# B2 ---------------------------------------------------------------------------------------------------------------- which module each stage reads
def test_stage3_reads_the_lookup_from_the_new_module_not_from_stage4(monkeypatch):
    asked = []
    monkeypatch.setattr(epmc, "europepmc_abstract", lambda p, **kw: asked.append(("sources.europepmc", p.doi)) or {"abstract": "z" * 50, "via": "doi"})
    monkeypatch.setattr(s4, "europepmc_abstract", lambda p, **kw: asked.append(("stage4", p.doi)) or {"abstract": "w" * 50, "via": "doi"})
    out, recovered = _recover_missing_abstracts([_paper("10.1/b2")])
    assert asked == [("sources.europepmc", "10.1/b2")] and recovered == [True] and out[0].abstract == "z" * 50


def test_known_change_patching_stage4_europepmc_abstract_no_longer_reaches_stage3_but_still_reaches_stage4(monkeypatch):
    """RECORDED CHANGE (C3): Stage 3's default lookup used to be imported from assessment.stage4 at call time, so patching `stage4.europepmc_abstract` also changed
    Stage 3; it now comes from sources.europepmc.  No test or script in the repository relied on that.  Stage 4's own fallback still uses the name in its namespace."""
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _Resp({"resultList": {"result": [{"doi": "10.1/k", "abstractText": "r" * 60}]}}))
    stub_calls = []
    monkeypatch.setattr(s4, "europepmc_abstract", lambda p, **kw: stub_calls.append(p.doi) or {"abstract": "w" * 50, "via": "doi"})
    out, recovered = _recover_missing_abstracts([_paper("10.1/k")])
    assert stub_calls == [] and out[0].abstract == "r" * 60                          # Stage 3 went to the real lookup
    papers, source, failures = s4._apply_abstract_fallback([_paper("10.1/k2")], s4.Stage4Config(abstract_lookup=None), s4._Counter())
    assert stub_calls == ["10.1/k2"] and source[s4._paper_key(papers[0])] == "europepmc" and failures == {}


# B3 ---------------------------------------------------------------------------------------------------------------- the moved private names, in their new home
def test_the_moved_helpers_behave_the_same_in_the_new_module():
    assert epmc._strip_markup("<p>A &lt; B &amp;  C</p>\n<i>d</i>") == "A < B & C d"
    assert epmc._strip_markup(None) == "" and epmc._strip_markup("<br/>") == ""
    assert epmc._NS_EPMC == "stage4_epmc_abstract"
    assert epmc._CACHE_NAMESPACE == "europepmc_methods_snippet" and epmc._NS_EPMC != epmc._CACHE_NAMESPACE      # two clients, two namespaces: never mixed


# B4 ---------------------------------------------------------------------------------------------------------------- every in-repo user of Stage 4 still resolves
def _stage4_references():
    """(file, name) for every `<alias>.<name>` and `from cgscout.assessment.stage4 import <name>` in src/, tests/ and benchmark/ (alias = whatever the file binds the module to)."""
    out = []
    for base in (ROOT / "src", ROOT / "tests", ROOT / "benchmark"):
        for f in base.rglob("*.py"):
            tree = ast.parse(f.read_text(), filename=str(f))
            aliases = set()
            for n in ast.walk(tree):
                if isinstance(n, ast.Import):
                    aliases |= {a.asname or a.name for a in n.names if a.name == "cgscout.assessment.stage4" and a.asname}
                elif isinstance(n, ast.ImportFrom):
                    if n.module == "cgscout.assessment":
                        aliases |= {a.asname or a.name for a in n.names if a.name == "stage4"}
                    elif n.module == "cgscout.assessment.stage4":
                        out += [(f, a.name) for a in n.names]
            for n in ast.walk(tree):
                if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in aliases:
                    out.append((f, n.attr))
    return out


def test_every_name_the_repository_takes_from_stage4_still_exists():
    refs = _stage4_references()
    assert any(name == "europepmc_abstract" for _, name in refs), "the scan found none of the generation scripts' S4.europepmc_abstract: the scan itself is broken"
    missing = sorted({(str(f.relative_to(ROOT)), name) for f, name in refs if not hasattr(s4, name)})
    assert not missing, missing


# B5 ---------------------------------------------------------------------------------------------------------------- layering
def _imported_modules(path):
    mods = set()
    for n in ast.walk(ast.parse(path.read_text())):
        if isinstance(n, ast.Import):
            mods |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.level == 0:
            mods.add(n.module)
    return mods


def test_stage3_no_longer_imports_stage4_anywhere_not_even_inside_a_function():
    mods = _imported_modules(PKG / "screening" / "filter.py")
    assert not [m for m in mods if m.startswith("cgscout.assessment")], sorted(mods)
    assert "cgscout.sources.europepmc" in mods


def test_sources_europepmc_depends_on_nothing_above_the_sources_layer():
    mods = _imported_modules(PKG / "sources" / "europepmc.py")
    internal = {m for m in mods if m.startswith("cgscout")}
    assert internal <= {"cgscout", "cgscout.models"}, sorted(internal)               # `from cgscout import cache` shows up as module "cgscout"


# B6 ---------------------------------------------------------------------------------------------------------------- the clients that were already there are untouched
def test_the_methods_snippet_client_keeps_its_signatures_and_constants():
    assert str(inspect.signature(epmc.fetch_methods_text)) == "(doi: 'Optional[str]') -> 'Optional[str]'"
    assert str(inspect.signature(epmc.snippet_around)) == "(text: 'str', markers: 'tuple[str, ...]') -> 'Optional[str]'"
    assert epmc._REQUEST_TIMEOUT == 10.0 and epmc._MAX_METHODS_TEXT_CHARS == 40_000 and epmc._SNIPPET_WINDOW_CHARS == 300
    assert epmc.fetch_methods_text(None) is None and epmc.fetch_methods_text("") is None
