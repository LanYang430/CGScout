"""docs/architecture.md names real things: every `module.function` and every `path/file.py` in backticks exists, and all nine prompts are mentioned.  Offline."""
import importlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "cgscout"
TEXT = (ROOT / "docs" / "architecture.md").read_text()
TOKENS = sorted(set(re.findall(r"`([^`\n]+)`", TEXT)))
MODULES = {p.stem: ".".join(("cgscout",) + p.relative_to(PKG).with_suffix("").parts) for p in PKG.rglob("*.py") if p.name != "__init__.py"}


def _dotted():
    out = []
    for t in TOKENS:
        t = t.removesuffix("()")
        if re.fullmatch(r"[a-z0-9_]+(\.[A-Za-z_][A-Za-z0-9_]*)+", t) and t.split(".")[0] in MODULES and t.split(".")[-1] not in ("py", "md", "json", "txt"):
            out.append(t)
    return out


def _resolve(token):
    parts = token.split(".")
    obj = importlib.import_module(MODULES[parts[0]])
    for name in parts[1:]:
        obj = getattr(obj, name)
    return obj


def test_the_document_names_a_meaningful_number_of_functions():
    assert len(_dotted()) >= 20


@pytest.mark.parametrize("token", _dotted())
def test_every_module_function_in_the_document_exists(token):
    assert _resolve(token) is not None, token


def test_every_file_path_in_the_document_exists():
    missing = []
    for t in TOKENS:
        if re.fullmatch(r"[A-Za-z0-9_/]+\.py", t):
            if not (list(PKG.glob(t)) or list(PKG.rglob(Path(t).name))) and not (ROOT / t).exists():
                missing.append(t)
    assert not missing, missing


def test_all_nine_prompts_are_in_the_stage_table():
    rows = [l for l in TEXT.splitlines() if l.startswith("|")]
    for i in range(1, 10):
        assert any(re.search(rf"\bP{i}\b", r) for r in rows), f"P{i}"
