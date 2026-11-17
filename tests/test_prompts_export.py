"""prompts/ (the nine system prompts as text files) must be in sync with the string constants in the code, which are the single source.  Offline.

The export script is `prompts/export_prompts.py`; a prompt edited in the code without `python prompts/export_prompts.py --write` fails here."""
import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "prompts"


def _export():
    spec = importlib.util.spec_from_file_location("export_prompts", PROMPTS / "export_prompts.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


EX = _export()


def test_prompts_directory_is_in_sync_with_the_code():
    assert EX.check() == [], "run `python prompts/export_prompts.py --write`"
    assert EX.main(["--check"]) == 0


@pytest.mark.parametrize("key", sorted(EX.WHERE))
def test_each_file_is_exactly_the_constant(key):
    import importlib
    mod, attr, _, _ = EX.WHERE[key]
    text = getattr(importlib.import_module(mod), attr)
    assert (PROMPTS / EX.filename(key)).read_bytes() == text.encode("utf-8")


def test_nine_prompts_numbered_in_order_and_no_stray_files():
    assert list(EX.WHERE) == [f"P{i}" for i in range(1, 10)]
    assert sorted(p.name for p in PROMPTS.glob("P[0-9]_*.txt")) == sorted(EX.filename(k) for k in EX.WHERE)


def test_sha256sums_is_in_the_sha256sum_format_and_matches_the_files():
    lines = (PROMPTS / "SHA256SUMS").read_text().splitlines()
    assert len(lines) == 9
    for line in lines:
        digest, name = line.split("  ")
        assert hashlib.sha256((PROMPTS / name).read_bytes()).hexdigest() == digest


def test_index_lists_every_prompt_with_its_hash_and_version_constant():
    index = (PROMPTS / "INDEX.md").read_text()
    for key, (mod, attr, role, ver) in EX.WHERE.items():
        row = next(l for l in index.splitlines() if l.startswith(f"| {key} |"))
        assert f"`{mod.replace('cgscout.', '')}.{attr}`" in row and (PROMPTS / EX.filename(key)).exists()
        assert (ver is None) == ("(none" in row)
