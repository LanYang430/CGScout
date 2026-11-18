"""Shared pytest fixtures for tests/.

Every test gets its own fresh, isolated cache directory (cgscout.cache
reads CGSCOUT_CACHE_DIR fresh on every call, never at import time -- see
cgscout/cache.py) -- this is required, not just convenient: several
existing extractor tests intentionally reuse the same default paper/prompt
fixture across different success/failure scenarios in the same file, and
without isolation a cache entry written by an earlier test would silently
be replayed for a later one expecting a different (fake) response. It also
guarantees no test ever reads or writes the real project `.cache/`.
"""
import pytest


@pytest.fixture(autouse=True)
def _isolated_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CGSCOUT_CACHE_DIR", str(tmp_path / "test-cache"))
