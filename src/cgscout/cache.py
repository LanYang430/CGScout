"""Minimal, dependency-free, file-based cache for dev/test iteration speed.

Persists across PROCESSES (not just within one run) under a `.cache/`
directory at the repo root, or `CGSCOUT_CACHE_DIR` (legacy: `BEAM_CACHE_DIR`) if set. This is a dev-speed
cache, not a correctness/production cache: it has no opinion about what is
"safe" to store -- every call site decides that for itself (e.g. "only cache
a search result with zero source errors", "only cache a successfully-parsed
LLM extraction"). This module only does two things: hash whatever tuple of
strings a caller gives it into a stable key, and store/retrieve a JSON-
serializable value under that key.

Key composition is entirely the CALLER's responsibility. Anything that
should invalidate a cached entry when it changes (a paper's own content, the
exact prompt text, the exact schema, the model name, a structural_class, a
query string, ...) must be included in the parts passed to `content_hash()`
-- this module has no way to know what matters to a given call site.

Never raises on a cache-layer problem (missing file, corrupt JSON, an
unwritable directory) -- any such failure degrades to "cache miss" /
"silently skip caching this value", exactly like a network retrieval or an
LLM call failing degrades elsewhere in this codebase. A cache should never
be the reason a real computation crashes.

Every entry is stored with its write timestamp regardless of whether the
caller ever uses it (`set()` always records `_cached_at`; the on-disk shape
never differs between a TTL and a non-TTL caller). Whether that timestamp
is ever CHECKED is entirely opt-in per call: `get(..., ttl_seconds=None)`
(the default) never expires an entry -- appropriate for a cache keyed on
content that fully determines the answer (a paper's own text + the exact
prompt/schema/model that produced it -- see transfer_entity_extraction.py /
stage4_v2.py, neither passes ttl_seconds). `get(..., ttl_seconds=N)`
treats an entry older than N seconds as a miss -- appropriate for a cache
keyed on a QUERY whose real-world answer can change over time as new papers
get indexed or a ranking shifts (see search.py, which does pass ttl_seconds).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from cgscout import _env

def _source_checkout_root() -> Optional[Path]:
    """The repository root when running from a source checkout (src/cgscout/cache.py -> <root>/pyproject.toml), else None
    (an installed package: its files are not inside a repository)."""
    root = Path(__file__).resolve().parent.parent.parent
    return root if (root / "pyproject.toml").exists() and (root / "src" / "cgscout").is_dir() else None


def _default_cache_dir() -> Path:
    """Re-reads CGSCOUT_CACHE_DIR (legacy name BEAM_CACHE_DIR) every call (not cached at import time) so
    tests can point this at a tmp directory via monkeypatch/env without
    needing to change any call site's signature.  Default: `<repo root>/.cache`
    in a source checkout (unchanged from before the package layout), else
    `~/.cache/cgscout` for an installed package."""
    override = _env.get("CACHE_DIR")
    if override:
        return Path(override)
    root = _source_checkout_root()
    return (root / ".cache") if root else (Path.home() / ".cache" / "cgscout")


def content_hash(*parts: str) -> str:
    """Stable hash of an ORDERED tuple of strings. NUL-separated internally
    so ("ab", "c") and ("a", "bc") never collide. The caller decides what
    goes in; anything not included here will NOT invalidate the cache when
    it changes."""
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _path_for(namespace: str, key: str, cache_dir: Optional[Path]) -> Path:
    base = cache_dir if cache_dir is not None else _default_cache_dir()
    return base / namespace / f"{key}.json"


def get(
    namespace: str, key: str, *, cache_dir: Optional[Path] = None, ttl_seconds: Optional[float] = None,
) -> Optional[Any]:
    """Returns the cached value, or None on any miss -- including a missing
    file, a corrupt/unreadable one, invalid JSON, a missing/malformed
    `_cached_at`, or (only when `ttl_seconds` is given) an entry older than
    that many seconds. Never raises. `ttl_seconds=None` (default) means no
    expiry at all -- an entry is a hit for as long as it exists on disk."""
    path = _path_for(namespace, key, cache_dir)
    try:
        with open(path, "r") as f:
            entry = json.load(f)
        if ttl_seconds is not None:
            cached_at = entry["_cached_at"]
            if (time.time() - cached_at) > ttl_seconds:
                return None  # expired -- treated exactly like a miss, caller re-fetches and overwrites
        return entry["value"]
    except Exception:  # noqa: BLE001 -- a cache-layer problem is a miss, never a crash
        return None


def set(namespace: str, key: str, value: Any, *, cache_dir: Optional[Path] = None) -> None:
    """Writes atomically (tmp file + rename) so a crash mid-write can never
    leave a corrupt entry -- `get()` would otherwise silently and
    permanently treat a half-written file as a miss forever, but a
    concurrent reader could race a partial read of the same path without
    the tmp+rename step. Failing to cache is never fatal to the caller: any
    write error here is swallowed, matching `get()`'s own never-raises
    contract. Always records `_cached_at` (the write timestamp), regardless
    of whether this particular caller ever uses a TTL -- so a namespace can
    start being TTL-checked later without a storage-format migration."""
    path = _path_for(namespace, key, cache_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        with open(tmp, "w") as f:
            json.dump({"_cached_at": time.time(), "value": value}, f)
        tmp.replace(path)
    except Exception:  # noqa: BLE001 -- a failed cache WRITE must never break the real computation
        pass
