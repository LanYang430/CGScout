"""Unit tests for cgscout/cache.py -- the generic file-based dev-speed
cache. Pure filesystem behavior, no network, no LLM. Each test passes its
own tmp_path as cache_dir explicitly (in addition to tests/conftest.py's
autouse CGSCOUT_CACHE_DIR isolation) so this file's assertions don't depend on
that fixture existing.
"""
from __future__ import annotations

import json
import time

from cgscout import cache


def _write_raw_entry(namespace, key, cache_dir, entry: dict) -> None:
    """Bypasses cache.set() to write an arbitrary (possibly malformed/
    backdated) envelope directly -- how the TTL tests simulate an old entry
    without needing to actually sleep."""
    path = cache._path_for(namespace, key, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(entry, f)


def test_get_is_a_miss_when_nothing_was_ever_set(tmp_path):
    assert cache.get("ns", "some-key", cache_dir=tmp_path) is None


def test_set_then_get_roundtrips_the_exact_value(tmp_path):
    value = {"a": 1, "b": ["x", "y"], "c": None}
    cache.set("ns", "k1", value, cache_dir=tmp_path)
    assert cache.get("ns", "k1", cache_dir=tmp_path) == value


def test_different_namespaces_do_not_collide_on_the_same_key(tmp_path):
    cache.set("ns_a", "same-key", {"who": "a"}, cache_dir=tmp_path)
    cache.set("ns_b", "same-key", {"who": "b"}, cache_dir=tmp_path)
    assert cache.get("ns_a", "same-key", cache_dir=tmp_path) == {"who": "a"}
    assert cache.get("ns_b", "same-key", cache_dir=tmp_path) == {"who": "b"}


def test_a_corrupt_cache_file_is_treated_as_a_miss_not_a_crash(tmp_path):
    path = cache._path_for("ns", "bad-key", tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json")
    assert cache.get("ns", "bad-key", cache_dir=tmp_path) is None


def test_set_writes_atomically_no_leftover_tmp_file(tmp_path):
    cache.set("ns", "k", {"v": 1}, cache_dir=tmp_path)
    files = list((tmp_path / "ns").iterdir())
    assert [f.name for f in files] == ["k.json"]  # no .tmp* left behind


def test_default_cache_dir_honors_beam_cache_dir_env_var(tmp_path, monkeypatch):
    override = tmp_path / "wherever"
    monkeypatch.setenv("CGSCOUT_CACHE_DIR", str(override))
    cache.set("ns", "k", {"v": 1})  # no explicit cache_dir -- must use the env var
    assert (override / "ns" / "k.json").exists()
    assert cache.get("ns", "k") == {"v": 1}


# --- content_hash: the part every call site actually relies on for correct
# cache-invalidation semantics -------------------------------------------

def test_content_hash_is_deterministic():
    assert cache.content_hash("a", "b", "c") == cache.content_hash("a", "b", "c")


def test_content_hash_is_order_sensitive():
    assert cache.content_hash("a", "b") != cache.content_hash("b", "a")


def test_content_hash_does_not_let_parts_bleed_into_each_other():
    """("ab", "c") and ("a", "bc") must not collide just because naive
    concatenation would produce the same string -- this is exactly the
    property every (prompt_hash, schema_hash, user_message, model, ...)
    cache key in this codebase depends on for correctness."""
    assert cache.content_hash("ab", "c") != cache.content_hash("a", "bc")


def test_content_hash_changes_when_any_single_part_changes():
    base = cache.content_hash("prompt-v1", "schema-v1", "same paper text", "qwen3:8b")
    changed_prompt = cache.content_hash("prompt-v2", "schema-v1", "same paper text", "qwen3:8b")
    changed_schema = cache.content_hash("prompt-v1", "schema-v2", "same paper text", "qwen3:8b")
    changed_paper = cache.content_hash("prompt-v1", "schema-v1", "different paper text", "qwen3:8b")
    changed_model = cache.content_hash("prompt-v1", "schema-v1", "same paper text", "qwen3:14b")
    assert len({base, changed_prompt, changed_schema, changed_paper, changed_model}) == 5


# --- TTL: opt-in per get() call, `set()` always records _cached_at --------

def test_set_always_records_cached_at_timestamp(tmp_path):
    before = time.time()
    cache.set("ns", "k", {"v": 1}, cache_dir=tmp_path)
    after = time.time()
    path = cache._path_for("ns", "k", tmp_path)
    entry = json.loads(path.read_text())
    assert before <= entry["_cached_at"] <= after
    assert entry["value"] == {"v": 1}


def test_no_ttl_by_default_an_arbitrarily_old_entry_is_still_a_hit(tmp_path):
    _write_raw_entry("ns", "k", tmp_path, {"_cached_at": 0.0, "value": {"v": "ancient"}})  # epoch 0
    assert cache.get("ns", "k", cache_dir=tmp_path) == {"v": "ancient"}  # no ttl_seconds passed -- never expires


def test_ttl_hit_when_entry_is_within_the_window(tmp_path):
    cache.set("ns", "k", {"v": "fresh"}, cache_dir=tmp_path)
    assert cache.get("ns", "k", cache_dir=tmp_path, ttl_seconds=3600) == {"v": "fresh"}


def test_ttl_miss_when_entry_is_older_than_the_window(tmp_path):
    stale_cached_at = time.time() - 7200  # 2h ago
    _write_raw_entry("ns", "k", tmp_path, {"_cached_at": stale_cached_at, "value": {"v": "stale"}})
    assert cache.get("ns", "k", cache_dir=tmp_path, ttl_seconds=3600) is None  # 1h TTL -- expired


def test_ttl_boundary_matches_the_documented_24h_retrieval_default(tmp_path):
    just_inside = time.time() - (24 * 60 * 60 - 60)   # 23h59m ago
    just_outside = time.time() - (24 * 60 * 60 + 60)  # 24h01m ago
    _write_raw_entry("ns", "inside", tmp_path, {"_cached_at": just_inside, "value": "kept"})
    _write_raw_entry("ns", "outside", tmp_path, {"_cached_at": just_outside, "value": "expired"})
    ttl = 24 * 60 * 60
    assert cache.get("ns", "inside", cache_dir=tmp_path, ttl_seconds=ttl) == "kept"
    assert cache.get("ns", "outside", cache_dir=tmp_path, ttl_seconds=ttl) is None


def test_ttl_missing_cached_at_is_a_miss_not_a_crash(tmp_path):
    _write_raw_entry("ns", "k", tmp_path, {"value": {"v": 1}})  # malformed: no _cached_at at all
    assert cache.get("ns", "k", cache_dir=tmp_path, ttl_seconds=3600) is None


def test_expired_entry_can_be_overwritten_by_a_fresh_set(tmp_path):
    stale_cached_at = time.time() - 7200
    _write_raw_entry("ns", "k", tmp_path, {"_cached_at": stale_cached_at, "value": {"v": "stale"}})
    assert cache.get("ns", "k", cache_dir=tmp_path, ttl_seconds=3600) is None  # confirm it reads as expired

    cache.set("ns", "k", {"v": "refreshed"}, cache_dir=tmp_path)  # caller re-fetches and overwrites, as designed

    assert cache.get("ns", "k", cache_dir=tmp_path, ttl_seconds=3600) == {"v": "refreshed"}
