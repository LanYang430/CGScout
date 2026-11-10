"""CGSCOUT_<NAME> is the environment variable name; the legacy BEAM_<NAME> keeps working (one-time notice) -- cgscout._env."""
import pytest

from cgscout import _env, cache


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in ("CGSCOUT_CACHE_DIR", "BEAM_CACHE_DIR", "CGSCOUT_LLM_MODEL", "BEAM_LLM_MODEL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(_env, "_WARNED", set())


def test_unset_gives_the_default():
    assert _env.get("LLM_MODEL") is None
    assert _env.get("LLM_MODEL", "x") == "x"


def test_new_name_is_read():
    import os
    os.environ["CGSCOUT_LLM_MODEL"] = "m-new"
    try:
        assert _env.get("LLM_MODEL", "d") == "m-new"
    finally:
        del os.environ["CGSCOUT_LLM_MODEL"]


def test_legacy_name_still_works_and_notifies_once(monkeypatch, capsys):
    monkeypatch.setenv("BEAM_LLM_MODEL", "m-old")
    assert _env.get("LLM_MODEL", "d") == "m-old"
    assert _env.get("LLM_MODEL", "d") == "m-old"
    err = capsys.readouterr().err
    assert err.count("BEAM_LLM_MODEL is deprecated; use CGSCOUT_LLM_MODEL") == 1


def test_new_name_wins_over_the_legacy_one(monkeypatch, capsys):
    monkeypatch.setenv("BEAM_LLM_MODEL", "m-old")
    monkeypatch.setenv("CGSCOUT_LLM_MODEL", "m-new")
    assert _env.get("LLM_MODEL") == "m-new"
    assert capsys.readouterr().err == ""


def test_an_empty_value_is_a_value_like_os_environ_get(monkeypatch):
    monkeypatch.setenv("CGSCOUT_LLM_MODEL", "")
    monkeypatch.setenv("BEAM_LLM_MODEL", "m-old")
    assert _env.get("LLM_MODEL", "d") == ""


def test_cache_dir_honours_both_names(monkeypatch, tmp_path):
    monkeypatch.setenv("BEAM_CACHE_DIR", str(tmp_path / "old"))
    assert cache._default_cache_dir() == tmp_path / "old"
    monkeypatch.setenv("CGSCOUT_CACHE_DIR", str(tmp_path / "new"))
    assert cache._default_cache_dir() == tmp_path / "new"
