"""The scripts that read the author's labelled Stage-3 CSV exports (not in the repository) have no built-in path: a command-line option or an
environment variable names the data, a missing value is an explicit error, and importing a module never reads the environment or the disk."""
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

BB = Path(__file__).resolve().parents[1]            # benchmark/generation
ROOT = BB.parents[1]
sys.path.insert(0, str(BB))

import l3_verify  # noqa: E402
import overlap_audit  # noqa: E402
import propose14  # noqa: E402
import stage3_data  # noqa: E402

SCRIPTS = [BB / "l3_verify.py", BB / "overlap_audit.py", BB / "propose14.py", ROOT / "benchmark" / "training" / "train_sim_classifier.py"]
HOME_PREFIXES = ("/" + "Users/", "/" + "home/")                 # built by concatenation so that this file does not contain them itself


def _env(**extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CGSCOUT_STAGE3", "BEAM_STAGE3"))}
    env.update(PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1", **extra)
    return env


def test_data_dir_prefers_the_option_then_the_environment(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    monkeypatch.setenv(stage3_data.ENV_NAME, str(b))
    assert stage3_data.data_dir(str(a)) == a
    assert stage3_data.data_dir() == b


def test_data_dir_without_a_value_or_a_directory_is_an_explicit_error(tmp_path, monkeypatch):
    monkeypatch.delenv(stage3_data.ENV_NAME, raising=False)
    with pytest.raises(SystemExit) as e:
        stage3_data.data_dir()
    assert stage3_data.ENV_NAME in str(e.value) and "--stage3-data-dir" in str(e.value) and "beam_stage3_training_ready_yes_no.csv" in str(e.value)
    with pytest.raises(SystemExit) as e:
        stage3_data.data_dir(str(tmp_path / "missing"))
    assert "not a directory" in str(e.value)


def test_the_environment_is_read_at_call_time_not_at_import(tmp_path, monkeypatch):
    monkeypatch.delenv(stage3_data.ENV_NAME, raising=False)
    with pytest.raises(SystemExit):
        stage3_data.data_dir()
    monkeypatch.setenv(stage3_data.ENV_NAME, str(tmp_path))      # set after the module was imported
    assert stage3_data.data_dir() == tmp_path


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_help_runs_without_data_and_names_the_option(script):
    r = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True, env=_env(), cwd=str(ROOT), timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    assert ("--train-csv" if script.name == "train_sim_classifier.py" else "--stage3-data-dir") in r.stdout


@pytest.mark.parametrize("script", [BB / "l3_verify.py", BB / "overlap_audit.py", BB / "propose14.py"], ids=lambda p: p.name)
def test_generation_scripts_stop_before_any_work_when_no_data_dir_is_given(script):
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=_env(), cwd=str(ROOT), timeout=120)
    assert r.returncode != 0
    assert stage3_data.ENV_NAME in r.stderr and "Traceback" not in r.stderr


def test_trainer_stops_with_an_explanation_when_no_csv_is_given_and_reads_both_env_spellings(tmp_path):
    trainer = ROOT / "benchmark" / "training" / "train_sim_classifier.py"
    r = subprocess.run([sys.executable, str(trainer)], capture_output=True, text=True, env=_env(), cwd=str(ROOT), timeout=120)
    assert r.returncode != 0 and "CGSCOUT_STAGE3_TRAIN_CSV" in r.stderr and "Traceback" not in r.stderr
    csv = tmp_path / "t.csv"
    csv.write_text("doi,title,abstract,simulation_by_authors,label_note\n")
    code = ("import sys; sys.path.insert(0, %r); import train_sim_classifier as T; print(T.train_csv())" % str(trainer.parent))
    for name in ("CGSCOUT_STAGE3_TRAIN_CSV", "BEAM_STAGE3_TRAIN_CSV"):
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=_env(**{name: str(csv)}), cwd=str(ROOT), timeout=120)
        assert r.returncode == 0 and r.stdout.strip() == str(csv), (name, r.stderr[-1500:])


def test_no_module_level_path_constant_and_no_personal_absolute_path_in_benchmark_or_src():
    import overlap_audit
    assert not hasattr(overlap_audit, "DL")
    for py in sorted((ROOT / "benchmark").rglob("*.py")) + sorted((ROOT / "src").rglob("*.py")):
        text = py.read_text(encoding="utf-8")
        for prefix in HOME_PREFIXES:
            assert prefix not in text, f"{py.relative_to(ROOT)} contains the absolute path prefix {prefix!r}"


def _write_exports(directory):
    for name in stage3_data.FILE_NAMES:
        (directory / name).write_text("doi,title,case,simulation_by_authors\n10.1/ABC,A title,case1: seed,YES\n", encoding="utf-8")


def test_each_script_reads_the_exports_from_the_directory_it_is_given(tmp_path, monkeypatch):
    monkeypatch.delenv(stage3_data.ENV_NAME, raising=False)
    _write_exports(tmp_path)
    ov = l3_verify.classifier_overlap(str(tmp_path))
    assert any(v.startswith("training:case1") for v in ov["10.1/abc"])
    assert "10.1/abc" in propose14.overlap_dois(str(tmp_path))
    fake = types.ModuleType("sim_classifier_train")           # the trainer's old module name, which overlap_audit imports lazily
    fake.load_train_rows = lambda: []
    monkeypatch.setitem(sys.modules, "sim_classifier_train", fake)
    data, used = overlap_audit.load_all(str(tmp_path))
    assert set(data) == set(overlap_audit.FILES) and data["train557"][0]["doi"] == "10.1/ABC" and used == set()
