"""Golden replay: the whole retrieval -> Stage 3 -> Stage 4 pipeline, run offline against a FIXED fixture, must reproduce the stored outputs byte for byte.

The fixture is large and is not part of the repository: build it with benchmark/evaluation/build_replay_fixture.py and point CGSCOUT_REPLAY_FIXTURE at it
(CGSCOUT_REPLAY_CASES=a,b,c optionally restricts the cases).  Without a fixture this test is skipped.  The replay runs in a fresh interpreter (it blocks the
network and patches modules), with no LLM and no network access; a cache miss, an uncached LLM call or a network attempt in a covered path fails the comparison.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = os.environ.get("CGSCOUT_REPLAY_FIXTURE")
pytestmark = pytest.mark.skipif(not FIXTURE or not Path(FIXTURE or "/nonexistent").is_dir(),
                                reason="no replay fixture (set CGSCOUT_REPLAY_FIXTURE to a directory built by benchmark/evaluation/build_replay_fixture.py)")


def test_replay_reproduces_the_golden_outputs(tmp_path):
    fx = Path(FIXTURE)
    golden = {p.stem: p for p in sorted((fx / "golden" / "cases").glob("*.json"))}
    wanted = [c for c in os.environ.get("CGSCOUT_REPLAY_CASES", ",".join(golden)).split(",") if c]
    assert wanted and set(wanted) <= set(golden), (wanted, sorted(golden))
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": os.pathsep.join([str(ROOT / "src")] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))}
    proc = subprocess.run([sys.executable, str(ROOT / "benchmark" / "evaluation" / "replay.py"), str(fx), str(tmp_path / "out"), ",".join(wanted)],
                          capture_output=True, text=True, env=env, cwd=str(ROOT))
    assert proc.returncode == 0, proc.stderr[-3000:]
    for pid in wanted:
        assert (tmp_path / "out" / "cases" / f"{pid}.json").read_bytes() == golden[pid].read_bytes(), f"case {pid} differs from the golden output"
