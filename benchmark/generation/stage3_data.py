"""Where the labelled Stage-3 CSV exports live.  Benchmark code only; importing this module has no side effects.

The four exports (training_ready 557 rows, external_test 519 rows, the 89-row label set and the 91-row silver set) are the author's own data and are
not in the repository.  Point `CGSCOUT_STAGE3_DATA_DIR` at the directory that holds them, or pass `--stage3-data-dir` to the script."""
from __future__ import annotations

import os
from pathlib import Path

ENV_NAME = "CGSCOUT_STAGE3_DATA_DIR"
FILE_NAMES = ("beam_stage3_training_ready_yes_no.csv", "beam_stage3_external_test_clean_519.csv",
              "beam_stage3_training_labels_89.csv", "beam_stage3_silver_labels_91_web_reviewed.csv")


def data_dir(explicit: str | None = None) -> Path:
    """The directory with the four CSV exports: `explicit` (the --stage3-data-dir option) if given, else $CGSCOUT_STAGE3_DATA_DIR, else a clear error.
    The environment is read at call time, never at import time."""
    value = explicit or os.environ.get(ENV_NAME)
    if not value:
        raise SystemExit(f"The labelled Stage-3 CSV exports are not part of the repository. Set {ENV_NAME} to the directory that holds "
                         f"{', '.join(FILE_NAMES)} (or pass --stage3-data-dir).")
    d = Path(value).expanduser()
    if not d.is_dir():
        raise SystemExit(f"Stage-3 data directory {str(d)!r} (from --stage3-data-dir or {ENV_NAME}) is not a directory.")
    return d
