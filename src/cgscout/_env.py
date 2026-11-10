"""Environment variables of cgscout: `CGSCOUT_<NAME>` is the name; the legacy `BEAM_<NAME>` is still read (a one-time notice on stderr when only the
legacy name is set), so existing scripts and caches keep working.  Semantics are exactly those of os.environ.get: a variable that is present
(even empty) wins, the new name before the legacy one."""
import os
import sys

_WARNED: set = set()


def get(name: str, default=None):
    new, old = f"CGSCOUT_{name}", f"BEAM_{name}"
    if new in os.environ:
        return os.environ[new]
    if old in os.environ:
        if old not in _WARNED:
            _WARNED.add(old)
            print(f"cgscout: environment variable {old} is deprecated; use {new}", file=sys.stderr)
        return os.environ[old]
    return default
