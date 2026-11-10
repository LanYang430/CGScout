"""Provenance of a run: which P1/P2 code produced a frozen plan, which code and configuration a run executed with.  No network, no LLM, no pipeline import.

A frozen plan (plans_frozen/<pid>.json, written by freeze_plans.py) carries a `provenance` block:
  * p1p2_source_sha256   sha256 of the three modules that define P1 (ScientificPlanner + its session) and P2 (ExpertPriorPlanner): the prompts, the pydantic output
                         schemas (their descriptions are part of what the LLM sees) and the clarification / history format;
  * model, effort        what the plans were generated with;
  * cgscout_version, git_commit, git_dirty   informational.
run_benchmark.py refuses to start when a selected case's plan has no provenance (frozen by an older freeze_plans.py, or copied from another run), when the hashes differ from the
code that is about to run, when the model / effort differ from the evaluation profile, or when the plan was made for a different query text.  Editing a prompt (or any other line of
those three modules) after the freeze therefore invalidates the plans on purpose: re-freeze in a fresh run directory.  The same plans can be used for any number of resumed runs.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

SCHEMA = 1
P1P2_MODULES = ("cgscout.planning.scientific_planner", "cgscout.planning.expert_prior_planner", "cgscout.planning.session")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _module_file(name: str) -> Path:
    spec = importlib.util.find_spec(name)           # imports only the parent packages (cgscout, cgscout.planning), not the module itself
    if spec is None or not spec.origin:
        raise RuntimeError(f"cannot locate module {name}")
    return Path(spec.origin)


def p1p2_sources() -> dict:
    """{module name: sha256 of its source file} for the P1/P2 modules of the code that is importable right now."""
    return {name: sha256_file(_module_file(name)) for name in P1P2_MODULES}


def package_sha256() -> str:
    """One hash over every source file of the cgscout package (+ the shipped Stage 3 classifier): identifies the exact code a run executed."""
    import cgscout
    root = Path(cgscout.__file__).resolve().parent
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file() and (p.suffix == ".py" or p.suffix == ".joblib") and "__pycache__" not in p.parts:
            h.update(p.relative_to(root).as_posix().encode()); h.update(b"\0"); h.update(sha256_file(p).encode()); h.update(b"\0")
    return h.hexdigest()


def git_state(cwd=None) -> dict:
    """{'commit': HEAD or None, 'dirty': True/False/None} of the checkout this file lives in (None = not a checkout / git missing)."""
    cwd = Path(cwd) if cwd else Path(__file__).resolve().parent
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=cwd, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=cwd, check=True).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:   # noqa: BLE001 -- provenance is simply absent
        return {"commit": None, "dirty": None}


def stamp(*, model: str, effort: str) -> dict:
    """The provenance block freeze_plans.py writes into every plan file."""
    import cgscout
    g = git_state()
    return {"schema": SCHEMA, "p1p2_source_sha256": p1p2_sources(), "model": model, "effort": effort,
            "cgscout_version": cgscout.__version__, "git_commit": g["commit"], "git_dirty": g["dirty"]}


def check_plan(plan: dict, *, pid: str, query: str, model: str, effort: str, current: dict | None = None) -> list:
    """Why this frozen plan must not be used by a run with the current code / profile / queries.csv (empty list = usable)."""
    problems = []
    prov = plan.get("provenance")
    if not isinstance(prov, dict):
        return ["no provenance block (frozen by an older freeze_plans.py, or copied from another run)"]
    if prov.get("schema") != SCHEMA:
        problems.append(f"provenance schema {prov.get('schema')!r}, expected {SCHEMA}")
    cur = p1p2_sources() if current is None else current
    frozen = prov.get("p1p2_source_sha256") or {}
    for name in P1P2_MODULES:
        if frozen.get(name) != cur.get(name):
            problems.append(f"{name}: plan frozen with source {str(frozen.get(name))[:12]}, current code {str(cur.get(name))[:12]}")
    if prov.get("model") != model:
        problems.append(f"plan frozen with model {prov.get('model')!r}, the run uses {model!r}")
    if prov.get("effort") != effort:
        problems.append(f"plan frozen with effort {prov.get('effort')!r}, the run uses {effort!r}")
    if plan.get("pid") != pid:
        problems.append(f"plan file belongs to case {plan.get('pid')!r}, not {pid!r}")
    if plan.get("query") != query:
        problems.append("the plan was made for a different query text than queries.csv has")
    return problems


def check_plans(plans_dir, rows: dict, selected: list, *, model: str, effort: str) -> dict:
    """Check the plan file of every selected case.  rows = {pid: queries.csv row}.  Returns {'missing': [pid], 'problems': {pid: [reason]}, 'checked': n, ...summary}."""
    cur = p1p2_sources()
    missing, problems, commits, dirty, created = [], {}, set(), set(), []
    for pid in selected:
        p = Path(plans_dir) / f"{pid}.json"
        if not p.exists():
            missing.append(pid); continue
        try:
            plan = json.loads(p.read_text(encoding="utf-8"))
        except ValueError as e:
            problems[pid] = [f"unreadable plan file: {e}"]; continue
        why = check_plan(plan, pid=pid, query=rows[pid]["Latest query (v2)"], model=model, effort=effort, current=cur)
        if why:
            problems[pid] = why
        prov = plan.get("provenance") or {}
        commits.add(prov.get("git_commit")); dirty.add(prov.get("git_dirty")); created.append(plan.get("created"))
    ok = len(selected) - len(missing) - len(problems)
    return {"missing": missing, "problems": problems, "checked": len(selected), "usable": ok, "p1p2_source_sha256": cur,
            "git_commits": sorted(str(c) for c in commits), "git_dirty": sorted(str(d) for d in dirty),
            "created_range": [min((c for c in created if c), default=None), max((c for c in created if c), default=None)]}


def format_plan_problems(res: dict, plans_dir, limit: int = 8) -> str:
    lines = [f"plans in {plans_dir}: {res['usable']} of {res['checked']} usable"]
    if res["missing"]:
        lines.append(f"  no plan file for {len(res['missing'])} case(s): {', '.join(res['missing'][:limit])}{' ...' if len(res['missing']) > limit else ''}")
    for i, (pid, why) in enumerate(res["problems"].items()):
        if i >= limit:
            lines.append(f"  ... and {len(res['problems']) - limit} more case(s) with problems"); break
        lines.append(f"  {pid}: " + "; ".join(why))
    lines.append("Plans must come from freeze_plans.py run with the FINAL P1/P2 code (after the last edit of the three modules).  Missing plans: run freeze_plans.py.  "
                 "Stale or foreign plans: use a fresh run directory (or delete plans_frozen/) and freeze again.")
    return "\n".join(lines)
