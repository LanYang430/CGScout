"""run_benchmark.py: the command line really starts, a 200-case selection cannot silently shrink or grow, the configuration that runs is the PAPER profile (and is recorded in
full), an incomplete run never exits 0, stale frozen plans are refused, and a run resumes under the same code.

Two layers, both offline (no network, no LLM):
  * subprocess tests launch the script exactly as docs/reproduce.md does (real __main__, real configure(), real plan checks); they only use cases that need no API call
    (refusals, everything already finished, spend ceiling already reached);
  * in-process tests run the real main() with run_query / the OpenAlex budget object / the LLM backend replaced by stubs, to exercise selection, retries, ceiling and resume.
The pipeline itself (run_query) is verified separately by the offline golden replay (tests/test_golden_replay.py).
"""
import dataclasses
import json
import types

import pytest

from bench_support import PAPER, PV, ROOT, pid_of, run_script, write_plans, write_queries, write_result  # noqa: F401  (bench_support puts benchmark/evaluation on sys.path)

import arms  # noqa: E402

# --- what the evaluated CUR configuration IS, spelled out: the paper profile plus every library default it leaves alone -----------------------------------------------------
# (a change of a library default, e.g. max_papers_per_query, or of a PAPER value changes what "the paper run" means: it must fail here and be changed on purpose)
EFFECTIVE_BUDGETS = {'max_queries_per_route': 3, 'max_papers_per_query': 20, 'max_papers_per_route': 60, 'structural_max_discovered_systems': 150, 'structural_hop2_max_systems': 30,
                     'structural_max_papers_to_extract': 150, 'hop1_extraction_max_workers': 2, 'structural_total_budget': None, 'max_total_unique_papers': 120,
                     'flat_anchor_short_context': True, 'flat_query_hygiene': True, 'need_semantic_enabled': True, 'need_semantic_limit': 20, 'need_semantic_include_context': False,
                     'stage3_recover_abstracts': True, 'query_construction_v2': False}
EFFECTIVE_STAGE4 = {'screen_batch_size': 12, 'screen_effort': 'medium', 'deep_cap': 500, 'reject_min_confidence': 'medium', 'abstract_fallback': True, 'abstract_lookup': None,
                    'abstract_workers': 3, 'extract_effort': 'medium', 'judge_effort': 'medium', 'workers': 6, 'timeout_s': 150.0, 'screen_abstract_chars': 4000, 'retries': 1}
CLASSIFIER_SHA256 = "7f986a6195775efafe6b13b1e414ce114a84e6b78970ab197b9bcb2aeafd2bb7"     # screening/sim_classifier_model.joblib, byte-identical to the evaluated runs' file


# ===================================================================== selection (pure function) =========================================================================

IDS = [pid_of(i) for i in range(1, 201)]


def test_select_pids_full_run_needs_the_row_count():
    with pytest.raises(arms.PreflightError, match="--expect-rows"):
        arms.select_pids(IDS, None, None)
    with pytest.raises(arms.PreflightError, match="200 rows|has 200"):
        arms.select_pids(IDS, None, 199)
    assert arms.select_pids(IDS, None, 200) == (IDS, True)


@pytest.mark.parametrize("value", ["", " ", ",", " , ,"])
def test_select_pids_empty_value_is_an_error_not_everything(value):
    with pytest.raises(arms.PreflightError, match="empty"):
        arms.select_pids(IDS, value, None)


@pytest.mark.parametrize("value", ["B999", "B010,B999", "b010"])
def test_select_pids_unknown_id_is_an_error_not_silently_dropped(value):
    with pytest.raises(arms.PreflightError, match="not in queries.csv"):
        arms.select_pids(IDS, value, None)


def test_select_pids_subset_keeps_file_order_and_dedupes():
    assert arms.select_pids(IDS, "B020, B010,B020", None) == (["B010", "B020"], False)
    assert arms.select_pids(IDS, "B010", 200) == (["B010"], False)
    with pytest.raises(arms.PreflightError):
        arms.select_pids(IDS, "B010", 201)             # --expect-rows is checked for subsets too


def test_read_rows_rejects_a_bad_queries_file(tmp_path, monkeypatch):
    monkeypatch.setattr(arms, "INPUT", tmp_path / "queries.csv")
    with pytest.raises(arms.PreflightError, match="not found"):
        arms.read_rows()
    (tmp_path / "queries.csv").write_text("Paper ID,Latest query (v2)\nB1,q\n")
    with pytest.raises(arms.PreflightError, match="lacks column"):
        arms.read_rows()
    (tmp_path / "queries.csv").write_text("Paper ID,Latest query (v2),DOI,OpenAlex ID\nB1,q,d,w\nB1,q2,d2,w2\n")
    with pytest.raises(arms.PreflightError, match="duplicate"):
        arms.read_rows()


# ===================================================================== the real command line (subprocess) ===============================================================

def test_cli_starts_and_prints_help(tmp_path):
    r = run_script("run_benchmark.py", ["--help"], tmp_path / "run")
    assert r.returncode == 0 and "--expect-rows" in r.stdout and "--pids" in r.stdout, r.stderr[-1500:]


def test_cli_launch_exactly_as_documented_reaches_main(tmp_path):
    """Regression: `python run_benchmark.py CUR ...` used to die with AttributeError ('NoneType' has no attribute 'run_guarded') before main() ran."""
    run = tmp_path / "run"; write_queries(run, 5)
    r = run_script("run_benchmark.py", ["CUR"], run)
    assert "Traceback" not in r.stderr and "AttributeError" not in r.stderr, r.stderr[-1500:]
    assert r.returncode == 2 and "not started" in r.stderr and "--expect-rows" in r.stderr, r.stderr[-1500:]       # main() ran and refused: no row count
    assert not (run / "results").exists() and not (run / "run_config_CUR.json").exists()


def test_cli_refuses_stale_plans_before_any_work(tmp_path):
    run = tmp_path / "run"; write_queries(run, 5)
    for label, kw in (("no provenance", dict(drop_provenance=True)), ("different query text", dict(query_override="some other query"))):
        write_plans(run, **kw)
        r = run_script("run_benchmark.py", ["CUR", "--expect-rows", "5"], run, env_extra={"CGSCOUT_STOP_USD": "40"})
        assert r.returncode == 2 and "not started" in r.stderr and "B001" in r.stderr, (label, r.stderr[-1500:])
        assert not (run / "results").exists() and not (run / "run_config_CUR.json").exists(), label            # nothing was started, nothing spent


def test_cli_resume_with_everything_finished_exits_0_and_records_each_start(tmp_path):
    run = tmp_path / "run"; pids = write_queries(run, 5); write_plans(run)
    for pid in pids:
        write_result(run, pid, "OK", 0.09)
    for n in (1, 2):                                                   # a restart of a finished run: same code, same plans -> accepted, nothing re-run
        r = run_script("run_benchmark.py", ["CUR", "--expect-rows", "5"], run, env_extra={"CGSCOUT_STOP_USD": "40"})
        assert r.returncode == 0 and "DONE" in r.stdout, r.stdout[-1500:] + r.stderr[-1500:]
        assert len((run / "run_starts_CUR.jsonl").read_text().splitlines()) == n
    cfg = json.loads((run / "run_config_CUR.json").read_text())
    assert cfg["selection"]["full_run"] is True and cfg["selection"]["n_selected"] == 5 and cfg["stop_usd_explicit"] is True
    assert cfg["effective_config"]["budgets"] == EFFECTIVE_BUDGETS and cfg["effective_config"]["stage4"] == EFFECTIVE_STAGE4
    # B4 (consolidation 2b): the record of the entry defaults keeps its old key, keys, key order and values although the entry is now agent.retrieve
    ed = cfg["effective_config"]["entry_defaults"]
    assert list(ed["run_expert_prior_retrieval"].items()) == [("budgets", None), ("case_id", None), ("stage3_classifier_threshold", 0.5)]
    assert cfg["effective_config"]["stage3_classifier"]["threshold"] == 0.5 and set(ed) == {"run_expert_prior_retrieval", "run_stage4_v2"}


def test_cli_ceiling_already_reached_is_an_error_not_success(tmp_path):
    run = tmp_path / "run"; pids = write_queries(run, 5); write_plans(run)
    write_result(run, pids[0], "OK", 6.0)
    r = run_script("run_benchmark.py", ["CUR", "--expect-rows", "5"], run, env_extra={"CGSCOUT_STOP_USD": "5"})
    assert r.returncode == 4 and "INCOMPLETE" in r.stderr and "B002" in r.stderr, r.stdout[-1500:] + r.stderr[-1500:]
    assert not (run / "results" / "CUR" / "B002.json").exists()


# ===================================================================== in-process, stubbed pipeline =====================================================================

class Bench:
    def __init__(self, run, rb):
        self.run, self.rb = run, rb
        self.ran, self.cost, self.status = [], 0.0, {}           # status: pid -> "OK" | "ERROR" | ... | [status of attempt 1, attempt 2, ...]

    def main(self, *args):
        return self.rb.main(list(args))

    def config(self):
        return json.loads((self.run / "run_config_CUR.json").read_text())


@pytest.fixture
def bench(tmp_path, monkeypatch):
    import run_benchmark as rb
    import cgscout.retrieval.transfer as orch                                      # P3's expand_terminology is looked up in retrieval/transfer.py
    run = tmp_path / "run"; run.mkdir()
    for name, value in (("ROOT", run), ("INPUT", run / "queries.csv"), ("SHARED_CACHE", run / "cache_work"), ("PLANS", run / "plans_frozen")):
        monkeypatch.setattr(arms, name, value)
    for k in ("CGSCOUT_CACHE_DIR", "CGSCOUT_USAGE_LOG_PATH", "CGSCOUT_LLM_MODEL", "CGSCOUT_LLM_EFFORT", "PYTHONDONTWRITEBYTECODE"):
        monkeypatch.setenv(k, "x")                                  # configure() overwrites these; setting them first makes pytest restore the originals afterwards
    monkeypatch.setenv("OPENALEX_STRICT_BUDGET", "1")
    for k in ("CGSCOUT_STOP_USD", "NEW37_STOP_USD"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(orch, "expand_terminology", orch.expand_terminology)      # configure() wraps it with the terminology memo; restore afterwards
    b = Bench(run, rb)
    real_configure = rb.configure

    def configure(arm):
        real_configure(arm)                                           # the REAL configure: environment, imports, profile, ceiling, budgets
        rb.RL = types.SimpleNamespace(harness_guard=lambda *a, **k: None, run_guarded=lambda fn, *a, **k: fn(*a, **k))
        rb.U = types.SimpleNamespace(exhausted=lambda: None, budget_status=lambda force=False: {"state": "ok", "remaining_credits": 10 ** 6},
                                     ledger_path=lambda: run / "no_ledger", clear_exhausted=lambda: None, OpenAlexBudgetExhausted=type("OpenAlexBudgetExhausted", (Exception,), {}))
        rb.get_backend = lambda: types.SimpleNamespace(model=PAPER["llm"]["model"], default_effort=PAPER["llm"]["effort"])
        rb.backend_label = lambda backend: "STUB"

    def run_query(row, bw, s4cfg):
        pid = row["Paper ID"]; b.ran.append(pid)
        st = b.status.get(pid, "OK")
        if isinstance(st, list):
            st = st[min(b.ran.count(pid) - 1, len(st) - 1)]
        return {"pid": pid, "status": st, "seconds": 0.0, "llm": {"cost_usd": b.cost}, "seed": {}}

    monkeypatch.setattr(rb, "configure", configure)
    monkeypatch.setattr(rb, "run_query", run_query)
    return b


def _setup(bench, n, monkeypatch, stop=None):
    write_queries(bench.run, n); write_plans(bench.run)
    if stop is not None:
        monkeypatch.setenv("CGSCOUT_STOP_USD", str(stop))


def test_full_200_case_run_runs_every_case_once(bench, monkeypatch):
    _setup(bench, 200, monkeypatch, stop=100); bench.cost = 0.088
    assert bench.main("CUR", "--expect-rows", "200") == 0
    assert bench.ran == [pid_of(i) for i in range(1, 201)]
    cfg = bench.config()
    assert cfg["selection"] == {"full_run": True, "n_rows": 200, "n_selected": 200, "expect_rows": 200, "pids": None}
    assert len(list((bench.run / "results" / "CUR").glob("*.json"))) == 200


def test_a_run_cut_short_by_the_spend_ceiling_is_an_error_and_resumes_where_it_stopped(bench, monkeypatch, capsys):
    _setup(bench, 200, monkeypatch, stop=5); bench.cost = 0.088            # the built-in $5 would stop a 200-case run after 57 cases (see the next test)
    assert bench.main("CUR", "--expect-rows", "200") == 4
    first = list(bench.ran)
    assert len(first) == 57 and "INCOMPLETE: 143 of 200" in capsys.readouterr().err
    monkeypatch.setenv("CGSCOUT_STOP_USD", "100")                          # operator raises the ceiling and starts again
    bench.ran.clear()
    assert bench.main("CUR", "--expect-rows", "200") == 0
    assert not set(first) & set(bench.ran) and len(first) + len(bench.ran) == 200       # finished cases are never re-run, nothing is skipped
    assert len(list((bench.run / "results" / "CUR").glob("*.json"))) == 200
    assert len((bench.run / "run_starts_CUR.jsonl").read_text().splitlines()) == 2


def test_cases_still_failing_after_three_passes_are_an_error(bench, monkeypatch, capsys):
    _setup(bench, 3, monkeypatch, stop=40); bench.status = {"B002": "ERROR"}
    assert bench.main("CUR", "--expect-rows", "3") == 4
    assert bench.ran.count("B002") == 3 and bench.ran.count("B001") == 1 and bench.ran.count("B003") == 1
    err = capsys.readouterr().err
    assert "INCOMPLETE: 1 of 3" in err and "B002" in err


def test_a_degraded_case_that_succeeds_on_retry_completes_the_run(bench, monkeypatch):
    _setup(bench, 3, monkeypatch, stop=40); bench.status = {"B002": ["DEGRADED", "OK"]}
    assert bench.main("CUR", "--expect-rows", "3") == 0
    assert bench.ran.count("B002") == 2


def test_refused_and_not_run_clarification_are_final_results(bench, monkeypatch):
    _setup(bench, 3, monkeypatch, stop=40); bench.status = {"B001": "REFUSED", "B003": "NOT_RUN_CLARIFICATION"}
    assert bench.main("CUR", "--expect-rows", "3") == 0
    assert sorted(bench.ran) == ["B001", "B002", "B003"]                      # once each: neither is retried


def test_pids_subset_runs_only_those_cases_with_the_smoke_test_default_ceiling(bench, monkeypatch):
    _setup(bench, 200, monkeypatch)
    assert bench.main("CUR", "--pids", "B020,B010") == 0
    assert bench.ran == ["B010", "B020"]
    cfg = bench.config()
    assert cfg["selection"]["full_run"] is False and cfg["selection"]["pids"] == ["B010", "B020"] and cfg["stop_usd"] == 5.0 and cfg["stop_usd_explicit"] is False


@pytest.mark.parametrize("pids", ["", " , ", "B999", "B010,B999"])
def test_empty_or_unknown_pids_start_nothing(bench, monkeypatch, capsys, pids):
    _setup(bench, 200, monkeypatch, stop=100)
    assert bench.main("CUR", "--pids", pids) == 2
    assert bench.ran == [] and not (bench.run / "results").exists()
    assert "--pids" in capsys.readouterr().err


@pytest.mark.parametrize("args, stop, expected", [
    (["CUR"], 100, "--expect-rows"),                                    # a full run must say how many cases it expects
    (["CUR", "--expect-rows", "199"], 100, "has 200 rows"),             # a truncated / extended queries.csv
    (["CUR", "--expect-rows", "200"], None, "CGSCOUT_STOP_USD"),        # a full run must set its spend ceiling explicitly
    (["CUR", "--expect-rows", "200"], 0, "positive"),
])
def test_full_run_preconditions(bench, monkeypatch, capsys, args, stop, expected):
    _setup(bench, 200, monkeypatch, stop=stop)
    assert bench.main(*args) == 2
    assert bench.ran == [] and not (bench.run / "results").exists()
    assert expected in capsys.readouterr().err


def test_finished_cases_are_not_run_again(bench, monkeypatch):
    _setup(bench, 5, monkeypatch, stop=40)
    for pid in ("B001", "B002"):
        write_result(bench.run, pid, "OK")
    write_result(bench.run, "B003", "ERROR")                              # an ERROR checkpoint is retried
    assert bench.main("CUR", "--expect-rows", "5") == 0
    assert bench.ran == ["B003", "B004", "B005"]


# --- configuration: PAPER is the single source and the record shows every effective value -----------------------------------------------------------------------------------

def test_effective_configuration_is_the_paper_profile_and_the_evaluated_cur_run(bench, monkeypatch):
    _setup(bench, 2, monkeypatch, stop=40)
    assert bench.main("CUR", "--expect-rows", "2") == 0
    cfg = bench.config(); eff = cfg["effective_config"]
    # the values the evaluated CUR run used: every retrieval / Stage-4 field, the classifier file, the LLM
    assert eff["budgets"] == EFFECTIVE_BUDGETS
    assert eff["stage4"] == EFFECTIVE_STAGE4
    assert eff["llm"] == {"model": "gpt-5.6-luna", "default_effort": "medium", "lift_explicit_low_effort": True, "lift_target_effort": "medium"}
    assert eff["stage3_classifier"] == {"profile": "shipped", "file": "sim_classifier_model.joblib", "sha256": CLASSIFIER_SHA256, "threshold": 0.5}
    assert eff["profile"] == "paper" and cfg["profile"] == "paper" and cfg["classifier"] == "shipped"
    assert cfg["budgets"] == PAPER["budgets"] and PAPER["stage4"].items() <= eff["stage4"].items()
    # ... and what differs from the library defaults is exactly what the profile names
    assert set(cfg["differs_from_library_defaults"]["budgets"]) == {"need_semantic_enabled"}
    assert set(cfg["differs_from_library_defaults"]["stage4"]) == {"screen_effort", "extract_effort", "deep_cap"}
    # the legacy keys of the original run_config are still there, with the original values
    assert cfg["stage4_config"] == {"screen_effort": "medium", "extract_effort": "medium", "judge_effort": "medium", "deep_cap": 500, "screen_batch_size": 12,
                                    "reject_min_confidence": "medium", "workers": 6}
    assert cfg["model"] == "gpt-5.6-luna" and cfg["default_effort"] == "medium" and cfg["effective_config_sha256"] and cfg["code_sha256"]
    # environment the pipeline modules read
    import os
    assert os.environ["CGSCOUT_LLM_MODEL"] == PAPER["llm"]["model"] and os.environ["CGSCOUT_LLM_EFFORT"] == PAPER["llm"]["effort"]


def test_paper_profile_is_the_single_source_of_every_setting(bench, monkeypatch):
    """Change the profile -> the run (environment, budgets, Stage-4 config, effort lifting, the record) follows; nothing is written twice in the runner."""
    import os
    monkeypatch.setitem(PAPER["llm"], "model", "other-model"); monkeypatch.setitem(PAPER["llm"], "effort", "high"); monkeypatch.setitem(PAPER["llm"], "lift_low_effort_to_medium", False)
    monkeypatch.setitem(PAPER["stage4"], "deep_cap", 123); monkeypatch.setitem(PAPER["stage4"], "judge_effort", "high"); monkeypatch.setitem(PAPER["stage4"], "screen_abstract_chars", 1500)
    monkeypatch.setitem(PAPER["budgets"], "max_papers_per_query", 33)
    _setup(bench, 2, monkeypatch, stop=40)                                 # plans are stamped with the (patched) profile model / effort
    assert bench.main("CUR", "--expect-rows", "2") == 0
    eff = bench.config()["effective_config"]
    assert os.environ["CGSCOUT_LLM_MODEL"] == "other-model" and os.environ["CGSCOUT_LLM_EFFORT"] == "high"
    assert bench.rb.DEEP_CAP == 123
    assert eff["budgets"]["max_papers_per_query"] == 33
    assert eff["stage4"]["deep_cap"] == 123 and eff["stage4"]["judge_effort"] == "high" and eff["stage4"]["screen_abstract_chars"] == 1500
    assert eff["llm"]["lift_explicit_low_effort"] is False and eff["llm"]["lift_target_effort"] == "high"
    assert arms.budgets_for("CUR", __import__("cgscout.retrieval.direct.route_plan", fromlist=["x"])).max_papers_per_query == 33


def test_effort_unification_wrapper_follows_the_profile(bench):
    rb = bench.rb
    seen = []
    inner = types.SimpleNamespace(structured=lambda **kw: seen.append(kw.get("effort")), model="m")
    w = rb.EffortUnifiedBackend(inner)                                      # defaults: lift 'low' -> 'medium' (what the evaluated runs did)
    w.structured(effort="low"); w.structured(effort="high"); w.structured(effort=None)
    assert seen == ["medium", "high", None] and w.overrides == 1 and w.calls == 3 and w.model == "m"
    seen.clear(); w2 = rb.EffortUnifiedBackend(inner, lift_to="high"); w2.structured(effort="low")
    w3 = rb.EffortUnifiedBackend(inner, enabled=False); w3.structured(effort="low")
    assert seen == ["high", "low"] and w3.overrides == 0


def test_a_restart_with_changed_configuration_or_code_is_flagged(bench, monkeypatch, capsys):
    _setup(bench, 2, monkeypatch, stop=40)
    assert bench.main("CUR", "--expect-rows", "2") == 0
    monkeypatch.setitem(PAPER["budgets"], "max_papers_per_query", 33)       # configuration drift between two starts of the same run directory
    assert bench.main("CUR", "--expect-rows", "2") == 0
    out = capsys.readouterr().out
    assert "WARNING: effective_config_sha256 differs from the previous start" in out
    starts = [json.loads(line) for line in (bench.run / "run_starts_CUR.jsonl").read_text().splitlines()]
    assert len(starts) == 2 and starts[0]["effective_config_sha256"] != starts[1]["effective_config_sha256"] and starts[0]["code_sha256"] == starts[1]["code_sha256"]


def test_arm_is_derived_from_the_profile_and_stage4_config_helper():
    from cgscout.assessment.stage4 import Stage4Config
    cur = arms.ARMS["CUR"]
    assert cur["profile"] == "paper" and arms.profile_for("CUR") is PAPER
    assert cur["budgets"] == PAPER["budgets"] and cur["screen_chars"] == PAPER["stage4"]["screen_abstract_chars"] and cur["classifier"] == PAPER["stage3_classifier"]
    assert dataclasses.asdict(arms.stage4_config("CUR")) == EFFECTIVE_STAGE4 == dataclasses.asdict(Stage4Config(**PAPER["stage4"]))
