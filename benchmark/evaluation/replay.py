"""Offline golden replay: run the paper-profile pipeline (retrieval -> Stage 3 -> Stage 4) for a few cases against a FIXED fixture, with
  * every LLM call forbidden (a cache miss raises and is recorded),
  * every network connection forbidden (recorded),
  * the Stage-3 classifier replaced by a fixed one built from the stored per-paper `stage3_kept` flags,
  * P3 (terminology expansion) replayed from terminology_memo.json, exactly as run_benchmark.py does.
The fixture (NOT in the repository; build it with build_replay_fixture.py) holds: queries.csv, plans_frozen/, terminology_memo.json, cache_work/, results/CUR/ (stored
per-case results, used only for the classifier stub) and golden/cases/ (the expected outputs).  Nothing in the fixture is modified: the cache is copied.
usage: python replay.py <fixture_dir> <out_dir> <pid,pid,...>     (cgscout must be importable: pip install . or PYTHONPATH=src)
Writes <out_dir>/cases/<pid>.json (byte-compared against <fixture>/golden/cases/<pid>.json by tests/test_golden_replay.py) and <out_dir>/summary.json.
"""
import sys, os, json, re, time, shutil, socket, dataclasses, csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEW37 = Path(sys.argv[1]).resolve()               # the fixture directory (kept under this historical name below)
OUT = Path(sys.argv[2]).resolve()
PIDS = sys.argv[3].split(",")
if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "cases").mkdir(parents=True)

# ---- fixture: private copy of the cache with refreshed timestamps (24h TTL on openalex_request / search_and_dedupe) ----
CACHE = OUT / "cache"
shutil.copytree(NEW37 / "cache_work", CACHE, ignore=shutil.ignore_patterns("*.lock", "openalex_usage.jsonl", "openalex_semantic.ts", "openalex_budget_state.json"))
_now = time.time()
for p in CACHE.rglob("*.json"):
    try:
        d = json.loads(p.read_text())
        if isinstance(d, dict) and "_cached_at" in d:
            d["_cached_at"] = _now
            p.write_text(json.dumps(d))
    except Exception:
        pass
BEFORE = {str(p.relative_to(CACHE)) for p in CACHE.rglob("*") if p.is_file()}

os.environ.update(CGSCOUT_CACHE_DIR=str(CACHE), CGSCOUT_USAGE_LOG_PATH=str(OUT / "llm_usage.jsonl"), CGSCOUT_LLM_MODEL="gpt-5.6-luna",
                  CGSCOUT_LLM_EFFORT="medium", PYTHONDONTWRITEBYTECODE="1", OPENALEX_STRICT_BUDGET="1")
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))                      # harness.py, recorder.py, arms.py
os.environ["CGSCOUT_BENCH_DIR"] = str(NEW37)       # arms.py reads queries / frozen plans from the fixture directory

# ---- no network at all ----
NET = []
def _blocked(*a, **k):
    NET.append(repr(a)[:120]); raise ConnectionError("replay: network access is forbidden")
socket.socket.connect = _blocked
socket.getaddrinfo = _blocked

import harness as RL                             # noqa: E402  (was regression_lib)
import cgscout.retrieval.direct.route_plan as RP         # noqa: E402
import cgscout.retrieval.orchestrator as TB              # noqa: E402
from cgscout.retrieval.related_systems.terminology import TerminologyExpansion   # noqa: E402
from cgscout.planning.expert_prior_planner import ExpertPriorOutput    # noqa: E402
from cgscout.retrieval.direct.from_planner import evidence_plan_from_planner_output   # noqa: E402
from cgscout.planning.scientific_planner import PlannerOutput     # noqa: E402
from recorder import Recorder                        # noqa: E402  (benchmark/evaluation/recorder.py)
import arms as AR                                  # noqa: E402  (benchmark/evaluation/arms.py)
U = RL.U

BUDGETS = AR.budgets_for("CUR", RP)          # both from cgscout.config.PAPER, like run_benchmark.py
S4CFG = AR.stage4_config("CUR")


def norm_doi(d):
    d = (d or "").strip().lower()
    for p in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "http://dx.doi.org/", "doi:"):
        if d.startswith(p): d = d[len(p):]
    return d.rstrip(".,;")

def pkey(p):
    return norm_doi(getattr(p, "doi", None)) or (getattr(p, "title", "") or "").lower()

def is_seed(p, seed_doi, seed_oa):
    d = norm_doi(getattr(p, "doi", None))
    oa = (getattr(getattr(p, "external_ids", None), "openalex_id", None) or "").split("/")[-1]
    return bool((d and d == seed_doi) or (oa and oa == seed_oa))


class StrictBackend:
    name = "openai"; model = "gpt-5.6-luna"; default_effort = "medium"
    def __init__(self): self.attempts = []
    def structured(self, *, system, user, schema, label, effort=None, purpose=None, case_id=None, timeout=None):
        self.attempts.append({"label": label, "case_id": case_id, "user_head": user[:100]})
        raise RuntimeError(f"replay: uncached LLM call attempted ({label})")

BACKEND = StrictBackend()

MEMO = json.loads((NEW37 / "terminology_memo.json").read_text())
MEMO_MISS = []
def _memo_expand(system_term, *, call):
    if system_term in MEMO:
        return TerminologyExpansion.model_validate(MEMO[system_term])
    MEMO_MISS.append(system_term); raise RuntimeError("replay: terminology memo miss")
TB.expand_terminology = _memo_expand

STUB, UNKNOWN = {}, []
def _stub_classify(papers):
    out = []
    for p in papers:
        k = pkey(p)
        if k not in STUB: UNKNOWN.append(k); out.append(0.5)
        else: out.append(0.99 if STUB[k] else 0.01)
    return out
_orig_filter = TB.filter_papers
TB.filter_papers = lambda papers, **kw: _orig_filter(papers, classify_fn=_stub_classify, **kw)

_VOL = re.compile(r"(seconds|elapsed|^ts$|_ts$|latency|timestamp|started|duration)", re.I)
def ser(o):
    if hasattr(o, "model_dump"): return ser(o.model_dump(mode="json"))
    if dataclasses.is_dataclass(o) and not isinstance(o, type): return ser(dataclasses.asdict(o))
    if isinstance(o, dict): return {str(k): ser(v) for k, v in o.items() if not _VOL.search(str(k))}
    if isinstance(o, (list, tuple)): return [ser(x) for x in o]
    if isinstance(o, (set, frozenset)): return sorted((ser(x) for x in o), key=lambda x: json.dumps(x, sort_keys=True))
    if o is None or isinstance(o, (str, int, float, bool)): return o
    return repr(o)


rows = {r["Paper ID"]: r for r in csv.DictReader(open(NEW37 / "queries.csv", encoding="utf-8-sig"))}
summary = {"fixture": str(NEW37), "cases": {}}
for pid in PIDS:
    row = rows[pid]; fp = AR.frozen_plan(pid); stored = json.load(open(NEW37 / "results" / "CUR" / f"{pid}.json"))
    STUB.clear(); STUB.update({(norm_doi(e["doi"]) or (e["title"] or "").lower()): bool(e["stage3_kept"]) for e in stored["pool"]})
    seed_doi, seed_oa = norm_doi(row["DOI"]), row["OpenAlex ID"]
    run_label = f"phase1_{pid}"      # the label is part of the recorded output (run["case_id"]): keep it, the golden files contain it
    eff = {k: fp["planner"][k] for k in ("system", "target_component", "scientific_objective", "search_context", "search_axes")}
    ep = ExpertPriorOutput.model_validate(fp["expert_prior"])
    po = PlannerOutput(**{k: fp["planner"].get(k) for k in eff}, clarification_needed=False)
    plan = RP.build_route_retrieval_plan(evidence_plan_from_planner_output(po), BUDGETS)
    roles = {}
    for q in plan.route_queries: roles.setdefault(q.query, []).append([q.route, q.query_role])
    REC = Recorder(lambda: U.run_usage(run_label), lambda p_: is_seed(p_, seed_doi, seed_oa), roles).install()
    n_llm0, n_net0, n_unk0 = len(BACKEND.attempts), len(NET), len(UNKNOWN)
    err = None
    try:
        run, agent_result = RL.run_plan_checked(BACKEND, eff, run_label, {"seed": [seed_doi]}, user_request=fp.get("user_request", row["Latest query (v2)"]),
                                  stage4=True, stage4_config=S4CFG, expert_prior=ep, budgets=BUDGETS, max_tries=1)
    except BaseException as e:   # noqa: BLE001
        run, agent_result, err = None, None, repr(e)
    finally:
        REC.uninstall()
    res_, s4_ = (agent_result.retrieval, agent_result.assessment) if agent_result is not None else (None, None)
    out = {"pid": pid, "error": err, "run": ser(run), "res": ser(res_), "s4": ser(s4_), "requests": ser(REC.records)}
    (OUT / "cases" / f"{pid}.json").write_text(json.dumps(out, sort_keys=True, ensure_ascii=False, indent=1))
    summary["cases"][pid] = {"error": err, "llm_attempts": len(BACKEND.attempts) - n_llm0, "net_attempts": len(NET) - n_net0, "unknown_classifier_papers": len(UNKNOWN) - n_unk0,
                             "n_requests": len(REC.records), "degraded": (run or {}).get("degraded"), "n_retained": len(getattr(res_, "retained", []) or [])}
    print(pid, summary["cases"][pid], flush=True)

summary["llm_attempt_labels"] = sorted({a["label"] for a in BACKEND.attempts})
summary["memo_miss"] = sorted(set(MEMO_MISS))
summary["net_samples"] = NET[:3]
summary["loaded_repo_modules"] = sorted(n for n in sys.modules if n == "cgscout" or n.startswith("cgscout."))
summary["mellea_loaded"] = "mellea" in sys.modules
summary["new_cache_files"] = sorted(str(p.relative_to(CACHE)) for p in CACHE.rglob("*") if p.is_file() and str(p.relative_to(CACHE)) not in BEFORE)
(OUT / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
print("loaded repo modules:", len(summary["loaded_repo_modules"]), "| mellea loaded:", summary["mellea_loaded"], "| new cache files:", len(summary["new_cache_files"]), "| llm attempts:", len(BACKEND.attempts), "| net attempts:", len(NET))
