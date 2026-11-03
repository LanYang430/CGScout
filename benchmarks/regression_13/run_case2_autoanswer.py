"""One-off Case 2 run: if the (unmodified) ScientificPlanner asks 'specific IDP vs generic CG models', answer with
the fixed pre-agreed reply and continue the full pipeline (retrieval -> Stage 3 -> Stage 4). Harness-only script:
no change to the planner or any pipeline stage.  Result is stored like a regression label (summary works on it)."""
import json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import regression_lib as RL  # noqa: E402
from planner_session import PlannerSession  # noqa: E402
from llm_backend import get_backend  # noqa: E402

U = RL.U
LABEL = "s4_case2_autoanswer"
ANSWER = "No specific IDP sequence; search for generic CG models for temperature-driven / LCST-like phase separation."


def main():
    c = next(x for x in json.loads((HERE / "cases.json").read_text())["cases"] if x["id"] == 2)
    targets = {k: v["dois"] for k, v in c["targets"].items()}
    RL.harness_guard(40, what=f"regression:{LABEL}")
    out_dir = HERE / "results" / LABEL
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = get_backend()
    run_label = f"reg_{LABEL}_c2_r1"
    rec = {"case": 2, "short": c["short"], "rep": 1, "query": c["query"], "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "clarification": None}
    sess = PlannerSession(backend=backend, case_id=run_label)
    t = sess.submit_user_message(c["query"])
    if t.status == "awaiting_user":
        q = t.display_question_to_user
        print("PLANNER ASKED:", q, flush=True)
        ql = q.lower()
        if not ("generic" in ql and ("specific" in ql or "sequence" in ql or "which idp" in ql)):
            rec["status"] = "awaiting_clarification"; rec["planner"] = {"status": "awaiting_clarification", "question": q}
            rec["clarification"] = {"question": q, "answered": False, "why": "not the pre-agreed 'specific IDP vs generic' question"}
            (out_dir / "case2_rep1.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False)); print("different question -> stopped"); return
        t = sess.submit_user_message(ANSWER)
        rec["clarification"] = {"question": q, "answered": True, "answer": ANSWER, "status_after": t.status}
        print("ANSWERED ->", t.status, flush=True)
    else:
        print("planner did not ask for clarification", flush=True)
    if t.status != "ready":
        rec["status"] = "awaiting_clarification"; rec["planner"] = {"status": "awaiting_clarification", "question": t.display_question_to_user}
    else:
        p = t.plan
        rec["planner"] = {"status": "ok", "system": p.system, "target_component": p.target_component, "scientific_objective": p.scientific_objective,
                          "search_context": p.search_context, "search_axes": p.search_axes}
        rec["status"] = "ran"
        rec["planner_effective"] = {k: rec["planner"][k] for k in ("system", "target_component", "scientific_objective", "search_context")}
        rec["run"] = RL.run_plan_checked(backend, rec["planner_effective"], run_label, targets, user_request=c["query"], stage4=True)
    (out_dir / "case2_rep1.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False))
    print("-> stored", out_dir / "case2_rep1.json")


if __name__ == "__main__":
    RL.run_guarded(main)
