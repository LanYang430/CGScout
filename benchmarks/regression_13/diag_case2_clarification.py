"""Case 2 diagnosis (LLM-only, no retrieval, production code untouched): how often does the ScientificPlanner ask
for clarification on the Case 2 query, what exactly does it say (scientific_ambiguity / target_component /
clarification_question), and does answering it yield a plan that retrieval can use directly?"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import regression_lib  # noqa: F401,E402  (sets up sys.path for the mellea_integration modules)
from planner_session import PlannerSession  # noqa: E402
from llm_backend import get_backend  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
START = int(sys.argv[2]) if len(sys.argv) > 2 else 1
OUT = sys.argv[3] if len(sys.argv) > 3 else 'diag_case2_clarification.json'
query = next(c for c in json.loads((HERE / "cases.json").read_text())["cases"] if c["id"] == 2)["query"]
ANSWER = "No specific IDP -- I want generic coarse-grained models that can capture IDP phase separation that is driven by increasing temperature (LCST-like)."
backend = get_backend()
out = []
for i in range(START, START + N):
    s = PlannerSession(backend=backend, case_id=f"diag_case2_{i}")
    t = s.submit_user_message(query)
    p = t.plan
    rec = {"draw": i, "status": t.status, "system": p.system, "target_component": p.target_component,
           "scientific_objective": p.scientific_objective, "scientific_ambiguity": p.scientific_ambiguity,
           "clarification_needed": p.clarification_needed, "clarification_question": p.clarification_question,
           "search_context": p.search_context, "search_axes": p.search_axes}
    if t.status == "awaiting_user":
        t2 = s.submit_user_message(ANSWER)
        p2 = t2.plan
        rec["after_answer"] = {"status": t2.status, "system": p2.system, "scientific_objective": p2.scientific_objective,
                               "search_context": p2.search_context, "clarification_needed": p2.clarification_needed,
                               "clarification_question": p2.clarification_question}
    out.append(rec)
    print(f"draw {i}: {t.status}", flush=True)
(HERE / "results" / OUT).write_text(json.dumps({"query": query, "answer_used": ANSWER, "draws": out}, indent=2, ensure_ascii=False))
n = sum(r["status"] == "awaiting_user" for r in out)
print(f"\nclarification asked in {n}/{N} draws")
