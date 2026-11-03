"""Complex-objective validation of the canonical schema (Case 6 protein-RNA
vesicles, Case 9 protein-DNA). Offline experiment; production untouched.

Stages (CLI):  capture | canon | retention | retrieval <A|B>
  capture   : 8 fresh ScientificPlanner runs per case -> results/co_planner_raw.json
  canon     : canonicalize every ok record under variants A and B -> results/co_canon.json
  retention : does the downstream-visible text still carry each stated
              condition/constraint? raw fields vs canonical-rendered fields
  retrieval : full end-to-end retrieval (incl. 3B where chosen), raw vs
              canonical input, first 3 ok records per case
"""
import sys, json, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
RES = HERE / "results"; RES.mkdir(exist_ok=True)

import regression_lib as RL  # noqa: E402  (also puts mellea_integration on sys.path)
from llm_backend import get_backend, backend_label  # noqa: E402
from planner_session import PlannerSession  # noqa: E402
from canon_lib import (normalize_system, VARIANTS, build_user_message, render_for_expert_prior)  # noqa: E402

CASES = {
    "case6_protein_rna_vesicles": dict(
        message="I want to model internal organization and morphology of multiphase protein-RNA condensates, not just whether they phase separate. What CG approaches could do this?",
        targets={"Protein-RNA vesicles (JACS Au 2026)": ["10.1021/jacsau.6c00322"]}),
    "case9_protein_dna": dict(
        message="I want to predict sequence-specific protein-DNA recognition and binding affinities while retaining a simulation-based physical model. What CG approaches could be suitable?",
        targets={"Protein-DNA structure-sequence (eLife 2025)": ["10.7554/elife.105565", "10.1101/2024.05.26.595895"]}),
}
N_REPS = 8
N_RETRIEVAL = 3

# What the user actually stated, as checks over the text a downstream consumer sees.
def _has(rx):
    return lambda t: bool(re.search(rx, t, re.I))

CHECKS = {
    "case6_protein_rna_vesicles": {
        "internal organization": _has(r"organi[sz]ation|internal structure|spatial"),
        "morphology": _has(r"morpholog"),
        "multiphase": _has(r"multi[- ]?phase|multiple phases|coexisting phases"),
        "protein + RNA": lambda t: bool(re.search(r"\bRNA\b", t)) and bool(re.search(r"protein", t, re.I)),
        "CONTRAST: more than whether they phase separate":
            lambda t: bool(re.search(r"not just|not only|beyond|rather than|more than|besides|in addition to|going beyond|whether", t, re.I))
            and bool(re.search(r"phase[- ]separat", t, re.I)),
    },
    "case9_protein_dna": {
        "sequence-specific": _has(r"sequence[- ]specific|sequence[- ]dependent|sequence"),
        "recognition": _has(r"recogni"),
        "binding affinities": _has(r"affinit"),
        "protein + DNA": lambda t: bool(re.search(r"\bDNA\b", t)) and bool(re.search(r"protein", t, re.I)),
        "CONSTRAINT: simulation-based physical model": _has(r"physical|physics|simulation[- ]based|mechanistic"),
    },
}


def stage_capture():
    backend = get_backend()
    print(f"LLM backend: {backend_label(backend)}", flush=True)
    out = {}
    for short, c in CASES.items():
        out[short] = []
        for rep in range(1, N_REPS + 1):
            s = PlannerSession(backend=backend, case_id=f"co_{short}_rep{rep}")
            t = s.submit_user_message(c["message"])
            if t.status == "awaiting_user":
                rec = {"rep": rep, "status": "awaiting_clarification", "clarification_question": t.display_question_to_user}
            else:
                p = t.plan
                rec = {"rep": rep, "status": "ok", "system": p.system, "target_component": p.target_component,
                       "scientific_objective": p.scientific_objective, "search_context": p.search_context,
                       "search_axes": p.search_axes}
            out[short].append(rec)
            print(f"{short} rep{rep}: {json.dumps(rec, ensure_ascii=False)}", flush=True)
        (RES / "co_planner_raw.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))


def stage_canon():
    backend = get_backend()
    raw = json.loads((RES / "co_planner_raw.json").read_text())
    prev = json.loads((RES / "co_canon.json").read_text()) if (RES / "co_canon.json").exists() else {}
    out = {}
    for short, runs in raw.items():
        out[short] = []
        for rec in runs:
            if rec["status"] != "ok":
                out[short].append({"rep": rec["rep"], "status": rec["status"]}); continue
            norm = normalize_system(rec["system"])
            row = next((r for r in prev.get(short, []) if r.get("rep") == rec["rep"] and r.get("status") == "ok"), None) \
                or {"rep": rec["rep"], "status": "ok", "raw": rec, "norm": norm}
            for v, (schema, prompt) in VARIANTS.items():
                if v in row:
                    row[v]["rendered"] = render_for_expert_prior(norm, row[v]["canon"], rec)  # re-render with fixed renderer
                    continue
                canon = backend.structured(system=prompt, user=build_user_message(rec, norm), schema=schema,
                                           label=f"canonical_{v}_complex", purpose="offline_schema_validation",
                                           case_id=f"co_canon_{short}_{rec['rep']}_{v}").model_dump()
                row[v] = {"canon": canon, "rendered": render_for_expert_prior(norm, canon, rec)}
            out[short].append(row)
            print(f"{short} rep{rec['rep']} variants={[v for v in VARIANTS if v in row]}", flush=True)
        (RES / "co_canon.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))


def _visible(d):  # text a downstream consumer is handed
    return " ".join([d["system"], d["scientific_objective"], " ".join(d.get("search_context") or [])])


def stage_retention():
    d = json.loads((RES / "co_canon.json").read_text())
    for short, rows in d.items():
        ok = [r for r in rows if r["status"] == "ok"]
        n_clar = len(rows) - len(ok)
        print(f"\n{'='*100}\n{short}: {len(ok)} ok / {n_clar} clarification\n{'='*100}")
        arms = {"RAW": lambda r: _visible(r["raw"]), "CANON-A": lambda r: _visible(r["A"]["rendered"]),
                "CANON-B": lambda r: _visible(r["B"]["rendered"]), "CANON-C": lambda r: _visible(r["C"]["rendered"])}
        print(f"{'check':50s} " + " ".join(f"{a:>8s}" for a in arms))
        for name, fn in CHECKS[short].items():
            cells = []
            for a, getter in arms.items():
                cells.append(f"{sum(fn(getter(r)) for r in ok)}/{len(ok)}")
            print(f"{name:50s} " + " ".join(f"{c:>8s}" for c in cells))
        for r in ok:
            print(f"\n rep{r['rep']}")
            print(f"   RAW obj : {r['raw']['scientific_objective']}")
            print(f"   A render: {r['A']['rendered']['scientific_objective']}")
            print(f"   B render: {r['B']['rendered']['scientific_objective']}")
            print(f"   C render: {r['C']['rendered']['scientific_objective']}")


def stage_retrieval(variant):
    import os as _os
    n_arms = len(_os.environ.get("ARMS", "raw,canonical").split(","))
    a0, b0 = (int(x) for x in _os.environ.get("REP_SLICE", f"0:{N_RETRIEVAL}").split(":"))
    RL.harness_guard(len(CASES) * (b0 - a0) * n_arms * 25, what="complex_objective")
    backend = get_backend()
    print(f"LLM backend: {backend_label(backend)}  canonical variant={variant}", flush=True)
    d = json.loads((RES / "co_canon.json").read_text())
    results = []
    import os
    arms = os.environ.get("ARMS", "raw,canonical").split(",")
    a_, b_ = (int(x) for x in os.environ.get("REP_SLICE", f"0:{N_RETRIEVAL}").split(":"))
    tag = os.environ.get("TAG", "")
    outp = RES / f"co_retrieval_{variant}_{sys.argv[3] if len(sys.argv) > 3 else 'all'}{tag}.json"
    only = sys.argv[3] if len(sys.argv) > 3 else None
    for short, rows in d.items():
        if only and only not in short:
            continue
        ok = [r for r in rows if r["status"] == "ok"][a_:b_]
        for row in ok:
            for arm in arms:
                fields = {k: row["raw"][k] for k in ("system", "target_component", "scientific_objective", "search_context")} \
                    if arm == "raw" else row[variant]["rendered"]
                cid = f"co_{short}_{arm}_{variant}_{row['rep']}"
                print(f"\n{'='*90}\n{short} {arm} rep{row['rep']}\n input={fields}", flush=True)
                try:
                    r = RL.run_plan(backend, fields, cid, CASES[short]["targets"])
                except Exception as e:  # noqa: BLE001
                    import traceback; traceback.print_exc()
                    r = {"case_id": cid, "error": str(e)}
                r.update({"case": short, "arm": arm, "rep": row["rep"]})
                results.append(r)
                print(json.dumps({k: r.get(k) for k in ("expert_prior", "targets", "pool", "retained", "error")}, ensure_ascii=False, indent=1), flush=True)
                outp.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "capture": stage_capture()
    elif stage == "canon": stage_canon()
    elif stage == "retention": stage_retention()
    elif stage == "retrieval": RL.run_guarded(stage_retrieval, sys.argv[2])
    print("DONE", flush=True)
