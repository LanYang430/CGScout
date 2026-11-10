"""Render data/pilot_v0.jsonl as a readable markdown sheet for the human pick (title hidden, so the choice is about
how natural the question sounds, not about how well it matches the paper).  Fill `selected_candidate` (1/2/3) or
`selected_query` in data/<name>.csv afterwards."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME = sys.argv[1] if len(sys.argv) > 1 else "pilot_v1"
recs = [json.loads(l) for l in (HERE / "data" / f"{NAME}.jsonl").read_text().splitlines() if l.strip()]
out = ["# BEAM-bench v1 pilot -- pick one question per case (judge NATURALNESS, not fit to the paper)\n"]
for r in recs:
    need = r["need_card"]["specific_need" if r["case_type"] == "L1" else "abstracted_need"]
    out.append(f"## {r['case_id']}  {r['case_type']}  ({r['domain']})")
    if r["case_type"] == "L3":
        ev = r.get("l3_evidence") or {}
        out.append(f"- seed system X: {r['transfer_source_system']}  ->  researcher's system Y: **{r['transfer_target_system']}**  "
                   f"(verified_no_direct={r['verified_no_direct']}; papers checked={ev.get('n_unique_papers_judged')}, transfer axis={ev.get('transfer_axis')}, confidence={ev.get('confidence')})")
    out.append(f"- need: {need['system_class']} | {need['scientific_process']}")
    for i, (q, f) in enumerate(zip(r["query_candidates"], r["leakage_flags"]), 1):
        out.append(f"{i}. {q}" + (f"  **[flags: {', '.join(f)}]**" if f else ""))
    out.append("")
(HERE / "data" / f"{NAME}_review.md").write_text("\n".join(out))
print("wrote", HERE / "data" / f"{NAME}_review.md")
