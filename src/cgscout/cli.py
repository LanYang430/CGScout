"""cgscout command-line entry point (PlannerSession -> retrieval -> Stage 3 ->
Stage 4). The old A/B/Guard/Gate/C/D intake path (chat_with_beam.py,
concept_resolver_session.py) is historical code kept in the git tag
`legacy-pre-cgscout`.

Live path (cgscout.agent.run does the work; this module is only the terminal loop and the printing):
    PlannerSession (one ScientificPlanner call + thin router)
    -> ExpertPriorPlanner.propose()                         [agent.propose_prior]
    -> retrieval (direct routes, 3A, 3B, merge) + Stage 3   [agent.retrieve]
    -> Stage 4 assessment                                   [agent.assess, only when something was retained]

Stage 4 (assessment/stage4.py): an LLM evaluates the scientific relevance of every
Stage-3-retained paper against the user's problem (graded 0-3, evidence-based
explanation with a verbatim supporting quote) and the output is ranked by that
assessment -- retrieval route and publication year only break exact ties.
`--no-stage4` falls back to the old behaviour: the Stage-3 list ordered by
rank_for_display() (route tier + year), which is NOT a relevance ranking.

    cgscout [--case-id ID] [--no-stage4]        (or: python -m cgscout.cli)
"""
from __future__ import annotations

import argparse


from cgscout import agent  # noqa: E402
from cgscout.planning.session import PlannerSession  # noqa: E402
from cgscout.llm import get_backend, backend_label, LLMBackend  # noqa: E402
from cgscout.planning.scientific_planner import PlannerOutput  # noqa: E402
from cgscout.retrieval.orchestrator import rank_for_display  # noqa: E402

BANNER = (
    "BEAM (scientific-planner-v2 -- Plan B live path)\n"
    "Type 'quit' to exit, 'new' to start a new session.\n\n"
    "BEAM: Hi! I'm an agent for discovering protein coarse-grained models "
    "and related simulation literature.\n"
)


def _print_plan(plan: PlannerOutput) -> None:
    print(f"BEAM: (plan) system={plan.system!r}"
          + (f" target_component={plan.target_component!r}" if plan.target_component else ""))
    print(f"      objective={plan.scientific_objective!r}")
    if plan.search_context:
        print(f"      search_context={plan.search_context!r}")
    if plan.search_axes:
        print(f"      search_axes={plan.search_axes!r}")


def print_result(out: agent.AgentResult) -> None:
    result, s4 = out.retrieval, out.assessment
    if s4 is not None:                 # Stage 4 ran: the gate said yes (it was asked for and something was retained)
        w = s4.workload
        print("\nBEAM: Evidence for your specific problem (ranked by scientific relevance)")
        print(f"  {len(result.direct_papers) + len(result.transfer_papers)} candidates -> {w['n_retained']} kept by the cheap filter -> "
              f"{w['n_recommended']} recommended, {w['n_background']} background, {w['n_rejected']} judged not relevant"
              + (f", {len(s4.errors)} could not be evaluated" if s4.errors else ""))
        for a in s4.recommended:
            j = a.judgment
            print(f"\n  [{a.rank}] {a.title}  ({a.year})  grade {a.final_grade}/3"
                  + (f"  [{j.evidence_type.replace('_', ' ')}]" if j else ""))
            if a.doi:
                print(f"      doi={a.doi}")
            if j:
                print(f"      why: {j.explanation}")
                print(f"      evidence: \"{j.supporting_quote}\"" + ("" if a.quote_grounded else "  (quote not verified in text)"))
        if s4.background:
            print("\n  Background (weaker applicability):")
            for a in s4.background[:10]:
                print(f"    - {a.title} ({a.year})")
        return
    print("\nBEAM: Evidence for your specific problem")
    ranked = rank_for_display(result.retained)
    if not ranked:
        print("  (no candidates found across direct or transfer evidence)")
        return
    print(f"  {len(result.direct_papers) + len(result.transfer_papers)} candidates -> "
          f"{len(result.retained)} worth reading ({len(result.rejected)} filtered for lacking "
          f"explicit CG evidence)\n")
    for i, p in enumerate(ranked, 1):
        print(f"  [{i}] {p.title}")
        meta = f"      {getattr(p, 'year', None)}"
        venue = getattr(p, "venue", None)
        if venue:
            meta += f"  {venue}"
        doi = getattr(p, "doi", None)
        if doi:
            meta += f"  doi={doi}"
        print(meta)
        print(f"      found via: {', '.join(p.query_categories)}")
        if p.matched_keywords:
            print(f"      CG evidence: {', '.join(p.matched_keywords)}")
        print()


def handle_turn(session: PlannerSession, text: str, backend: LLMBackend, *, case_id: str | None = None,
                use_stage4: bool = True) -> None:
    out = agent.run(text, backend=backend, session=session, case_id=case_id,
                    gate=None if use_stage4 else (lambda res: False),      # default gate: Stage 4 only when something was retained
                    on_plan=_print_plan)
    if isinstance(out, agent.Clarification):
        print(f"BEAM: {out.question}")
    else:
        print_result(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="BEAM v2 -- scientific-planner live path")
    parser.add_argument("--case-id", default=None,
                         help="tag every LLM call this session makes in the usage log")
    parser.add_argument("--no-stage4", action="store_true",
                        help="skip the Stage 4 LLM relevance evaluation (old Stage-3 list ordered by route tier + year)")
    args = parser.parse_args()

    # One backend, resolved ONCE for this entire process -- threaded into
    # every module that needs an LLM call (ScientificPlanner,
    # ExpertPriorPlanner, Hop-1 related-system extraction, terminology
    # expansion) for the rest of this run, including across "new" sessions.
    # No module below this point calls get_backend() itself. See llm_
    # backend.get_backend()'s own docstring on the "one run, one backend"
    # invariant.
    backend = get_backend()
    print(BANNER)
    print(f"LLM backend: {backend_label(backend)}")
    session = PlannerSession(backend=backend, case_id=args.case_id)
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in ("quit", "exit"):
            break
        if text.lower() == "new":
            session = PlannerSession(backend=backend, case_id=args.case_id)
            print("BEAM: (started a new session)")
            continue
        try:
            handle_turn(session, text, backend, case_id=args.case_id, use_stage4=not args.no_stage4)
        except Exception as e:  # noqa: BLE001 -- surface, don't crash the loop
            print(f"BEAM: (error) {e}")


if __name__ == "__main__":
    raise SystemExit(main())
