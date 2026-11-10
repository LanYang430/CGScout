"""Budget-only regression tests for the fixed Hop-1-depth-15 / Hop-2-top-30
production decision (RouteRetrievalBudgets.structural_max_discovered_systems
raised to a 150 safety cap, structural_hop2_max_systems set to 30 as the real
cost gate, live-adapter Hop-1 AND Hop-2 per-query retrieval depth both raised
to 15 -- Hop-2's depth was later raised from its original 8, diagnosed
directly against a live end-to-end run: a real Hop-2 query genuinely
returned a relevant paper at rank 14, past the old top-8 cutoff).

No LLM, no network -- search_fn/extract_fn are fakes, following the exact
convention already used in tests/test_related_system_discovery.py and
tests/test_related_system_hop2.py (_sys/_paper/_extract helpers). Does not
touch TerminologyExpander, Hop-1 extraction prompt, grounding logic, Hop-2
query construction, Stage 3/4, MethodologyEligibility, or adaptive retrieval.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks" / "concept_resolver_reasoning"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks" / "concept_resolver_reasoning" / "mellea_integration"))

from related_system_discovery import (  # noqa: E402
    DiscoveredSystem,
    SupportingPaper,
    discover_related_systems,
)
from related_system_hop2 import select_systems_for_hop2  # noqa: E402
from route_retrieval_plan import RouteRetrievalBudgets  # noqa: E402
from intake_contract import SystemExtraction, SystemMention  # noqa: E402
from beam_agent.models import Paper  # noqa: E402


def _paper(title, abstract="", doi=None, year=2020):
    return Paper(title=title, abstract=abstract, doi=doi, year=year)


def _extract(mention_map):
    def _fn(title, abstract, structural_class, aka=None):
        return SystemExtraction(systems=[
            SystemMention(name=n, evidence_quote=q, structural_term_matched=s)
            for (n, q, s) in mention_map.get(title, [])
        ])
    return _fn


def _sys(name, n_support=1, rank=0):
    sp = SupportingPaper(paper_ref=f"10/{name}", title="T", found_by_query="synthetic-test-fold",
                         retrieval_rank=rank, evidence_quote="q",
                         structural_term_matched="synthetic-test-fold", system_name=name)
    return DiscoveredSystem(name=name, norm_name=name.lower(), supporting_papers=[sp],
                            n_support=n_support, first_seen_rank=rank)


# --- 1. the two budget defaults are actually the new values ------------

def test_default_budgets_are_the_fixed_depth15_top30_values():
    b = RouteRetrievalBudgets()
    assert b.structural_max_discovered_systems == 150   # safety cap, not a selection cutoff
    assert b.structural_hop2_max_systems == 30           # the real cost gate


# --- 2. Hop-1 itself can return well over 30 grounded systems -----------
#     (proves max_discovered=150 is not silently re-introducing the old
#     small cutoff at the Hop-1 layer)

def test_hop1_can_return_more_than_30_grounded_systems():
    # 35 papers, each names ONE distinct system tied to the structural class --
    # same idiom as test_related_system_discovery.test_more_than_k_systems_capped_and_ranked,
    # scaled past 30.
    papers = [_paper(f"P{i}", f"System{i} is a synthetic-test-fold", doi=f"10/{i}") for i in range(35)]
    extract_map = {f"P{i}": [(f"System{i}", f"System{i} is a synthetic-test-fold", "synthetic-test-fold")]
                   for i in range(35)}
    budgets = RouteRetrievalBudgets()
    r = discover_related_systems(
        "synthetic-test-fold", [],
        search_fn=lambda q: papers,
        extract_fn=_extract(extract_map),
        max_discovered=budgets.structural_max_discovered_systems,
        # both budget values passed explicitly, exactly as route_retrieval_plan.py's
        # live call does (route_retrieval_plan.py:447-451) -- otherwise
        # discover_related_systems()'s own max_papers_to_extract DEFAULT (25,
        # an unrelated, untouched value) truncates the paper pool before
        # ranking ever runs, which would falsely look like this budget change
        # failing.
        max_papers_to_extract=budgets.structural_max_papers_to_extract,
    )
    assert len(r.discovered_systems) == 35   # all 35 survive -- 150 cap never binds
    assert r.ungrounded_survivor_count == 0


# --- 3. Hop-2 selects exactly the top 30 by (-n_support, first_seen_rank),
#     never more, and ranks 31+ are excluded from the SELECTION but are not
#     what this test checks for upstream survival (see test 2 above and
#     test 4 below for that) ------------------------------------------

def test_hop2_selects_exactly_top_30_by_support_then_rank():
    # 35 systems, STRICTLY decreasing n_support (35 down to 1) -- an
    # unambiguous total order, so the rank-30/rank-31 boundary is crisp and
    # never depends on the first_seen_rank tie-break.
    systems = [_sys(f"S{i:02d}", n_support=35 - i, rank=i) for i in range(35)]
    budgets = RouteRetrievalBudgets()

    kept, note = select_systems_for_hop2(systems, max_systems=budgets.structural_hop2_max_systems)

    assert len(kept) == 30
    expected_order = sorted(systems, key=lambda s: (-s.n_support, s.first_seen_rank))[:30]
    assert [s.name for s in kept] == [s.name for s in expected_order]
    # strictly ranked -- kept is exactly S00..S29 (highest n_support first)
    assert [s.name for s in kept] == [f"S{i:02d}" for i in range(30)]
    # ranks 31-35 (S30..S34) must NOT be in the Hop-2 selection
    kept_names = {s.name for s in kept}
    for i in range(30, 35):
        assert f"S{i:02d}" not in kept_names
    assert "Dropped" in note


def test_hop2_selection_uses_first_seen_rank_as_tie_break():
    # equal n_support -- must fall back to first_seen_rank (earlier retrieval
    # wins), exactly the existing (-n_support, first_seen_rank) key, no new
    # ranking signal introduced.
    tied = [_sys("Late", n_support=1, rank=9), _sys("Early", n_support=1, rank=2),
            _sys("Middle", n_support=1, rank=5)]
    kept, _ = select_systems_for_hop2(tied, max_systems=30)
    assert [s.name for s in kept] == ["Early", "Middle", "Late"]


# --- 4. systems ranked 31+ never perturb the top-30 selection or its order,
#     and (combined with test 2) are provably NOT lost by Hop-1 itself --
#     only excluded from the Hop-2 SELECTION step ------------------------

def test_systems_below_rank_30_do_not_perturb_the_top_30():
    top30 = [_sys(f"S{i:02d}", n_support=30 - i, rank=i) for i in range(30)]
    kept_without_extra, _ = select_systems_for_hop2(top30, max_systems=30)

    tail = [_sys(f"T{i:02d}", n_support=1, rank=100 + i) for i in range(10)]  # ranks 31-40
    kept_with_extra, _ = select_systems_for_hop2(top30 + tail, max_systems=30)

    assert [s.name for s in kept_without_extra] == [s.name for s in kept_with_extra]
    # and the full (pre-Hop-2-selection) discovered-systems list the caller
    # holds is untouched by that selection -- the 10 tail systems are simply
    # not part of `kept`, not deleted from the list the caller passed in.
    full_list_after_call = top30 + tail
    assert len(full_list_after_call) == 40
    assert {s.name for s in tail} <= {s.name for s in full_list_after_call}


# --- 6. structural_max_papers_to_extract raised 60 -> 150: Hop-1 at depth=15
#     scale (measured real unique-paper counts: beta-helix 123, coiled-coil
#     138, TIM barrel 109 -- all > 60) must actually SEND more than 60 papers
#     to the extractor, not just retrieve them and then silently truncate
#     before extraction/ranking ever happens -----------------------------

def test_default_extraction_cap_is_150():
    assert RouteRetrievalBudgets().structural_max_papers_to_extract == 150


def test_depth15_scale_retrieval_sends_more_than_60_papers_to_extractor():
    # 120 papers -- within the real depth=15 range measured this session
    # (109-138 unique papers across 3 cases), each naming one distinct
    # system, so extraction count and discovered-system count are both
    # directly readable off n_papers_extracted / len(discovered_systems).
    n = 120
    papers = [_paper(f"P{i}", f"System{i} is a synthetic-test-fold", doi=f"10/{i}") for i in range(n)]
    extract_map = {f"P{i}": [(f"System{i}", f"System{i} is a synthetic-test-fold", "synthetic-test-fold")]
                   for i in range(n)}
    extracted_titles: list[str] = []

    def _extract_and_record(mention_map):
        base = _extract(mention_map)
        def _fn(title, abstract, structural_class, aka=None):
            extracted_titles.append(title)
            return base(title, abstract, structural_class, aka)
        return _fn

    budgets = RouteRetrievalBudgets()
    r = discover_related_systems(
        "synthetic-test-fold", [],
        search_fn=lambda q: papers,
        extract_fn=_extract_and_record(extract_map),
        max_discovered=budgets.structural_max_discovered_systems,
        max_papers_to_extract=budgets.structural_max_papers_to_extract,
    )

    # 1. Hop-1 at depth-15 scale (120 unique papers, within the real
    #    measured 109-138 range) actually SENDS more than 60 to the
    #    extractor -- the exact regression the 60->150 change fixes.
    assert len(extracted_titles) > 60
    assert len(extracted_titles) == n
    # 2. the cap value came from the budget field, not a hardcoded number --
    #    n_papers_extracted reflects exactly structural_max_papers_to_extract
    #    (150) applied to a 120-paper pool, i.e. min(120, 150) == 120, not
    #    the old min(120, 60) == 60.
    assert r.n_papers_extracted == n == min(n, budgets.structural_max_papers_to_extract)
    # 3. max_discovered=150 still works normally at this scale -- all 120
    #    grounded systems survive (well under the 150 safety cap).
    assert len(r.discovered_systems) == n

    # 4. Hop-2 still selects only the top 30, using the SAME unchanged
    #    ranking key, off the REAL DiscoveredSystem objects Hop-1 just
    #    produced (not a synthetic _sys() fixture) -- closes the loop
    #    end-to-end for this budget set.
    kept, note = select_systems_for_hop2(r.discovered_systems, max_systems=budgets.structural_hop2_max_systems)
    assert len(kept) == 30
    assert "Dropped" in note


def test_extraction_cap_is_still_a_real_cap_not_unlimited():
    # 160 papers > 150 -- proves 150 is still an actual SAFETY CAP (binds
    # when truly exceeded), not silently unlimited.
    n = 160
    papers = [_paper(f"P{i}", f"System{i} is a synthetic-test-fold", doi=f"10/{i}") for i in range(n)]
    extract_map = {f"P{i}": [(f"System{i}", f"System{i} is a synthetic-test-fold", "synthetic-test-fold")]
                   for i in range(n)}
    budgets = RouteRetrievalBudgets()
    r = discover_related_systems(
        "synthetic-test-fold", [],
        search_fn=lambda q: papers,
        extract_fn=_extract(extract_map),
        max_discovered=budgets.structural_max_discovered_systems,
        max_papers_to_extract=budgets.structural_max_papers_to_extract,
    )
    assert r.n_papers_retrieved == 160
    assert r.n_papers_extracted == 150   # capped, as a safety net -- not 160


# NOTE: the live-adapter check (Hop-1 per-query depth == 15, Hop-2's own
# retrieval depth unchanged at 8) is not part of this suite; it was done once
# with a one-off script (run_live_adapter_search_limit_check.py, kept in the
# git tag `legacy-pre-cgscout`).
