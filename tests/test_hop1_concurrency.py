"""Unit tests for this session's Hop-1 bounded-concurrency addition:
discover_related_systems(..., max_workers=N) runs extract_fn calls
concurrently (concurrent.futures.ThreadPoolExecutor) while the grounding/
merge step always stays sequential, in the ORIGINAL to_extract order --
first_seen_rank/supporting_papers order/ranking/provenance must never
depend on completion order, concurrent or not.

No LLM, no network -- extract_fn is a fake (optionally sleeping, to prove
genuine concurrency happened, not just that the parameter was accepted).
Reuses the exact _paper/_extract helper convention already used in
tests/test_related_system_discovery.py.
"""
from __future__ import annotations

import random
import time


from cgscout.retrieval.related_systems.discovery import discover_related_systems  # noqa: E402
from cgscout.retrieval.related_systems.extraction import SystemExtraction, SystemMention  # noqa: E402
from cgscout.models import Paper  # noqa: E402


def _paper(title, abstract="", doi=None, year=2020):
    return Paper(title=title, abstract=abstract, doi=doi, year=year)


def _search(mapping):
    return lambda q: mapping.get(q, [])


def _extract_with_delay(mention_map, delay_by_title=None, fail_titles=None):
    """Same convention as test_related_system_discovery.py's `_extract()`,
    plus an optional per-title artificial delay (to prove real concurrency)
    and an optional set of titles that raise (to prove per-call failure
    isolation under concurrency)."""
    delay_by_title = delay_by_title or {}
    fail_titles = fail_titles or set()

    def _fn(title, abstract, structural_class, aka=None):
        if title in fail_titles:
            raise RuntimeError(f"simulated failure for {title!r}")
        time.sleep(delay_by_title.get(title, 0))
        return SystemExtraction(systems=[
            SystemMention(name=n, evidence_quote=q, structural_term_matched=s)
            for (n, q, s) in mention_map.get(title, [])
        ])
    return _fn


def _make_scenario(n_papers: int):
    """n_papers, each naming ONE distinct system, ranked 0..n_papers-1 by
    retrieval order (matches how to_extract is ordered in production)."""
    papers = [_paper(f"Paper{i}", f"System{i} is a synthetic-test-fold, paper {i}", doi=f"10/{i}")
              for i in range(n_papers)]
    mention_map = {
        f"Paper{i}": [(f"System{i}", f"System{i} is a synthetic-test-fold", "synthetic-test-fold")]
        for i in range(n_papers)
    }
    search_fn = _search({"synthetic-test-fold": papers, "synthetic-test-fold structure": []})
    return papers, mention_map, search_fn


# --- correctness: concurrent output must be identical to sequential -------

def test_concurrent_result_identical_to_sequential_multi_system():
    _papers, mention_map, search_fn = _make_scenario(12)
    extract_fn = _extract_with_delay(mention_map)

    seq = discover_related_systems("synthetic-test-fold", [], search_fn=search_fn, extract_fn=extract_fn,
                                   max_papers_to_extract=25, max_discovered=25, max_workers=1)
    conc = discover_related_systems("synthetic-test-fold", [], search_fn=search_fn, extract_fn=extract_fn,
                                    max_papers_to_extract=25, max_discovered=25, max_workers=4)

    assert len(seq.discovered_systems) == len(conc.discovered_systems) == 12
    seq_shape = [(d.name, d.n_support, d.first_seen_rank,
                  tuple(sp.paper_ref for sp in d.supporting_papers)) for d in seq.discovered_systems]
    conc_shape = [(d.name, d.n_support, d.first_seen_rank,
                   tuple(sp.paper_ref for sp in d.supporting_papers)) for d in conc.discovered_systems]
    assert seq_shape == conc_shape, "concurrent run must produce a byte-identical result, order included"
    assert seq.raw_candidate_names == conc.raw_candidate_names
    assert seq.ungrounded_survivor_count == conc.ungrounded_survivor_count == 0


def test_concurrent_result_identical_with_randomized_completion_order():
    """Deliberately makes LATER-ranked papers finish FIRST (via delay), so a
    naive implementation that merged in completion order (instead of
    original to_extract order) would produce a visibly different result."""
    _papers, mention_map, search_fn = _make_scenario(8)
    # earlier-ranked papers sleep LONGER, so completion order is reversed
    delay_by_title = {f"Paper{i}": (8 - i) * 0.01 for i in range(8)}
    extract_fn = _extract_with_delay(mention_map, delay_by_title=delay_by_title)

    seq = discover_related_systems("synthetic-test-fold", [], search_fn=search_fn, extract_fn=extract_fn,
                                   max_papers_to_extract=25, max_discovered=25, max_workers=1)
    conc = discover_related_systems("synthetic-test-fold", [], search_fn=search_fn, extract_fn=extract_fn,
                                    max_papers_to_extract=25, max_discovered=25, max_workers=4)

    seq_order = [d.name for d in seq.discovered_systems]
    conc_order = [d.name for d in conc.discovered_systems]
    assert seq_order == conc_order, "merge order must stay tied to to_extract order, not completion order"
    assert [d.first_seen_rank for d in conc.discovered_systems] == [d.first_seen_rank for d in seq.discovered_systems]


# --- genuine concurrency (not just parameter acceptance) -------------------

def test_workers_2_is_measurably_faster_than_workers_1_for_slow_calls():
    n = 6
    delay = 0.15
    _papers, mention_map, search_fn = _make_scenario(n)
    delay_by_title = {f"Paper{i}": delay for i in range(n)}
    extract_fn = _extract_with_delay(mention_map, delay_by_title=delay_by_title)

    t0 = time.time()
    discover_related_systems("synthetic-test-fold", [], search_fn=search_fn, extract_fn=extract_fn,
                             max_papers_to_extract=25, max_discovered=25, max_workers=1)
    seq_time = time.time() - t0

    t0 = time.time()
    discover_related_systems("synthetic-test-fold", [], search_fn=search_fn, extract_fn=extract_fn,
                             max_papers_to_extract=25, max_discovered=25, max_workers=2)
    conc_time = time.time() - t0

    assert conc_time < seq_time * 0.75, (
        f"workers=2 ({conc_time:.2f}s) should be meaningfully faster than workers=1 "
        f"({seq_time:.2f}s) for {n} calls x {delay}s each -- concurrency isn't actually happening"
    )


# --- failure isolation under concurrency -----------------------------------

def test_one_failing_call_does_not_affect_others_under_concurrency():
    _papers, mention_map, search_fn = _make_scenario(6)
    extract_fn = _extract_with_delay(mention_map, fail_titles={"Paper3"})

    result = discover_related_systems("synthetic-test-fold", [], search_fn=search_fn, extract_fn=extract_fn,
                                      max_papers_to_extract=25, max_discovered=25, max_workers=3)

    assert len(result.discovered_systems) == 5  # Paper3's System3 never contributed
    assert not any(d.name == "System3" for d in result.discovered_systems)
    assert len(result.extraction_errors) == 1
    assert "Paper3" in result.extraction_errors[0] or "10/3" in result.extraction_errors[0]


def test_progress_fn_fires_exactly_once_per_paper_under_concurrency():
    _papers, mention_map, search_fn = _make_scenario(10)
    extract_fn = _extract_with_delay(mention_map, fail_titles={"Paper2", "Paper7"})
    calls: list[tuple[int, int]] = []

    discover_related_systems("synthetic-test-fold", [], search_fn=search_fn, extract_fn=extract_fn,
                             max_papers_to_extract=25, max_discovered=25, max_workers=4,
                             progress_fn=lambda done, total: calls.append((done, total)))

    assert len(calls) == 10  # once per paper, success or failure
    assert all(total == 10 for _done, total in calls)
    assert sorted(done for done, _total in calls) == list(range(1, 11))  # every count 1..10 exactly once


# --- default behavior unchanged --------------------------------------------

def test_default_max_workers_is_sequential():
    """No max_workers passed -- must reproduce the exact previous
    (sequential, no ThreadPoolExecutor at all) behavior."""
    _papers, mention_map, search_fn = _make_scenario(5)
    extract_fn = _extract_with_delay(mention_map)
    result = discover_related_systems("synthetic-test-fold", [], search_fn=search_fn, extract_fn=extract_fn,
                                      max_papers_to_extract=25, max_discovered=25)
    assert len(result.discovered_systems) == 5
