"""Unit tests for postprocess.py (cleanup section) -- the conservative Hop-1 post-processing
layer (suffix duplicate merge / exact fragment blocklist / objective-aware
interaction-duplicate merge) added this session. Runs strictly between
discover_related_systems() and the Hop-2 query building -- does NOT touch either;
these tests exercise cleanup_discovered_systems() directly against
DiscoveredSystem fixtures, following the same _sys()-helper convention
already used in tests/test_route_retrieval_budgets.py.

Fixture data below reproduces the REAL 20-system beta-helix/folding sample
this session diagnosed (see the session's own controlled Hop-1 schema
comparison), so these tests are a direct regression check against a real
observed case, not just synthetic examples.
"""
from __future__ import annotations


from cgscout.retrieval.related_systems.discovery import DiscoveredSystem, SupportingPaper  # noqa: E402
from cgscout.retrieval.related_systems.postprocess import cleanup_discovered_systems  # noqa: E402


def _sys(name, rank=0, n_support=1, quote=None):
    sp = SupportingPaper(paper_ref=f"ref-{name}-{rank}", title="T", found_by_query="q",
                         retrieval_rank=rank, evidence_quote=quote or f"quote for {name}",
                         structural_term_matched="", system_name=name)
    return DiscoveredSystem(name=name, norm_name=name.lower(), supporting_papers=[sp],
                            n_support=n_support, first_seen_rank=rank)


_REAL_SAMPLE = [
    ("PKHD1", 0), ("Chlamydia", 4), ("pylori", 4), ("double-stranded", 12),
    ("HET-s(218–289)", 27), ("LytA", 39), ("CRD", 40), ("short repeats (SRs)", 40),
    ("long repeats (LRs)", 40), ("Polymers", 59), ("CdrA", 84), ("TPS", 88),
    ("RIV", 91), ("XadA", 94), ("pertactin", 94), ("P22 tailspike", 95),
    ("EspP-BamA interaction", 100), ("EspP", 105),
    ("of the Escherichia coli Autotransporter EspP", 110), ("vtaA", 112),
]


def _real_sample():
    return [_sys(name, rank) for name, rank in _REAL_SAMPLE]


# --- end-to-end against the real observed 20-system sample -----------------

def test_real_sample_folding_objective_predicted_outcome():
    cleaned, log = cleanup_discovered_systems(_real_sample(), clean_objective="folding")
    names = {d.name for d in cleaned}

    # exactly 3 removed: 1 fragment drop, 1 duplicate merge, 1 interaction merge
    assert len(_real_sample()) - len(cleaned) == 3
    assert "double-stranded" not in names
    assert "of the Escherichia coli Autotransporter EspP" not in names
    assert "EspP-BamA interaction" not in names
    assert "EspP" in names

    espp = next(d for d in cleaned if d.name == "EspP")
    assert espp.n_support == 3   # its own + the title-residue duplicate + the interaction candidate
    assert espp.first_seen_rank == 100  # min(105, 100, 110)
    assert len(espp.supporting_papers) == 3

    # explicit boundary items must survive untouched
    for kept in ("CRD", "TPS", "Polymers", "short repeats (SRs)", "long repeats (LRs)",
                 "Chlamydia", "pylori", "pertactin"):
        assert kept in names, f"{kept!r} must be left alone -- not this module's decision to make"

    assert len(log) == 3


def test_real_sample_binding_objective_does_not_merge_interaction_candidate():
    """The exact same fixture, but an interaction-type objective must leave
    'EspP-BamA interaction' standalone -- interaction may BE the thing being
    studied."""
    cleaned, log = cleanup_discovered_systems(_real_sample(), clean_objective="protein-protein binding")
    names = {d.name for d in cleaned}
    assert "EspP-BamA interaction" in names
    assert not any(l.startswith("interaction_merge") for l in log)
    # the objective-independent rules (fragment + suffix duplicate) still fire
    assert "double-stranded" not in names
    assert "of the Escherichia coli Autotransporter EspP" not in names


# --- 1. suffix duplicate merge, isolated -----------------------------------

def test_suffix_duplicate_merges_into_existing_short_name():
    systems = [_sys("EspP", rank=105), _sys("of the Escherichia coli Autotransporter EspP", rank=110)]
    cleaned, log = cleanup_discovered_systems(systems, clean_objective="")
    assert [d.name for d in cleaned] == ["EspP"]
    assert cleaned[0].n_support == 2
    assert cleaned[0].first_seen_rank == 105
    assert len(log) == 1 and log[0].startswith("duplicate_merge:")


def test_suffix_merge_target_must_meet_minimum_length():
    """A candidate ending with a very short (<4 alnum chars) existing name
    must NOT merge into it -- avoids trivial/short accidental suffix
    collisions becoming merge targets."""
    systems = [_sys("RIV", rank=0), _sys("Some Other RIV", rank=1)]
    cleaned, log = cleanup_discovered_systems(systems, clean_objective="")
    assert {d.name for d in cleaned} == {"RIV", "Some Other RIV"}
    assert log == []


def test_suffix_merge_requires_word_boundary_not_raw_substring():
    """"PreEspPX" containing "EspP" as a raw substring but NOT at a word
    boundary must not merge."""
    systems = [_sys("EspP", rank=0), _sys("PreEspPX", rank=1)]
    cleaned, log = cleanup_discovered_systems(systems, clean_objective="")
    assert {d.name for d in cleaned} == {"EspP", "PreEspPX"}
    assert log == []


def test_suffix_merge_does_not_fire_between_unrelated_names():
    systems = [_sys("short repeats (SRs)", rank=40), _sys("long repeats (LRs)", rank=40)]
    cleaned, log = cleanup_discovered_systems(systems, clean_objective="")
    assert {d.name for d in cleaned} == {"short repeats (SRs)", "long repeats (LRs)"}
    assert log == []


# --- 2. exact fragment blocklist, isolated ----------------------------------

def test_fragment_blocklist_drops_only_exact_known_cases():
    systems = [_sys("related", rank=12), _sys("double-stranded", rank=12), _sys("CRD", rank=40)]
    cleaned, log = cleanup_discovered_systems(systems, clean_objective="")
    assert {d.name for d in cleaned} == {"CRD"}
    assert len(log) == 2
    assert all(l.startswith("fragment_drop:") for l in log)


def test_fragment_blocklist_does_not_drop_on_short_length_or_caps_or_domain_words():
    """Explicit non-goal: short names, ALL-CAPS names, and names containing
    'domain'/'repeat'/'polymer' must survive untouched."""
    systems = [
        _sys("CRD", rank=40), _sys("TPS", rank=88), _sys("Polymers", rank=59),
        _sys("short repeats (SRs)", rank=40), _sys("long repeats (LRs)", rank=40),
        _sys("some domain", rank=1),
    ]
    cleaned, _log = cleanup_discovered_systems(systems, clean_objective="")
    assert {d.name for d in cleaned} == {s.name for s in systems}


def test_single_stranded_is_not_preemptively_blocked():
    """Per the explicit scoping decision: 'single-stranded' is a predictable
    sibling of the observed 'double-stranded' case but was never itself
    observed, so it must NOT be blocked yet."""
    systems = [_sys("single-stranded", rank=1)]
    cleaned, log = cleanup_discovered_systems(systems, clean_objective="")
    assert [d.name for d in cleaned] == ["single-stranded"]
    assert log == []


# --- 3. objective-aware interaction merge, isolated -------------------------

def test_interaction_merge_only_for_monomer_structural_objectives():
    systems = [_sys("EspP", rank=105), _sys("EspP-BamA interaction", rank=100)]
    for obj in ("folding", "unfolding", "conformational stability"):
        cleaned, log = cleanup_discovered_systems(list(systems), clean_objective=obj)
        assert {d.name for d in cleaned} == {"EspP"}, f"expected merge for objective={obj!r}"
        assert any(l.startswith("interaction_merge") for l in log)


def test_interaction_merge_skipped_for_interaction_type_objectives():
    systems = [_sys("EspP", rank=105), _sys("EspP-BamA interaction", rank=100)]
    for obj in ("binding", "protein-protein interaction", "translocation", "docking"):
        cleaned, log = cleanup_discovered_systems(list(systems), clean_objective=obj)
        assert {d.name for d in cleaned} == {"EspP", "EspP-BamA interaction"}, f"unexpected merge for objective={obj!r}"
        assert log == []


def test_interaction_merge_skipped_for_unrecognized_or_empty_objective():
    """Safer default when the objective can't be confidently classified:
    do NOT merge."""
    systems = [_sys("EspP", rank=105), _sys("EspP-BamA interaction", rank=100)]
    for obj in ("", "some unrelated objective text"):
        cleaned, log = cleanup_discovered_systems(list(systems), clean_objective=obj)
        assert {d.name for d in cleaned} == {"EspP", "EspP-BamA interaction"}
        assert log == []


def test_interaction_merge_leaves_standalone_interaction_candidate_with_no_known_system():
    """An interaction-shaped candidate that names no ALREADY-DISCOVERED
    system must be left alone -- nothing to merge it into."""
    systems = [_sys("SystemX-SystemY interaction", rank=0)]
    cleaned, log = cleanup_discovered_systems(systems, clean_objective="folding")
    assert [d.name for d in cleaned] == ["SystemX-SystemY interaction"]
    assert log == []


# --- ordering / re-ranking ---------------------------------------------------

def test_result_is_re_ranked_by_n_support_then_first_seen_rank_after_merging():
    """Merging can raise a survivor's n_support -- the SAME existing
    (-n_support, first_seen_rank) tie-break must be re-applied afterward so
    Hop-2's system selection sees a correctly ordered list."""
    systems = [
        _sys("Alpha", rank=0, n_support=1),
        _sys("Beta", rank=1, n_support=1),
        _sys("of the Alpha", rank=50, n_support=1),  # will merge into Alpha, bumping it to n_support=2
    ]
    cleaned, _log = cleanup_discovered_systems(systems, clean_objective="")
    assert [d.name for d in cleaned] == ["Alpha", "Beta"]
    assert cleaned[0].n_support == 2


def test_provenance_is_never_silently_lost():
    """Every merge/drop must produce exactly one audit log line; supporting
    papers are consolidated (never discarded) on a merge. `_real_sample()`
    is called TWICE deliberately: cleanup_discovered_systems() merges by
    mutating target.supporting_papers IN PLACE, so measuring "before" from
    the SAME objects after the call would double-count -- the merge
    sources' own (untouched) lists still exist even though they're no
    longer in `cleaned`."""
    total_supporting_before = sum(len(s.supporting_papers) for s in _real_sample())
    cleaned, log = cleanup_discovered_systems(_real_sample(), clean_objective="folding")
    assert len(log) == len(_real_sample()) - len(cleaned)
    total_supporting_after = sum(len(s.supporting_papers) for s in cleaned)
    # the 1 fragment drop's paper is gone; the 2 merges' papers are preserved
    # (consolidated onto their targets), so exactly 1 supporting paper is lost
    assert total_supporting_before - total_supporting_after == 1
