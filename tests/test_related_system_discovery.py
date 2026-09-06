"""Pure unit tests for related_system_discovery deterministic grounding/merge.
No LLM, no network -- search_fn and extract_fn are fakes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks" / "concept_resolver_reasoning"))

from related_system_discovery import (  # noqa: E402
    build_discovery_queries,
    discover_related_systems,
)
from intake_contract import SystemExtraction, SystemMention  # noqa: E402
from beam_agent.models import Paper  # noqa: E402


def _paper(title, abstract="", doi=None, year=2020):
    return Paper(title=title, abstract=abstract, doi=doi, year=year)


def _search(mapping):
    return lambda q: mapping.get(q, [])


def _extract(mention_map):
    """paper.title -> list[SystemMention tuples]"""
    def _fn(title, abstract, structural_class, aka=None):
        return SystemExtraction(systems=[
            SystemMention(name=n, evidence_quote=q, structural_term_matched=s)
            for (n, q, s) in mention_map.get(title, [])
        ])
    return _fn


# --- query building -----------------------------------------------------

def test_discovery_queries_are_objective_free():
    qs = build_discovery_queries("200-aa beta-helix protein", ["beta-solenoid", "beta-roll"])
    joined = " ".join(qs).lower()
    assert "folding" not in joined and "coarse" not in joined and "200" not in joined
    assert "beta-helix protein" in qs and "beta-solenoid" in qs
    assert "beta-solenoid structure" in qs
    assert len(qs) == len(set(qs))


# --- grounding ---------------------------------------------------------

def test_well_grounded_mention_is_discovered_with_provenance():
    papers = [_paper("Pertactin beta-helix folding mechanism",
                     "The passenger domain of pertactin forms a right-handed parallel beta-helix.",
                     doi="10.1/x")]
    r = discover_related_systems(
        "beta-helix protein", ["beta-solenoid"],
        search_fn=_search({"beta-helix protein": papers}),
        extract_fn=_extract({"Pertactin beta-helix folding mechanism": [
            ("pertactin", "pertactin forms a right-handed parallel beta-helix", "beta-helix"),
        ]}),
    )
    assert r.ungrounded_survivor_count == 0
    assert len(r.discovered_systems) == 1
    ds = r.discovered_systems[0]
    assert ds.name == "pertactin" and ds.n_support == 1
    sp = ds.supporting_papers[0]
    assert sp.paper_ref == "10.1/x"
    assert sp.evidence_quote == "pertactin forms a right-handed parallel beta-helix"
    assert sp.name_in_quote is True


def test_quote_not_in_paper_is_dropped():
    papers = [_paper("Some beta-helix paper", "This is about beta-helix proteins in general.")]
    r = discover_related_systems(
        "beta-helix", [],
        search_fn=_search({"beta-helix": papers}),
        extract_fn=_extract({"Some beta-helix paper": [
            ("HET-s", "HET-s prion forms a beta-solenoid", "beta-solenoid"),  # not in paper text
        ]}),
    )
    assert r.discovered_systems == []
    assert any("verbatim substring of the paper" in d.reason for d in r.dropped)


def test_name_not_in_paper_is_dropped():
    papers = [_paper("T", "a right-handed parallel beta-helix is described here")]
    r = discover_related_systems(
        "beta-helix", [],
        search_fn=_search({"beta-helix": papers}),
        extract_fn=_extract({"T": [
            ("Pertactin", "a right-handed parallel beta-helix", "beta-helix"),  # quote ok, name absent
        ]}),
    )
    assert r.discovered_systems == []
    assert any("name is not a verbatim substring" in d.reason for d in r.dropped)


def test_generic_and_class_names_are_dropped():
    papers = [_paper("T", "the protein forms a beta-helix; this beta-helix protein is common")]
    r = discover_related_systems(
        "beta-helix protein", ["beta-solenoid"],
        search_fn=_search({"beta-helix protein": papers}),
        extract_fn=_extract({"T": [
            ("the protein", "the protein forms a beta-helix", "beta-helix"),
            ("beta-helix protein", "this beta-helix protein is common", "beta-helix"),
        ]}),
    )
    assert r.discovered_systems == []
    reasons = " ".join(d.reason for d in r.dropped)
    assert "generic" in reasons or "structural" in reasons


def test_greek_plural_hyphen_variants_of_the_class_are_dropped():
    # names that are just the structural class restated: Greek letter, plural,
    # unicode dash, "+ generic head word" -- none are concrete systems
    text = ("beta-solenoid proteins are common; β‐rolls and β‐helices "
            "occur; β-solenoid proteins self-assemble; beta solenoid proteins too")
    papers = [_paper("T", text)]
    r = discover_related_systems(
        "beta-helix protein", ["beta-solenoid"],
        search_fn=_search({"beta-helix protein": papers}),
        extract_fn=_extract({"T": [
            ("β‐Rolls", "β‐rolls and β‐helices occur", "β‐rolls"),
            ("β‐Helices", "β‐rolls and β‐helices occur", "β‐helices"),
            ("beta solenoid proteins", "beta solenoid proteins too", "beta solenoid"),
            ("β-solenoid proteins", "β-solenoid proteins self-assemble", "β-solenoid"),
        ]}),
    )
    assert r.discovered_systems == []
    assert len(r.dropped) == 4
    assert all("structural" in d.reason for d in r.dropped)


def test_concrete_named_system_survives_alongside_class_variants():
    text = ("PKHD1 encodes a protein with parallel beta-helix repeats. "
            "Separately, beta-helix proteins are widespread.")
    papers = [_paper("PKHD1 study", text, doi="10/pkhd1")]
    r = discover_related_systems(
        "beta-helix protein", [],
        search_fn=_search({"beta-helix protein": papers}),
        extract_fn=_extract({"PKHD1 study": [
            ("PKHD1", "PKHD1 encodes a protein with parallel beta-helix repeats", "beta-helix"),
            ("beta-helix proteins", "beta-helix proteins are widespread", "beta-helix"),
        ]}),
    )
    assert [ds.name for ds in r.discovered_systems] == ["PKHD1"]
    assert any("structural" in d.reason for d in r.dropped)
    assert r.ungrounded_survivor_count == 0


# --- deterministic structural-class-modifier stripping -----------------
# A candidate is NOT dropped just for containing the structural term; a
# conservative deterministic strip peels the class modifier and the concrete
# name is re-grounded against the (unchanged) evidence quote.

def test_strip_class_modifier_pertactin():
    ab = ("The Pertactin β-helix has been a folding model for decades. "
          "Pertactin forms a right-handed parallel β-helix.")
    papers = [_paper("Pertactin structure", ab, doi="10/p")]
    r = discover_related_systems(
        "beta-helix protein", ["beta-solenoid"],
        search_fn=_search({"beta-helix protein": papers}),
        extract_fn=_extract({"Pertactin structure": [
            ("Pertactin β-helix", "Pertactin forms a right-handed parallel β-helix", "β-helix"),
        ]}),
    )
    assert [ds.name for ds in r.discovered_systems] == ["Pertactin"]
    sp = r.discovered_systems[0].supporting_papers[0]
    assert sp.name_as_extracted == "Pertactin β-helix"
    assert sp.evidence_quote == "Pertactin forms a right-handed parallel β-helix"  # unchanged
    assert sp.paper_ref == "10/p"
    assert r.ungrounded_survivor_count == 0


def test_strip_class_modifier_p22_tailspike():
    ab = ("The P22 tailspike beta-helix domain is a classic model. "
          "The P22 tailspike folds into a large right-handed parallel beta-helix.")
    papers = [_paper("P22 tailspike", ab, doi="10/t")]
    r = discover_related_systems(
        "beta-helix protein", [],
        search_fn=_search({"beta-helix protein": papers}),
        extract_fn=_extract({"P22 tailspike": [
            ("P22 tailspike beta-helix domain",
             "The P22 tailspike folds into a large right-handed parallel beta-helix", "beta-helix"),
        ]}),
    )
    assert [ds.name for ds in r.discovered_systems] == ["P22 tailspike"]
    assert r.discovered_systems[0].supporting_papers[0].name_as_extracted == "P22 tailspike beta-helix domain"
    assert r.ungrounded_survivor_count == 0


def test_strip_class_modifier_pectate_lyase():
    ab = ("The pectate lyase β-helix domain was among the first solved. "
          "Pectate lyase is a single-stranded right-handed β-helix.")
    papers = [_paper("Pectate lyase", ab, doi="10/pl")]
    r = discover_related_systems(
        "beta-helix protein", [],
        search_fn=_search({"beta-helix protein": papers}),
        extract_fn=_extract({"Pectate lyase": [
            ("pectate lyase β-helix domain",
             "Pectate lyase is a single-stranded right-handed β-helix", "β-helix"),
        ]}),
    )
    assert [ds.name for ds in r.discovered_systems] == ["pectate lyase"]
    assert r.ungrounded_survivor_count == 0


def test_strip_class_modifier_does_not_fabricate_from_bare_class():
    papers = [_paper("T", "A beta-helix protein was characterised here. beta-helix protein folds vary.")]
    r = discover_related_systems(
        "beta-helix protein", ["beta-solenoid"],
        search_fn=_search({"beta-helix protein": papers}),
        extract_fn=_extract({"T": [
            ("beta-helix protein", "A beta-helix protein was characterised here", "beta-helix"),
        ]}),
    )
    assert r.discovered_systems == []
    assert r.dropped[0].cleaned_name == ""
    assert "structural-class" in r.dropped[0].reason


def test_strip_class_modifier_rejects_class_plural():
    papers = [_paper("T", "β-solenoid proteins self-assemble into fibres. β-solenoid proteins are studied.")]
    r = discover_related_systems(
        "beta-helix protein", ["beta-solenoid"],
        search_fn=_search({"beta-helix protein": papers}),
        extract_fn=_extract({"T": [
            ("β-solenoid proteins", "β-solenoid proteins self-assemble into fibres", "β-solenoid"),
        ]}),
    )
    assert r.discovered_systems == []
    assert r.dropped[0].cleaned_name == ""


def test_strip_class_modifier_does_not_truncate_a_named_family():
    # "Type V secretory proteins" -- "Type" is part of the name, "proteins" is
    # not adjacent to a class term, so nothing is peeled and the name is kept
    # whole (not mangled to "V secretory").
    ab = ("The turn of the screw. Variations of the abundant beta-solenoid motif "
          "in passenger domains of Type V secretory proteins.")
    papers = [_paper("Type V secretory proteins", ab, doi="10/tv")]
    r = discover_related_systems(
        "beta-helix protein", ["beta-solenoid"],
        search_fn=_search({"beta-helix protein": papers}),
        extract_fn=_extract({"Type V secretory proteins": [
            ("Type V secretory proteins",
             "the abundant beta-solenoid motif in passenger domains of Type V secretory proteins",
             "beta-solenoid"),
        ]}),
    )
    assert [ds.name for ds in r.discovered_systems] == ["Type V secretory proteins"]
    assert r.ungrounded_survivor_count == 0


def test_residue_range_is_not_stripped_as_a_class_modifier():
    # "PrP 27-30" -- the digits are part of the entity name (a proteolytic
    # fragment), NOT a class modifier. Only "beta-helical fold" is peeled.
    ab = ("PrP 27-30 is the protease-resistant core of the prion. "
          "The PrP 27-30 beta-helix is left-handed. "
          "PrP 27-30 forms a parallel left-handed beta-helix.")
    papers = [_paper("PrP 27-30", ab, doi="10/prp")]
    r = discover_related_systems(
        "beta-helix protein", ["beta-helical fold"],
        search_fn=_search({"beta-helix protein": papers}),
        extract_fn=_extract({"PrP 27-30": [
            ("PrP 27-30 beta-helix",
             "PrP 27-30 forms a parallel left-handed beta-helix", "beta-helix"),
        ]}),
    )
    assert [ds.name for ds in r.discovered_systems] == ["PrP 27-30"]
    assert r.discovered_systems[0].supporting_papers[0].name_as_extracted == "PrP 27-30 beta-helix"
    assert r.ungrounded_survivor_count == 0


def test_unicode_dash_variant_name_still_grounds_in_quote():
    # name carries an en-dash (from the title); the quote uses a hyphen (abstract).
    # the cleaned-name-in-quote check folds dash shape -- it must NOT drop this.
    ab = ("Amyloid fibrils of the HET-s(218–289) prion form a solenoid. "
          "HET-s(218-289) forms a left-handed beta solenoid.")
    papers = [_paper("HET-s(218–289) fibrils", ab, doi="10/hets")]
    r = discover_related_systems(
        "beta-helix protein", ["beta-solenoid"],
        search_fn=_search({"beta-helix protein": papers}),
        extract_fn=_extract({"HET-s(218–289) fibrils": [
            ("HET-s(218–289)", "HET-s(218-289) forms a left-handed beta solenoid", "beta solenoid"),
        ]}),
    )
    assert [ds.name for ds in r.discovered_systems] == ["HET-s(218–289)"]
    assert r.discovered_systems[0].supporting_papers[0].evidence_quote == \
        "HET-s(218-289) forms a left-handed beta solenoid"
    assert r.ungrounded_survivor_count == 0


def test_class_notation_and_class_plus_enzymes_are_dropped():
    # "(betaalpha)8 (TIM) barrels" and "TIM-barrel enzymes" are the class
    # restated in notation / with a generic head -- not concrete systems.
    # "narbonin" in the same corpus is a real named protein and must survive.
    text = ("Examples include (betaalpha)8 (TIM) barrels. The seven TIM-barrel "
            "enzymes were analysed. The 2S globulin narbonin consists of an "
            "alpha/beta barrel structure.")
    papers = [_paper("T", text, doi="10/tim")]
    r = discover_related_systems(
        "TIM-barrel fold protein", ["TIM barrel", "alpha-beta barrel"],
        search_fn=_search({"TIM-barrel fold protein": papers}),
        extract_fn=_extract({"T": [
            ("(betaalpha)8 (TIM) barrels", "Examples include (betaalpha)8 (TIM) barrels", "(betaalpha)8 (TIM) barrels"),
            ("TIM-barrel enzymes", "The seven TIM-barrel enzymes were analysed", "TIM-barrel"),
            ("narbonin", "the 2S globulin narbonin consists of an alpha/beta barrel", "alpha/beta barrel"),
        ]}),
    )
    assert [ds.name for ds in r.discovered_systems] == ["narbonin"]
    assert {d.name for d in r.dropped} == {"(betaalpha)8 (TIM) barrels", "TIM-barrel enzymes"}
    assert r.ungrounded_survivor_count == 0


def test_structural_term_must_be_inside_the_quote():
    papers = [_paper("T", "Pectate lyase C is an enzyme. Separately, a beta-helix is a fold.")]
    r = discover_related_systems(
        "beta-helix", [],
        search_fn=_search({"beta-helix": papers}),
        extract_fn=_extract({"T": [
            ("Pectate lyase C", "Pectate lyase C is an enzyme", "beta-helix"),  # struct term not in quote
        ]}),
    )
    assert r.discovered_systems == []
    assert any("structural_term_matched not inside" in d.reason for d in r.dropped)


def test_same_system_from_two_papers_merges_and_counts_support():
    p1 = _paper("A", "Pertactin is a beta-helix protein", doi="10.1/a")
    p2 = _paper("B", "the crystal structure of pertactin reveals a beta-helix", doi="10.1/b")
    r = discover_related_systems(
        "beta-helix", [],
        search_fn=_search({"beta-helix": [p1], "beta-helix structure": [p2]}),
        extract_fn=_extract({
            "A": [("Pertactin", "Pertactin is a beta-helix", "beta-helix")],
            "B": [("pertactin", "pertactin reveals a beta-helix", "beta-helix")],
        }),
    )
    assert len(r.discovered_systems) == 1
    assert r.discovered_systems[0].n_support == 2
    assert r.ungrounded_survivor_count == 0


def test_more_than_k_systems_capped_and_ranked():
    # 10 papers, each names a distinct system once; #3 named twice -> ranks first
    papers = [_paper(f"P{i}", f"System{i} is a beta-helix", doi=f"10/{i}") for i in range(10)]
    papers.append(_paper("P3b", "System3 also a beta-helix", doi="10/3b"))
    extract_map = {f"P{i}": [(f"System{i}", f"System{i} is a beta-helix", "beta-helix")] for i in range(10)}
    extract_map["P3b"] = [("System3", "System3 also a beta-helix", "beta-helix")]
    r = discover_related_systems(
        "beta-helix", [],
        search_fn=lambda q: papers,
        extract_fn=_extract(extract_map),
        max_discovered=8,
    )
    assert len(r.discovered_systems) == 8
    assert r.discovered_systems[0].name == "System3" and r.discovered_systems[0].n_support == 2
    assert r.ungrounded_survivor_count == 0
