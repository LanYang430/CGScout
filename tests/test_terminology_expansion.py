"""Pure unit tests for terminology_expansion deterministic helpers. No LLM."""


from cgscout.retrieval.related_systems.terminology import filter_instance_terms, strip_size_qualifiers  # noqa: E402


def test_strip_size_qualifiers():
    assert strip_size_qualifiers("200-aa beta-helix protein") == "beta-helix protein"
    assert strip_size_qualifiers("~200 residue beta-helix protein") == "beta-helix protein"
    assert strip_size_qualifiers("100-residue disordered linker") == "disordered linker"
    assert strip_size_qualifiers("beta-helix protein") == "beta-helix protein"
    assert strip_size_qualifiers("50 kDa membrane protein") == "membrane protein"


def test_filter_keeps_structural_and_class_terms():
    kept, dropped = filter_instance_terms([
        "beta-solenoid", "parallel beta-helix", "beta-helical fold",
        "autotransporter passenger domain", "pectin lyase-like fold",
        "single-stranded left-handed beta-helix", "right-handed beta-helix",
    ])
    assert dropped == []
    assert "beta-solenoid" in kept and "autotransporter passenger domain" in kept


def test_filter_drops_specific_protein_and_gene_names():
    kept, dropped = filter_instance_terms([
        "Pertactin", "P22 tailspike", "EspP", "pectate lyase C",
        "SARS-CoV E", "1QRE", "filamentous haemagglutinin adhesin",
    ])
    for bad in ["Pertactin", "P22 tailspike", "EspP", "pectate lyase C", "SARS-CoV E", "1QRE"]:
        assert bad in dropped, f"{bad} should be dropped as an instance"
    # "...adhesin" has a structural/class morpheme -> kept
    assert "filamentous haemagglutinin adhesin" in kept


def test_filter_drops_terms_repeating_the_objective():
    kept, dropped = filter_instance_terms(
        ["beta-solenoid", "beta-helix folding"], ban_substrings=("folding",)
    )
    assert "beta-helix folding" in dropped
    assert "beta-solenoid" in kept


def test_filter_dedupes_case_insensitively():
    kept, _ = filter_instance_terms(["Beta-Solenoid", "beta-solenoid", "BETA-SOLENOID"])
    assert len(kept) == 1
