"""The fields of RetrievedPaper and the signature of the accumulator after the old structural provenance is removed (zero API, zero network).

Phase 5 / C2d-1: `RetrievedPaper.hop1_provenance` / `hop1_provenances` (always None / [] on the direct route, since the only call that could fill them passed None) and
the accumulator parameter and helper that fed them are gone.  The route provenance that downstream code reads (`found_by`, `provenance_index`, the Paper fields that
`retrieve_direct` writes) is pinned by tests/test_route_provenance.py, which passed unchanged before and after."""
import inspect

import pytest

import cgscout.retrieval.direct.route_plan as RP
from cgscout.retrieval.direct.evidence_plan import EvidenceRoutePlan, RouteTerm
from cgscout.retrieval.direct.route_plan import RouteProvenanceEntry, RouteRetrievalBudgets

REMOVED = ("hop1_provenance", "hop1_provenances")


def test_retrieved_paper_has_exactly_the_four_remaining_fields():
    assert list(RP.RetrievedPaper.model_fields) == ["paper_ref", "title", "year", "found_by"]


@pytest.mark.parametrize("name", REMOVED)
def test_a_removed_field_is_in_no_dump_no_schema_and_not_readable(name):
    plan = EvidenceRoutePlan(scientific_objective="folding pathway", system_description="EspP passenger domain",
                             process_terms=[RouteTerm(term="folding pathway", route_type="process", source="USER_GROUNDED")])
    from cgscout.models import Paper
    res = RP.run_route_retrieval(plan, flat_search_fn=lambda q, l: [Paper(title="p", year=2001, doi="10.1/x", abstract="x" * 80)])
    assert res.papers and name not in res.papers[0].model_dump()
    assert name not in RP.RetrievedPaper.model_json_schema()["properties"]
    assert name not in RP.RouteRetrievalResult.model_json_schema()["$defs"]["RetrievedPaper"]["properties"]
    with pytest.raises(AttributeError):
        getattr(res.papers[0], name)


def test_removed_fields_passed_to_the_constructor_are_ignored_not_stored():
    rp = RP.RetrievedPaper(paper_ref="r", title="t", hop1_provenance={"anything": 1}, hop1_provenances=[{"anything": 1}])
    assert not any(hasattr(rp, n) for n in REMOVED) and not set(rp.model_dump()) & set(REMOVED)


def test_hop1_provenance_is_no_longer_part_of_the_route_retrieval_schema():
    assert "Hop1Provenance" not in RP.RouteRetrievalResult.model_json_schema().get("$defs", {})
    assert not hasattr(RP, "Hop1Provenance")


def test_the_accumulator_add_signature_has_no_provenance_parameter_and_the_helper_is_gone():
    assert list(inspect.signature(RP._Accumulator.add).parameters) == ["self", "ref", "title", "year", "entry", "route_new_count", "route_cap"]
    assert not hasattr(RP._Accumulator, "_add_hop1_provenance")
    acc = RP._Accumulator(RouteRetrievalBudgets())
    e = RouteProvenanceEntry(route="process", route_term="x", query="q", query_role="r", retrieval_rank=0)
    assert acc.add("p", "P", 2020, e, 0, 5) == (True, True) and acc.add("p", "P", 2020, e, 0, 5) == (True, False)
    with pytest.raises(TypeError):                              # the old call shape, with the provenance argument (None) before the counters
        acc.add("q", "Q", 2020, e, None, 0, 5)
