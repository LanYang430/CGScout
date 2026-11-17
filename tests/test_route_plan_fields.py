"""The fields of RouteRetrievalPlan after the structural scaffolding is removed (zero API, zero network).

DRAFT for Phase 5 / C2b-1.  The five `structural_branch_*` fields of RouteRetrievalPlan were filled by build_route_retrieval_plan and read by nothing except the guard
of run_route_retrieval (which now looks at EvidenceRoutePlan.structural_terms directly).  These tests describe the new model and record the two compatibility changes:
a removed field is neither stored nor readable any more, and a structural term list that is not made of RouteTerm objects (possible only by assigning to the field
after construction: pydantic does not validate assignment) now reaches the guard instead of failing earlier inside the primary-term selection."""
import pytest

import cgscout.retrieval.direct.route_plan as RP
from cgscout.models import Paper
from cgscout.retrieval.direct.evidence_plan import EvidenceRoutePlan, RouteTerm

RUN, BUILD = RP.run_route_retrieval, RP.build_route_retrieval_plan
REMOVED = ("structural_branch_enabled", "structural_branch_structural_class", "structural_branch_structural_class_source",
           "structural_branch_also_known_as", "structural_branch_secondary_terms")


def evidence_plan():
    return EvidenceRoutePlan(scientific_objective="folding pathway of the beta-helix", system_description="EspP passenger domain",
                             process_terms=[RouteTerm(term="folding pathway", route_type="process", source="USER_GROUNDED")])


def test_the_plan_has_exactly_the_five_remaining_fields():
    assert list(RP.RouteRetrievalPlan.model_fields) == ["system_description", "scientific_objective", "route_queries", "budgets", "empty_routes"]


@pytest.mark.parametrize("name", REMOVED)
def test_a_removed_field_is_in_no_dump_no_schema_and_not_readable(name):
    plan = BUILD(evidence_plan())
    assert name not in plan.model_dump() and name not in RP.RouteRetrievalPlan.model_json_schema()["properties"]
    assert name not in RP.RouteRetrievalResult.model_json_schema()["$defs"]["RouteRetrievalPlan"]["properties"]      # the result embeds the plan
    with pytest.raises(AttributeError):
        getattr(plan, name)


def test_passing_a_removed_field_to_the_constructor_is_ignored_not_stored():
    plan = RP.RouteRetrievalPlan(system_description="s", scientific_objective="o", structural_branch_enabled=True, structural_branch_secondary_terms=[{"term": "x"}])
    assert not any(hasattr(plan, n) for n in REMOVED) and not set(plan.model_dump()) & set(REMOVED)


def test_a_structural_term_list_of_plain_strings_now_reaches_the_guard_after_the_flat_routes():
    ep = evidence_plan()
    ep.structural_terms = ["beta-helix"]                  # not RouteTerm objects: only possible by assignment after construction
    calls = []

    def search(query, limit):
        calls.append(query)
        return [Paper(title="p", year=2001, doi=f"10.1/{len(calls)}", abstract="x" * 80)]
    with pytest.raises(NotImplementedError):              # before the change: AttributeError ('str' object has no attribute 'source') before any search
        RUN(ep, flat_search_fn=search)
    assert calls
