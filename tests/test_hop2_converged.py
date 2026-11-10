"""Converged Hop-2 (no LLM, no network): fake backend + fake search.  What is pinned down:
  * Hop-2 queries are SHORT and built from structured fields, never from the class phrase / objective sentence / taxon;
  * every normalized field is validated deterministically (anchor grounded in Hop-1 evidence, process from the objective,
    descriptor identity-only);
  * normally 2 primary queries per system, the fallback only when A+B return almost nothing, at most 2 systems per axis,
    entity/polymer/complex duplicates merged;
  * nothing in the module is special-cased to a particular system.
"""
import itertools
from pathlib import Path


import cgscout.retrieval.related_systems.hop2 as H  # noqa: E402

OBJ = "Model tension-dependent gating and predict bilayer-stress-driven transitions from nonconducting to conducting states"


_COUNTER = itertools.count()


class FakeBackend:
    model = "fake-1"

    def __init__(self, items=None, fail=False):
        self.items, self.fail, self.calls = items or [], fail, 0
        self.name = f"fake-{next(_COUNTER)}"    # the normalization cache is keyed by backend name: keep each fake's outputs separate

    def structured(self, *, system, user, schema, label, effort=None, purpose=None, case_id=None, timeout=None):
        self.calls += 1
        if self.fail:
            raise RuntimeError("down")
        return schema(items=[H._AnchorItem(**i) for i in self.items])


def sysin(index, label, ctx=None, quotes=(), titles=(), n=1, rank=0, etype="specific_protein_or_system"):
    return H.SystemInput(index=index, label=label, entity_type=etype, identity_context=ctx, evidence_quotes=list(quotes),
                         source_titles=list(titles), n_support=n, first_seen_rank=rank)


GOOD = sysin(0, "Zed1 channel", ctx="a membrane channel", quotes=["the Zed1 channel opens under membrane tension"], titles=["Gating of the Zed1 channel"])


def norm(items, systems=(GOOD,), objective=OBJ):
    return H.normalize_search_anchors(list(systems), objective, FakeBackend(items))


def item(i=0, anchor="Zed1 channel", process="tension-dependent gating", desc="", **kw):
    kw.setdefault("descriptor_scope", "whole_system_identity" if desc else "none")
    return dict(index=i, search_anchor=anchor, core_process=process, stable_descriptor=desc, **kw)


# --------------------------------------------------------------------- query construction
def test_primary_queries_are_a_and_b_in_the_agreed_shape_and_fallback_is_c():
    n = norm([item(desc="membrane channel")])[0]
    a, b = H.primary_queries(n)
    assert (a.role, a.query) == ("A", "Zed1 channel tension-dependent gating coarse-grained model")
    assert (b.role, b.query) == ("B", "Zed1 channel membrane channel coarse-grained model")
    assert H.fallback_query(n).query == "Zed1 channel coarse-grained model" and H.fallback_query(n).role == "C"


def test_queries_never_carry_the_class_phrase_the_objective_sentence_or_extra_qualifiers():
    n = norm([item(desc="membrane channel")])[0]
    for q in [*H.primary_queries(n), H.fallback_query(n)]:
        words = q.query.split()
        assert len(words) <= 8                                    # short
        assert "bilayer-stress-driven" not in q.query and "nonconducting" not in q.query and "predict" not in q.query


def test_no_descriptor_means_only_query_a_and_no_descriptor_permutations():
    n = norm([item(desc="")])[0]
    assert [q.role for q in H.primary_queries(n)] == ["A"]
    n2 = norm([item(process="", desc="")])[0]
    assert H.primary_queries(n2) == []                                # nothing to form A/B from


# --------------------------------------------------------------------- deterministic validation of the LLM fields
def test_core_process_must_come_from_the_objective():
    n = norm([item(process="allosteric ligand activation")])[0]
    assert n.core_process == "" and "process_not_from_objective" in n.flags
    n2 = norm([item(process="gating transitions")])[0]
    assert n2.core_process == "gating transitions"                    # words (lightly inflected) from the objective
    n3 = norm([item(process="one two three four five six")])[0]
    assert n3.core_process == ""


def test_anchor_must_be_grounded_short_and_not_a_bare_generic_word():
    assert norm([item(anchor="Invented protein")])[0].flags[:1] == ["anchor_not_grounded_in_hop1_evidence"]
    assert norm([item(anchor="the complex")])[0].flags[:1] == ["anchor_generic"]
    long = norm([item(anchor="Zed1 channel opens under membrane tension in cells")])[0]
    assert "anchor_length" in long.flags and long.search_anchor == "Zed1 channel membrane channel"      # label + (article-less) context
    assert all("anchor_fallback_to_label" in n.flags for n in (norm([item(anchor="Invented protein")])[0], long))


def test_a_bare_organism_label_falls_back_to_label_plus_context_and_the_model_may_normalize_it_properly():
    s = sysin(0, "Q. testus", ctx="70X ribosome", quotes=["the Q. testus 70X ribosome was solved"], titles=["Q. testus 70X ribosome"])
    n = H.normalize_search_anchors([s], "Determine how the ribosome rearranges", FakeBackend([item(anchor="Q. testus 70X ribosome", process="ribosome rearranges")]))[0]
    assert n.search_anchor == "Q. testus 70X ribosome"
    bad = H.normalize_search_anchors([s], "x", FakeBackend([item(anchor="Q. testus", process="")]))[0]    # model kept only the organism
    assert bad.search_anchor == "Q. testus"                           # grounded & non-generic, so allowed...
    unusable = H.normalize_search_anchors([s], "x", FakeBackend([item(anchor="zzz yyy")]))[0]
    assert unusable.search_anchor == "Q. testus 70X ribosome"         # ...but an unusable anchor falls back to label + context


def test_descriptor_must_be_short_grounded_identity_and_not_redundant():
    assert norm([item(desc="membrane channel")])[0].stable_descriptor == "membrane channel"
    assert norm([item(desc="evolutionary conserved gating transition mechanism")])[0].stable_descriptor == ""
    assert norm([item(desc="shared transfer rationale")])[0].stable_descriptor == ""          # not grounded in the evidence
    assert norm([item(desc="Zed1 channel")])[0].stable_descriptor == ""                       # redundant with the anchor


def test_normalization_failure_degrades_to_label_only_not_an_exception():
    n = H.normalize_search_anchors([GOOD], OBJ, FakeBackend(fail=True))[0]
    assert n.search_anchor == "Zed1 channel membrane channel"
    assert any(f.startswith("normalization_unavailable") for f in n.flags) and n.core_process == "" and n.stable_descriptor == ""


def test_normalization_is_one_batched_cached_call():
    be = FakeBackend([item(0), item(1, anchor="Zed2 channel")])
    systems = [GOOD, sysin(1, "Zed2 channel", quotes=["the Zed2 channel is homologous"], titles=["Zed2 channel"])]
    H.normalize_search_anchors(systems, OBJ, be)
    H.normalize_search_anchors(systems, OBJ, be)
    assert be.calls == 1


# --------------------------------------------------------------------- selection / merging / execution
def test_entity_and_polymer_forms_merge_and_at_most_two_systems_enter_hop2():
    ns = [H.NormalizedSystem(index=i, label=l, search_anchor=a, n_support=n, first_seen_rank=r)
          for i, (l, a, n, r) in enumerate([("dynx", "dynx", 2, 0), ("dynx polymer", "dynx", 1, 1), ("Drpy", "Drpy", 1, 2), ("Oxz", "Oxz", 1, 3)])]
    sel = H.merge_and_select(ns)
    assert [s.search_anchor for s in sel] == ["dynx", "Drpy"] and sel[0].n_support == 3 and len(sel) == H.HOP2_MAX_SYSTEMS_PER_AXIS


def test_state_or_form_words_do_not_make_a_different_system_for_selection():
    ns = [H.NormalizedSystem(index=i, label=l, search_anchor=a, n_support=n, first_seen_rank=r)
          for i, (l, a, n, r) in enumerate([("dynx", "dynx", 1, 0), ("dynx oligomer", "dynx oligomer", 1, 1), ("dynx tetramer", "dynx tetramer", 1, 2), ("Oxz", "Oxz", 1, 3)])]
    sel = H.merge_and_select(ns)
    assert [s.search_anchor for s in sel] == ["dynx", "Oxz"] and sel[0].n_support == 3          # the three forms became one system


def test_descriptor_prompt_excludes_fold_and_structural_class_terms():
    assert "fold, architecture or other structural-class term" in H._SYSTEM_PROMPT.replace("\\\n", " ").replace("\n", " ")


def _runner(counts):
    calls = []

    def search(q):
        calls.append(q)
        return [f"{q}#{i}" for i in range(counts.get(q.split()[-3], 0))] if False else counts(q)
    return search, calls


def test_fallback_runs_only_when_a_and_b_return_almost_nothing():
    n = norm([item(desc="membrane channel")])[0]
    ref = lambda p: p
    # plenty of results -> no fallback
    plans, _ = H.run_converged_hop2([n], lambda q: [f"{q}-{i}" for i in range(5)], ref)
    assert [q["role"] for q in plans[0].queries] == ["A", "B"] and not plans[0].fallback_used
    # nothing -> fallback C
    plans, _ = H.run_converged_hop2([n], lambda q: [], ref)
    assert [q["role"] for q in plans[0].queries] == ["A", "B", "C"] and plans[0].fallback_used
    # very few (<= threshold) distinct papers -> fallback C
    plans, _ = H.run_converged_hop2([n], lambda q: ["same-1", "same-2"], ref)
    assert plans[0].fallback_used and plans[0].n_papers == 2
    # just above the threshold -> no fallback
    plans, _ = H.run_converged_hop2([n], lambda q: ["p1", "p2", "p3"], ref)
    assert not plans[0].fallback_used


def test_when_no_primary_query_can_be_formed_c_is_the_single_primary_query_not_a_fallback():
    n = norm([item(process="", desc="")])[0]
    plans, _ = H.run_converged_hop2([n], lambda q: ["p1"], lambda p: p)
    assert [q["role"] for q in plans[0].queries] == ["C"] and not plans[0].fallback_used


def test_search_errors_are_recorded_and_do_not_crash():
    n = norm([item(desc="membrane channel")])[0]

    def boom(q):
        raise RuntimeError("search down")
    plans, found = H.run_converged_hop2([n], boom, lambda p: p)
    assert all(q["error"] for q in plans[0].queries) and found[n.label] == {}


def test_found_by_queries_provenance_is_kept_per_paper():
    n = norm([item(desc="membrane channel")])[0]
    plans, found = H.run_converged_hop2([n], lambda q: ["shared", f"only-{q}"], lambda p: p)
    assert len(found[n.label]["shared"]) == 2                         # found by both A and B


# --------------------------------------------------------------------- no name-level hardcoding
def test_module_has_no_system_specific_strings():
    src = Path(H.__file__).read_text().lower()
    for name in ("dynamin", "mscs", "espp", "pertactin", "ribosome", "polymerase", "pol ii", "mgm1", "msl10", "flagell", "archaell", "autotransporter"):
        assert name not in src.replace("ribosomal", ""), name         # (the prompt text mentions none of these either)


def test_without_a_descriptor_plain_c_is_the_second_primary_query():
    n = norm([item(desc="")])[0]
    assert n.stable_descriptor == ""
    plans, _ = H.run_converged_hop2([n], lambda q: [f"{q}-{i}" for i in range(5)], lambda p: p)
    assert [q["role"] for q in plans[0].queries] == ["A", "C"] and not plans[0].fallback_used
    assert plans[0].queries[1]["query"] == H.fallback_query(n).query
    # even with no results at all: still at most two queries (C is not run a second time)
    plans, _ = H.run_converged_hop2([n], lambda q: [], lambda p: p)
    assert [q["role"] for q in plans[0].queries] == ["A", "C"]


def test_without_a_process_plain_c_is_the_second_primary_query():
    n = norm([item(process="", desc="membrane channel")])[0]
    plans, _ = H.run_converged_hop2([n], lambda q: ["p1", "p2", "p3"], lambda p: p)
    assert [q["role"] for q in plans[0].queries] == ["B", "C"]


def test_descriptor_must_be_declared_a_whole_system_identity_characteristic():
    for scope in ("part_or_partner", "state_or_condition", "none", ""):
        n = norm([item(desc="membrane channel", descriptor_scope=scope)])[0]
        assert n.stable_descriptor == "" and "descriptor_not_system_identity" in n.flags
        # rejected descriptor -> plain C becomes the second primary (A + C), no third query
        plans, _ = H.run_converged_hop2([n], lambda q: [f"{q}-{i}" for i in range(5)], lambda p: p)
        assert [q["role"] for q in plans[0].queries] == ["A", "C"]
    ok = norm([item(desc="membrane channel", descriptor_scope="whole_system_identity")])[0]
    assert ok.stable_descriptor == "membrane channel"


def test_prompt_states_the_whole_system_principle_without_naming_systems():
    assert "WHOLE system" in H._SYSTEM_PROMPT and "subunit or subcomponent" in H._SYSTEM_PROMPT and "binding partner" in H._SYSTEM_PROMPT
