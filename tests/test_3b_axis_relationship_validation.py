"""3B relationship validation after the Q15-Q20 Hop-1 audit.  No LLM, no network: fake backend / fake search.

What these pin down (general behaviour, no system names are special-cased anywhere in production code):
  1. the extractor judges "is this entity related to the class" against an AXIS-specific criterion and requires a
     verbatim class_membership_quote -- there is no fold/shape word list any more;
  2. a concrete named member whose name also appears in the class label ("MscS" in "MscS-related channels") is a valid
     related system, while an abstract class label or the exact class phrase is still rejected;
  3. the older (non-concrete / old-pipeline) behaviour is unchanged;
  4. the ExpertPrior prompt asks related-system-discovery classes to be abstract enough to cross species/domains.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent / "benchmarks" / "concept_resolver_reasoning"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mellea_integration"))

import pytest  # noqa: E402

import transfer_entity_extraction as TE  # noqa: E402
from related_system_discovery import discover_related_systems  # noqa: E402
from beam_agent.models import Paper  # noqa: E402

TITLE = "Membrane fission by a GTPase"
ABSTRACT = ("Dynamin-related protein 1 (Drp1), a large GTPase of the dynamin superfamily, constricts membranes after "
            "GTP hydrolysis. A second assembly is built from a different enzyme and is unrelated.")


class FakeBackend:
    name, model = "fake", "fake-1"

    def __init__(self, mentions):
        self.mentions, self.users, self.calls = mentions, [], 0

    def structured(self, *, system, user, schema, label, effort=None, purpose=None, case_id=None, timeout=None):
        self.calls += 1
        self.users.append(user)
        return schema(systems=[TE.RichSystemMentionLLM(**m) for m in self.mentions])


def mention(name, quote, membership, etype="specific_protein_or_system"):
    return dict(name=name, evidence_quote=quote, entity_type=etype, identity_context=None, class_membership_quote=membership)


GOOD = mention("Drp1", "Dynamin-related protein 1 (Drp1), a large GTPase of the dynamin superfamily",
               "a large GTPase of the dynamin superfamily")


def run(mentions, basis="family_homology", cls="dynamin-family membrane fission GTPases"):
    be = FakeBackend(mentions)
    res = TE.extract_related_systems_rich(TITLE, ABSTRACT, cls, backend=be, basis=basis)
    return res, be


# ------------------------------------------------------------------ 1. axis-aligned validation in the extractor
@pytest.mark.parametrize("basis", ["family_homology", "distinctive_structure", "physical_process",
                                   "chemistry_representation", "environment_context", None])
def test_every_basis_requires_a_verbatim_membership_quote_and_no_word_list(basis):
    res, _ = run([GOOD,
                  mention("Other enzyme", "a different enzyme", "an invented sentence that is not in the paper"),
                  mention("Another", "A second assembly is built from a different enzyme", "")], basis=basis)
    assert [m.name for m in res.systems] == ["Drp1"]
    # the kept quote has no fold/shape vocabulary at all -- the old morpheme gate would have dropped it
    assert not any(w in res.systems[0].class_membership_quote.lower() for w in ("helix", "barrel", "beta", "fold", "coil"))


def test_each_axis_gets_its_own_criterion_in_the_user_message():
    seen = {}
    for basis in TE.AXIS_RELATIONSHIP_CRITERIA:
        _, be = run([GOOD], basis=basis)
        seen[basis] = be.users[0]
    crit = {b: TE.relationship_criterion(b) for b in TE.AXIS_RELATIONSHIP_CRITERIA}
    assert len(set(crit.values())) == len(crit)                       # five different criteria
    for b, user in seen.items():
        assert f"RELATIONSHIP CRITERION: {crit[b]}" in user
    assert "homolog" in crit["family_homology"] and "architecture" in crit["distinctive_structure"]
    assert "process" in crit["physical_process"] and "chemi" in crit["chemistry_representation"] and "environment" in crit["environment_context"]
    assert "belonging to, or being an example of" in TE.relationship_criterion(None)       # unknown basis -> generic criterion


def test_results_are_cached_per_criterion_not_shared_across_axes():
    be = FakeBackend([GOOD])
    TE.extract_related_systems_rich(TITLE, ABSTRACT, "c", backend=be, basis="family_homology")
    TE.extract_related_systems_rich(TITLE, ABSTRACT, "c", backend=be, basis="family_homology")
    assert be.calls == 1                                              # same axis -> cache hit
    TE.extract_related_systems_rich(TITLE, ABSTRACT, "c", backend=be, basis="physical_process")
    assert be.calls == 2                                              # different criterion -> fresh judgement


def test_prompt_asks_for_membership_evidence_and_allows_a_member_named_inside_the_class_label():
    p = TE.RICH_SYSTEM_EXTRACTION_SYSTEM_PROMPT
    assert "class_membership_quote" in p and "RELATIONSHIP CRITERION" in p
    assert "even when its name also appears inside the class label" in p.replace("\\\n", " ").replace("\n", " ")


# ------------------------------------------------------------------ 2. grounding: concrete member vs abstract class label
def _paper(title, abstract):
    return Paper(title=title, abstract=abstract, doi="10/x", year=2020)


def _disc(mentions, cls, aka=()):
    ab = ("The mechanosensitive ion channel MscS opens under membrane tension. Its eukaryotic homolog MSL1 from Arabidopsis "
          "thaliana has also been studied. Mechanosensitive channels are widespread.")
    p = _paper("Channels", ab)
    ms = [SimpleNamespace(**{"structural_term_matched": "", "identity_context": None, "class_membership_quote": None, **m}) for m in mentions]
    return discover_related_systems(cls, list(aka), search_fn=lambda q: [p],
                                    extract_fn=lambda t, a, c, k=None: SimpleNamespace(systems=ms))


CLS = "MscS-related mechanosensitive ion channels"


def test_concrete_member_named_inside_the_class_label_is_a_valid_related_system():
    r = _disc([dict(name="MscS", evidence_quote="mechanosensitive ion channel MscS", entity_type="specific_protein_or_system",
                    class_membership_quote="The mechanosensitive ion channel MscS opens under membrane tension")], CLS)
    assert [d.name for d in r.discovered_systems] == ["MscS"] and r.ungrounded_survivor_count == 0


def test_abstract_class_label_and_the_exact_class_phrase_are_still_rejected():
    r = _disc([dict(name="Mechanosensitive channels", evidence_quote="Mechanosensitive channels are widespread",
                    entity_type="protein_family_or_class", class_membership_quote="Mechanosensitive channels are widespread"),
               dict(name="MscS-related mechanosensitive ion channels", evidence_quote="MscS-related mechanosensitive ion channels",
                    entity_type="specific_protein_or_system", class_membership_quote="x")], CLS)
    assert r.discovered_systems == []


def test_old_pipeline_mentions_without_entity_type_keep_the_old_rule():
    r = _disc([dict(name="MscS", evidence_quote="mechanosensitive ion channel MscS")], CLS)       # no entity_type -> not 'concrete'
    assert r.discovered_systems == [] and "structural-class" in r.dropped[0].reason


def test_validated_membership_quote_replaces_the_old_structural_vocabulary_tie_test():
    quote_without_class_words = "its eukaryotic homolog MSL1 from Arabidopsis thaliana"
    with_membership = _disc([dict(name="MSL1", evidence_quote=quote_without_class_words, entity_type="specific_protein_or_system",
                                  class_membership_quote=quote_without_class_words)], CLS)
    assert [d.name for d in with_membership.discovered_systems] == ["MSL1"]
    without = _disc([dict(name="MSL1", evidence_quote=quote_without_class_words, entity_type="specific_protein_or_system")], CLS)
    assert without.discovered_systems == [] and "ties the system" in without.dropped[0].reason


def test_concrete_entity_still_has_a_class_modifier_peeled_off_its_name():
    ab = "The Pertactin beta-helix has been a folding model. Pertactin forms a right-handed parallel beta-helix."
    p = _paper("Pertactin", ab)
    m = SimpleNamespace(name="Pertactin beta-helix", evidence_quote="Pertactin forms a right-handed parallel beta-helix",
                        structural_term_matched="beta-helix", entity_type="specific_protein_or_system", identity_context=None,
                        class_membership_quote="Pertactin forms a right-handed parallel beta-helix")
    r = discover_related_systems("beta-helix protein", ["beta-solenoid"], search_fn=lambda q: [p],
                                 extract_fn=lambda t, a, c, k=None: SimpleNamespace(systems=[m]))
    assert [d.name for d in r.discovered_systems] == ["Pertactin"]


# ------------------------------------------------------------------ 3. ExpertPrior: transferable class abstraction
def test_expert_prior_prompt_requires_classes_one_level_above_the_target():
    import expert_prior_planner as EP
    p = EP.EXPERT_PRIOR_SYSTEM_PROMPT.replace("\\\n", " ").replace("\n", " ")
    assert "abstraction ABOVE the target itself" in p
    assert "species, organism, taxonomic domain" in p and "too narrow" in p
