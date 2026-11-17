"""Consolidation safety net, Phase 0 -- P3 -> P5: the REAL Hop-1 discovery and the REAL P5 extractor.  Offline, no network; the only scripted parts are the
LLM answer (RichSystemExtractionLLM) and the search results.

Why this file exists: in the first scenario family `discover_related_systems` is a fake that hands the extractor only `expansion_terms[:3]`, so the
P3 aliases (which sit AFTER the three combination terms) never reached any P5 message there.  Production passes the WHOLE list: it is part of the P5
user message and therefore of the P5 cache key `content_hash(prompt_hash, schema_hash, user_message, backend.name, backend.model)`.  Here the real code
builds both, and the results are pinned in tests/fixtures/consolidation/baseline_p5_traces.json (written once from tag consol-0-baseline).

What a refactor of the 3B chain must keep:
  * the P5 message lists  [class + each retrieval anchor, class + objective] FIRST and the P3 aliases AFTER them, in that order, all of them;
  * changing, dropping or reordering a P3 alias changes the P5 message and the P5 cache key (a result is never served for different input);
  * a failed P3 leaves only the combination terms and keeps the `terminology_expansion:` error;
  * the axis basis selects the RELATIONSHIP CRITERION appended to the message (and so the key);
  * a warm second run serves every P5 answer from the cache (no second LLM call).
"""
import json
import re

import pytest

import consolidation_layout as LY
import consolidation_world as W
from cgscout import cache
from cgscout.retrieval.related_systems import extraction as X

BASE = json.loads(W.FIXTURE_P5.read_text())
COMBOS = ["alpha helix repeat repeat fold", "alpha helix repeat stacked strands", "alpha helix repeat folding mechanism"]   # class + each anchor, class + objective
LABEL = "hop1_extract_rich"
AKA_LINE = "Related terms and constructed search phrases for this class: "


def _p5_llm(trace, run=0):
    r = trace["runs"][run]
    return [e for e in trace["events"][r["events_from"]:r["events_to"]] if e["kind"] == "llm" and e["label"] == LABEL]


def _p5_gets(trace, run=0):
    r = trace["runs"][run]
    return [e for e in trace["events"][r["events_from"]:r["events_to"]] if e["kind"] == "cache.get" and e["ns"] == X._CACHE_NAMESPACE]


def _aka(user):
    line = next(l for l in user.splitlines() if l.startswith(AKA_LINE))
    return re.findall(r'"([^"]*)"', line[len(AKA_LINE):])


def _by_title(calls):
    """{paper title: user message}: pairs the same paper across scenarios (the order of the concurrent P5 units follows the cache key, which differs)."""
    return {re.search(r"\nTITLE: (.*)\n", c["user"]).group(1): c["user"] for c in calls}


def _key(user, backend_name="mock", backend_model="mock-1"):
    return cache.content_hash(X._RICH_PROMPT_HASH, X._RICH_SCHEMA_HASH, user, backend_name, backend_model)


def test_fixture_covers_every_p5_scenario():
    assert set(BASE) == set(W.SCENARIOS_P5)


@pytest.mark.parametrize("name", sorted(W.SCENARIOS_P5))
def test_p5_trace_equals_baseline(name):
    got = W.run_scenario(name)
    why = LY.diff_against_baseline(got, BASE[name], LY.active_layout())
    assert not why, f"{name}: {why}"


# ---------------------------------------------------------------------------------------------------------------- the pinned baseline facts
def test_p3_aliases_reach_the_real_p5_message():
    calls = _p5_llm(BASE["P5R01_p3_aliases_reach_p5"])
    assert len(calls) == 4                                                      # one P5 call per Hop-1 paper (shared + 3 hits)
    for c in calls:
        assert _aka(c["user"]) == COMBOS + W.ALIASES_A                          # combination terms first, then ALL the aliases, in P3's order (7 > 3)
        assert c["schema"] == "RichSystemExtractionLLM" and c["purpose"] == "hop1_related_system_extraction" and c["case_id"] == "T"
        assert c["timeout"] == X._EXTRACTION_TIMEOUT_SECONDS
        assert c["system_sha"] == W._h(X.RICH_SYSTEM_EXTRACTION_SYSTEM_PROMPT)  # the real P5 system prompt is what is sent
        assert c["user"].startswith('CLASS: "alpha helix repeat"\n' + AKA_LINE)
        assert c["user"].endswith("RELATIONSHIP CRITERION: " + X.AXIS_RELATIONSHIP_CRITERIA["distinctive_structure"])


def test_p5_cache_key_is_the_documented_formula():
    for name, t in BASE.items():
        for run in range(len(t["runs"])):
            from_calls = {_key(c["user"]) for c in _p5_llm(t, run)}
            from_cache = {g["key"] for g in _p5_gets(t, run)}
            assert from_calls <= from_cache, f"{name}: a P5 call's key is not content_hash(prompt, schema, user, backend.name, backend.model)"
            # and the key really carries the whole message: the four papers have four different messages -> four keys
    assert len({g["key"] for g in _p5_gets(BASE["P5R01_p3_aliases_reach_p5"])}) == 4


def test_different_p3_aliases_give_different_p5_input_and_keys():
    a, b, c = (BASE[n] for n in ("P5R01_p3_aliases_reach_p5", "P5R02_other_aliases", "P5R03_p3_fails_no_aliases"))
    ka, kb, kc = ({g["key"] for g in _p5_gets(t)} for t in (a, b, c))
    assert ka.isdisjoint(kb) and ka.isdisjoint(kc) and kb.isdisjoint(kc)
    assert _aka(_p5_llm(b)[0]["user"]) == COMBOS + ["helical repeat", "solenoid protein"]
    assert _aka(_p5_llm(c)[0]["user"]) == COMBOS                                 # P3 failed: only the combination terms
    assert [l["error"] for l in c["runs"][0]["result"]["axis_logs"]] == ["terminology_expansion: p3 boom"]   # the P3 error survives a successful axis


def test_basis_selects_the_relationship_criterion_in_the_message_and_key():
    a, e = BASE["P5R01_p3_aliases_reach_p5"], BASE["P5R05_basis_physical_process"]
    ma, me = _by_title(_p5_llm(a)), _by_title(_p5_llm(e))
    assert set(ma) == set(me) and len(ma) == 4
    for title in ma:
        ua, ue = ma[title], me[title]
        assert ua != ue and ue.endswith("RELATIONSHIP CRITERION: " + X.AXIS_RELATIONSHIP_CRITERIA["physical_process"])
        assert ua.rsplit("\n\nRELATIONSHIP CRITERION:", 1)[0] == ue.rsplit("\n\nRELATIONSHIP CRITERION:", 1)[0]       # only the criterion differs
    assert {g["key"] for g in _p5_gets(a)}.isdisjoint({g["key"] for g in _p5_gets(e)})


def test_warm_second_run_serves_p5_from_the_cache():
    t = BASE["P5R04_warm_second_run"]
    cold, warm = _p5_gets(t, 0), _p5_gets(t, 1)
    assert [g["hit"] for g in cold] == [False] * 4 and [g["hit"] for g in warm] == [True] * 4
    assert [g["key"] for g in cold] == [g["key"] for g in warm]
    assert len(_p5_llm(t, 0)) == 4 and _p5_llm(t, 1) == []
    assert t["llm_calls"] == 5                                                    # 4 P5 + 1 P6, nothing on the second run


# ---------------------------------------------------------------------------------------------------------------- the same facts, executed live on the current code
@pytest.fixture
def live(tmp_path, monkeypatch):
    """Runs a P5 scenario on the CURRENT code with edited inputs, each time on a cold cache of its own."""
    n = {"i": 0}

    def run(aliases, name="P5R01_p3_aliases_reach_p5"):
        n["i"] += 1
        monkeypatch.setenv("CGSCOUT_CACHE_DIR", str(tmp_path / f"live-cache-{n['i']}"))
        t = W.run_scenario(name, overrides={"aliases": aliases})
        return t, {g["key"] for g in _p5_gets(t)}, _p5_llm(t)
    return run


def test_live_changing_aliases_changes_the_p5_input_and_keys(live):
    """Not stored data: the current code is run with edited alias lists and the effect on the REAL P5 message and key is measured."""
    _live = live
    _, k0, c0 = _live(W.ALIASES_A)
    for label, aliases in {
            "replace the last alias": W.ALIASES_A[:-1] + ["a different alias"],
            "drop the first alias": W.ALIASES_A[1:],
            "reorder the aliases": list(reversed(W.ALIASES_A)),
            "append an alias": W.ALIASES_A + ["one more alias"]}.items():
        _, k1, c1 = _live(aliases)
        assert k1.isdisjoint(k0), f"{label}: the P5 cache keys must change (a stored answer must never be served for a different message)"
        assert all(_aka(c["user"]) == COMBOS + aliases for c in c1), f"{label}: the aliases must reach every P5 message in P3's order"
        assert set(_by_title(c1).values()).isdisjoint(_by_title(c0).values())


def test_live_same_aliases_reproduce_the_same_p5_input_and_keys(live):
    _, k1, c1 = live(W.ALIASES_A)
    _, k2, c2 = live(W.ALIASES_A)
    assert k1 == k2 and len(c1) == len(c2) == 4 and _by_title(c1) == _by_title(c2)


# ---------------------------------------------------------------------------------------------------------------- the extractor on its own
class _Recorder:
    name, model = "mock", "mock-1"

    def __init__(self):
        self.calls = []

    def structured(self, *, system, user, schema, label, effort=None, purpose=None, case_id=None, timeout=None):
        self.calls.append(dict(system=system, user=user, label=label, purpose=purpose, case_id=case_id, timeout=timeout))
        return schema(systems=[])


def test_extractor_message_and_key_are_built_from_the_whole_alias_list():
    title, abstract, cls = "A study", "Aquila protein adopts an alpha helix repeat fold.", "alpha helix repeat"
    aka = ["t1", "t2", "t3", "t4", "t5"]
    be = _Recorder()
    X.extract_related_systems_rich(title, abstract, cls, aka, backend=be, basis="family_homology", case_id="c")
    (call,) = be.calls
    expected = (X.system_extraction_user_message(title, abstract, cls, aka) + "\n\nRELATIONSHIP CRITERION: " + X.AXIS_RELATIONSHIP_CRITERIA["family_homology"])
    assert call["user"] == expected and _aka(call["user"]) == aka
    assert call["system"] == X.RICH_SYSTEM_EXTRACTION_SYSTEM_PROMPT and call["label"] == LABEL and call["case_id"] == "c"
    assert cache.get(X._CACHE_NAMESPACE, _key(expected)) is not None                       # written under exactly the documented key
    # the same call again is a cache hit; a different alias list is not
    X.extract_related_systems_rich(title, abstract, cls, aka, backend=be, basis="family_homology", case_id="c")
    assert len(be.calls) == 1
    X.extract_related_systems_rich(title, abstract, cls, aka[:-1], backend=be, basis="family_homology", case_id="c")
    assert len(be.calls) == 2 and _aka(be.calls[1]["user"]) == aka[:-1]
    # backend name / model are part of the key: another model never reads this entry
    other = _Recorder()
    other.model = "mock-2"
    X.extract_related_systems_rich(title, abstract, cls, aka, backend=other, basis="family_homology", case_id="c")
    assert len(other.calls) == 1 and _key(expected) != _key(expected, backend_model="mock-2")
