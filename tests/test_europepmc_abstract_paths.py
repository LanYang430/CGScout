"""Stage 3's abstract recovery with the PRODUCTION lookup (Europe PMC through `urllib.request.urlopen`, mocked -- no network, no API).

Every other Stage 3 recovery test injects `abstract_lookup=`, so the default path of `screening.filter._recover_missing_abstracts` -- the function-local import of
the Europe PMC lookup and the MIN_ABSTRACT_CHARS rule -- was never executed by the suite.  These tests execute it, from the public behaviour only (what a paper
looks like afterwards), so they hold whichever module the lookup lives in.  Patching is done on `urllib.request.urlopen` itself, never on a module of the package."""
import json
import urllib.error
import urllib.parse
import urllib.request

import pytest

from cgscout.models import ExternalIds, Paper
from cgscout.screening.filter import _recover_missing_abstracts, filter_papers


class _Resp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode()

    def read(self, n=-1):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _answer(doi, text):
    return _Resp({"resultList": {"result": [{"doi": doi, "abstractText": text}]}})


@pytest.fixture
def epmc(monkeypatch):
    """Patch urlopen: `answers` maps a DOI to the abstractText Europe PMC 'returns' (a DOI not in it raises URLError); `calls` lists the DOIs asked for."""
    state = {"answers": {}, "calls": []}

    def fake(req, timeout=None):
        ident = urllib.parse.unquote(req.full_url).split('DOI:"', 1)[1].split('"', 1)[0]
        state["calls"].append(ident)
        if ident not in state["answers"]:
            raise urllib.error.URLError("down")
        return _answer(ident, state["answers"][ident])
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return state


def _p(doi, abstract=None, title="Assembly of a target system"):
    return Paper(title=title, abstract=abstract, doi=doi, sources=["openalex"])


def test_a_missing_abstract_is_recovered_from_europepmc_and_the_input_paper_is_not_mutated(epmc):
    epmc["answers"]["10.1/a"] = "<p>" + "We run coarse-grained simulations of the target system. " * 3 + "</p>"
    p = _p("10.1/a")
    out, recovered = _recover_missing_abstracts([p])
    assert recovered == [True] and epmc["calls"] == ["10.1/a"]
    assert out[0].abstract == ("We run coarse-grained simulations of the target system. " * 3).strip()
    assert out[0] is not p and p.abstract is None


def test_the_recovered_abstract_must_have_at_least_40_characters(epmc):
    epmc["answers"].update({"10.1/r40": "x" * 40, "10.1/r39": "x" * 39})
    papers = [_p("10.1/r40"), _p("10.1/r39")]
    out, recovered = _recover_missing_abstracts(papers)
    assert recovered == [True, False]
    assert out[0].abstract == "x" * 40 and out[1].abstract is None and out[1] is papers[1]


def test_a_failed_lookup_leaves_the_paper_exactly_as_retrieved_and_raises_nothing(epmc):
    p = _p("10.1/down")                                                      # not in `answers` -> URLError
    out, recovered = _recover_missing_abstracts([p])
    assert recovered == [False] and out[0] is p and epmc["calls"] == ["10.1/down"]


def test_papers_with_a_usable_abstract_or_without_any_identifier_cause_no_request(epmc):
    have = _p("10.1/have", abstract="We run coarse-grained simulations of the target system in detail.")
    no_ids = Paper(title="No identifiers", abstract=None)
    out, recovered = _recover_missing_abstracts([have, no_ids])
    assert recovered == [False, False] and out[0] is have and out[1] is no_ids and epmc["calls"] == []


def test_a_pmid_only_paper_is_looked_up_through_the_pmid_query(monkeypatch):
    urls = []
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout=None: (urls.append(urllib.parse.unquote(req.full_url)), _Resp({"resultList": {"result": [{"abstractText": "y" * 60}]}}))[1])
    only_pmid = Paper(title="PMID only", abstract=None, external_ids=ExternalIds(pmid="12345"))
    out, recovered = _recover_missing_abstracts([only_pmid])
    assert recovered == [True] and out[0].abstract == "y" * 60 and "EXT_ID:12345 AND SRC:MED" in urls[0]


def test_a_successful_answer_is_cached_so_a_second_recovery_makes_no_request(epmc):
    epmc["answers"]["10.1/c"] = "x" * 80
    _recover_missing_abstracts([_p("10.1/c")])
    out, recovered = _recover_missing_abstracts([_p("10.1/c")])
    assert recovered == [True] and out[0].abstract == "x" * 80 and epmc["calls"] == ["10.1/c"]


def test_filter_papers_with_recover_abstracts_uses_the_production_lookup_by_default(epmc):
    text = "We run coarse-grained molecular dynamics simulations of the target system and analyse its assembly pathway."
    epmc["answers"]["10.1/f"] = text
    p = _p("10.1/f")
    retained, rejected, verdicts = filter_papers([p], classify_fn=lambda ps: [1.0] * len(ps), fulltext_lookup_budget=0, recover_abstracts=True)
    assert epmc["calls"] == ["10.1/f"] and verdicts[0].abstract_recovered is True
    assert [r.abstract for r in retained + rejected] == [text] and p.abstract is None
    epmc["calls"].clear()
    filter_papers([_p("10.1/g")], classify_fn=lambda ps: [1.0] * len(ps), fulltext_lookup_budget=0)         # recover_abstracts off (the default): no lookup at all
    assert epmc["calls"] == []
