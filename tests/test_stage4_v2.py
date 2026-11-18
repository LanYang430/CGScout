"""Stage 4 (v2): graded LLM relevance evaluation. A fake backend stands in for the LLM -- no network.

What these tests pin down (the design requirements, not implementation details):
  * Tier 1 is a REJECT-ONLY gate: only a confident reject (positive evidence of irrelevance) stops a paper;
    pass, insufficient_evidence (no/too-short abstract, model cannot tell) and low-confidence rejects ALL
    continue to Tier 2 (the cap is only a safety valve);
  * Tier 2 may still report insufficient evidence: recorded internally as `missing_evidence` -- never a grade, never
    a rejection, never customer-facing;
  * a paper with no usable OpenAlex abstract gets ONE Europe PMC lookup (cached); if it supplies an abstract the
    paper is evaluated normally, otherwise it is `missing_evidence` and costs no LLM call;
  * the final rank comes from the Tier-2 relevance assessment -- route and year only break exact ties;
  * a supporting quote that is not in the paper's text demotes the grade;
  * hard caps (experimental -> 0, review -> <=1, non-CG computational -> <=2, no abstract -> <=1);
  * results are cached by paper content + problem context (a re-run makes ZERO LLM calls);
  * an LLM failure is status="error" -- never a silent 0 and never counted as 'not relevant'.
"""
import json
import urllib.request

import pytest


from cgscout.models import Paper  # noqa: E402
import cgscout.assessment.stage4 as s4  # noqa: E402
from cgscout.assessment.stage4 import ExtractedFacts, Judgment, ScreenBatch, ScreenItem, Stage4Config, run_stage4_v2  # noqa: E402


class Plan:
    system = "a beta-helical protein"
    target_component = None
    scientific_objective = "folding"
    search_context = []


class Axis:
    confidence = "high"
    basis = "distinctive_structure"
    scientific_class = "beta-helical passenger domains"
    discriminative_features = ["right-handed parallel beta-helix"]
    objective_relevance = "folding of the same fold"


class Prior:
    transfer_axes = [Axis()]


def paper(title, abstract="A coarse-grained simulation of something studied here.", year=2024, cats=("transfer_direct",), nq=1):
    return Paper(title=title, abstract=abstract, year=year, doi=f"10.1/{abs(hash(title)) % 10**8}",
                 query_categories=list(cats), found_by_queries=[f"q{i}" for i in range(nq)])


class FakeBackend:
    """Tier 1: 'G0...' -> reject (high), 'LOWREJ...' -> reject (low confidence), 'INSUF...' -> insufficient_evidence,
    anything else -> pass. Tier 2: titles starting 'G3'/'G2'/'G1'/'G0' get that grade."""
    name, model = "fake", "fake-1"

    def __init__(self, fail_titles=(), quote=None, comp=None):
        self.calls = []
        self.fail_titles = set(fail_titles)
        self.quote = quote
        self.comp = comp or {}

    def structured(self, *, system, user, schema, label, effort=None, purpose=None, case_id=None, timeout=None):
        self.calls.append(label)
        if schema is ScreenBatch:
            items = []
            for line in user.splitlines():
                if line.startswith("[") and "] Title: " in line:
                    idx = int(line[1:line.index("]")]); title = line.split("Title: ", 1)[1]
                    if title in self.fail_titles:
                        raise RuntimeError("boom")
                    decision = ("reject" if title.startswith(("G0", "LOWREJ")) else
                                "insufficient_evidence" if title.startswith("INSUF") else "pass")
                    items.append(ScreenItem(index=idx, decision=decision, computational=self.comp.get(title, "coarse_grained"),
                                            confidence="low" if title.startswith("LOWREJ") else "high", reason="r"))
            return ScreenBatch(items=items)
        if schema is ExtractedFacts:
            return ExtractedFacts(method={"model_class": ["coarse-grained"], "study_role": ["applied"]},
                                  system={"studied_systems": ["x"], "family_or_fold": [], "protein_or_system_names": []})
        if schema is Judgment:
            title = next(l.split("Title: ", 1)[1] for l in user.splitlines() if l.startswith("Title: "))
            g = int(title[1]) if title[:1] == "G" and title[1].isdigit() else 0
            return Judgment(verdict="insufficient_evidence" if title.startswith("NEEDV") else "graded",
                            computational=self.comp.get(title, "coarse_grained"), methods_used=["m"], system_relation="related_system",
                            process_relation="same_process", evidence_type="related_system_transfer",
                            supports_applicability="yes" if g >= 2 else "no", grade=g, explanation="e",
                            supporting_quote=self.quote if self.quote is not None else "coarse-grained simulation of something")
        raise AssertionError(schema)


def no_lookup(p):
    return {"abstract": None, "via": "doi"}


def run(papers, backend=None, **cfg):
    backend = backend or FakeBackend()
    cfg.setdefault("abstract_lookup", no_lookup)          # tests never touch the network
    return run_stage4_v2(Plan(), Prior(), papers, backend, config=Stage4Config(**cfg), case_id="t"), backend


def test_every_retained_paper_is_screened_and_graded():
    papers = [paper("G3 direct"), paper("G2 transfer"), paper("G1 background"), paper("G0 off topic")]
    res, _ = run(papers)
    grades = {a.title: a.final_grade for a in res.assessments}
    assert grades == {"G3 direct": 3, "G2 transfer": 2, "G1 background": 1, "G0 off topic": 0}
    assert [a for a in res.assessments if a.title == "G0 off topic"][0].status == "tier1_rejected"
    assert [a.title for a in res.recommended] == ["G3 direct", "G2 transfer"]
    assert [a.rank for a in res.assessments] == [1, 2, 3, 4]
    assert res.workload["n_recommended"] == 2 and res.workload["n_rejected"] == 1 and res.workload["n_background"] == 1


def test_every_paper_that_passes_tier1_is_deeply_evaluated_and_only_rejects_stop():
    res, be = run([paper("G3 a"), paper("G1 b"), paper("G0 c")])
    assert be.calls.count("stage4_judge") == 2 and be.calls.count("stage4_extract") == 2     # the grade-1-ish paper is NOT filtered out
    low = [a for a in res.assessments if a.title.startswith("G1")][0]
    assert low.judgment is not None and low.status == "assessed" and low.final_grade == 1
    rej = [a for a in res.assessments if a.title.startswith("G0")][0]
    assert rej.judgment is None and rej.status == "tier1_rejected" and rej.final_grade == 0


def test_final_rank_follows_relevance_not_route_or_year():
    # the most relevant paper is the OLDEST and arrives by a transfer route; flat + newest papers are weaker
    papers = [paper("G2 flat newest", year=2026, cats=("system_specific_cg",)),
              paper("G3 oldest transfer", year=2009, cats=("transfer_direct",)),
              paper("G2 flat newer", year=2025, cats=("process",))]
    res, _ = run(papers)
    assert [a.title for a in res.assessments][0] == "G3 oldest transfer"


def test_year_and_route_only_break_exact_ties():
    papers = [paper("G2 older", year=2015, nq=3), paper("G2 newer", year=2025, nq=3)]
    res, _ = run(papers)
    assert [a.title for a in res.assessments] == ["G2 newer", "G2 older"]   # identical relevance -> year is the last tie-break


def test_safety_cap_leaves_unevaluated_papers_ungraded_not_irrelevant():
    papers = [paper(f"G2 p{i}", cats=("system_specific_cg",), nq=1 + i) for i in range(5)]
    res, _ = run(papers, deep_cap=2)
    assert res.workload["deep_cap_binding"] and res.workload["n_deep_skipped_by_cap"] == 3
    skipped = [a for a in res.assessments if a.deep_skipped_by_cap]
    assert all(a.status == "not_evaluated_cap" and a.final_grade is None and a.rank is None for a in skipped)
    assert not any(a in res.rejected or a in res.background or a in res.recommended for a in skipped)
    assert {a.title for a in res.assessments if a.deep_candidate} == {"G2 p4", "G2 p3"}      # ties broken by support, not by route


def test_quote_not_in_text_demotes_the_grade():
    res, _ = run([paper("G3 liar")], backend=FakeBackend(quote="a sentence the paper never contains"))
    a = res.assessments[0]
    assert a.quote_grounded is False and a.final_grade == 2 and "quote_not_found_in_text_demoted_1" in a.flags


def test_quote_check_is_whitespace_and_case_insensitive_and_supports_ellipsis():
    p = paper("T", abstract="We  develop a Coarse-Grained model of the passenger domain and validate it against MD.")
    assert s4.quote_grounded("coarse-grained model of the passenger domain", p)
    assert s4.quote_grounded("We develop a coarse-grained model ... validate it against MD", p)
    assert not s4.quote_grounded("hallucinated sentence about something", p)
    assert not s4.quote_grounded("short", p)


@pytest.mark.parametrize("comp,expected", [("not_computational", 0), ("review_or_perspective", 1), ("other_computational", 2), ("coarse_grained", 3)])
def test_rubric_hard_caps_are_enforced_even_if_the_llm_overgrades(comp, expected):
    res, _ = run([paper("G3 x")], backend=FakeBackend(comp={"G3 x": comp}))
    assert res.assessments[0].final_grade == expected


def test_no_abstract_and_no_fallback_hit_is_missing_evidence_with_no_grade_and_no_llm_call():
    res, be = run([paper("G3 no abstract", abstract="")])
    a = res.assessments[0]
    assert a.status == "missing_evidence" and a.final_grade is None and a.rank is None and not a.recommended
    assert a.abstract_source == "none" and "missing_evidence_no_abstract" in a.flags
    assert be.calls == [] and a.screen is None and a.judgment is None       # not screened, not judged on a title alone
    assert a in res.missing_evidence and a not in res.rejected and a not in res.background and a not in res.recommended
    assert res.workload["n_missing_evidence"] == 1 and res.workload["n_rejected"] == 0
    assert res.workload["n_abstract_lookup_attempted"] == 1 and res.workload["n_abstract_recovered_europepmc"] == 0


def test_europepmc_abstract_is_used_when_openalex_has_none_and_the_paper_is_evaluated_normally():
    seen = []

    def lookup(p):
        seen.append(p.title)
        return {"abstract": "Recovered abstract: a coarse-grained simulation of something studied here, the passenger domain folding.", "via": "doi"}
    res, be = run([paper("G3 recovered", abstract=None), paper("G2 has one")], abstract_lookup=lookup)
    a = [x for x in res.assessments if x.title == "G3 recovered"][0]
    assert a.abstract_source == "europepmc" and a.status == "assessed" and a.final_grade == 3 and a.recommended
    assert a.judgment is not None and a.screen is not None                   # went through Tier 1 and Tier 2
    assert seen == ["G3 recovered"]                                          # the paper that HAS an abstract was never looked up
    assert res.workload["n_abstract_recovered_europepmc"] == 1 and res.workload["n_abstract_lookup_attempted"] == 1
    assert [x.abstract_source for x in res.assessments if x.title == "G2 has one"] == ["openalex"]


def test_fallback_can_be_disabled_and_is_never_called_when_every_paper_has_an_abstract():
    def boom(p):
        raise AssertionError("must not be called")
    res, _ = run([paper("G3 a"), paper("G2 b")], abstract_lookup=boom)
    assert res.workload["n_abstract_lookup_attempted"] == 0
    res2, _ = run([paper("G3 c", abstract="")], abstract_lookup=boom, abstract_fallback=False)
    assert res2.assessments[0].status == "missing_evidence" and res2.workload["n_abstract_lookup_attempted"] == 0


def test_a_too_short_openalex_abstract_counts_as_missing_and_triggers_the_lookup():
    res, be = run([paper("G2 stub", abstract="Poster abstract.")])
    assert res.assessments[0].status == "missing_evidence" and res.workload["n_abstract_lookup_attempted"] == 1 and be.calls == []


def test_lookup_failure_is_missing_evidence_flagged_as_failed_not_a_silent_no_abstract():
    def fail(p):
        raise TimeoutError("network down")
    res, be = run([paper("G3 x", abstract="")], abstract_lookup=fail)
    a = res.assessments[0]
    assert a.status == "missing_evidence" and "abstract_lookup_failed" in a.flags and be.calls == []
    assert res.workload["n_abstract_lookup_failed"] == 1 and a.final_grade is None


def test_tier2_can_report_insufficient_evidence_recorded_as_missing_evidence_not_a_grade():
    res, be = run([paper("NEEDV vague", abstract="A short but present abstract text about something."), paper("G3 clear")])
    n = [a for a in res.assessments if a.title.startswith("NEEDV")][0]
    assert n.status == "missing_evidence" and n.judgment is not None and n.judgment.verdict == "insufficient_evidence"
    assert n.final_grade is None and n.rank is None and not n.recommended
    assert n in res.missing_evidence and n not in res.rejected and n not in res.background
    assert res.workload["n_missing_evidence"] == 1 and res.workload["n_rejected"] == 0


def test_model_declared_tier1_insufficient_continues_to_tier2():
    res, be = run([paper("INSUF vague"), paper("G3 clear")])
    v = [a for a in res.assessments if a.title.startswith("INSUF")][0]
    assert v.screen.decision == "insufficient_evidence" and v.judgment is not None      # not killed at Tier 1
    assert v.status != "tier1_rejected"                    # (the fake tier 2 then grades it 0 on its own merits)
    assert be.calls.count("stage4_judge") == 2


def test_low_confidence_reject_does_not_kill_the_paper():
    res, be = run([paper("LOWREJ unsure"), paper("G0 sure")])
    unsure = [a for a in res.assessments if a.title.startswith("LOWREJ")][0]
    sure = [a for a in res.assessments if a.title.startswith("G0")][0]
    assert unsure.status == "assessed" and "tier1_low_confidence_reject_passed" in unsure.flags     # passed on to the deep tier
    assert unsure.judgment is not None
    assert sure.status == "tier1_rejected"
    assert res.workload["n_tier1_low_conf_reject_passed"] == 1


def test_insufficient_papers_are_listed_after_the_ranked_ones_and_before_errors():
    res, _ = run([paper("NEEDV a", abstract="A short but present abstract text about something."), paper("G3 b")])
    assert [a.title for a in res.assessments] == ["G3 b", "NEEDV a"]
    assert [a.rank for a in res.assessments] == [1, None]


def test_tier1_prompt_is_reject_only_and_versioned():
    sysmsg = s4._SCREEN_SYSTEM
    assert "WHEN IN DOUBT, PASS" in sysmsg and "insufficient_evidence" in sysmsg and "positively" in sysmsg.lower()
    assert "3 = DIRECT" not in sysmsg                      # Tier 1 no longer grades 0-3
    assert s4.SCREEN_PROMPT_VERSION.endswith("-3")


def test_second_run_is_served_entirely_from_cache():
    papers = [paper("G3 a"), paper("G2 b"), paper("G0 c")]
    first, be1 = run(papers)
    second, be2 = run(papers)
    assert len(be1.calls) > 0 and be2.calls == []
    assert [(a.title, a.final_grade, a.rank) for a in first.assessments] == [(a.title, a.final_grade, a.rank) for a in second.assessments]
    assert second.workload["cache_hits"]["screen"] == 3 and second.workload["cache_hits"]["judge"] == 2    # G0 is rejected at Tier 1


def test_cache_is_keyed_by_the_problem_so_a_different_problem_is_not_served_stale_grades():
    papers = [paper("G3 a")]
    run(papers)

    class OtherPlan(Plan):
        scientific_objective = "binding"
    be = FakeBackend()
    run_stage4_v2(OtherPlan(), Prior(), papers, be, config=Stage4Config(), case_id="t")
    assert be.calls                                       # re-evaluated for the new problem


def test_llm_failure_is_an_error_not_a_silent_zero():
    papers = [paper("G3 ok"), paper("G3 broken")]
    res, _ = run(papers, backend=FakeBackend(fail_titles={"G3 broken"}), retries=0)
    broken = [a for a in res.assessments if a.title == "G3 broken"][0]
    assert broken.status == "error" and broken.rank is None and not broken.recommended
    assert broken not in res.rejected and broken in res.errors     # NOT counted as 'not relevant'
    assert res.workload["errors"] >= 1
    assert res.assessments[-1].title == "G3 broken"                 # errors are listed last, unranked


def test_duplicate_papers_are_assessed_once():
    p = paper("G3 dup")
    res, be = run([p, p.model_copy()])
    assert len(res.assessments) == 1 and be.calls.count("stage4_judge") == 1


def test_screening_is_batched():
    res, be = run([paper(f"G0 p{i}") for i in range(25)], screen_batch_size=12)
    assert be.calls.count("stage4_screen") == 3            # 12 + 12 + 1, not 25 calls


def test_problem_context_carries_the_transfer_hypotheses_but_no_route_labels():
    ctx = s4.build_problem_context(Plan(), Prior(), "my request")
    assert "my request" in ctx and "beta-helical passenger domains" in ctx and "right-handed parallel beta-helix" in ctx
    assert "flat" not in ctx.lower() and "3A" not in ctx


# ----------------------------------------------------- the real Europe PMC lookup (urlopen mocked)
class _Resp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode()

    def read(self, n=-1):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _epmc(result):
    return {"resultList": {"result": result}}


def test_europepmc_lookup_by_doi_strips_markup_and_is_cached(monkeypatch):
    calls = []

    def fake(req, timeout=None):
        calls.append(req.full_url)
        return _Resp(_epmc([{"doi": "10.9/Abc", "abstractText": "<h4>Background</h4><p>We built a <i>coarse-grained</i> model of folding.</p>"}]))
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    p = Paper(title="T", doi="10.9/ABC", abstract=None)
    out = s4.europepmc_abstract(p)
    assert out["via"] == "doi" and "coarse-grained model of folding" in out["abstract"] and "<" not in out["abstract"]
    assert "DOI%3A%2210.9/abc%22" in calls[0] or "DOI%3A%2210.9%2Fabc%22" in calls[0]
    assert s4.europepmc_abstract(p) == out and len(calls) == 1               # second call served from the cache


def test_europepmc_negative_answers_are_cached_but_failures_are_not(monkeypatch):
    n = {"calls": 0}

    def none_found(req, timeout=None):
        n["calls"] += 1
        return _Resp(_epmc([]))
    monkeypatch.setattr(urllib.request, "urlopen", none_found)
    p = Paper(title="T", doi="10.9/none")
    assert s4.europepmc_abstract(p)["abstract"] is None and s4.europepmc_abstract(p)["abstract"] is None and n["calls"] == 1

    def failing(req, timeout=None):
        n["calls"] += 1
        raise OSError("boom")
    monkeypatch.setattr(urllib.request, "urlopen", failing)
    q = Paper(title="Q", doi="10.9/fail")
    for _ in range(2):
        with pytest.raises(OSError):
            s4.europepmc_abstract(q)
    assert n["calls"] == 3                                                    # the failure was retried, never cached as 'no abstract'


def test_europepmc_rejects_a_record_whose_doi_does_not_match(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _Resp(_epmc([{"doi": "10.1/other", "abstractText": "x" * 100}])))
    assert s4.europepmc_abstract(Paper(title="T", doi="10.9/mine"))["abstract"] is None


def test_europepmc_falls_back_to_pmid_and_does_nothing_without_any_identifier(monkeypatch):
    urls = []
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout=None: (urls.append(req.full_url), _Resp(_epmc([{"abstractText": "Abstract text " * 5}])))[1])
    from cgscout.models import ExternalIds
    out = s4.europepmc_abstract(Paper(title="T", external_ids=ExternalIds(pmid="12345")))
    assert out["via"] == "pmid" and out["abstract"] and "EXT_ID" in urllib_unquote(urls[0])
    assert s4.europepmc_abstract(Paper(title="no ids")) is None and len(urls) == 1


def urllib_unquote(u):
    import urllib.parse
    return urllib.parse.unquote(u)


# --- the Europe PMC abstract lookup, pinned in detail (written BEFORE the function moves to sources/europepmc.py; every test below reaches it through
# `s4.europepmc_abstract`, which exists before and after the move, and patches `urllib.request.urlopen` itself, so none of them depends on where the code lives) ----
from cgscout import cache  # noqa: E402

# Literal digests of cache.content_hash("epmc-abstract-v1", via, ident), computed on the code before the move: a cache written by an earlier run must keep hitting.
_EPMC_KEY_DOI_ABC = "fb59a314b14aacfb6e06230f7ad325769728e9778a1828e1d34900a5753f16f5"       # ("epmc-abstract-v1", "doi", "10.9/abc")
_EPMC_KEY_PMID_12345 = "f3b600ef944133b37f63a8193969ea8287fe61d580845ce828ce6ac308b17189"    # ("epmc-abstract-v1", "pmid", "12345")
_EPMC_NS = "stage4_epmc_abstract"


class _SpyResp(_Resp):
    reads = []

    def read(self, n=-1):
        _SpyResp.reads.append(n)
        return super().read(n)


_MARKUP_CASES = [
    ("<h4>Background</h4><p>We built a <i>coarse-grained</i> model.</p>", "Background We built a coarse-grained model."),
    ("A &lt; B &amp; C &#945; &gt; D", "A < B & C α > D"),                       # entities are unescaped AFTER tags are stripped ...
    ("3 &lt;tag&gt; kept", "3 <tag> kept"),                                           # ... so an escaped tag-like text survives
    ("line one\n\n  line   two\t.\r\nthree", "line one line two . three"),
    ("<b><i>nested <u>tags</u></i></b>", "nested tags"),
    ("  plain  ", "plain"),
    ("<p></p><br/>", None),                                                            # nothing left after stripping -> "no abstract"
    ("", None),
    (None, None),
]


@pytest.mark.parametrize("text,expected", _MARKUP_CASES)
def test_europepmc_abstract_text_is_cleaned_exactly_like_this(monkeypatch, text, expected):
    rec = {"doi": "10.9/MK"}
    if text is not None:
        rec["abstractText"] = text
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _Resp(_epmc([rec])))
    assert s4.europepmc_abstract(Paper(title="T", doi="10.9/mk")) == {"abstract": expected, "via": "doi"}


def test_europepmc_request_has_exactly_this_url_headers_timeout_and_read_limit(monkeypatch):
    seen = []

    def fake(req, timeout=None):
        seen.append({"url": req.full_url, "ua": req.get_header("User-agent"), "accept": req.get_header("Accept"), "timeout": timeout, "method": req.get_method()})
        return _SpyResp(_epmc([{"doi": "10.9/req", "abstractText": "x" * 50}]))
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    _SpyResp.reads.clear()
    s4.europepmc_abstract(Paper(title="T", doi="10.9/req"))
    s4.europepmc_abstract(Paper(title="T2", doi="10.9/req2"), timeout=3.5)
    from cgscout.models import ExternalIds
    s4.europepmc_abstract(Paper(title="T3", external_ids=ExternalIds(pmid="12345")))
    base = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query="
    tail = "&resultType=core&format=json&pageSize=1"
    assert [c["url"] for c in seen] == [base + "DOI%3A%2210.9/req%22" + tail, base + "DOI%3A%2210.9/req2%22" + tail, base + "EXT_ID%3A12345%20AND%20SRC%3AMED" + tail]
    assert {c["ua"] for c in seen} == {"CGScout-stage4-abstract-fallback/0.1"}
    assert {c["accept"] for c in seen} == {"application/json"} and {c["method"] for c in seen} == {"GET"}
    assert [c["timeout"] for c in seen] == [20.0, 3.5, 20.0]                              # default 20 s, a given timeout is forwarded
    assert _SpyResp.reads == [2_000_000, 2_000_000, 2_000_000]                            # the response body is read with this cap


def test_europepmc_answers_are_cached_under_this_namespace_key_and_file_layout(monkeypatch, tmp_path):
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _Resp(_epmc([{"doi": "10.9/abc", "abstractText": "<p>Hello abstract</p>"}])))
    out = s4.europepmc_abstract(Paper(title="T", doi="10.9/abc"))
    assert out == {"abstract": "Hello abstract", "via": "doi"}
    assert cache.get(_EPMC_NS, _EPMC_KEY_DOI_ABC) == out                                  # namespace and key are the stored literals, not recomputed
    cache_dir = tmp_path / "test-cache"                                                   # tests/conftest.py's per-test CGSCOUT_CACHE_DIR
    assert (cache_dir / _EPMC_NS / f"{_EPMC_KEY_DOI_ABC}.json").is_file()
    assert json.loads((cache_dir / _EPMC_NS / f"{_EPMC_KEY_DOI_ABC}.json").read_text())["value"] == out
    from cgscout.models import ExternalIds
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _Resp(_epmc([{"abstractText": "pmid abstract text " * 3}])))
    out2 = s4.europepmc_abstract(Paper(title="P", external_ids=ExternalIds(pmid="12345")))
    assert out2["via"] == "pmid" and cache.get(_EPMC_NS, _EPMC_KEY_PMID_12345) == out2


def test_europepmc_found_but_without_abstract_is_cached_and_a_failure_writes_nothing(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _Resp(_epmc([{"doi": "10.9/abc"}])))
    assert s4.europepmc_abstract(Paper(title="T", doi="10.9/abc")) == {"abstract": None, "via": "doi"}
    assert cache.get(_EPMC_NS, _EPMC_KEY_DOI_ABC) == {"abstract": None, "via": "doi"}
    cache_miss_before = cache.get(_EPMC_NS, _EPMC_KEY_PMID_12345)
    assert cache_miss_before is None

    def failing(req, timeout=None):
        raise OSError("down")
    monkeypatch.setattr(urllib.request, "urlopen", failing)
    from cgscout.models import ExternalIds
    with pytest.raises(OSError):
        s4.europepmc_abstract(Paper(title="P", external_ids=ExternalIds(pmid="12345")))
    assert cache.get(_EPMC_NS, _EPMC_KEY_PMID_12345) is None                              # the failure left no entry


def test_europepmc_doi_is_stripped_and_lowercased_for_the_query_and_the_cache_key(monkeypatch):
    urls = []
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout=None: (urls.append(req.full_url), _Resp(_epmc([{"doi": "10.9/ABC", "abstractText": "x" * 60}])))[1])
    first = s4.europepmc_abstract(Paper(title="T", doi="  10.9/AbC \n"))
    assert "DOI%3A%2210.9/abc%22" in urls[0]
    assert cache.get(_EPMC_NS, _EPMC_KEY_DOI_ABC) == first
    assert s4.europepmc_abstract(Paper(title="T", doi="10.9/abc")) == first and len(urls) == 1      # same record, served from the cache


def test_europepmc_prefers_the_doi_over_the_pmid_when_a_paper_has_both(monkeypatch):
    from cgscout.models import ExternalIds
    urls = []
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout=None: (urls.append(urllib_unquote(req.full_url)), _Resp(_epmc([{"doi": "10.9/both", "abstractText": "x" * 60}])))[1])
    out = s4.europepmc_abstract(Paper(title="T", doi="10.9/both", external_ids=ExternalIds(pmid="777")))
    assert out["via"] == "doi" and 'DOI:"10.9/both"' in urls[0] and "EXT_ID" not in urls[0]


def test_the_usable_abstract_threshold_is_40_characters_after_stripping():
    assert s4.MIN_ABSTRACT_CHARS == 40
    cases = {None: False, "": False, "x" * 39: False, "x" * 40: True, "  " + "x" * 39 + "  ": False, "  " + "x" * 40 + " ": True}
    for text, usable in cases.items():
        assert s4.has_usable_abstract(Paper(title="T", abstract=text)) is usable, repr(text)


def test_stage4_uses_the_default_europepmc_lookup_and_accepts_exactly_40_recovered_characters(monkeypatch):
    def fake(req, timeout=None):
        ident = urllib_unquote(req.full_url).split('DOI:"', 1)[1].split('"', 1)[0]
        text = {"10.7/r40": "x" * 40, "10.7/r39": "x" * 39}[ident]
        return _Resp(_epmc([{"doi": ident, "abstractText": text}]))
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    ok = paper("G3 recovered40", abstract=None).model_copy(update={"doi": "10.7/r40"})
    short = paper("G3 recovered39", abstract=None).model_copy(update={"doi": "10.7/r39"})
    res, be = run([ok, short], abstract_lookup=None)                                      # None = the production default: the real europepmc_abstract
    by = {a.title: a for a in res.assessments}
    assert by["G3 recovered40"].abstract_source == "europepmc" and by["G3 recovered40"].status == "assessed"
    assert by["G3 recovered39"].status == "missing_evidence" and by["G3 recovered39"].abstract_source == "none"
    assert res.workload["n_abstract_lookup_attempted"] == 2 and res.workload["n_abstract_recovered_europepmc"] == 1


# --- tier-1 sees the whole abstract (regression: a 1200-character cut hid the computational sentence of long abstracts) ------------------------------
def _long_abstract_paper():
    return Paper(title="Force-induced opening of a lever arm", sources=["openalex"], doi="10.0000/long",
                 abstract=("Single-molecule pulling experiments were performed. " * 30) + "Coarse-grained Brownian dynamics detailed the opening of the ankle.")


def test_default_screen_message_contains_the_end_of_a_long_abstract():
    p = _long_abstract_paper()
    assert len(p.abstract) > 1200
    msg = s4._screen_message("CTX", [(1, p)], Stage4Config())
    assert "Coarse-grained Brownian dynamics" in msg and "..." not in msg.split("Abstract:")[1]


def test_screen_abstract_cap_is_still_honoured_and_part_of_the_cache_key():
    p = _long_abstract_paper()
    msg = s4._screen_message("CTX", [(1, p)], Stage4Config(screen_abstract_chars=1200))
    assert "Coarse-grained Brownian dynamics" not in msg and msg.rstrip().endswith("...")
    a = s4.cache.content_hash(s4.SCREEN_PROMPT_VERSION, "low", "1200", "ctx", "k", "h")
    b = s4.cache.content_hash(s4.SCREEN_PROMPT_VERSION, "low", "4000", "ctx", "k", "h")
    assert a != b
