"""Thin, read-only client for Europe PMC's REST API -- fetches ONLY a
paper's Methods-like section text (never the whole full text), for the ONE
narrow purpose Stage 3's CG-evidence gate needs it for: a paper's CG/force-
field usage is very often stated only in Methods, never in its own
title/abstract (real diagnosed case: "Structural basis of outer membrane
biogenesis by the TamAB translocase" -- its abstract says only "molecular
dynamics simulations"; "Martini" and "coarse-grained" appear 9 and 8 times
in its Methods section, zero times in title+abstract). No API key, no
registration -- Europe PMC's REST API is free and open.

Docs: https://europepmc.org/RestfulWebService

This is a NEW capability -- nothing in beam_agent/sources/ reached past
title/abstract before this. Deliberately narrow scope:
  - looks up a paper by DOI only (not by title/author -- avoids ambiguous
    matches)
  - only attempts a fetch for open-access, Europe-PMC-indexed papers
    (isOpenAccess == "Y" and inEPMC == "Y") -- never tries to scrape a
    paywalled page or a PDF
  - extracts <sec> elements whose type/title look like Methods (including
    "Simulation Methods", "Supplementary Methods" variants) and returns
    their concatenated TEXT ONLY, truncated to a snippet -- callers never
    receive (and this module never processes) the whole document
  - never raises: network errors, XML parse errors, missing sections, or
    "paper not found" all return None -- a caller treats None exactly like
    "no additional evidence available", same fail-open philosophy every
    other source in this package already follows (e.g. openalex.py's
    retry-then-explicit-exception is the deliberate EXCEPTION to this, for
    rate-limit accounting; this module has no equivalent budget to track)

Cached via beam_agent.cache (dev-speed + avoids re-hitting Europe PMC for
the same DOI across runs) -- keyed on DOI alone, no TTL (a paper's own
Methods text does not change after publication).

The abstract lookup (`europepmc_abstract`, with `_strip_markup`, the cache namespace `_NS_EPMC` and the usable-abstract threshold `MIN_ABSTRACT_CHARS`) was moved
here verbatim from assessment/stage4.py (Phase 5 / C3) so that both Stage 3 (abstract recovery) and Stage 4 (abstract fallback) reach it without Stage 3
importing Stage 4.  It is a different client from fetch_methods_text: it uses urllib, looks a paper up by DOI (else PMID) in the search endpoint, and RAISES on
a transport / HTTP failure, so a caller can tell "no abstract exists" from "could not ask"; its answers are cached under its own namespace.
"""
from __future__ import annotations

import html
import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Optional

import requests

from cgscout import cache
from cgscout.models import Paper

_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
_FULLTEXT_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
_CACHE_NAMESPACE = "europepmc_methods_snippet"
# A Methods section is rarely huge -- generous enough that a real paper's
# combined Methods content is never cut off mid-subsection (the earlier,
# much smaller cap truncated BEFORE the simulation-specific subsection was
# even reached on a real paper -- protein expression/purification/
# structure-determination subsections came first in document order and
# alone exceeded it). This is the text CHECKED for markers, not what gets
# returned/logged -- see _make_snippet() for the actual short window
# returned around a match.
_MAX_METHODS_TEXT_CHARS = 40_000
_SNIPPET_WINDOW_CHARS = 300  # chars of context on each side of a matched marker, for the returned/logged snippet
_REQUEST_TIMEOUT = 10.0

_METHODS_TITLE_RE = re.compile(
    r"method|simulation|supplementary|materials", re.IGNORECASE
)


def _find_pmcid(doi: str) -> Optional[str]:
    try:
        resp = requests.get(
            _SEARCH_URL,
            params={"query": f"DOI:{doi}", "format": "json"},
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json().get("resultList", {}).get("result", [])
    except Exception:  # noqa: BLE001 -- network/parse problem here is a miss, not a crash
        return None
    for r in results:
        if r.get("isOpenAccess") == "Y" and r.get("inEPMC") == "Y" and r.get("pmcid"):
            return r["pmcid"]
    return None


def _extract_methods_text(xml_bytes: bytes) -> Optional[str]:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return None
    chunks: list[str] = []
    for sec in root.iter("sec"):
        title_el = sec.find("title")
        title_text = "".join(title_el.itertext()) if title_el is not None else ""
        sec_type = sec.attrib.get("sec-type", "")
        if _METHODS_TITLE_RE.search(title_text) or _METHODS_TITLE_RE.search(sec_type):
            text = " ".join(t.strip() for t in sec.itertext() if t.strip())
            if text:
                chunks.append(text)
    if not chunks:
        return None
    return " ".join(chunks)[:_MAX_METHODS_TEXT_CHARS]


def fetch_methods_text(doi: Optional[str]) -> Optional[str]:
    """Returns up to ~40,000 characters of concatenated Methods-like
    section text for `doi` -- the text a caller checks for CG/force-field
    markers -- or None if the DOI is missing, the paper isn't found / isn't
    open-access in Europe PMC, no Methods-like section was found, or
    anything about the fetch/parse fails. Never raises. This is
    intentionally NOT itself a short "snippet" -- see snippet_around() to
    build one, once a caller knows which marker(s) actually matched."""
    if not doi:
        return None
    cache_key = cache.content_hash(doi)
    cached = cache.get(_CACHE_NAMESPACE, cache_key)
    if cached is not None:
        return cached.get("text")  # may itself be None -- a cached "no Methods text found" miss

    text: Optional[str] = None
    pmcid = _find_pmcid(doi)
    if pmcid:
        try:
            resp = requests.get(_FULLTEXT_URL.format(pmcid=pmcid), timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
            text = _extract_methods_text(resp.content)
        except Exception:  # noqa: BLE001 -- fetch/parse failure is a miss, not a crash
            text = None

    cache.set(_CACHE_NAMESPACE, cache_key, {"text": text})
    return text


def snippet_around(text: str, markers: tuple[str, ...]) -> Optional[str]:
    """A short (~600-char) window of `text` centered on the first
    occurrence of any of `markers` (case-insensitive) -- the actual
    reportable/loggable "snippet", as opposed to fetch_methods_text()'s
    much larger marker-search text. None if no marker is found."""
    low = text.lower()
    best_idx = None
    for m in markers:
        idx = low.find(m.lower())
        if idx != -1 and (best_idx is None or idx < best_idx):
            best_idx = idx
    if best_idx is None:
        return None
    start = max(0, best_idx - _SNIPPET_WINDOW_CHARS)
    end = min(len(text), best_idx + _SNIPPET_WINDOW_CHARS)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return prefix + text[start:end].strip() + suffix


# ===== moved verbatim from assessment/stage4.py (Phase 5 / C3): the abstract lookup used by Stage 3 recovery and the Stage 4 fallback =====
_NS_EPMC = "stage4_epmc_abstract"
MIN_ABSTRACT_CHARS = 40     # below this there is no usable abstract


def _strip_markup(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


def europepmc_abstract(p: Paper, *, timeout: float = 20.0) -> Optional[dict]:
    """Look the paper up in Europe PMC by DOI (else PMID) and return {"abstract": str|None, "via": "doi"|"pmid"}.
    Returns None when the paper has neither identifier (nothing to look up). Raises on a transport / HTTP
    failure so the caller can tell 'no abstract exists' from 'could not ask'. Successful answers -- including
    'found, but no abstract' -- are cached, failures never are."""
    doi = (getattr(p, "doi", None) or "").strip().lower()
    pmid = (getattr(getattr(p, "external_ids", None), "pmid", None) or "").strip()
    if doi:
        via, ident, query = "doi", doi, f'DOI:"{doi}"'
    elif pmid:
        via, ident, query = "pmid", pmid, f"EXT_ID:{pmid} AND SRC:MED"
    else:
        return None
    key = cache.content_hash("epmc-abstract-v1", via, ident)
    hit = cache.get(_NS_EPMC, key)
    if hit is not None:
        return hit
    url = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=" + urllib.parse.quote(query)
           + "&resultType=core&format=json&pageSize=1")
    req = urllib.request.Request(url, headers={"User-Agent": "BEAM-stage4-abstract-fallback/0.1", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:        # URLError / HTTPError / timeout propagate
        data = json.loads(r.read(2_000_000).decode("utf-8", "replace"))
    results = (data.get("resultList") or {}).get("result") or []
    rec = results[0] if results else None
    if rec is not None and via == "doi" and (rec.get("doi") or "").strip().lower() != doi:
        rec = None                                                  # not the same record -> do not borrow its abstract
    abstract = _strip_markup(rec.get("abstractText") or "") if rec else ""
    out = {"abstract": abstract or None, "via": via}
    cache.set(_NS_EPMC, key, out)
    return out
