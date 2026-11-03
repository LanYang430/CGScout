"""Metadata coverage audit (read-only; no production code, no benchmark logic):
for every Stage-3-kept paper whose OpenAlex abstract is missing/unusable in a regression label, ask three other
public sources *by DOI* whether they have an abstract -- Crossref, Europe PMC (covers PubMed/PMC/preprints) and
the publisher landing page (doi.org redirect; meta tags / JSON-LD only, nothing saved) -- and record the work type.
Polite: one request per source per paper, ~1 s apart, identifying User-Agent, short timeouts. A blocked or failed
request is recorded as 'unknown', never as 'no abstract'.   Usage: abstract_coverage_audit.py [--label s4_tier1v2]"""
import argparse, html, json, os, re, sys, time
import urllib.parse, urllib.request, urllib.error
from pathlib import Path

HERE = Path(__file__).resolve().parent
MIN_USABLE = 40       # same threshold Stage 4 uses for 'usable abstract'
SUBSTANTIVE = 200     # chars: a real abstract rather than a stub


def ua():
    m = os.environ.get("OPENALEX_MAILTO")
    return "BEAM-metadata-audit/0.1" + (f" (mailto:{m})" if m else "")


def get(url, accept="application/json", timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": ua(), "Accept": accept})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(600_000).decode("utf-8", "replace"), r.geturl()
    except urllib.error.HTTPError as e:
        return e.code, "", url
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}", url


def strip_tags(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


def crossref(doi):
    st, body, _ = get("https://api.crossref.org/works/" + urllib.parse.quote(doi))
    if st == 404:
        return {"status": "not_in_crossref", "abstract_chars": 0}
    if st != 200:
        return {"status": f"unknown(http {st})", "abstract_chars": None}
    m = json.loads(body)["message"]
    ab = strip_tags(m.get("abstract") or "")
    ab = re.sub(r"^abstract[:\s]*", "", ab, flags=re.I)
    return {"status": "ok", "abstract_chars": len(ab), "type": m.get("type"), "subtype": m.get("subtype"), "page": m.get("page"),
            "container": (m.get("container-title") or [None])[0], "title": (m.get("title") or [None])[0]}


def europepmc(doi):
    q = urllib.parse.quote(f'DOI:"{doi}"')
    st, body, _ = get(f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={q}&resultType=core&format=json&pageSize=1")
    if st != 200:
        return {"status": f"unknown(http {st})", "abstract_chars": None}
    res = json.loads(body).get("resultList", {}).get("result", [])
    if not res:
        return {"status": "not_indexed", "abstract_chars": 0}
    r = res[0]
    ab = strip_tags(r.get("abstractText") or "")
    return {"status": "ok", "abstract_chars": len(ab), "source": r.get("source"), "pubType": r.get("pubType"), "pmid": r.get("pmid"), "pmcid": r.get("pmcid")}


META = re.compile(r'<meta\s+[^>]*?(?:name|property)=["\'](citation_abstract|dc\.description|description|og:description|twitter:description)["\'][^>]*?content=["\']([^"\']*)["\']', re.I | re.S)
META2 = re.compile(r'<meta\s+[^>]*?content=["\']([^"\']*)["\'][^>]*?(?:name|property)=["\'](citation_abstract|dc\.description|description|og:description|twitter:description)["\']', re.I | re.S)


def publisher(doi, title):
    st, body, final = get("https://doi.org/" + urllib.parse.quote(doi), accept="text/html", timeout=25)
    if st is None or st >= 400:
        return {"status": f"blocked/unknown(http {st})", "chars": None, "final_host": urllib.parse.urlparse(final).netloc}
    best = 0; kind = None
    cands = [(k.lower(), v) for k, v in META.findall(body)] + [(k.lower(), v) for v, k in META2.findall(body)]
    for k, v in cands:
        t = strip_tags(v)
        if t.lower().startswith(title.lower()[:40]):
            continue                      # description that merely repeats the title
        if len(t) > best:
            best, kind = len(t), k
    for m in re.finditer(r'"(?:abstract|description)"\s*:\s*"((?:[^"\\]|\\.){80,})"', body):    # JSON-LD
        t = strip_tags(json.loads('"' + m.group(1) + '"')) if '\\' not in m.group(1)[:0] else m.group(1)
        if len(t) > best:
            best, kind = len(t), "json-ld"
    return {"status": "ok", "chars": best, "via": kind, "final_host": urllib.parse.urlparse(final).netloc}


def classify(r, cr):
    t, page, title = r.get("oa_type"), (cr or {}).get("page") or "", r["title"]
    if t == "conference-abstract" or re.match(r"^(BPS\d{4}|\d[A-Z]-\d+)", title) or (re.fullmatch(r"\d+a", str(page)) if page else False) \
            or re.match(r"10\.1016/j\.bpj\.\d{4}\.11\.\d+$", (r.get("doi") or "")):
        return "meeting/conference abstract"
    if t == "book-chapter" or (cr or {}).get("type") in ("book-chapter",):
        return "book chapter"
    if t is None and not r.get("doi"):
        return "no-DOI record (type unknown)"
    return "research/review article"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--label", default="s4_tier1v2"); a = ap.parse_args()
    rows = json.loads((HERE / "results" / "abstract_audit_input.json").read_text())
    out = []
    for i, r in enumerate(rows, 1):
        rec = dict(r)
        if r.get("doi"):
            rec["crossref"] = crossref(r["doi"]); time.sleep(0.5)
            rec["europepmc"] = europepmc(r["doi"]); time.sleep(0.5)
            rec["publisher"] = publisher(r["doi"], r["title"]); time.sleep(1.0)
        rec["work_class"] = classify(r, rec.get("crossref"))
        out.append(rec)
        print(f"[{i}/{len(rows)}] {rec['work_class'][:12]:12s} CR={rec.get('crossref',{}).get('abstract_chars')} EPMC={rec.get('europepmc',{}).get('abstract_chars')} PUB={rec.get('publisher',{}).get('chars')} {r['title'][:60]}", flush=True)
    (HERE / "results" / "abstract_audit.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
