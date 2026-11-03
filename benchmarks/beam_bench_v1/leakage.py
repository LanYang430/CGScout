"""Simple, transparent leakage checks for generated need cards and queries. No classifier: every flag names the
exact string that triggered it, and flags never delete anything -- the human reviewer sees them.

Checks (query or card text  vs  the seed paper):
  doi             a DOI pattern
  author          a seed author's surname (>=4 letters) as a whole word
  model_name      a CG model / force-field name: a small lexicon + names the abstract itself attaches to "model" /
                  "force field" (capitalised, upper-case or digit-containing tokens), minus generic abbreviations and
                  minus words the need card legitimately uses for the researcher's own system
  title_ngram     >=4 consecutive words shared with the paper title        (cards: >=5)
  copied_phrase   queries: >=4 consecutive words shared with the abstract that contain a RARE word and do not occur
                  in the need card the query was written from (the query writer never saw the abstract, so such an
                  overlap is a coincidence or a memorised phrase).  cards: >=6 consecutive words copied.
A word is RARE if it occurs in <3% of a background pool of related abstracts (the seed-candidate pool).
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

DOI_RE = re.compile(r"10\.\d{4,9}/\S+|\bdoi\b", re.I)
TOKEN_RE = re.compile(r"[a-z0-9α-ω]+")

MODEL_LEXICON = {
    "martini", "sirah", "awsem", "unres", "cabs", "hps", "mpipi", "calvados", "elnedin", "spica", "oxdna", "oxrna",
    "3spn", "3spn2", "hire", "hirerna", "cgdna", "primo", "opep", "mrsfm", "kb", "go", "gō", "gromos", "ff14sb",
    "pace", "bmw", "emap", "triple", "tiny", "swm", "gaussian", "smog", "sbm", "cgmd", "rosetta", "cafemol", "lammps",
}
# model-name patterns are only taken from tokens that are NOT in this generic list
GENERIC_CAPS = {
    "cg", "md", "dna", "rna", "idp", "idps", "idr", "idrs", "atp", "gtp", "lcst", "ucst", "pdb", "nmr", "fret", "saxs",
    "gpcr", "aa", "afm", "sars", "cov", "covid", "mrna", "trna", "mhc", "cnn", "ml", "ai", "ph", "lipid", "pmf",
    "mc", "bd", "dpd", "sds", "peg", "usa", "ii", "iii", "iv", "vi", "vii", "viii", "ix", "xi", "xii",
}
STOP = set("a an the of in on for to and or with by as at from is are was were be been that this these those it its "
           "into onto via using use used can could would should we our their they which what how why when where while "
           "also both each such than then there here more most other any some many much very".split())


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower().replace("-", " ").replace("–", " ").replace("—", " "))


def ngrams(toks: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def longest_shared_run(a: list[str], b: list[str]) -> tuple[int, str]:
    """Longest run of consecutive tokens common to a and b (O(len(a)*len(b)), texts are short)."""
    best, best_end = 0, 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best, best_end = cur[j], i
        prev = cur
    return best, " ".join(a[best_end - best:best_end])


def build_rare_words(background_abstracts: Iterable[str], max_df: float = 0.03) -> set[str]:
    """Words that occur in fewer than `max_df` of the background abstracts (rare = discriminative)."""
    docs = [set(tokens(t)) for t in background_abstracts if t]
    if not docs:
        return set()
    df: dict[str, int] = {}
    for d in docs:
        for w in d:
            df[w] = df.get(w, 0) + 1
    limit = max_df * len(docs)
    seen = set().union(*docs)
    return {w for w in seen if df[w] <= limit and w not in STOP and len(w) > 2}


def abstract_model_names(abstract: str) -> set[str]:
    """Names the abstract attaches to 'model' / 'force field' (e.g. 'the SIRAH force field', 'Martini 3 model')."""
    out = set()
    for m in re.finditer(r"\b([A-Z][A-Za-z0-9À-ſ]*[A-Z0-9][A-Za-z0-9]*|[A-Z]{2,}[a-z0-9]*)\b\s*(?:\d+\s*)?(?:force[- ]?field|model|ff|potential|framework)\b", abstract or ""):
        w = m.group(1).lower()
        if w not in GENERIC_CAPS and w not in STOP:
            out.add(w)
    return out


def _surnames(authors: list[str]) -> set[str]:
    out = set()
    for a in authors or []:
        parts = re.findall(r"[A-Za-zÀ-ſ'\-]+", a)
        if parts and len(parts[-1]) >= 4:
            out.add(parts[-1].lower().replace("-", " ").split()[-1])
    return out


def check_text(text: str, *, title: str, abstract: str, authors: list[str], rare: set[str],
               card_text: Optional[str] = None, allowed_words: Iterable[str] = (), is_card: bool = False) -> list[str]:
    """Return a list of human-readable flags (empty = clean). `allowed_words`: words that legitimately belong to the
    researcher's own system (taken from the need card) and therefore are not 'model names'."""
    flags: list[str] = []
    tt = tokens(text)
    tset = set(tt)
    if DOI_RE.search(text or ""):
        flags.append(f"doi:{DOI_RE.search(text).group(0)!r}")
    for s in sorted(_surnames(authors) & tset):
        flags.append(f"author:{s!r}")
    allowed = {w.lower() for w in allowed_words}
    names = (MODEL_LEXICON | abstract_model_names(abstract)) - allowed
    for n in sorted(names & tset):
        flags.append(f"model_name:{n!r}")
    title_t = tokens(title)
    n_title = 5 if is_card else 4
    run, phrase = longest_shared_run(tt, title_t)
    if run >= n_title:
        flags.append(f"title_ngram:{phrase!r}")
    abs_t = tokens(abstract)
    if is_card:
        run, phrase = longest_shared_run(tt, abs_t)
        if run >= 6:
            flags.append(f"copied_phrase:{phrase!r}")
    else:
        card_t = tokens(card_text or "")
        card_grams = ngrams(card_t, 4) if card_t else set()
        for g in sorted(ngrams(tt, 4) & ngrams(abs_t, 4)):
            if g in card_grams:
                continue
            if any(w in rare for w in g):
                flags.append(f"copied_phrase:{' '.join(g)!r}")
    return flags
