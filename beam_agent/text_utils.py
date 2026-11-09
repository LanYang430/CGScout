"""Shared lightweight text helpers: tokenization and stopword filtering.

Deliberately dumb on purpose -- regex tokenization plus a small stopword list,
no NLP libraries, no LLM. Good enough for query trimming and keyword-overlap
relevance scoring; both `query_expansion.py` and `relevance_filter.py` build on
this so the two stay consistent about what counts as a "meaningful word."
"""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    {
        "a", "an", "the", "of", "for", "with", "in", "on", "to", "and", "or",
        "using", "based", "study", "studies", "via", "from", "by", "into",
        "is", "are", "was", "were", "be", "been", "being", "that", "this",
        "these", "those", "as", "at", "can", "its", "it", "we", "our", "us",
        "new", "novel", "toward", "towards", "approach", "approaches",
        "analysis", "such", "than", "their", "which", "how", "what",
    }
)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-]+")


def tokenize(text: str) -> list[str]:
    """Lowercased word tokens (letters + internal hyphens only, digits dropped)."""
    return _WORD_RE.findall(text.lower())


def keywords(text: str, *, min_len: int = 3) -> set[str]:
    """Tokenize and drop stopwords / very short tokens."""
    return {t for t in tokenize(text) if t not in STOPWORDS and len(t) >= min_len}


def _stem_word(word: str, *, prefix_len: int = 4) -> str:
    """Truncate a single hyphen-free word to its first `prefix_len` characters."""
    return word if len(word) <= prefix_len else word[:prefix_len]


def stem(token: str, *, prefix_len: int = 4) -> str:
    """Crude, deliberately dumb stemmer for cross-text keyword matching.

    NOT a real stemmer -- no Porter/Snowball morphology rules, just a
    fixed-length prefix per hyphen-separated part. That's enough to unify
    common regular forms (protein/proteins, fold/folding/folded) and, more
    importantly, catches irregular derivations a suffix-stripper would miss
    entirely: "helix" and "helical" share only a 4-letter prefix, not a
    suffix relationship, so no amount of stripping "-ing"/"-ed"/"-s" off
    either one makes them equal.

    The cost is occasional false-positive collisions between unrelated words
    that happen to share a prefix (model/modern both stem to "mode"). That
    tradeoff is intentional: RelevanceFilter is designed to over-retain
    rather than over-reject (see its module docstring) -- a collision here
    just means one extra paper reaches manual/LLM review; a missed match
    means a paper silently vanishes before anyone sees it.
    """
    return "-".join(_stem_word(part, prefix_len=prefix_len) for part in token.split("-"))


def stemmed_keywords(text: str, *, min_len: int = 3) -> set[str]:
    """Like `keywords()`, but each token is reduced via `stem()` first.

    Use this (not `keywords()`) whenever comparing two *different* texts for
    topical overlap -- e.g. a user query against a paper's title/abstract.
    Keep using `keywords()` when working within one text (e.g. trimming a
    query string), where exact surface tokens are what you want to preserve.
    """
    return {stem(t) for t in keywords(text, min_len=min_len)}


def core_keyword_string(text: str) -> str:
    """Rebuild a shorter query string keeping only non-stopword tokens, in order.

    Preserves each surviving word's original surface form (case, attached
    punctuation) rather than the normalized token -- this is meant to feed
    back into a search API, not to be a canonical key.
    """
    kw = keywords(text)
    kept = []
    for raw_word in re.findall(r"\S+", text):
        normalized = re.sub(r"[^A-Za-z\-]", "", raw_word).lower()
        if normalized in kw:
            kept.append(raw_word)
    return " ".join(kept)


# Methodology synonym clusters (canonical term, [synonyms]) -- vocabulary, not protein names.  Lives here (not in concept_resolver.py, where it
# was first defined) so that the Stage-3 filter (relevance_filter_handoff.py) can read it without importing the v1 concept resolver.
# concept_resolver.py imports it back under its old private name.
METHODOLOGY_TERM_CLUSTERS: list[tuple[str, list[str]]] = [
    ("coarse-grained", ["cg", "coarse grained"]),
    ("all-atom", ["all atom", "atomistic"]),
    ("molecular dynamics", ["md simulation", "md"]),
    ("machine learning", ["ml", "deep learning", "neural network"]),
]
