"""Stage 3 high-confidence review/perspective filter -- production version
of the standalone `cheap_review_filter_v2.py` diagnostic (validated this
session against a combined ~1076-paper labeled population: 20/444 training
rows and 84/511 test rows flagged this narrow version alone).

Deliberately narrow -- ONLY the two highest-confidence signal categories,
per explicit instruction, everything else (venue-name regex, "recent
advances in"/"we summarize"/"an overview of" phrasing, the eLife peer-
review-artifact title pattern) left out as too prone to false positives:

  1. OpenAlex's own `type` field says "review" (Paper.oa_type -- see
     beam_agent/sources/openalex.py's `_SELECT`/`_to_paper()`). OpenAlex
     normalizes publisher-level sub-types ("Review", "Mini Review",
     "Advanced Review", ...) into this one coarse category; there is no
     more granular field to check beyond it. None for any non-OpenAlex
     source, or if OpenAlex omitted it -- never treated as a match.
  2. The abstract explicitly self-declares with one of exactly these four
     phrasings: "in this review", "this review", "here we review",
     "we review".

Runs BEFORE the classifier in relevance_filter_handoff.filter_papers() --
a flagged paper is dropped without ever spending an embedding call.
"""
from __future__ import annotations

import re

from cgscout.models import Paper

REVIEW_PHRASE_RE = re.compile(
    r"\bin this review\b|\bthis review\b|\bhere,? we review\b|\bwe review\b",
    re.IGNORECASE,
)


def review_filter_detail(paper: Paper) -> dict:
    type_hit = paper.oa_type == "review"
    m = REVIEW_PHRASE_RE.search(paper.abstract or "")
    return {
        "flagged": type_hit or bool(m),
        "type_hit": type_hit,
        "matched_phrase": m.group(0) if m else "",
    }


def is_high_confidence_review(paper: Paper) -> bool:
    return review_filter_detail(paper)["flagged"]
