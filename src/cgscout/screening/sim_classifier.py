"""Stage 3 experiment-vs-simulation classifier -- production runtime
scorer for the validated PCA(60)+StandardScaler+LogisticRegression
pipeline (5-fold GroupKFold CV ROC-AUC 0.9160 on the cleaned 424-row
training set; external held-out 427-row test set ROC-AUC 0.8477, 89.2%
recall / 61.3% precision at the default 0.5 threshold -- see
sim_classifier_train.py's module docstring for the full validation
history this session ran against beam_stage3_training_ready_yes_no.csv /
beam_stage3_external_test_clean_519.csv).

Answers ONE question per paper: "did the authors themselves perform a
computational/molecular simulation" (title+abstract only -- it was never
trained or evaluated on full text, so this module never fetches it).
It has NO signal about topical relevance to a specific query or system --
that is scored separately, upstream, by the existing keyword-overlap gate
in filter.py.

Dependencies (openai, scikit-learn, joblib, numpy; scikit-learn is needed
to read sim_classifier_model.joblib) are imported lazily inside functions so
importing this module -- or filter.py, which imports it --
never requires these packages unless a classifier score is actually
requested.
"""
from __future__ import annotations

from pathlib import Path

from cgscout.models import Paper

# The single knob a caller is expected to change -- kept as one obvious
# module-level constant (matching this file's own
# _DEFAULT_FULLTEXT_LOOKUP_BUDGET-style convention) rather than buried
# inside scoring logic. filter_papers() reads this as its default but
# accepts an override per call.
SIM_CLASSIFIER_THRESHOLD = 0.5

EMBED_MODEL = "text-embedding-3-small"
MODEL_PATH = Path(__file__).resolve().parent / "sim_classifier_model.joblib"

_model = None  # lazy singleton, see _load_model()


class ClassifierUnavailable(RuntimeError):
    """Raised when the persisted model artifact (MODEL_PATH) is missing --
    e.g. sim_classifier_train.py was never run in this checkout. Distinct
    from a generic RuntimeError so a caller can choose to fail open
    (keep, unscored) rather than crash Stage 3 entirely."""


def _load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise ClassifierUnavailable(
                f"sim classifier model artifact not found at {MODEL_PATH} -- "
                f"run sim_classifier_train.py once to build it"
            )
        import joblib  # local import -- see module docstring
        _model = joblib.load(MODEL_PATH)
    return _model


def embed_texts(texts: list[str]) -> "np.ndarray":  # noqa: F821 -- numpy typed lazily
    """Batched OpenAI embedding call, same pattern as llm.py's
    OpenAI() client: reads OPENAI_API_KEY from env, never a literal key."""
    import numpy as np
    from openai import OpenAI  # local import -- see module docstring

    client = OpenAI()
    out: list[list[float]] = [None] * len(texts)  # type: ignore[list-item]
    BATCH = 200
    for start in range(0, len(texts), BATCH):
        batch = texts[start:start + BATCH]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        for i, d in zip(range(start, start + len(batch)), resp.data):
            out[i] = d.embedding
    return np.array(out)


def _paper_text(paper: Paper) -> str:
    return f"{paper.title}\n\n{paper.abstract or ''}"


def classify_papers(papers: list[Paper]) -> list[float]:
    """P(the authors themselves performed a simulation), one score per
    paper, same order as `papers`. One batched embedding call for the
    whole list -- the efficient path when scoring many candidates at
    once (filter_papers()'s normal use)."""
    if not papers:
        return []
    model = _load_model()
    X = embed_texts([_paper_text(p) for p in papers])
    return model.predict_proba(X)[:, 1].tolist()


def default_classify_fn(papers: list[Paper]) -> list[float]:
    """The real classifier, as a `classify_fn` matching the signature
    filter.py expects -- its default when a caller
    (production code) doesn't inject a fake one (tests do, to stay
    network-free; see tests/test_relevance_filter_handoff.py)."""
    return classify_papers(papers)
