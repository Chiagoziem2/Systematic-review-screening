"""Active-learning screening loop.

Simulates a human screener working through a corpus, where a model re-ranks the
unscreened pile as labels come in. Core question: how much of the corpus can you
avoid reading and still find (almost) all the relevant papers?

WHAT CHANGED FROM v1
    query_relevance is now YOUR argmax generalised to top-k (np.argsort), so the
    model can refit every `batch_size` records instead of every single record.
    With batch_size=1 it is exactly your original argmax. This is purely an
    efficiency change: Brouwer_2019 has 38,114 records, and refitting 38,114 times
    is not tractable, but refitting ~760 times at batch_size=50 is.
    Also added: swappable classifier (`model_factory`) and vectorizer settings
    (`vectorizer_kwargs`) so variants can be tested without editing this file.

DESIGN DECISIONS YOU SHOULD BE ABLE TO DEFEND

  1. TF-IDF is fitted ONCE on the full corpus, not refitted on the labelled subset.
     Uses unlabelled *text*, never unlabelled *labels* -- transductive, not leakage.
     Realistic here: in a real review you do hold all candidate abstracts up front.

  2. The seed set is forced to contain >=1 positive and >=1 negative. A classifier
     cannot fit on one class, and reviewers do start from known relevant papers.
     This mildly flatters the method -- say so in the write-up.

  3. batch_size > 1 means the model is slightly staler within a batch, so it should
     perform marginally WORSE than batch_size=1, not better. If batching ever looks
     better, suspect a bug. It is a speed/accuracy trade, and the size of that trade
     is worth measuring rather than assuming.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# --------------------------------------------------------------------------
# Text preparation
# --------------------------------------------------------------------------

def build_text(df: pd.DataFrame) -> pd.Series:
    """Concatenate title + abstract. Missing abstracts become empty strings rather
    than being dropped, so a title-only record is still screenable.
    """
    return (df["title"].fillna("") + " " + df["abstract"].fillna("")).str.strip()


def vectorize(texts: pd.Series, **kwargs):
    """TF-IDF matrix for the whole corpus. See design decision 1."""
    params = dict(stop_words="english", min_df=2, ngram_range=(1, 1), sublinear_tf=True)
    params.update(kwargs)
    return TfidfVectorizer(**params).fit_transform(texts)


# --------------------------------------------------------------------------
# Seed selection
# --------------------------------------------------------------------------

def seed_indices(y: np.ndarray, n_seed: int, rng: np.random.Generator) -> list[int]:
    """Random initial labelled set, guaranteed >=1 positive and >=1 negative."""
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("Dataset must contain both classes.")

    seed = [int(rng.choice(pos)), int(rng.choice(neg))]
    remaining = np.setdiff1d(np.arange(len(y)), seed)
    extra = rng.choice(remaining, size=max(0, n_seed - 2), replace=False)
    return seed + [int(i) for i in extra]


# --------------------------------------------------------------------------
# Query strategies -- now return k indices instead of 1
# --------------------------------------------------------------------------

def query_random(X, labelled_idx, y_labelled, unlabelled_idx, rng, k=1) -> np.ndarray:
    """BASELINE: k random unscreened records, model ignored entirely."""
    k = min(k, len(unlabelled_idx))
    return rng.choice(unlabelled_idx, size=k, replace=False)


def default_model():
    """The classifier used unless a variant is passed in."""
    return LogisticRegression(max_iter=1000, class_weight="balanced")


def make_relevance_query(model_factory=default_model):
    """Build a relevance sampler using a given classifier.

    This is YOUR sampler, with two changes: the classifier is injectable (so you
    can test naive Bayes / SVM without editing code), and argmax is generalised to
    top-k via argsort. At k=1 the behaviour is identical to what you wrote.

    Still relevance sampling, not uncertainty sampling: we take the records with the
    HIGHEST P(include), because the objective is finding positives early (recall /
    ranking), not improving classifier accuracy per label spent.
    """
    def query_relevance(X, labelled_idx, y_labelled, unlabelled_idx, rng, k=1):
        model = model_factory()
        model.fit(X[labelled_idx], y_labelled)
        probs = model.predict_proba(X[unlabelled_idx])
        pos_col = list(model.classes_).index(1)
        scores = probs[:, pos_col]

        k = min(k, len(unlabelled_idx))
        top_positions = np.argsort(scores)[-k:]      # k highest-scoring
        return unlabelled_idx[top_positions]

    return query_relevance


# Convenience: the default relevance sampler.
query_relevance = make_relevance_query()


# --------------------------------------------------------------------------
# The screening simulation
# --------------------------------------------------------------------------

def run_screening(
    df: pd.DataFrame,
    query_fn,
    n_seed: int = 10,
    random_state: int = 42,
    batch_size: int = 1,
    vectorizer_kwargs: dict | None = None,
    max_records: int | None = None,
    stop_at_recall: float | None = None,
) -> pd.DataFrame:
    """Simulate screening the corpus in the order chosen by `query_fn`.

    batch_size: records screened per model refit. 1 = refit after every record
        (ideal, slow). Larger = faster but slightly staler model. Use 1 for small
        datasets, ~50 for datasets in the tens of thousands.

    stop_at_recall: stop once this recall is reached (e.g. 0.99). Everything after
        that point is irrelevant to effort@95, so screening it wastes compute --
        Walker_2018 would otherwise screen ~47,000 records past the answer. The
        returned curve is truncated, so do NOT set this if you want the full
        recall-vs-effort figure.

    Returns a log with one row per batch:
        n_screened / pct_screened  -- the effort axis
        n_found / recall           -- the recall axis
    """
    rng = np.random.default_rng(random_state)

    y = df["label_included"].to_numpy().astype(int)
    X = vectorize(build_text(df), **(vectorizer_kwargs or {}))
    n_total = len(y)
    n_pos_total = int(y.sum())

    labelled = seed_indices(y, n_seed, rng)
    unlabelled = np.setdiff1d(np.arange(n_total), labelled)

    def row():
        found = int(y[labelled].sum())
        return {
            "n_screened": len(labelled),
            "pct_screened": 100 * len(labelled) / n_total,
            "n_found": found,
            "recall": found / n_pos_total,
        }

    rows = [row()]  # seed counts as effort; excluding it would flatter the method

    limit = max_records if max_records is not None else n_total
    while len(unlabelled) > 0 and len(labelled) < limit:
        k = min(batch_size, len(unlabelled), limit - len(labelled))
        picks = np.atleast_1d(query_fn(X, labelled, y[labelled], unlabelled, rng, k))

        if not np.all(np.isin(picks, unlabelled)):
            raise ValueError("query_fn returned already-screened or invalid indices.")

        labelled.extend(int(i) for i in picks)
        unlabelled = unlabelled[~np.isin(unlabelled, picks)]
        rows.append(row())

        if stop_at_recall is not None and rows[-1]["recall"] >= stop_at_recall:
            break

    return pd.DataFrame(rows)


def effort_to_recall(log: pd.DataFrame, target_recall: float = 0.95) -> float:
    """% of corpus screened to reach `target_recall`. Lower is better."""
    hit = log[log["recall"] >= target_recall]
    return float(hit.iloc[0]["pct_screened"]) if len(hit) else float("nan")


def wss(log: pd.DataFrame, target_recall: float = 0.95) -> float:
    """Work Saved over Sampling at `target_recall`, as a percentage.

    WSS@95 = (% you would have screened at random to hit 95% recall)
             - (% you actually screened)
    Random screening reaches recall r after ~r of the corpus, so the first term is
    ~95%. This is the standard Cohen (2006) formulation. It is sometimes quoted with
    a (1 - target) correction term; the two differ by a few points, so state which
    definition you used -- reviewers do check.
    """
    return 100 * target_recall - effort_to_recall(log, target_recall)
