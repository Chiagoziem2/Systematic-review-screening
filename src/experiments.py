"""Multi-seed experiments.

A single run with a single random_state is an anecdote, not a result: the whole
outcome depends on which 10 records happened to land in the seed set. Everything
here exists to replace one number with a distribution.

Usage:
    python3 -m src.experiments seeds Nelson_2002      # step 1: seed spread
    python3 -m src.experiments variants Nelson_2002   # step 2: model variants
    python3 -m src.experiments batch Nelson_2002      # step 3: batching cost check

HOW TO READ THE OUTPUT
    A variant is only better if its mean beats the other's mean by MORE than the
    seed-to-seed spread. If the ranges overlap heavily, you have found noise, not
    an improvement. This is the single easiest way to fool yourself in this project.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from .data import load_dataset
from .screening import (
    run_screening,
    query_random,
    make_relevance_query,
    effort_to_recall,
    wss,
)

DEFAULT_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


# --------------------------------------------------------------------------
# Model variants (step 2)
# --------------------------------------------------------------------------

def logreg():
    return LogisticRegression(max_iter=1000, class_weight="balanced")


def naive_bayes():
    """ASReview's default. Often strong on sparse TF-IDF at low prevalence, and
    much faster to fit. Note it has no class_weight -- it handles imbalance through
    the prior instead."""
    return MultinomialNB()


def svm():
    """LinearSVC has no predict_proba, so it is wrapped in a calibrator. That
    calibration does an internal cross-validation on the labelled set, which is
    slow and unstable when very few records are labelled -- expect noise early on.
    """
    return CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3)


VARIANTS = {
    "logreg_unigram": dict(model=logreg, vec={}),
    "logreg_bigram": dict(model=logreg, vec=dict(ngram_range=(1, 2))),
    "naive_bayes": dict(model=naive_bayes, vec={}),
}


# --------------------------------------------------------------------------
# Core sweep
# --------------------------------------------------------------------------

def sweep_seeds(
    df: pd.DataFrame,
    seeds=DEFAULT_SEEDS,
    model_factory=logreg,
    vectorizer_kwargs=None,
    batch_size: int = 1,
    include_random: bool = True,
) -> pd.DataFrame:
    """Run relevance sampling (and optionally random) across several seeds.

    Returns one row per (strategy, seed) with effort@95 and WSS@95.
    """
    query_relevance = make_relevance_query(model_factory)
    rows = []

    for seed in seeds:
        log = run_screening(
            df, query_relevance, random_state=seed,
            batch_size=batch_size, vectorizer_kwargs=vectorizer_kwargs,
        )
        rows.append({
            "strategy": "relevance",
            "seed": seed,
            "effort_at_95": effort_to_recall(log, 0.95),
            "wss_at_95": wss(log, 0.95),
        })

        if include_random:
            log_r = run_screening(
                df, query_random, random_state=seed,
                batch_size=batch_size, vectorizer_kwargs=vectorizer_kwargs,
            )
            rows.append({
                "strategy": "random",
                "seed": seed,
                "effort_at_95": effort_to_recall(log_r, 0.95),
                "wss_at_95": wss(log_r, 0.95),
            })

    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame, by: str = "strategy") -> pd.DataFrame:
    """Mean / SD / min / max of effort@95. The SD is the number that decides whether
    any difference you see is real."""
    g = results.groupby(by)["effort_at_95"]
    out = pd.DataFrame({
        "n_seeds": g.count(),
        "mean_effort": g.mean().round(1),
        "sd": g.std().round(1),
        "min": g.min().round(1),
        "max": g.max().round(1),
    })
    out["mean_wss"] = results.groupby(by)["wss_at_95"].mean().round(1)
    return out.sort_values("mean_effort")


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def cmd_seeds(name: str) -> None:
    """Step 1: how much does the result move with the seed?"""
    df = load_dataset(name)
    print(f"{name}: {len(df)} records, {int(df.label_included.sum())} positives "
          f"({100 * df.label_included.mean():.2f}%)")
    print(f"Running {len(DEFAULT_SEEDS)} seeds x 2 strategies...\n")

    results = sweep_seeds(df)
    results.to_csv(f"data/seeds_{name}.csv", index=False)

    print(summarise(results).to_string())
    print("\nPer-seed effort@95 (relevance):")
    rel = results[results.strategy == "relevance"]
    print("  " + "  ".join(f"{v:.1f}" for v in rel.effort_at_95))
    print(f"\nSaved to data/seeds_{name}.csv")


def cmd_variants(name: str, seeds=(0, 1, 2, 3, 4)) -> None:
    """Step 2: does a model variant beat the seed noise?"""
    df = load_dataset(name)
    print(f"{name}: comparing {len(VARIANTS)} variants over {len(seeds)} seeds\n")

    all_rows = []
    for label, cfg in VARIANTS.items():
        t0 = time.time()
        res = sweep_seeds(
            df, seeds=list(seeds), model_factory=cfg["model"],
            vectorizer_kwargs=cfg["vec"], include_random=False,
        )
        res["variant"] = label
        all_rows.append(res)
        print(f"  {label:<18} done in {time.time() - t0:.0f}s")

    results = pd.concat(all_rows)
    results.to_csv(f"data/variants_{name}.csv", index=False)

    print()
    print(summarise(results, by="variant").to_string())
    print("\nA variant only wins if the gap between means exceeds the SD column.")
    print(f"Saved to data/variants_{name}.csv")


def cmd_batch(name: str, sizes=(1, 10, 25, 50), seeds=(0, 1, 2)) -> None:
    """Step 3: what does batching cost in accuracy, and save in time?

    Needed before touching Brouwer_2019 (38,114 records). Batching should be
    slightly WORSE per design decision 3 -- this measures how much worse.
    """
    df = load_dataset(name)
    print(f"{name}: {len(df)} records. Batch size vs cost/benefit\n")

    rows = []
    for bs in sizes:
        t0 = time.time()
        res = sweep_seeds(df, seeds=list(seeds), batch_size=bs, include_random=False)
        elapsed = time.time() - t0
        rows.append({
            "batch_size": bs,
            "mean_effort_at_95": round(res.effort_at_95.mean(), 1),
            "sd": round(res.effort_at_95.std(), 1),
            "seconds_per_seed": round(elapsed / len(seeds), 1),
        })
        print(f"  batch_size={bs:<4} mean effort@95={rows[-1]['mean_effort_at_95']}%  "
              f"{rows[-1]['seconds_per_seed']}s/seed")

    out = pd.DataFrame(rows)
    out.to_csv(f"data/batch_{name}.csv", index=False)
    print(f"\nSaved to data/batch_{name}.csv")


COMMANDS = {"seeds": cmd_seeds, "variants": cmd_variants, "batch": cmd_batch}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "seeds"
    dataset = sys.argv[2] if len(sys.argv) > 2 else "Nelson_2002"
    if cmd not in COMMANDS:
        print(f"Unknown command '{cmd}'. Options: {', '.join(COMMANDS)}")
        sys.exit(1)
    COMMANDS[cmd](dataset)
