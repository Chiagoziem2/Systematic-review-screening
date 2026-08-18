"""Why do some datasets resist prioritisation?

Moran_2021 has 2.13% prevalence and WSS@95 of 13.3. Hall_2012 has 1.18% prevalence
and WSS@95 of 92.0. Prevalence predicts the opposite of what happened, so something
else is driving performance on those datasets.

HYPOTHESIS UNDER TEST
    The included studies in poor-performing datasets are textually heterogeneous --
    they do not resemble each other -- so a linear model trained on some positives
    cannot rank the remaining positives highly.

WHY THREE STATISTICS, NOT ONE
    "Positives are similar to each other" is not on its own meaningful: in a corpus
    on a narrow topic, EVERYTHING is similar to everything. So this computes:

      pos_cohesion  mean pairwise cosine similarity among positives
      background    mean pairwise cosine among a random sample of all records
                    -- the control. Without it, cohesion is uninterpretable.
      separation    pos_cohesion minus mean cosine(positive, negative)
                    -- how distinguishable positives are from the rest. This is
                    the statistic I would expect to predict WSS, but that is a
                    prediction, not a finding.

WHAT WOULD FALSIFY THE HYPOTHESIS
    If Moran_2021 and Chou_2003 do NOT sit at the bottom of the separation ranking,
    the heterogeneity story is wrong and you need a different explanation. Record
    that outcome rather than reaching for a second hypothesis to rescue the first.

Usage:
    python3 -m src.cohesion           # compute for all 26 datasets, join to sweep
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .data import list_datasets, load_dataset
from .screening import build_text, vectorize

RNG = np.random.default_rng(0)
MAX_SAMPLE = 300   # cap pairwise comparisons; 300x300 is plenty and keeps it fast


def _sample(idx: np.ndarray, n: int = MAX_SAMPLE) -> np.ndarray:
    """Subsample so pairwise similarity stays tractable on 48k-record datasets."""
    if len(idx) <= n:
        return idx
    return RNG.choice(idx, size=n, replace=False)


def _mean_offdiag(sim: np.ndarray) -> float:
    """Mean of a similarity matrix excluding the diagonal (self-similarity = 1)."""
    n = sim.shape[0]
    if n < 2:
        return float("nan")
    return float((sim.sum() - np.trace(sim)) / (n * n - n))


def cohesion_stats(name: str) -> dict:
    """Compute the three similarity statistics for one dataset."""
    df = load_dataset(name)
    X = vectorize(build_text(df))
    y = df["label_included"].to_numpy().astype(int)

    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    allx = np.arange(len(y))

    pos_s = _sample(pos)
    neg_s = _sample(neg)
    all_s = _sample(allx)

    pos_cohesion = _mean_offdiag(cosine_similarity(X[pos_s]))
    background = _mean_offdiag(cosine_similarity(X[all_s]))
    pos_to_neg = float(cosine_similarity(X[pos_s], X[neg_s]).mean())

    return {
        "dataset": name,
        "n_positives": len(pos),
        "pos_cohesion": round(pos_cohesion, 4),
        "background": round(background, 4),
        "pos_to_neg": round(pos_to_neg, 4),
        "separation": round(pos_cohesion - pos_to_neg, 4),
        "cohesion_ratio": round(pos_cohesion / background, 2) if background else np.nan,
    }


def main() -> None:
    print("Computing textual cohesion for 26 datasets...\n")
    rows = [cohesion_stats(n) for n in list_datasets()]
    coh = pd.DataFrame(rows)

    try:
        sweep = pd.read_csv("data/sweep.csv")
        merged = coh.merge(
            sweep[["dataset", "prevalence_pct", "wss_at_95", "n_records"]],
            on="dataset",
        )
    except FileNotFoundError:
        print("data/sweep.csv not found -- run `python3 -m src.sweep` first.")
        coh.to_csv("data/cohesion.csv", index=False)
        return

    merged = merged.sort_values("separation")
    merged.to_csv("data/cohesion.csv", index=False)

    cols = ["dataset", "prevalence_pct", "wss_at_95",
            "pos_cohesion", "background", "separation", "cohesion_ratio"]
    print(merged[cols].to_string(index=False))

    # --- Does separation predict WSS, and does it survive controlling for prevalence?
    from scipy import stats

    print("\n--- correlations with WSS@95 (n=26) ---")
    for col in ["separation", "pos_cohesion", "cohesion_ratio", "prevalence_pct"]:
        rho, p = stats.spearmanr(merged[col], merged["wss_at_95"])
        print(f"  {col:<16} rho={rho:+.2f}  p={p:.4f}")

    # Partial correlation: separation vs WSS, controlling for log-prevalence.
    # Regress both on log-prevalence, correlate the residuals.
    logprev = np.log10(merged["prevalence_pct"])
    res_wss = merged["wss_at_95"] - np.poly1d(
        np.polyfit(logprev, merged["wss_at_95"], 1))(logprev)
    res_sep = merged["separation"] - np.poly1d(
        np.polyfit(logprev, merged["separation"], 1))(logprev)
    rho_p, p_p = stats.spearmanr(res_sep, res_wss)
    print(f"\n  separation vs WSS, controlling for prevalence: rho={rho_p:+.2f} p={p_p:.4f}")
    print("  (this is the number that matters -- prevalence already explains a lot)")

    print("\n--- where do the anomalies sit on separation? (1 = lowest) ---")
    merged = merged.reset_index(drop=True)
    for target in ["Moran_2021", "Chou_2003", "Chou_2004", "Hall_2012", "Brouwer_2019"]:
        if target in set(merged.dataset):
            r = merged.index[merged.dataset == target][0] + 1
            w = merged.loc[merged.dataset == target, "wss_at_95"].iloc[0]
            print(f"  {target:<18} separation rank {r:>2}/26   WSS@95 = {w}")

    print("\nSaved to data/cohesion.csv")


if __name__ == "__main__":
    main()
