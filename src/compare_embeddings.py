"""TF-IDF vs sentence-transformer embeddings, targeted at the cohesion hypothesis.

NOT a full 26-dataset rerun. The literature already shows TF-IDF wins on average
(see src/embeddings.py docstring for citations) -- replicating that broadly would
mostly confirm a known result. This instead tests a specific, directional
prediction from src/cohesion.py: that embeddings help most where lexical cohesion
among positives is low, and least where it is already high.

Two groups, chosen from data/cohesion.csv (cohesion_ratio: positives' similarity to
each other, relative to the corpus background):

    LOW-COHESION (test group -- predict embeddings help here)
        Moran_2021    ratio 1.27  (lowest in the collection)
        Sep_2021      ratio 1.30
        Menon_2022    ratio 1.50
        Nelson_2002   ratio 1.62  (also the development dataset throughout)

    HIGH-COHESION (control group -- predict little or no change)
        Wolters_2018       ratio 4.45
        Leenaars_2019       ratio 5.24  (highest in the collection)
        Hall_2012           ratio 3.98
        Bos_2018            ratio 3.91

WHAT WOULD CONFIRM VS FALSIFY THE HYPOTHESIS
    Confirms: mean WSS delta (embeddings - TF-IDF) is positive and larger in the
    low-cohesion group than the high-cohesion group.
    Falsifies: no difference between groups, or embeddings lose everywhere (which
    would just reproduce the published result), or -- the more interesting failure
    -- embeddings lose MORE on the low-cohesion group, which would suggest those
    datasets are simply hard for any method, not specifically hard for lexical ones.
    Report whichever happens. A clean confirmation is the least likely outcome
    given how consistently the literature found TF-IDF winning outright.

Usage:
    python3 -m src.compare_embeddings              # 3 seeds, all 8 datasets
    python3 -m src.compare_embeddings --model allenai/specter2_base
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from .data import load_dataset
from .screening import run_screening, query_relevance, build_text, vectorize, wss
from .embeddings import build_embeddings

LOW_COHESION = ["Moran_2021", "Sep_2021", "Menon_2022", "Nelson_2002"]
HIGH_COHESION = ["Wolters_2018", "Leenaars_2019", "Hall_2012", "Bos_2018"]
SEEDS = [0, 1, 2]


def run_one(name: str, model_name: str, seeds=SEEDS) -> pd.DataFrame:
    df = load_dataset(name)
    text = build_text(df)

    print(f"  {name}: embedding {len(df)} records with {model_name}...")
    t0 = time.time()
    X_emb = build_embeddings(name, text, model_name=model_name)
    print(f"    done in {time.time() - t0:.0f}s")

    X_tfidf = vectorize(text)

    rows = []
    for seed in seeds:
        for label, X in [("tfidf", X_tfidf), ("embedding", X_emb)]:
            log = run_screening(
                df, query_relevance, random_state=seed,
                stop_at_recall=0.99, precomputed_X=X,
            )
            rows.append({
                "dataset": name, "seed": seed, "feature": label,
                "wss_at_95": round(wss(log, 0.95), 1),
            })
    return pd.DataFrame(rows)


def main() -> None:
    model_name = "all-MiniLM-L6-v2"
    if "--model" in sys.argv:
        model_name = sys.argv[sys.argv.index("--model") + 1]

    print(f"Model: {model_name}\n")

    all_rows = []
    for group, names in [("low_cohesion", LOW_COHESION), ("high_cohesion", HIGH_COHESION)]:
        for name in names:
            res = run_one(name, model_name)
            res["group"] = group
            all_rows.append(res)

    results = pd.concat(all_rows, ignore_index=True)
    results.to_csv("data/embedding_comparison.csv", index=False)

    # Per-dataset means, then the delta that decides the hypothesis.
    pivot = results.groupby(["group", "dataset", "feature"])["wss_at_95"].mean().unstack()
    pivot["delta"] = (pivot["embedding"] - pivot["tfidf"]).round(1)
    pivot = pivot.round(1)
    print("\n" + pivot.to_string())

    group_means = results.groupby(["group", "feature"])["wss_at_95"].mean().unstack()
    group_means["delta"] = (group_means["embedding"] - group_means["tfidf"]).round(1)
    print("\n--- group means ---")
    print(group_means.round(1).to_string())

    low_delta = pivot.loc["low_cohesion", "delta"].mean()
    high_delta = pivot.loc["high_cohesion", "delta"].mean()
    print(f"\nmean delta, low-cohesion group : {low_delta:+.1f}")
    print(f"mean delta, high-cohesion group: {high_delta:+.1f}")

    if low_delta > high_delta + 2:
        print("\n-> consistent with the hypothesis: embeddings help more where")
        print("   lexical cohesion is low. Still only 4 datasets per group --")
        print("   treat as suggestive, not confirmed.")
    elif low_delta < high_delta - 2:
        print("\n-> OPPOSITE of the hypothesis. Worth reporting as-is, not")
        print("   explaining away -- this would itself be a finding.")
    else:
        print("\n-> no clear group difference. Consistent with the published")
        print("   result that TF-IDF is simply hard to beat on this task, full stop.")

    print("\nSaved to data/embedding_comparison.csv")


if __name__ == "__main__":
    main()
