"""Screening burden at multiple recall targets, not just 95%.

WSS@95 answers "how much can I avoid screening while missing at most 1 in 20
relevant studies". For some reviews that miss rate is unacceptable. This computes
the same curve at 90/95/99/99.5% recall so the trade-off is visible rather than
asserted.

Usage:
    python3 -m src.burden_curve                        # 6 datasets spanning prevalence
    python3 -m src.burden_curve Nelson_2002 Moran_2021  # specific datasets
"""

from __future__ import annotations

import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import load_dataset
from .screening import run_screening, query_relevance, effort_to_recall

TARGETS = [0.90, 0.95, 0.99, 0.995]
DEFAULT_DATASETS = [
    "Brouwer_2019",        # lowest prevalence
    "van_de_Schoot_2018",  # validated against ASReview
    "Hall_2012",           # strong performer
    "Moran_2021",          # the anomaly
    "Sep_2021",            # high prevalence, poor WSS
    "Nelson_2002",         # development dataset
]


def main() -> None:
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or DEFAULT_DATASETS
    rows = []

    for name in names:
        df = load_dataset(name)
        log = run_screening(df, query_relevance, random_state=0, stop_at_recall=0.999)
        for t in TARGETS:
            rows.append({
                "dataset": name,
                "recall_target": t,
                "pct_screened": effort_to_recall(log, t),
            })
        print(f"  {name} done")

    results = pd.DataFrame(rows)
    results.to_csv("data/burden_curve.csv", index=False)

    pivot = results.pivot(index="dataset", columns="recall_target", values="pct_screened")
    print("\n% of corpus screened to reach each recall target:\n")
    print(pivot.round(1).to_string())

    fig, ax = plt.subplots(figsize=(7, 5))
    for name in names:
        sub = results[results.dataset == name].sort_values("recall_target")
        ax.plot(sub.pct_screened, 100 * sub.recall_target, marker="o", label=name)

    ax.set_xlabel("% of corpus screened")
    ax.set_ylabel("Recall (%)")
    ax.set_title("Screening burden vs recall target (single seed, illustrative)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("data/burden_curve.png", dpi=150)

    print("\nNote: single seed (0) per dataset -- illustrative, not a robust estimate.")
    print("Saved to data/burden_curve.csv and data/burden_curve.png")


if __name__ == "__main__":
    main()
