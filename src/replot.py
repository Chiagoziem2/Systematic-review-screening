"""Regenerate the headline figure from an existing data/sweep.csv.

Does NOT re-run the sweep -- it only redraws. Run after `python3 -m src.sweep`.

Why this exists: the original labelling rule (extremes of prevalence + single worst
WSS) was chosen before the cohesion analysis. The datasets the write-up actually
argues about -- Chou_2003, Chou_2004, Hall_2012 -- were left as anonymous dots, so a
reader following the text could not locate them. A figure should label the points
its accompanying argument names.

Usage:
    python3 -m src.replot
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# Points the write-up discusses by name, so they must be findable on the chart.
ARGUED = [
    "Moran_2021",          # lowest WSS, least textually distinct
    "Chou_2003",           # low prevalence, poor WSS, UNEXPLAINED
    "Chou_2004",           # low prevalence, poor WSS
    "Hall_2012",           # low prevalence, excellent WSS -- the contrast case
    "Brouwer_2019",        # lowest prevalence, excellent WSS
    "van_de_Schoot_2018",  # validated against ASReview
    "Nelson_2002",         # development dataset, high prevalence
]


def main() -> None:
    d = pd.read_csv("data/sweep.csv")

    # Datasets that defy the prevalence trend: below 2% prevalence but poor WSS.
    anomalous = set(d[(d.prevalence_pct < 2.2) & (d.wss_at_95 < 50)].dataset)

    fig, ax = plt.subplots(figsize=(9, 6))

    typical = d[~d.dataset.isin(anomalous)]
    ax.scatter(typical.prevalence_pct, typical.wss_at_95,
               s=70, alpha=0.75, edgecolor="black", linewidth=0.5,
               label="follows the trend", zorder=3)
    odd = d[d.dataset.isin(anomalous)]
    ax.scatter(odd.prevalence_pct, odd.wss_at_95,
               s=90, alpha=0.9, color="#c0392b", edgecolor="black", linewidth=0.6,
               marker="D", label="low prevalence, poor WSS", zorder=4)

    # Trend line on log-prevalence, with the correlation stated on the figure.
    logp = np.log10(d.prevalence_pct)
    fit = np.poly1d(np.polyfit(logp, d.wss_at_95, 1))
    xs = np.linspace(logp.min(), logp.max(), 100)
    ax.plot(10 ** xs, fit(xs), color="grey", linestyle="--", linewidth=1.2,
            zorder=2, label="linear fit (log prevalence)")

    rho, p = stats.spearmanr(d.prevalence_pct, d.wss_at_95)
    ax.text(0.03, 0.04, f"Spearman ρ = {rho:.2f}  (p = {p:.4f}),  n = 26",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))

    for _, r in d[d.dataset.isin(ARGUED)].iterrows():
        ax.annotate(r["dataset"], (r["prevalence_pct"], r["wss_at_95"]),
                    fontsize=8, xytext=(6, 5), textcoords="offset points", zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel("Prevalence — % of records that are relevant (log scale)")
    ax.set_ylabel("WSS@95 — work saved over random screening")
    ax.set_title("Screening prioritisation across 26 SYNERGY datasets")
    ax.grid(alpha=0.3, zorder=1)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 0.10), fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig("data/wss_vs_prevalence.png", dpi=150)

    print("Regenerated data/wss_vs_prevalence.png")
    print(f"Spearman rho = {rho:.2f}, p = {p:.4f}")
    print(f"Highlighted as anomalous: {sorted(anomalous)}")


if __name__ == "__main__":
    main()
