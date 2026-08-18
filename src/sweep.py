"""Full sweep across all 26 SYNERGY datasets.

The point of this file: two datasets do not establish a relationship, 26 do. The
headline figure for the whole project is WSS@95 plotted against prevalence, because
it demonstrates the thing most write-ups get wrong -- that WSS@95 is not a property
of the method alone, it is heavily conditioned on how rare the positives are.

Usage:
    python3 -m src.sweep              # all 26 datasets, ~3 seeds each
    python3 -m src.sweep --quick      # 1 seed, for a fast smoke test
    python3 -m src.sweep --coldstart  # quantify the warm-start assumption

TWO THINGS THIS MEASURES THAT A NAIVE SWEEP WOULD MISS

  1. Cold-start cost. Every result assumes the seed set contains a known positive.
     At 0.16% prevalence that assumption is doing enormous work: finding the first
     positive by random search alone costs, on average, (N+1)/(P+1) draws. This is
     reported as `coldstart_pct` -- effort you are NOT counting in effort@95. On
     some datasets it is larger than the reported effort itself. Report it.

  2. Batch size is scaled to corpus size, so results across datasets are NOT
     strictly like-for-like: bigger datasets use bigger batches and are therefore
     mildly penalised. The alternative (batch_size=1 everywhere) is not tractable
     on 48k records. Say which you did.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import list_datasets, load_dataset
from .screening import (
    run_screening,
    query_relevance,
    query_random,
    effort_to_recall,
    wss,
)


def pick_batch_size(n_records: int) -> int:
    """Keep the number of model refits roughly constant (~200-400) regardless of
    corpus size, so a 48k-record dataset does not take 100x longer than a 500-record
    one. This trades a little accuracy on large datasets -- see batch experiment.
    """
    if n_records <= 1000:
        return 1
    if n_records <= 5000:
        return 10
    if n_records <= 15000:
        return 25
    return 100


def coldstart_cost_pct(n_records: int, n_positives: int) -> float:
    """Expected % of the corpus you would screen at random before hitting your FIRST
    positive, i.e. the effort the warm-start assumption hides.

    Expected draws to first success when sampling without replacement from N items
    containing P successes is (N + 1) / (P + 1).
    """
    expected_draws = (n_records + 1) / (n_positives + 1)
    return 100 * expected_draws / n_records


def sweep(seeds=(0, 1, 2), target: float = 0.95) -> pd.DataFrame:
    """Run relevance sampling + random baseline over every SYNERGY dataset."""
    rows = []
    names = list_datasets()

    for i, name in enumerate(names, 1):
        df = load_dataset(name)
        n, n_pos = len(df), int(df["label_included"].sum())
        bs = pick_batch_size(n)
        t0 = time.time()

        eff_rel, eff_rnd = [], []
        for seed in seeds:
            log = run_screening(df, query_relevance, random_state=seed,
                                batch_size=bs, stop_at_recall=0.99)
            eff_rel.append(effort_to_recall(log, target))

            log_r = run_screening(df, query_random, random_state=seed,
                                  batch_size=bs, stop_at_recall=0.99)
            eff_rnd.append(effort_to_recall(log_r, target))

        rows.append({
            "dataset": name,
            "n_records": n,
            "n_positives": n_pos,
            "prevalence_pct": round(100 * n_pos / n, 2),
            "batch_size": bs,
            "effort_at_95": round(float(np.mean(eff_rel)), 1),
            "effort_sd": round(float(np.std(eff_rel)), 1),
            "wss_at_95": round(100 * target - float(np.mean(eff_rel)), 1),
            "random_effort": round(float(np.mean(eff_rnd)), 1),
            "coldstart_pct": round(coldstart_cost_pct(n, n_pos), 2),
            "seconds": round(time.time() - t0, 1),
        })
        print(f"[{i:2}/{len(names)}] {name:<24} "
              f"prev={rows[-1]['prevalence_pct']:>5}%  "
              f"WSS@95={rows[-1]['wss_at_95']:>5}  "
              f"({rows[-1]['seconds']}s)")

    return pd.DataFrame(rows)


def plot_wss_vs_prevalence(results: pd.DataFrame, path: str) -> None:
    """The headline figure: WSS@95 against prevalence, one point per dataset."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(results["prevalence_pct"], results["wss_at_95"],
               s=60, alpha=0.75, edgecolor="black", linewidth=0.5)

    # Label the extremes, which carry the argument.
    extremes = pd.concat([
        results.nsmallest(2, "prevalence_pct"),
        results.nlargest(2, "prevalence_pct"),
        results.nsmallest(1, "wss_at_95"),
    ]).drop_duplicates("dataset")
    for _, r in extremes.iterrows():
        ax.annotate(r["dataset"], (r["prevalence_pct"], r["wss_at_95"]),
                    fontsize=7.5, xytext=(4, 4), textcoords="offset points")

    ax.set_xscale("log")
    ax.set_xlabel("Prevalence — % of records that are relevant (log scale)")
    ax.set_ylabel("WSS@95 — work saved over random screening")
    ax.set_title("Screening prioritisation across 26 SYNERGY datasets")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nFigure saved to {path}")


def report_coldstart() -> None:
    """How much effort does the warm-start assumption hide, per dataset?"""
    rows = []
    for name in list_datasets():
        df = load_dataset(name)
        n, n_pos = len(df), int(df["label_included"].sum())
        rows.append({
            "dataset": name,
            "prevalence_pct": round(100 * n_pos / n, 2),
            "n_positives": n_pos,
            "coldstart_pct": round(coldstart_cost_pct(n, n_pos), 2),
        })
    out = pd.DataFrame(rows).sort_values("coldstart_pct", ascending=False)
    out.to_csv("data/coldstart.csv", index=False)
    print(out.to_string(index=False))
    print("\ncoldstart_pct = expected % of corpus screened at random to find the")
    print("FIRST positive. This effort is NOT counted in effort@95, because the")
    print("seed set is forced to contain a positive. State this as a limitation.")


def main() -> None:
    args = sys.argv[1:]

    if "--coldstart" in args:
        report_coldstart()
        return

    seeds = (0,) if "--quick" in args else (0, 1, 2)
    print(f"Sweeping 26 datasets with {len(seeds)} seed(s)...\n")

    results = sweep(seeds=seeds)
    results = results.sort_values("prevalence_pct")
    results.to_csv("data/sweep.csv", index=False)

    print("\n" + results[[
        "dataset", "n_records", "prevalence_pct", "batch_size",
        "effort_at_95", "wss_at_95", "coldstart_pct",
    ]].to_string(index=False))

    print(f"\nmean WSS@95 : {results.wss_at_95.mean():.1f}")
    print(f"range       : {results.wss_at_95.min():.1f} to {results.wss_at_95.max():.1f}")

    lo = results[results.prevalence_pct < 2]
    hi = results[results.prevalence_pct >= 5]
    print(f"\nWSS@95 where prevalence <2%  : {lo.wss_at_95.mean():.1f}  (n={len(lo)})")
    print(f"WSS@95 where prevalence >=5% : {hi.wss_at_95.mean():.1f}  (n={len(hi)})")

    plot_wss_vs_prevalence(results, "data/wss_vs_prevalence.png")
    print("Saved to data/sweep.csv")


if __name__ == "__main__":
    main()
