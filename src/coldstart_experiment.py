"""Empirical test of the warm-start cost, checked against the analytical estimate.

data/coldstart.csv already gives an ANALYTICAL estimate of the hidden warm-start
cost: expected draws to first positive under random search, (N+1)/(P+1). This runs
an actual random-start simulation -- no forced positive in the seed, screen
randomly until a positive turns up, THEN switch to relevance sampling -- and checks
whether total effort@95 (random phase + relevance phase) roughly matches
warm-start effort@95 + the analytical coldstart_pct. If they agree, that is real
validation of the earlier estimate, not just restated arithmetic.

Usage:
    python3 -m src.coldstart_experiment                      # 4 datasets
    python3 -m src.coldstart_experiment --seeds 0 1 2 3 4
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from .data import load_dataset
from .screening import (
    build_text, vectorize, seed_indices, query_relevance, run_screening,
    effort_to_recall, wss,
)

from .sweep import pick_batch_size

DEFAULT_DATASETS = ["Nelson_2002", "Bos_2018", "van_de_Schoot_2018"]


def run_random_start(df, random_state: int, batch_size: int = 1):
    """No forced positive. Screen randomly until BOTH classes have appeared, then
    switch to relevance sampling (batched for speed on larger datasets) for the
    rest. This is what screening actually looks like with no prior knowledge.
    """
    rng = np.random.default_rng(random_state)
    y = df["label_included"].to_numpy().astype(int)
    X = vectorize(build_text(df))
    n_total = len(y)

    order = rng.permutation(n_total)
    labelled: list[int] = []
    switched_at = None

    # Phase 1: pure random, until BOTH a positive and a negative have been seen
    # (a single-class seed cannot fit a classifier -- same constraint as seed_indices).
    for idx in order:
        labelled.append(int(idx))
        seen = y[labelled]
        if seen.sum() >= 1 and (seen == 0).sum() >= 1:
            switched_at = len(labelled)
            break

    if switched_at is None:
        return None  # no positive found at all -- shouldn't happen on these datasets

    # Phase 2: relevance sampling on the remainder, seeded with what phase 1 found.
    unlabelled = np.setdiff1d(np.arange(n_total), labelled)
    rows = [{
        "n_screened": len(labelled), "pct_screened": 100 * len(labelled) / n_total,
        "recall": y[labelled].sum() / y.sum(),
    }]
    while len(unlabelled) > 0 and rows[-1]["recall"] < 0.99:
        k = min(batch_size, len(unlabelled))
        picks = query_relevance(X, labelled, y[labelled], unlabelled, rng, k=k)
        labelled.extend(int(i) for i in np.atleast_1d(picks))
        unlabelled = unlabelled[~np.isin(unlabelled, picks)]
        rows.append({
            "n_screened": len(labelled), "pct_screened": 100 * len(labelled) / n_total,
            "recall": y[labelled].sum() / y.sum(),
        })

    return pd.DataFrame(rows), switched_at, 100 * switched_at / n_total


def main() -> None:
    argv = sys.argv[1:]
    seeds = [0, 1, 2]
    if "--seeds" in argv:
        i = argv.index("--seeds")
        seeds = [int(s) for s in argv[i + 1:] if s.isdigit()]
    names = [a for a in argv if not a.startswith("--")] or DEFAULT_DATASETS

    coldstart = pd.read_csv("data/coldstart.csv").set_index("dataset")
    rows = []

    for name in names:
        df = load_dataset(name)
        bs = pick_batch_size(len(df))
        warm_effs = []
        for seed in seeds:
            log = run_screening(df, query_relevance, random_state=seed,
                                batch_size=bs, stop_at_recall=0.99)
            warm_effs.append(effort_to_recall(log, 0.95))
        warm_mean = float(np.mean(warm_effs))

        random_effs, first_pos_costs = [], []
        for seed in seeds:
            result = run_random_start(df, seed, batch_size=bs)
            if result is None:
                continue
            log, switched_at, pct_to_first = result
            random_effs.append(effort_to_recall(log, 0.95))
            first_pos_costs.append(pct_to_first)

        analytical_cost = coldstart.loc[name, "coldstart_pct"] if name in coldstart.index else float("nan")

        rows.append({
            "dataset": name,
            "warm_start_effort95": round(warm_mean, 1),
            "random_start_effort95": round(float(np.mean(random_effs)), 1),
            "empirical_first_positive_pct": round(float(np.mean(first_pos_costs)), 2),
            "analytical_coldstart_pct": round(float(analytical_cost), 2),
            "warm_plus_analytical": round(warm_mean + analytical_cost, 1),
        })
        print(f"  {name} done")

    results = pd.DataFrame(rows)
    results.to_csv("data/coldstart_experiment.csv", index=False)
    print("\n" + results.to_string(index=False))
    print("\nIf 'random_start_effort95' is close to 'warm_plus_analytical', the")
    print("analytical hidden-cost estimate used elsewhere in this repo is validated.")
    print("Saved to data/coldstart_experiment.csv")


if __name__ == "__main__":
    main()
