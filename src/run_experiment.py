"""Run the screening experiment: relevance sampling vs random baseline.

Usage:
    python3 -m src.run_experiment                 # Nelson_2002 (small, dense -- start here)
    python3 -m src.run_experiment Sep_2021
    python3 -m src.run_experiment Brouwer_2019    # only once the loop works

Develop on a SMALL, HIGH-PREVALENCE dataset. Nelson_2002 is 366 records at 21.9%
prevalence, so a broken loop is obvious in seconds. Debugging on Brouwer_2019
(38,114 records, 0.16% prevalence, 62 positives) means waiting a long time to find
out you have a bug.
"""

from __future__ import annotations

import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import load_dataset
from .screening import run_screening, query_random, query_relevance, effort_to_recall


def main(name: str = "Nelson_2002", random_state: int = 42) -> None:
    df = load_dataset(name)
    n_pos = int(df["label_included"].sum())
    print(f"{name}: {len(df)} records, {n_pos} positives "
          f"({100 * n_pos / len(df):.2f}% prevalence)\n")

    print("Running random baseline...")
    log_random = run_screening(df, query_random, random_state=random_state)

    print("Running relevance sampling...")
    try:
        log_al = run_screening(df, query_relevance, random_state=random_state)
    except NotImplementedError as e:
        print(f"\n  query_relevance is not written yet: {e}")
        print("  Baseline ran fine, so the harness works. Write the sampler, rerun.\n")
        return

    e_al = effort_to_recall(log_al, 0.95)
    e_rand = effort_to_recall(log_random, 0.95)
    print(f"\n  % screened to reach 95% recall")
    print(f"    relevance sampling : {e_al:.1f}%")
    print(f"    random baseline    : {e_rand:.1f}%")
    print(f"    work saved         : {e_rand - e_al:.1f} percentage points")

    if e_al >= e_rand:
        print("\n  WARNING: relevance sampling is not beating random. Something is")
        print("  wrong -- check you are taking argmax (most likely positive), not")
        print("  argmin, and that you mapped the argmax back through unlabelled_idx.")

    plt.figure(figsize=(7, 5))
    plt.plot(log_al["pct_screened"], 100 * log_al["recall"],
             label="Relevance sampling", linewidth=2)
    plt.plot(log_random["pct_screened"], 100 * log_random["recall"],
             label="Random screening", linewidth=2, linestyle="--")
    plt.axhline(95, color="grey", linewidth=0.8, linestyle=":")
    plt.xlabel("% of corpus screened")
    plt.ylabel("% of relevant records found (recall)")
    plt.title(f"Screening prioritisation — {name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out = f"data/recall_effort_{name}.png"
    plt.savefig(out, dpi=150)
    print(f"\n  Figure saved to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Nelson_2002")
