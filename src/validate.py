"""Head-to-head validation against ASReview, the reference implementation.

Internal consistency proves only that the code agrees with itself. This runs
ASReview's own simulator on the same datasets under a matched configuration and
compares WSS@95 directly.

REQUIRES (not in requirements.txt -- validation only):
    pip install asreview asreview-insights

    NOTE: Python 3.9 resolves to ASReview 1.x; Python 3.10+ gets 2.x. The two have
    DIFFERENT command-line interfaces (v1 uses --n_prior_included and -m, v2 uses
    --n-prior-included and -c). This script detects the installed version and uses
    the right flags, so it works either way -- but v1 and v2 results are not
    perfectly comparable to each other. Record which version you ran.

Usage:
    python3 -m src.validate                             # 3 datasets, seed 0
    python3 -m src.validate Nelson_2002                 # one dataset
    python3 -m src.validate Nelson_2002 --seeds 0 1 2

RUNTIME
    Nelson_2002 (366 records): seconds per seed.
    van_de_Schoot_2018 (4,544) and Hall_2012 (8,793): minutes per seed.
    Start with Nelson_2002 to confirm the pipeline works before committing time.

WHAT "MATCHED CONFIGURATION" MEANS, AND WHAT IT DOES NOT
    Matched: logistic regression, TF-IDF features, `max` querier (relevance
    sampling -- note this is ASReview's DEFAULT query strategy, which independently
    corroborates the design choice made here).

    NOT matched, and these explain residual differences:
      - Seed set. ASReview takes exactly 1 positive + 1 negative. This code takes
        1 positive + 1 negative + 8 random, so our starting information varies more
        between runs. This is the likely cause of our higher seed-to-seed variance.
      - TF-IDF parameters (min_df, sublinear_tf) differ.
      - Balancing: v2 'balanced' vs v1 'double' are not the same strategy.
      - ASReview's default model is naive Bayes (v1) / elas_u4 (v2), not logistic.
        This script configures ASReview DOWN to match. A pass therefore means
        "matches the reference implementation under matched configuration", NOT
        "matches ASReview at its best". Do not overclaim it.

    Expect differences of a few percentage points running in BOTH directions.
    A consistent one-directional gap is what a systematic error looks like.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .data import load_dataset
from .screening import run_screening, query_relevance, wss

DEFAULT_DATASETS = ["Nelson_2002", "van_de_Schoot_2018", "Hall_2012"]
ASR_DIR = Path("asr")


def asreview_major_version() -> int | None:
    """1, 2, or None if ASReview is not installed."""
    try:
        out = subprocess.run(["asreview", "--version"],
                             capture_output=True, text=True, timeout=60)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    m = re.search(r"(\d+)\.", out.stdout + out.stderr)
    return int(m.group(1)) if m else None


def export_for_asreview(name: str) -> Path:
    """ASReview expects columns: title, abstract, included."""
    ASR_DIR.mkdir(exist_ok=True)
    path = ASR_DIR / f"{name}.csv"
    if path.exists():
        return path

    df = load_dataset(name)
    out = df[["title", "abstract", "label_included"]].copy()
    out.columns = ["title", "abstract", "included"]
    out["title"] = out["title"].fillna("")
    out["abstract"] = out["abstract"].fillna("")
    out.to_csv(path, index=False)
    return path


def build_command(version: int, csv_path: Path, out_path: Path, seed: int) -> list[str]:
    """v1 and v2 have incompatible flag names for the same options."""
    if version == 1:
        return [
            "asreview", "simulate", str(csv_path),
            "-m", "logistic", "-e", "tfidf", "-q", "max", "-b", "double",
            "--n_prior_included", "1", "--n_prior_excluded", "1",
            "--init_seed", str(seed), "--seed", str(seed),
            "-s", str(out_path),
        ]
    return [
        "asreview", "simulate", str(csv_path),
        "-c", "logistic", "-e", "tfidf", "-q", "max", "-b", "balanced",
        "--n-prior-included", "1", "--n-prior-excluded", "1",
        "--prior-seed", str(seed), "--seed", str(seed),
        "--output", str(out_path),
    ]


def run_asreview(name: str, seed: int, version: int) -> float | None:
    """Run ASReview's simulator and return its WSS@95, or None on failure."""
    csv_path = export_for_asreview(name)
    out_path = ASR_DIR / f"v{version}_{name}_s{seed}.asreview"

    if not out_path.exists():
        print(f"    running ASReview v{version} on {name} (seed {seed})...")
        try:
            subprocess.run(build_command(version, csv_path, out_path, seed),
                           check=True, capture_output=True, timeout=7200)
        except subprocess.CalledProcessError as e:
            print(f"    ASReview failed: {e.stderr.decode()[:300]}")
            return None
        except subprocess.TimeoutExpired:
            print("    ASReview timed out.")
            return None

    metrics = subprocess.run(["asreview", "metrics", str(out_path)],
                             capture_output=True, text=True)
    try:
        data = json.loads(metrics.stdout)
    except json.JSONDecodeError:
        print(f"    could not parse metrics: {metrics.stderr[:200]}")
        return None

    for item in data["data"]["items"]:
        if item["id"] == "wss":
            return round(100 * item["value"][0][1], 1)
    return None


def run_ours(name: str, seed: int) -> float:
    df = load_dataset(name)
    log = run_screening(df, query_relevance, random_state=seed, stop_at_recall=0.99)
    return round(wss(log, 0.95), 1)


def main() -> None:
    version = asreview_major_version()
    if version is None:
        print("ASReview not found. Install with:")
        print("    pip install asreview asreview-insights")
        return
    print(f"Detected ASReview v{version}.x\n")

    argv = sys.argv[1:]
    seeds = [0]
    if "--seeds" in argv:
        i = argv.index("--seeds")
        seeds = [int(s) for s in argv[i + 1:] if s.isdigit()]
        argv = argv[:i]
    datasets = [a for a in argv if not a.startswith("--")] or DEFAULT_DATASETS

    rows = []
    for name in datasets:
        for seed in seeds:
            print(f"  {name} (seed {seed})")
            ours = run_ours(name, seed)
            theirs = run_asreview(name, seed, version)
            rows.append({
                "dataset": name,
                "seed": seed,
                "asreview_version": version,
                "ours_wss95": ours,
                "asreview_wss95": theirs,
                "difference": round(ours - theirs, 1) if theirs is not None else None,
            })

    results = pd.DataFrame(rows)
    Path("data").mkdir(exist_ok=True)
    results.to_csv("data/validation.csv", index=False)
    print("\n" + results.to_string(index=False))

    valid = results.dropna(subset=["difference"])
    if len(valid):
        print(f"\nmean signed difference   : {valid.difference.mean():+.1f} pp")
        print(f"mean absolute difference : {valid.difference.abs().mean():.1f} pp")
        if len(valid) > 1:
            print(f"\nours     SD across seeds: {valid.ours_wss95.std():.1f}")
            print(f"asreview SD across seeds: {valid.asreview_wss95.std():.1f}")
            print("A much larger SD on our side points at the seed-set difference,")
            print("not at a ranking bug -- test by setting n_seed=2 in run_screening.")
        if len(valid) > 2 and (
            (valid.difference > 0).all() or (valid.difference < 0).all()
        ):
            print("\nNOTE: all differences run in the SAME direction -- check for")
            print("systematic bias before claiming agreement.")
    print("\nSaved to data/validation.csv")


if __name__ == "__main__":
    main()
