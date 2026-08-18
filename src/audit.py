"""Week-1 data audit.

Produces exactly the three numbers per dataset that decide whether the v1 plan
survives contact with reality:
  - n_records          : total citations to screen
  - pct_with_abstract  : share of records that actually have abstract text
  - prevalence_pct     : share of records that are positives (included)

No modelling. If pct_with_abstract is low on the datasets you care about, the
active-learning input distribution shifts and you must decide (in data.py) whether
missing-abstract records are dropped or handled title-only BEFORE writing the loop.

Usage:
    python -m src.audit                # audit all datasets
    python -m src.audit Chou_2003 ...  # audit named datasets only
"""

from __future__ import annotations

import sys
import pandas as pd

from .data import ensure_downloaded, list_datasets, load_dataset


def audit_one(name: str) -> dict:
    df = load_dataset(name, drop_missing_abstract=False)
    n = len(df)
    n_abs = int(df["abstract"].notna().sum())
    n_pos = int((df["label_included"] == 1).sum())
    return {
        "dataset": name,
        "n_records": n,
        "pct_with_abstract": round(100 * n_abs / n, 1) if n else 0.0,
        "n_positives": n_pos,
        "prevalence_pct": round(100 * n_pos / n, 2) if n else 0.0,
    }


def main(names: list[str] | None = None) -> pd.DataFrame:
    # Comment out if you have already downloaded once.
    ensure_downloaded()

    names = names or list_datasets()
    rows = []
    for name in names:
        try:
            rows.append(audit_one(name))
        except Exception as e:  # keep going; note which datasets failed to load
            rows.append({"dataset": name, "n_records": None, "error": str(e)[:80]})

    audit = pd.DataFrame(rows).sort_values("n_records", ascending=False, na_position="last")
    audit.to_csv("data/audit.csv", index=False)

    print(audit.to_string(index=False))
    ok = audit[audit["n_records"].notna()]
    if len(ok):
        print("\n--- ranges across datasets ---")
        print(f"records          : {int(ok.n_records.min())} - {int(ok.n_records.max())}")
        print(f"pct_with_abstract: {ok.pct_with_abstract.min()}% - {ok.pct_with_abstract.max()}%")
        print(f"prevalence       : {ok.prevalence_pct.min()}% - {ok.prevalence_pct.max()}%")
    return audit


if __name__ == "__main__":
    args = sys.argv[1:] or None
    main(args)
