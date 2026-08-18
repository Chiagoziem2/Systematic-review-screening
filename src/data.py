"""SYNERGY data loading.

Grounded in the real synergy-dataset v1.2 API (Dataset.to_frame), not reconstructed
from memory. The packaged datasets ship works locally (works_*.zip) with abstracts
stored as OpenAlex inverted indices, so there is NO live per-record network fetch:
to_frame() reconstructs abstract text offline.

Two facts that shape your modelling code, both surfaced by the Week-1 audit:
  1. A non-trivial fraction of records have NO abstract (null). You must decide,
     explicitly, whether to drop these or keep them title-only. load_dataset does
     NOT decide for you -- see `drop_missing_abstract`.
  2. Positive prevalence is very low (often 1-2%). This is why accuracy/AUC are the
     wrong headline metrics downstream.
"""

from __future__ import annotations

import synergy_dataset as sd
import pandas as pd


def ensure_downloaded(source: str = "dataverse") -> None:
    """Download the raw SYNERGY release if not already present.

    source="dataverse" is the canonical source (dataverse.nl). If that is blocked
    on your network, source="github" pulls the same release from the GitHub mirror
    (github.com/asreview/synergy-dataset). Run this ONCE; it is a no-op-ish re-download
    otherwise, so guard it yourself if you re-run frequently.
    """
    sd.download_raw_dataset(source=source)


def list_datasets() -> list[str]:
    """Names of all available SYNERGY datasets."""
    return [d.name for d in sd.iter_datasets()]


def load_dataset(name: str, drop_missing_abstract: bool = False) -> pd.DataFrame:
    """Load one SYNERGY dataset as a DataFrame.

    Columns: doi, title, abstract, label_included (1 = included at full-text screening).

    drop_missing_abstract:
        False (default) -> return everything, abstract may be None. YOUR modelling
                           code owns the decision of what to do with these.
        True            -> drop rows with no abstract. Convenient, but note this
                           changes the effective prevalence and screening task --
                           do NOT flip this on without recording what it removes.
    """
    df = sd.Dataset(name).to_frame()
    if drop_missing_abstract:
        df = df[df["abstract"].notna()].reset_index(drop=True)
    return df
