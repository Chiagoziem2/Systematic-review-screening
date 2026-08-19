"""Sentence-transformer embeddings, swappable in place of TF-IDF.

WHY THIS EXISTS, AND WHY IT IS NOT "DOES DEEP LEARNING BEAT TF-IDF"
    That question is already answered in the literature, consistently, across
    several independent studies on this exact task (ASReview / SYNERGY-style
    screening): TF-IDF (usually with naive Bayes or SVM) wins on WSS@95 in every
    comparison found, including against SBERT and SPECTER. See README for sources.
    Re-running that comparison here would mostly replicate a known result.

    The narrower, UNTESTED question this module exists to answer: does whatever
    advantage embeddings have concentrate specifically on datasets where included
    studies are lexically dissimilar to each other -- the datasets src/cohesion.py
    flagged as textually heterogeneous (Moran_2021, Sep_2021, Menon_2022)? TF-IDF
    cannot see that two abstracts describing the same finding in different words
    are related; a semantic embedding might. If the prediction is right, embeddings
    should help on the low-cohesion group and do little or nothing on the
    high-cohesion controls (Wolters_2018, Leenaars_2019, Hall_2012, Bos_2018).
    If embeddings help everywhere equally, or nowhere, the hypothesis is wrong.

REQUIRES (not in requirements.txt -- experimental extension, not core):
    pip install sentence-transformers

UNTESTED IN THIS SANDBOX
    This environment has no route to huggingface.co, so `embed_texts` below --
    the actual model download and encode call -- has NOT been run end to end here.
    Everything else in this file (caching, shape handling, the swap into
    run_screening via precomputed_X) IS tested, using a random-matrix stand-in for
    what embed_texts would return. You are the first to run the real thing.
    If it errors, the likely culprits are: package not installed, model name typo,
    or (rarely) a version mismatch between sentence-transformers and torch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path("data/embeddings")


def _cache_path(dataset_name: str, model_name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_model = model_name.replace("/", "_")
    return CACHE_DIR / f"{dataset_name}__{safe_model}.npy"


def embed_texts(texts: pd.Series, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Encode a series of strings into a dense embedding matrix (n_texts x d).

    all-MiniLM-L6-v2: 384 dimensions, small and fast, general-purpose -- good for
    proving the pipeline works. For a stronger test of the hypothesis, rerun with
    a scientific-paper model, e.g. model_name='allenai/specter2_base' (heavier,
    needs the `adapter-transformers` extra per its model card -- check before use).
    """
    from sentence_transformers import SentenceTransformer  # deferred: heavy import

    model = SentenceTransformer(model_name)
    return model.encode(
        texts.fillna("").tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


def build_embeddings(
    dataset_name: str,
    texts: pd.Series,
    model_name: str = "all-MiniLM-L6-v2",
    force: bool = False,
) -> np.ndarray:
    """Embeddings for one dataset, cached to disk. Re-running with different seeds
    or query strategies should NOT re-embed the same text -- that is the expensive
    step. Delete the .npy file (or pass force=True) to recompute.
    """
    path = _cache_path(dataset_name, model_name)
    if path.exists() and not force:
        return np.load(path)

    X = embed_texts(texts, model_name)
    np.save(path, X)
    print(f"  cached {X.shape} to {path}")
    return X
