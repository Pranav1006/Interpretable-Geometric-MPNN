"""
src/data/splitter.py
====================
Train / test splitting for IG-MPNN datasets.

Design decisions
----------------
* 80 / 20 train / test split (no separate validation split at this stage —
  val will be carved from train during hyperparameter search).
* Random shuffling with a fixed seed for reproducibility.
* Splits are persisted to ``data/splits/`` as JSON so every downstream
  script uses identical indices without re-computing them.
* ``make_splits`` is idempotent: if the split file already exists it is
  loaded rather than recomputed.

Usage
-----
    from src.data.splitter import make_splits, load_splits

    train_idx, test_idx = make_splits(
        n_samples=130831,
        splits_dir="data/splits",
        dataset_name="qm9",
        train_ratio=0.80,
        seed=42,
    )
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np


# ── Public API ───────────────────────────────────────────────────────────────

def make_splits(
    n_samples: int,
    splits_dir: str | Path,
    dataset_name: str = "qm9",
    train_ratio: float = 0.80,
    seed: int = 42,
    force: bool = False,
) -> tuple[list[int], list[int]]:
    """
    Create (or load) an 80/20 train/test split.

    Parameters
    ----------
    n_samples : int
        Total number of molecules in the dataset.
    splits_dir : str or Path
        Directory where split JSON files are stored (``data/splits/``).
    dataset_name : str
        Used to name the output file, e.g. ``qm9_split.json``.
    train_ratio : float
        Fraction of data used for training (default 0.80).
    seed : int
        Random seed for reproducibility (default 42).
    force : bool
        If True, recompute the split even if the file already exists.

    Returns
    -------
    train_idx : list[int]
    test_idx  : list[int]
    """
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    split_path = splits_dir / f"{dataset_name}_split.json"

    if split_path.exists() and not force:
        return load_splits(split_path)

    # ── Shuffle indices ──────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples).tolist()

    n_train = int(len(indices) * train_ratio)
    train_idx = indices[:n_train]
    test_idx  = indices[n_train:]

    # ── Persist ──────────────────────────────────────────────────────
    payload = {
        "dataset": dataset_name,
        "n_samples": n_samples,
        "train_ratio": train_ratio,
        "seed": seed,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "train_indices": train_idx,
        "test_indices": test_idx,
    }
    with open(split_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(
        f"[splitter] Split saved → {split_path}\n"
        f"           train={len(train_idx):,}  test={len(test_idx):,}"
    )
    return train_idx, test_idx


def load_splits(
    split_path: str | Path,
) -> tuple[list[int], list[int]]:
    """
    Load previously persisted split indices from a JSON file.

    Parameters
    ----------
    split_path : str or Path
        Full path to the ``*_split.json`` file.

    Returns
    -------
    train_idx : list[int]
    test_idx  : list[int]
    """
    with open(split_path, "r") as f:
        payload = json.load(f)

    train_idx = payload["train_indices"]
    test_idx  = payload["test_indices"]

    print(
        f"[splitter] Loaded split ← {Path(split_path).name}\n"
        f"           train={len(train_idx):,}  test={len(test_idx):,}"
    )
    return train_idx, test_idx
