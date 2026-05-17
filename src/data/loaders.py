"""
src/data/loaders.py
===================
Build PyG DataLoaders from a dataset and pre-computed split indices.

Usage
-----
    from src.data.datasets import QM9Dataset
    from src.data.splitter import make_splits
    from src.data.loaders import get_loaders

    ds = QM9Dataset(root="data", target_idx=0)
    train_idx, test_idx = make_splits(
        n_samples=len(ds),
        splits_dir="data/splits",
        dataset_name="qm9",
    )
    train_loader, test_loader = get_loaders(
        dataset=ds,
        train_idx=train_idx,
        test_idx=test_idx,
        batch_size=32,
    )
"""

from __future__ import annotations

from torch.utils.data import Subset
from torch_geometric.loader import DataLoader


def get_loaders(
    dataset,
    train_idx: list[int],
    test_idx: list[int],
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle_train: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Wrap split indices into ``torch_geometric.loader.DataLoader`` objects.

    Parameters
    ----------
    dataset : QM9Dataset (or any indexable sequence of PyG Data objects)
        The full dataset to slice.
    train_idx : list[int]
        Indices belonging to the training set (from ``make_splits``).
    test_idx : list[int]
        Indices belonging to the test set (from ``make_splits``).
    batch_size : int
        Number of graphs per mini-batch.
    num_workers : int
        Worker processes for the DataLoader.  0 = main process only
        (safest on Windows where multiprocessing can cause issues).
    shuffle_train : bool
        Whether to shuffle the training set each epoch.

    Returns
    -------
    train_loader : DataLoader
    test_loader  : DataLoader
    """
    train_subset = Subset(dataset, train_idx)
    test_subset  = Subset(dataset, test_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader
