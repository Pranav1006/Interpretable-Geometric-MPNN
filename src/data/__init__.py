"""
src/data
========
Data loading, featurization, and splitting for IG-MPNN.

Public API
----------
    QM9Dataset      — PyG Dataset wrapper for QM9
    GeometryTransform — Computes pairwise distances, angles, torsions
    make_splits     — Produces and persists 80/20 train/test index files
    get_loaders     — Returns (train_loader, test_loader) ready for training
"""

from .datasets import QM9Dataset
from .transforms import GeometryTransform
from .splitter import make_splits, load_splits
from .loaders import get_loaders

__all__ = [
    "QM9Dataset",
    "GeometryTransform",
    "make_splits",
    "load_splits",
    "get_loaders",
]
