"""
tests/conftest.py
=================
Shared pytest fixtures for IG-MPNN tests.
"""

import pytest
import torch
from torch_geometric.data import Data


@pytest.fixture()
def benzene_graph() -> Data:
    """
    Minimal aromatic ring graph for benzene (C6H6).
    Rough 3D positions placed on a hexagon of radius 1.4 Å.
    Used by model and explanation tests for a well-known structure.
    """
    import math
    n_heavy = 6
    positions = []
    for i in range(n_heavy):
        angle = 2 * math.pi * i / n_heavy
        positions.append([1.4 * math.cos(angle), 1.4 * math.sin(angle), 0.0])

    pos = torch.tensor(positions, dtype=torch.float)
    # Ring bonds (bidirectional)
    src = list(range(6)) + list(range(6))
    dst = [(i + 1) % 6 for i in range(6)] + [(i - 1) % 6 for i in range(6)]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    x = torch.zeros((6, 11))
    x[:, 1] = 1.0  # all carbon (index 1 in ATOM_TYPES one-hot)

    return Data(x=x, edge_index=edge_index, pos=pos)


@pytest.fixture()
def single_atom_graph() -> Data:
    """Edge-case: a single atom with no bonds."""
    return Data(
        x=torch.zeros((1, 11)),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        pos=torch.zeros((1, 3)),
        y=torch.tensor([0.0]),
    )
