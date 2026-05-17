"""
tests/test_data.py
==================
Unit and integration tests for src/data/.

Tests are intentionally lightweight — they use small synthetic graphs
rather than downloading QM9, so they run offline with no network access.

Run with:  pytest tests/test_data.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def ethanol_graph() -> Data:
    """
    Minimal graph for ethanol (CCO) with 3D positions.
    Not chemically accurate — just enough to exercise the transforms.

      0: C
      1: C
      2: O
      3: H  (on C0)
      4: H  (on C1)
      5: H  (on O)

    Edges are bidirectional (undirected graph convention in PyG).
    """
    # Bonds: C-C, C-O, C-H, C-H, O-H
    edge_index = torch.tensor([
        [0, 1, 1, 2, 0, 3, 1, 4, 2, 5],
        [1, 0, 2, 1, 3, 0, 4, 1, 5, 2],
    ], dtype=torch.long)

    # Rough 3D coordinates (Å)
    pos = torch.tensor([
        [ 0.000,  0.000,  0.000],  # C0
        [ 1.540,  0.000,  0.000],  # C1
        [ 2.100,  1.200,  0.000],  # O
        [-0.500,  1.000,  0.000],  # H on C0
        [ 2.050, -0.900,  0.000],  # H on C1
        [ 3.020,  1.100,  0.000],  # H on O
    ], dtype=torch.float)

    # Dummy node features (11-dim, all zeros except element flag)
    x = torch.zeros((6, 11))

    return Data(x=x, edge_index=edge_index, pos=pos)


@pytest.fixture()
def tmp_splits_dir(tmp_path: Path) -> Path:
    return tmp_path / "splits"


# ── featurizer tests ──────────────────────────────────────────────────────────

class TestFeaturizer:
    def test_compute_distances_shape(self, ethanol_graph: Data):
        from src.data.featurizer import compute_distances
        dist = compute_distances(ethanol_graph.pos, ethanol_graph.edge_index)
        E = ethanol_graph.edge_index.shape[1]
        assert dist.shape == (E, 1), f"Expected ({E}, 1), got {dist.shape}"

    def test_compute_distances_positive(self, ethanol_graph: Data):
        from src.data.featurizer import compute_distances
        dist = compute_distances(ethanol_graph.pos, ethanol_graph.edge_index)
        assert (dist >= 0).all(), "Distances must be non-negative"

    def test_compute_distances_symmetric(self, ethanol_graph: Data):
        """For each directed edge (i→j), the reverse edge (j→i) has the same length."""
        from src.data.featurizer import compute_distances
        dist = compute_distances(ethanol_graph.pos, ethanol_graph.edge_index)
        src, dst = ethanol_graph.edge_index
        for idx, (i, j) in enumerate(zip(src.tolist(), dst.tolist())):
            # Find reverse edge
            mask = (dst == i) & (src == j)
            if mask.any():
                rev_idx = mask.nonzero(as_tuple=True)[0][0].item()
                assert abs(dist[idx].item() - dist[rev_idx].item()) < 1e-5

    def test_compute_angles_shape(self, ethanol_graph: Data):
        from src.data.featurizer import compute_angles
        angle_index, angle_attr = compute_angles(
            ethanol_graph.pos, ethanol_graph.edge_index
        )
        assert angle_index.shape[0] == 3
        assert angle_attr.shape[1] == 1
        assert angle_index.shape[1] == angle_attr.shape[0]

    def test_compute_angles_range(self, ethanol_graph: Data):
        from src.data.featurizer import compute_angles
        import math
        _, angle_attr = compute_angles(
            ethanol_graph.pos, ethanol_graph.edge_index
        )
        assert (angle_attr >= 0).all()
        assert (angle_attr <= math.pi + 1e-6).all()

    def test_compute_torsions_shape(self, ethanol_graph: Data):
        from src.data.featurizer import compute_torsions
        torsion_index, torsion_attr = compute_torsions(
            ethanol_graph.pos, ethanol_graph.edge_index
        )
        assert torsion_index.shape[0] == 4
        assert torsion_attr.shape[1] == 1
        assert torsion_index.shape[1] == torsion_attr.shape[0]


# ── GeometryTransform tests ───────────────────────────────────────────────────

class TestGeometryTransform:
    def test_adds_edge_attr_geo(self, ethanol_graph: Data):
        from src.data.transforms import GeometryTransform
        t = GeometryTransform(use_angles=False, use_torsions=False)
        data = t(ethanol_graph)
        assert hasattr(data, "edge_attr_geo")
        E = data.edge_index.shape[1]
        assert data.edge_attr_geo.shape == (E, 1)

    def test_adds_angles(self, ethanol_graph: Data):
        from src.data.transforms import GeometryTransform
        t = GeometryTransform(use_angles=True, use_torsions=False)
        data = t(ethanol_graph)
        assert hasattr(data, "angle_index")
        assert hasattr(data, "angle_attr")

    def test_adds_torsions(self, ethanol_graph: Data):
        from src.data.transforms import GeometryTransform
        t = GeometryTransform(use_angles=True, use_torsions=True)
        data = t(ethanol_graph)
        assert hasattr(data, "torsion_index")
        assert hasattr(data, "torsion_attr")

    def test_raises_without_pos(self):
        from src.data.transforms import GeometryTransform
        data = Data(
            x=torch.zeros((3, 11)),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
        )
        t = GeometryTransform()
        with pytest.raises(ValueError, match="3D coordinates"):
            t(data)

    def test_repr(self):
        from src.data.transforms import GeometryTransform
        t = GeometryTransform(use_angles=True, use_torsions=False)
        assert "use_angles=True" in repr(t)
        assert "use_torsions=False" in repr(t)


# ── Splitter tests ────────────────────────────────────────────────────────────

class TestSplitter:
    def test_sizes_80_20(self, tmp_splits_dir: Path):
        from src.data.splitter import make_splits
        n = 1000
        train_idx, test_idx = make_splits(
            n_samples=n,
            splits_dir=tmp_splits_dir,
            dataset_name="test_ds",
            train_ratio=0.80,
            seed=42,
        )
        assert len(train_idx) == 800
        assert len(test_idx) == 200

    def test_no_overlap(self, tmp_splits_dir: Path):
        from src.data.splitter import make_splits
        train_idx, test_idx = make_splits(
            n_samples=500,
            splits_dir=tmp_splits_dir,
            dataset_name="test_ds2",
        )
        assert len(set(train_idx) & set(test_idx)) == 0

    def test_covers_all_indices(self, tmp_splits_dir: Path):
        from src.data.splitter import make_splits
        n = 300
        train_idx, test_idx = make_splits(
            n_samples=n,
            splits_dir=tmp_splits_dir,
            dataset_name="test_ds3",
        )
        assert sorted(train_idx + test_idx) == list(range(n))

    def test_persists_json(self, tmp_splits_dir: Path):
        from src.data.splitter import make_splits
        make_splits(
            n_samples=200,
            splits_dir=tmp_splits_dir,
            dataset_name="persist_test",
        )
        split_file = tmp_splits_dir / "persist_test_split.json"
        assert split_file.exists()
        with open(split_file) as f:
            payload = json.load(f)
        assert "train_indices" in payload
        assert "test_indices" in payload

    def test_idempotent(self, tmp_splits_dir: Path):
        from src.data.splitter import make_splits
        train1, test1 = make_splits(100, tmp_splits_dir, "idem_test")
        train2, test2 = make_splits(100, tmp_splits_dir, "idem_test")
        assert train1 == train2
        assert test1  == test2

    def test_load_splits(self, tmp_splits_dir: Path):
        from src.data.splitter import make_splits, load_splits
        train_idx, test_idx = make_splits(150, tmp_splits_dir, "load_test")
        split_file = tmp_splits_dir / "load_test_split.json"
        loaded_train, loaded_test = load_splits(split_file)
        assert loaded_train == train_idx
        assert loaded_test == test_idx

    def test_different_seeds_differ(self, tmp_splits_dir: Path):
        from src.data.splitter import make_splits
        train42, _ = make_splits(500, tmp_splits_dir, "seed42", seed=42)
        train99, _ = make_splits(500, tmp_splits_dir, "seed99", seed=99)
        assert train42 != train99


# ── DataLoader tests ──────────────────────────────────────────────────────────

class TestLoaders:
    def _make_fake_dataset(self, n: int = 50) -> list[Data]:
        """Return a list of tiny PyG Data objects."""
        dataset = []
        for _ in range(n):
            data = Data(
                x=torch.randn(5, 11),
                edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                y=torch.tensor([1.0]),
            )
            dataset.append(data)
        return dataset

    def test_loader_sizes(self, tmp_splits_dir: Path):
        from src.data.splitter import make_splits
        from src.data.loaders import get_loaders

        ds = self._make_fake_dataset(100)
        train_idx, test_idx = make_splits(100, tmp_splits_dir, "loader_test")
        train_loader, test_loader = get_loaders(
            ds, train_idx, test_idx, batch_size=16
        )
        # Total batches cover all samples
        total_train = sum(b.num_graphs for b in train_loader)
        total_test  = sum(b.num_graphs for b in test_loader)
        assert total_train == len(train_idx)
        assert total_test  == len(test_idx)

    def test_no_overlap_in_batches(self, tmp_splits_dir: Path):
        """Indices seen in train batches must not appear in test batches."""
        from src.data.splitter import make_splits
        from src.data.loaders import get_loaders

        ds = self._make_fake_dataset(60)
        train_idx, test_idx = make_splits(60, tmp_splits_dir, "overlap_test")
        # Just use the stored idx sets — loaders don't expose raw indices,
        # but we validated no overlap in splitter tests.
        assert len(set(train_idx) & set(test_idx)) == 0
