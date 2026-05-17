# src/data/ — Data Loading and Featurization

## Purpose
Handles downloading, processing, and featurizing molecular datasets into
PyTorch Geometric `Data` objects for use across all four pillars.

## Components

| File | Description |
|---|---|
| `__init__.py` | Public API: `QM9Dataset`, `GeometryTransform`, `make_splits`, `load_splits`, `get_loaders` |
| `datasets.py` | `QM9Dataset` — thin wrapper around `torch_geometric.datasets.QM9`; exposes a single regression target as `data.y` |
| `featurizer.py` | Atom, bond, and 3D geometry featurization; source of truth for feature dimensions and geometry math |
| `transforms.py` | `GeometryTransform` — PyG `BaseTransform` that adds bond lengths, angles, and torsion angles to each graph |
| `splitter.py` | `make_splits` / `load_splits` — produces and persists 80/20 train/test index JSON files |
| `loaders.py` | `get_loaders` — wraps split indices into `torch_geometric.loader.DataLoader` objects |

## Data Directories

| Directory | Contents |
|---|---|
| `data/raw/qm9/` | Downloaded QM9 SDF file (auto-managed by PyG) |
| `data/processed/qm9/` | Serialized PyG `Data` objects with geometry features (auto-managed by PyG) |
| `data/splits/` | `qm9_split.json` — shuffled train/test indices with metadata |

## Feature Dimensions

| Feature set | Dim | Notes |
|---|---|---|
| Node features (`data.x`) | 11 | Atomic number, element one-hot, charge, Hs, ring, aromaticity, degree |
| Bond features (`data.edge_attr`) | 4 | One-hot bond type: single / double / triple / aromatic |
| Bond lengths (`data.edge_attr_geo`) | (E, 1) | L2 distance in Å |
| Bond angles (`data.angle_attr`) | (T, 1) | Angle at centre atom, radians ∈ [0, π] |
| Torsion angles (`data.torsion_attr`) | (Q, 1) | Dihedral angle, radians ∈ (−π, π] |

## Split

- **80 % train / 20 % test**, seed = 42, shuffled once and frozen.
- Validation set will be carved from train during hyperparameter search.
- Split file: `data/splits/qm9_split.json`

## Usage

```python
from src.data import QM9Dataset, make_splits, get_loaders

ds = QM9Dataset(root="data", target_idx=0)   # 0 = dipole moment (μ)

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
```

Or run the entry-point script once:

```bash
python scripts/preprocess.py --config configs/preprocess_qm9.yaml
```

## Notes

- `GeometryTransform` is applied as a `pre_transform` — geometry is computed
  once at download time and cached in `data/processed/`.
- Re-running `preprocess.py` with `splitting.force: false` (default) reuses
  the existing split file; set `force: true` to regenerate.
- `num_workers: 0` is recommended on Windows to avoid multiprocessing issues.
