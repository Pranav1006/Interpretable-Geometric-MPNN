# src/data/ — Data Loading and Featurization

## Purpose
Handles downloading, processing, and featurizing molecular datasets into
PyTorch Geometric `Data` objects for use across all four pillars.

## Planned Components
- `datasets.py`      — Dataset classes (QM9, ESOL, Lipophilicity, etc.)
- `featurizer.py`    — Atom, bond, and 3D geometry featurization
- `transforms.py`    — PyG transforms (radius graph, geometry computation)
- `splitter.py`      — Train/val/test splitting strategies

## Data Directories
- `data/raw/`        — Downloaded source files
- `data/processed/`  — Serialized PyG graph objects
- `data/splits/`     — Saved split indices (JSON/numpy)

## Notes
- 3D coordinates required for geometric features — use conformer generation (RDKit ETKDG) if not available
- All processed datasets should include a metadata file describing featurization choices
