# src/model/ — Geometric MPNN

## Purpose
Implements the 3D-aware message passing neural network. Encodes molecular geometry
(bond lengths, angles, dihedral/torsion angles) alongside atom and bond features.

## Planned Components
- `mpnn.py`         — Top-level model class
- `layers.py`       — Geometric message passing layers
- `embeddings.py`   — Atom, bond, and geometry feature embeddings
- `readout.py`      — Graph-level pooling and property prediction head

## Design Notes
- Built in PyTorch / PyTorch Geometric
- Geometry encoded as continuous invariant scalars (distances, angles) rather than raw coordinates
- Architecture should remain differentiable end-to-end for explanation and counterfactual gradient flow
