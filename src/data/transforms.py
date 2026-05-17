"""
src/data/transforms.py
======================
PyG transforms that add geometric features to a Data object.

GeometryTransform
-----------------
Applied to every graph after it is loaded from the QM9 dataset.  It computes:

  • Bond lengths      → stored in ``data.edge_attr_geo``  (E, 1)
  • Bond angles       → stored in ``data.angle_index``    (3, T)
                                   ``data.angle_attr``    (T, 1)
  • Torsion angles    → stored in ``data.torsion_index``  (4, Q)
                                   ``data.torsion_attr``  (Q, 1)

All angles are in radians.  Distances are in Ångström.

Usage
-----
    from src.data.transforms import GeometryTransform
    transform = GeometryTransform(use_angles=True, use_torsions=True)
    data = transform(data)
"""

from __future__ import annotations

import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data

from .featurizer import compute_distances, compute_angles, compute_torsions


class GeometryTransform(BaseTransform):
    """
    Augment a PyG ``Data`` object with 3D geometric features.

    Parameters
    ----------
    use_angles : bool
        Whether to compute and store bond-angle triplets.
    use_torsions : bool
        Whether to compute and store dihedral-angle quadruplets.
        Requires ``use_angles=True`` to be meaningful (but is independent).
    """

    def __init__(
        self,
        use_angles: bool = True,
        use_torsions: bool = True,
    ) -> None:
        self.use_angles = use_angles
        self.use_torsions = use_torsions

    def forward(self, data: Data) -> Data:
        """
        PyG's BaseTransform requires ``forward`` to be implemented.
        ``__call__`` on BaseTransform delegates here automatically.
        """
        if data.pos is None:
            raise ValueError(
                "GeometryTransform requires 3D coordinates (data.pos). "
                "Ensure the dataset provides conformer positions."
            )

        pos = data.pos
        edge_index = data.edge_index

        # ── Bond lengths ─────────────────────────────────────────────
        dist = compute_distances(pos, edge_index)   # (E, 1)
        data.edge_attr_geo = dist

        # ── Bond angles ──────────────────────────────────────────────
        if self.use_angles:
            angle_index, angle_attr = compute_angles(pos, edge_index)
            data.angle_index = angle_index    # (3, T)
            data.angle_attr = angle_attr      # (T, 1)

        # ── Torsion angles ───────────────────────────────────────────
        if self.use_torsions:
            torsion_index, torsion_attr = compute_torsions(pos, edge_index)
            data.torsion_index = torsion_index   # (4, Q)
            data.torsion_attr = torsion_attr     # (Q, 1)

        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"use_angles={self.use_angles}, "
            f"use_torsions={self.use_torsions})"
        )
