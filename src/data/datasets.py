"""
src/data/datasets.py
====================
Dataset classes for IG-MPNN.

QM9Dataset
----------
A thin wrapper around ``torch_geometric.datasets.QM9`` that:

  1. Downloads and caches the raw QM9 SDF file to ``data/raw/``.
  2. Applies ``GeometryTransform`` to augment every graph with bond lengths,
     angles, and torsion angles computed from the conformer coordinates.
  3. Exposes a ``target`` property for easy access to a single regression
     target (default: ``mu``, index 0 — dipole moment in Debye).

QM9 target indices (following the PyG convention)
--------------------------------------------------
  0   μ     dipole moment          (Debye)
  1   α     isotropic polarizability (Bohr³)
  2   ε_HOMO HOMO energy           (eV)
  3   ε_LUMO LUMO energy           (eV)
  4   Δε    HOMO-LUMO gap          (eV)
  5   ⟨R²⟩  electronic spatial extent (Bohr²)
  6   ZPVE  zero-point vib. energy (eV)
  7   U₀    internal energy at 0K  (eV)
  8   U     internal energy at 298K (eV)
  9   H     enthalpy at 298K       (eV)
  10  G     free energy at 298K    (eV)
  11  Cv    heat capacity at 298K  (cal/mol/K)
  12–18  μ_x, μ_y, μ_z, α_xx … (individual components, rarely used)

Usage
-----
    from src.data.datasets import QM9Dataset

    ds = QM9Dataset(
        root="data",
        target_idx=0,       # dipole moment
        use_angles=True,
        use_torsions=True,
    )
    data = ds[0]
    print(data)             # Data(x, edge_index, edge_attr, pos,
                            #      edge_attr_geo, angle_index, angle_attr,
                            #      torsion_index, torsion_attr, y)
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.datasets import QM9
from torch_geometric.data import Data

from .transforms import GeometryTransform


# ── QM9 target metadata ──────────────────────────────────────────────────────

QM9_TARGETS: dict[int, dict] = {
    0:  {"name": "mu",   "unit": "D",        "desc": "Dipole moment"},
    1:  {"name": "alpha","unit": "a₀³",      "desc": "Isotropic polarizability"},
    2:  {"name": "homo", "unit": "eV",       "desc": "HOMO energy"},
    3:  {"name": "lumo", "unit": "eV",       "desc": "LUMO energy"},
    4:  {"name": "gap",  "unit": "eV",       "desc": "HOMO-LUMO gap"},
    5:  {"name": "r2",   "unit": "a₀²",      "desc": "Electronic spatial extent"},
    6:  {"name": "zpve", "unit": "eV",       "desc": "Zero-point vibrational energy"},
    7:  {"name": "u0",   "unit": "eV",       "desc": "Internal energy at 0K"},
    8:  {"name": "u298", "unit": "eV",       "desc": "Internal energy at 298K"},
    9:  {"name": "h298", "unit": "eV",       "desc": "Enthalpy at 298K"},
    10: {"name": "g298", "unit": "eV",       "desc": "Free energy at 298K"},
    11: {"name": "cv",   "unit": "cal/mol/K","desc": "Heat capacity at 298K"},
}


class QM9Dataset:
    """
    Wrapper around ``torch_geometric.datasets.QM9``.

    Parameters
    ----------
    root : str or Path
        Project-level data root (typically ``"data"``).
        QM9 will download to ``<root>/raw/qm9/`` and cache processed
        graphs to ``<root>/processed/qm9/``.
    target_idx : int
        Which of the 12 QM9 regression targets to expose as ``data.y``.
        See module docstring for the full list.  Default: 0 (dipole moment).
    use_angles : bool
        Whether ``GeometryTransform`` computes bond-angle triplets.
    use_torsions : bool
        Whether ``GeometryTransform`` computes torsion-angle quadruplets.
    """

    def __init__(
        self,
        root: str | Path = "data",
        target_idx: int = 0,
        use_angles: bool = True,
        use_torsions: bool = True,
    ) -> None:
        root = Path(root)
        qm9_root = root / "raw" / "qm9"

        self.target_idx = target_idx
        self.target_meta = QM9_TARGETS.get(target_idx, {})

        geo_transform = GeometryTransform(
            use_angles=use_angles,
            use_torsions=use_torsions,
        )

        # PyG downloads raw SDF + processes into Data objects automatically.
        # pre_transform runs once at download time (cheaper than re-running).
        self._dataset = QM9(
            root=str(qm9_root),
            pre_transform=geo_transform,
        )

    # ------------------------------------------------------------------
    # Sequence protocol — delegates to the underlying PyG dataset
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Data:
        data = self._dataset[idx]
        # QM9 stores all 19 targets in data.y of shape (1, 19).
        # Expose just the requested column as a scalar (1,) tensor.
        data.y = data.y[:, self.target_idx]
        return data

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def target_name(self) -> str:
        return self.target_meta.get("name", f"target_{self.target_idx}")

    @property
    def target_unit(self) -> str:
        return self.target_meta.get("unit", "")

    @property
    def num_node_features(self) -> int:
        return self._dataset.num_node_features

    @property
    def num_edge_features(self) -> int:
        return self._dataset.num_edge_features

    def __repr__(self) -> str:
        return (
            f"QM9Dataset("
            f"n={len(self)}, "
            f"target={self.target_name} [{self.target_unit}])"
        )
