"""
src/data/datasets.py
====================
Dataset classes for IG-MPNN.

QM9Dataset
----------
A thin wrapper around a patched QM9 class (``_SafeQM9``) that:

  1. Downloads and caches the raw QM9 SDF file to ``data/raw/``.
  2. Skips molecules that RDKit's SDMolSupplier cannot parse (returns None),
     which PyG's upstream QM9.process() does not guard against.
  3. Applies ``GeometryTransform`` to augment every graph with bond lengths,
     angles, and torsion angles computed from the conformer coordinates.
  4. Exposes a ``target`` property for easy access to a single regression
     target (default: ``mu``, index 0 — dipole moment in Debye).

Why ``_SafeQM9``
----------------
PyG's QM9.process() iterates the raw SDF via RDKit's SDMolSupplier, which
returns None for any molecule it cannot parse (malformed entries that slip
past the official uncharacterized.txt skip list).  The upstream loop calls
mol.GetNumAtoms() without a None check, crashing mid-processing.

``_SafeQM9`` subclasses QM9 and overrides process() with a single added guard:

    if mol is None:
        continue

Everything else in process() is identical to the upstream implementation so
behaviour is unchanged for well-formed molecules.

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

import sys
from pathlib import Path

import torch
from tqdm import tqdm
from torch_geometric.datasets import QM9
from torch_geometric.data import Data
from torch_geometric.utils import one_hot, scatter

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

# Copied from torch_geometric.datasets.qm9 so _SafeQM9.process() is
# self-contained and doesn't depend on PyG internals staying stable.
_HAR2EV = 27.211386246
_KCALMOL2EV = 0.04336414
_CONVERSION = torch.tensor([
    1., 1., _HAR2EV, _HAR2EV, _HAR2EV, 1., _HAR2EV, _HAR2EV, _HAR2EV,
    _HAR2EV, _HAR2EV, 1., _KCALMOL2EV, _KCALMOL2EV, _KCALMOL2EV,
    _KCALMOL2EV, 1., 1., 1.,
])


# ── Patched QM9 subclass ─────────────────────────────────────────────────────

class _SafeQM9(QM9):
    """
    QM9 with a None-mol guard in process().

    PyG's SDMolSupplier can return None for entries RDKit cannot parse.
    The upstream loop does not check for this before calling
    mol.GetNumAtoms(), causing an AttributeError mid-processing.
    This subclass overrides process() with a single added guard and is
    otherwise identical to the upstream implementation.
    """

    def process(self) -> None:
        try:
            from rdkit import Chem, RDLogger
            from rdkit.Chem.rdchem import BondType as BT
            from rdkit.Chem.rdchem import HybridizationType
            from torch_geometric.io import fs
            RDLogger.DisableLog('rdApp.*')
            WITH_RDKIT = True
        except ImportError:
            WITH_RDKIT = False

        if not WITH_RDKIT:
            print(
                "Using a pre-processed version of the dataset. Please "
                "install 'rdkit' to alternatively process the raw data.",
                file=sys.stderr,
            )
            from torch_geometric.io import fs
            data_list = fs.torch_load(self.raw_paths[0])
            data_list = [Data(**d) for d in data_list]
            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]
            self.save(data_list, self.processed_paths[0])
            return

        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        with open(self.raw_paths[1]) as f:
            target = [
                [float(x) for x in line.split(',')[1:20]]
                for line in f.read().split('\n')[1:-1]
            ]
            y = torch.tensor(target, dtype=torch.float)
            y = torch.cat([y[:, 3:], y[:, :3]], dim=-1)
            y = y * _CONVERSION.view(1, -1)

        with open(self.raw_paths[2]) as f:
            skip = [
                int(x.split()[0]) - 1
                for x in f.read().split('\n')[9:-2]
            ]

        suppl = Chem.SDMolSupplier(
            self.raw_paths[0], removeHs=False, sanitize=False
        )

        skipped_none = 0
        data_list = []
        for i, mol in enumerate(tqdm(suppl)):
            if i in skip:
                continue

            # ── THE FIX: upstream lacks this guard ───────────────────
            if mol is None:
                skipped_none += 1
                continue

            N = mol.GetNumAtoms()

            conf = mol.GetConformer()
            pos = torch.tensor(conf.GetPositions(), dtype=torch.float)

            type_idx, atomic_number, aromatic = [], [], []
            sp, sp2, sp3 = [], [], []
            for atom in mol.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])
                atomic_number.append(atom.GetAtomicNum())
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hyb = atom.GetHybridization()
                sp.append(1 if hyb == HybridizationType.SP else 0)
                sp2.append(1 if hyb == HybridizationType.SP2 else 0)
                sp3.append(1 if hyb == HybridizationType.SP3 else 0)

            z = torch.tensor(atomic_number, dtype=torch.long)

            rows, cols, edge_types = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                rows += [start, end]
                cols += [end, start]
                edge_types += 2 * [bonds[bond.GetBondType()]]

            edge_index = torch.tensor([rows, cols], dtype=torch.long)
            edge_type  = torch.tensor(edge_types, dtype=torch.long)
            edge_attr  = one_hot(edge_type, num_classes=len(bonds))

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type  = edge_type[perm]
            edge_attr  = edge_attr[perm]

            row, col = edge_index
            hs = (z == 1).to(torch.float)
            num_hs = scatter(hs[row], col, dim_size=N, reduce='sum').tolist()

            x1 = one_hot(torch.tensor(type_idx), num_classes=len(types))
            x2 = torch.tensor(
                [atomic_number, aromatic, sp, sp2, sp3, num_hs],
                dtype=torch.float,
            ).t().contiguous()
            x = torch.cat([x1, x2], dim=-1)

            name   = mol.GetProp('_Name')
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

            data = Data(
                x=x,
                z=z,
                pos=pos,
                edge_index=edge_index,
                smiles=smiles,
                edge_attr=edge_attr,
                y=y[i].unsqueeze(0),
                name=name,
                idx=i,
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        if skipped_none:
            print(
                f"[_SafeQM9] Skipped {skipped_none} molecule(s) that "
                "RDKit could not parse (SDMolSupplier returned None)."
            )

        self.save(data_list, self.processed_paths[0])


# ── Public dataset wrapper ───────────────────────────────────────────────────

class QM9Dataset:
    """
    Wrapper around ``_SafeQM9`` (a patched ``torch_geometric.datasets.QM9``).

    Parameters
    ----------
    root : str or Path
        Project-level data root (typically ``"data"``).
        QM9 will download to ``<root>/raw/qm9/`` and cache processed
        graphs to ``<root>/raw/qm9/processed/``.
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

        self._dataset = _SafeQM9(
            root=str(qm9_root),
            transform=geo_transform,
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
