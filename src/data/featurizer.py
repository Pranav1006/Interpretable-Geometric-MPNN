"""
src/data/featurizer.py
======================
Atom, bond, and 3D geometry featurization for molecular graphs.

All featurization choices are encoded here so that every other module
(datasets, transforms, tests) imports from one source of truth.

Atom features (11-dim per atom)
---------------------------------
  0   atomic number (Z), normalised by 9 (max in QM9 is F=9)
  1-5 one-hot: {H, C, N, O, F}
  6   formal charge
  7   number of implicit Hs (normalised by 4)
  8   is in ring (bool)
  9   is aromatic (bool)
  10  degree (normalised by 4)

Bond features (4-dim per directed edge)
-----------------------------------------
  0-3 one-hot: {single, double, triple, aromatic}

Geometry (computed by GeometryTransform, stored on Data)
---------------------------------------------------------
  edge_attr_geo[:, 0]  — bond length  (Å), L2 norm of Δpos
  Data.angle_index     — (3, num_angles) triplet indices
  Data.angle_attr      — (num_angles, 1) bond angles (radians)
  Data.torsion_index   — (4, num_torsions) quadruplet indices
  Data.torsion_attr    — (num_torsions, 1) torsion angles (radians)
"""

from __future__ import annotations

import torch
from torch import Tensor

# ── Constants ────────────────────────────────────────────────────────────────

ATOM_TYPES: list[int] = [1, 6, 7, 8, 9]          # H, C, N, O, F
ATOM_TYPES_SET: set[int] = set(ATOM_TYPES)
MAX_ATOMIC_NUM: float = 9.0                        # F is largest in QM9
MAX_HS: float = 4.0
MAX_DEGREE: float = 4.0

BOND_TYPE_TO_IDX: dict = {}                        # populated lazily (RDKit import)

ATOM_FEATURE_DIM: int = 11
BOND_FEATURE_DIM: int = 4


# ── Atom featurization ───────────────────────────────────────────────────────

def atom_features(atom) -> list[float]:
    """Return an 11-dim feature vector for an RDKit atom."""
    z = atom.GetAtomicNum()
    one_hot = [float(z == t) for t in ATOM_TYPES]
    return [
        z / MAX_ATOMIC_NUM,              # 0 — normalised atomic number
        *one_hot,                        # 1–5 — element identity
        float(atom.GetFormalCharge()),   # 6
        atom.GetTotalNumHs() / MAX_HS,  # 7
        float(atom.IsInRing()),          # 8
        float(atom.GetIsAromatic()),     # 9
        atom.GetDegree() / MAX_DEGREE,  # 10
    ]


# ── Bond featurization ───────────────────────────────────────────────────────

def _bond_type_idx(bond) -> int:
    from rdkit.Chem import BondType
    mapping = {
        BondType.SINGLE:   0,
        BondType.DOUBLE:   1,
        BondType.TRIPLE:   2,
        BondType.AROMATIC: 3,
    }
    return mapping.get(bond.GetBondType(), 0)


def bond_features(bond) -> list[float]:
    """Return a 4-dim one-hot feature vector for an RDKit bond."""
    idx = _bond_type_idx(bond)
    return [float(i == idx) for i in range(BOND_FEATURE_DIM)]


# ── Geometry helpers ─────────────────────────────────────────────────────────

def compute_distances(pos: Tensor, edge_index: Tensor) -> Tensor:
    """
    Compute pairwise L2 distances for each directed edge.

    Parameters
    ----------
    pos : (N, 3) float tensor of 3D coordinates.
    edge_index : (2, E) long tensor of directed edges.

    Returns
    -------
    dist : (E, 1) float tensor of Euclidean distances in Ångström.
    """
    src, dst = edge_index
    diff = pos[dst] - pos[src]                      # (E, 3)
    return diff.norm(dim=-1, keepdim=True)           # (E, 1)


def compute_angles(
    pos: Tensor, edge_index: Tensor
) -> tuple[Tensor, Tensor]:
    """
    Enumerate bond-angle triplets (i–j–k) and compute the angle at j.

    A triplet is valid when edges (i→j) and (j→k) both exist and i ≠ k.

    Returns
    -------
    angle_index : (3, T) — rows are [i, j, k]
    angle_attr  : (T, 1) — angle at j in radians ∈ [0, π]
    """
    src, dst = edge_index[0], edge_index[1]

    # Build neighbour lookup: j → {all i where (i→j) in edge_index}
    from collections import defaultdict
    neighbours: dict[int, list[int]] = defaultdict(list)
    for i, j in zip(src.tolist(), dst.tolist()):
        neighbours[j].append(i)

    triplets_i, triplets_j, triplets_k = [], [], []
    for j, in_neighbours in neighbours.items():
        for idx_a in range(len(in_neighbours)):
            for idx_b in range(idx_a + 1, len(in_neighbours)):
                i = in_neighbours[idx_a]
                k = in_neighbours[idx_b]
                triplets_i.append(i)
                triplets_j.append(j)
                triplets_k.append(k)

    if not triplets_i:
        empty_idx = torch.zeros((3, 0), dtype=torch.long)
        empty_attr = torch.zeros((0, 1), dtype=torch.float)
        return empty_idx, empty_attr

    ti = torch.tensor(triplets_i, dtype=torch.long)
    tj = torch.tensor(triplets_j, dtype=torch.long)
    tk = torch.tensor(triplets_k, dtype=torch.long)

    vi = pos[ti] - pos[tj]   # vector j→i
    vk = pos[tk] - pos[tj]   # vector j→k

    cos = (vi * vk).sum(dim=-1) / (
        vi.norm(dim=-1) * vk.norm(dim=-1) + 1e-8
    )
    angles = torch.acos(cos.clamp(-1.0, 1.0)).unsqueeze(-1)  # (T, 1)

    angle_index = torch.stack([ti, tj, tk], dim=0)            # (3, T)
    return angle_index, angles


def compute_torsions(
    pos: Tensor, edge_index: Tensor
) -> tuple[Tensor, Tensor]:
    """
    Enumerate dihedral-angle quadruplets (i–j–k–l) and compute the torsion.

    A quadruplet is valid when edges (i→j), (j→k), and (k→l) all exist,
    and i ≠ l.

    Returns
    -------
    torsion_index : (4, Q) — rows are [i, j, k, l]
    torsion_attr  : (Q, 1) — dihedral angle in radians ∈ (-π, π]
    """
    src, dst = edge_index[0], edge_index[1]

    from collections import defaultdict
    successors: dict[int, list[int]] = defaultdict(list)
    for i, j in zip(src.tolist(), dst.tolist()):
        successors[i].append(j)

    qi, qj, qk, ql = [], [], [], []
    for j in successors:
        for k in successors[j]:
            if k == j:
                continue
            for i in successors[j]:
                if i == k:
                    continue
                for l in successors[k]:
                    if l == j or l == i:
                        continue
                    qi.append(i)
                    qj.append(j)
                    qk.append(k)
                    ql.append(l)

    if not qi:
        empty_idx = torch.zeros((4, 0), dtype=torch.long)
        empty_attr = torch.zeros((0, 1), dtype=torch.float)
        return empty_idx, empty_attr

    ti = torch.tensor(qi, dtype=torch.long)
    tj = torch.tensor(qj, dtype=torch.long)
    tk = torch.tensor(qk, dtype=torch.long)
    tl = torch.tensor(ql, dtype=torch.long)

    b1 = pos[tj] - pos[ti]
    b2 = pos[tk] - pos[tj]
    b3 = pos[tl] - pos[tk]

    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)
    b2_norm = b2 / (b2.norm(dim=-1, keepdim=True) + 1e-8)
    m1 = torch.cross(n1, b2_norm, dim=-1)

    x = (n1 * n2).sum(dim=-1)
    y = (m1 * n2).sum(dim=-1)
    torsions = torch.atan2(y, x).unsqueeze(-1)    # (Q, 1)

    torsion_index = torch.stack([ti, tj, tk, tl], dim=0)  # (4, Q)
    return torsion_index, torsions
