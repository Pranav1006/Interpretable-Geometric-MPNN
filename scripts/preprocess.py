"""
scripts/preprocess.py
=====================
Download, featurize, and split the QM9 dataset.

This script is the single entry-point for setting up the data pipeline.
It should be run once before any training script.

Usage
-----
    python scripts/preprocess.py --config configs/preprocess_qm9.yaml

What it does
------------
  1. Downloads QM9 via PyG (SDF → processed Data objects with geometry).
  2. Applies GeometryTransform (bond lengths, angles, torsions) as a
     pre_transform so the computation happens once and is cached.
  3. Generates an 80/20 train/test split and persists the indices to
     data/splits/qm9_split.json.
  4. Verifies the loaders by iterating one batch from each split.
  5. Prints a summary of dataset statistics.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

# ── Make src/ importable when running from project root ──────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datasets import QM9Dataset
from src.data.splitter import make_splits
from src.data.loaders import get_loaders


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and preprocess QM9 for IG-MPNN."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/preprocess_qm9.yaml",
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def print_section(title: str) -> None:
    width = 60
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    data_cfg  = cfg["data"]
    split_cfg = cfg["splitting"]
    dl_cfg    = cfg["dataloader"]

    # ── 1. Download + featurize ──────────────────────────────────────
    print_section("Step 1 — Download & Featurize QM9")
    t0 = time.time()
    dataset = QM9Dataset(
        root=data_cfg["root_dir"],
        target_idx=data_cfg["target_idx"],
        use_angles=data_cfg["use_angles"],
        use_torsions=data_cfg["use_torsions"],
    )
    print(f"  Dataset : {dataset}")
    print(f"  Node features : {dataset.num_node_features}")
    print(f"  Edge features : {dataset.num_edge_features}")
    print(f"  Time elapsed  : {time.time() - t0:.1f}s")

    # ── 2. Example graph ─────────────────────────────────────────────
    print_section("Step 2 — Inspect Example Graph")
    sample = dataset[0]
    print(f"  {sample}")
    if hasattr(sample, "edge_attr_geo"):
        print(f"  Bond lengths  : {sample.edge_attr_geo.shape}  (E, 1)")
    if hasattr(sample, "angle_attr"):
        print(f"  Bond angles   : {sample.angle_attr.shape}     (T, 1)")
    if hasattr(sample, "torsion_attr"):
        print(f"  Torsion angles: {sample.torsion_attr.shape}   (Q, 1)")

    # ── 3. Split ─────────────────────────────────────────────────────
    print_section("Step 3 — Train / Test Split  (80 / 20)")
    train_idx, test_idx = make_splits(
        n_samples=len(dataset),
        splits_dir=data_cfg["splits_dir"],
        dataset_name=data_cfg["dataset"],
        train_ratio=split_cfg["train_ratio"],
        seed=split_cfg["seed"],
        force=split_cfg.get("force", False),
    )

    # ── 4. Build loaders + smoke test ────────────────────────────────
    print_section("Step 4 — Build DataLoaders & Smoke Test")
    train_loader, test_loader = get_loaders(
        dataset=dataset,
        train_idx=train_idx,
        test_idx=test_idx,
        batch_size=dl_cfg["batch_size"],
        num_workers=dl_cfg.get("num_workers", 0),
        shuffle_train=dl_cfg.get("shuffle_train", True),
    )

    train_batch = next(iter(train_loader))
    test_batch  = next(iter(test_loader))
    print(f"  Train batch : {train_batch}")
    print(f"  Test  batch : {test_batch}")

    # ── 5. Summary ───────────────────────────────────────────────────
    print_section("Summary")
    print(f"  Total molecules : {len(dataset):,}")
    print(f"  Train           : {len(train_idx):,}  ({len(train_idx)/len(dataset)*100:.0f}%)")
    print(f"  Test            : {len(test_idx):,}  ({len(test_idx)/len(dataset)*100:.0f}%)")
    print(f"  Target          : {dataset.target_name} [{dataset.target_unit}]")
    print(f"  Batch size      : {dl_cfg['batch_size']}")
    print(f"  Train batches   : {len(train_loader)}")
    print(f"  Test  batches   : {len(test_loader)}")
    print(f"\n  -- Pipeline ready.  Split saved to {data_cfg['splits_dir']}\n")


if __name__ == "__main__":
    main()
