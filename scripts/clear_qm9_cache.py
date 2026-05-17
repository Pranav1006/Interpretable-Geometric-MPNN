"""
scripts/clear_qm9_cache.py
==========================
Deletes the raw QM9 SDF files so that the next run of preprocess.py
triggers a clean re-download.

Run this once after a failed/interrupted preprocess run:
    python scripts/clear_qm9_cache.py

What it deletes
---------------
  data/raw/qm9/raw/gdb9.sdf
  data/raw/qm9/raw/gdb9.sdf.csv
  data/raw/qm9/raw/uncharacterized.txt
  data/raw/qm9/raw/QM9_README
  data/raw/qm9/processed/   (contents only, not the folder)

What it keeps
-------------
  data/splits/   — split indices are independent of the raw files
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

raw_dir       = ROOT / "data" / "raw" / "qm9" / "raw"
processed_dir = ROOT / "data" / "raw" / "qm9" / "processed"

deleted = []

for path in [raw_dir, processed_dir]:
    if path.exists():
        shutil.rmtree(path)
        path.mkdir()
        deleted.append(str(path))

if deleted:
    print("Cleared:")
    for p in deleted:
        print(f"  {p}")
    print("\nRun `python scripts/preprocess.py --config configs/preprocess_qm9.yaml` to re-download.")
else:
    print("Nothing to clear — directories did not exist.")
