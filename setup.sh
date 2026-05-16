#!/usr/bin/env bash
# =============================================================================
# setup.sh — IG-MPNN environment setup (CPU-only, macOS / Linux)
#
# Mirrors setup.bat exactly, but for Unix shells.
# Resolves the torch-scatter / torch-sparse "build wheels" failure by
# using the PyG CDN's pre-built wheels instead of compiling from source.
#
# Usage:
#   bash setup.sh          — create venv at .venv/ and install everything
#   bash setup.sh --check  — verify imports after install
# =============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
VENV_DIR=".venv"
TORCH_VERSION="2.10.0"
TORCH_TAG="torch-2.10.0"
# CPU-only wheel tag (change to cu118 / cu121 if you add a GPU later)
CUDA_TAG="cpu"
PYG_CDN="https://data.pyg.org/whl/${TORCH_TAG}+${CUDA_TAG}.html"

echo ""
echo "IG-MPNN Setup (CPU-only)"
echo "========================="
echo "PyTorch : ${TORCH_VERSION}"
echo "CUDA    : ${CUDA_TAG}"
echo "PyG CDN : ${PYG_CDN}"
echo "Venv    : ${VENV_DIR}/"
echo ""

# ── Create virtual environment ────────────────────────────────────────────────
if [ ! -d "${VENV_DIR}" ]; then
    echo "[1/5] Creating virtual environment at ${VENV_DIR}/ ..."
    python3 -m venv "${VENV_DIR}"
else
    echo "[1/5] Virtual environment already exists — skipping creation."
fi

# ── Activate ──────────────────────────────────────────────────────────────────
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# ── Upgrade pip / wheel / setuptools ──────────────────────────────────────────
echo "[2/5] Upgrading pip, wheel, setuptools ..."
pip install --upgrade pip wheel setuptools

# ── PyTorch CPU-only ──────────────────────────────────────────────────────────
echo "[3/5] Installing PyTorch ${TORCH_VERSION} (CPU-only) ..."
pip install "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu

# ── PyG extensions (pre-built wheels — no compiler required) ──────────────────
echo "[4/5] Installing PyG extensions from pre-built wheels ..."
pip install torch-scatter torch-sparse torch-cluster torch-geometric \
    --find-links "${PYG_CDN}"

# ── Remaining dependencies ────────────────────────────────────────────────────
echo "[5/5] Installing remaining dependencies ..."
pip install \
    rdkit \
    numpy scipy pandas scikit-learn \
    matplotlib seaborn \
    pyyaml tqdm \
    jupyterlab \
    pytest

# ── Optional: verification ────────────────────────────────────────────────────
if [ "${1-}" = "--check" ]; then
    echo ""
    echo "Running import check ..."
    python - <<'EOF'
import torch, torch_geometric, torch_scatter, torch_sparse
from rdkit import Chem
import numpy, scipy, pandas, sklearn, matplotlib, seaborn, yaml, tqdm
print("torch          :", torch.__version__)
print("torch_geometric:", torch_geometric.__version__)
print("torch_scatter  : ok")
print("torch_sparse   : ok")
print("rdkit          : ok")
print()
print("All imports succeeded.")
EOF
fi

echo ""
echo "Setup complete."
echo "Activate with:  source ${VENV_DIR}/bin/activate"
echo "Then verify:    bash setup.sh --check"
echo ""
