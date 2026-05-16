@echo off
:: ============================================================================
:: setup.bat — IG-MPNN environment setup (CPU-only, Windows)
::
:: Resolves the torch-scatter / torch-sparse "build wheels" failure by
:: fetching pre-built wheels directly from the PyG CDN instead of compiling
:: from source. All four PyG packages (torch-geometric, torch-scatter,
:: torch-sparse, torch-cluster) must reference the same torch + CUDA tag.
::
:: Usage:
::   setup.bat          — create venv at .venv\ and install everything
::   setup.bat --check  — verify imports after install
:: ============================================================================

setlocal EnableDelayedExpansion

:: ── Configuration ───────────────────────────────────────────────────────────
set VENV_DIR=.venv
set TORCH_VERSION=2.10.0
set TORCH_TAG=torch-2.10.0
:: CPU-only wheel tag (change to cu118 / cu121 if you add a GPU later)
set CUDA_TAG=cpu
set PYG_CDN=https://data.pyg.org/whl/%TORCH_TAG%+%CUDA_TAG%.html

echo.
echo IG-MPNN Setup ^(CPU-only, Windows^)
echo ====================================
echo PyTorch : %TORCH_VERSION%
echo CUDA    : %CUDA_TAG%
echo PyG CDN : %PYG_CDN%
echo Venv    : %VENV_DIR%\
echo.

:: ── Require Python 3.10+ ────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] python not found on PATH. Install Python 3.10+ and try again.
    exit /b 1
)

:: ── Create virtual environment ───────────────────────────────────────────────
if not exist %VENV_DIR%\ (
    echo [1/5] Creating virtual environment at %VENV_DIR%\ ...
    python -m venv %VENV_DIR%
) else (
    echo [1/5] Virtual environment already exists — skipping creation.
)

:: ── Activate ─────────────────────────────────────────────────────────────────
call %VENV_DIR%\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Could not activate virtual environment.
    exit /b 1
)

:: ── Upgrade pip / wheel / setuptools ─────────────────────────────────────────
echo [2/5] Upgrading pip, wheel, setuptools ...
pip install --upgrade pip wheel setuptools

:: ── PyTorch CPU-only ─────────────────────────────────────────────────────────
:: We pin the exact version so PyG wheels are guaranteed to match.
echo [3/5] Installing PyTorch %TORCH_VERSION% (CPU-only) ...
pip install torch==%TORCH_VERSION% --index-url https://download.pytorch.org/whl/cpu

:: ── PyG extensions (pre-built wheels — no compiler required) ─────────────────
:: torch-scatter and torch-sparse are built against a specific
:: (torch version, CUDA version) pair. The --find-links flag points pip at
:: the CDN page that hosts those exact wheels, so it never tries to compile.
echo [4/5] Installing PyG extensions from pre-built wheels ...
pip install torch-scatter torch-sparse torch-cluster torch-geometric ^
    --find-links %PYG_CDN%

:: ── Remaining dependencies ───────────────────────────────────────────────────
echo [5/5] Installing remaining dependencies ...
pip install ^
    rdkit ^
    numpy scipy pandas scikit-learn ^
    matplotlib seaborn ^
    pyyaml tqdm ^
    jupyterlab ^
    pytest

:: ── Optional: verification ───────────────────────────────────────────────────
if "%1"=="--check" (
    echo.
    echo Running import check ...

    :: Write the check script to a temp file (heredocs don't exist in .bat)
    set CHECK_SCRIPT=%TEMP%\igmpnn_check.py
    (
        echo import sys
        echo try:
        echo     import torch
        echo     print^("torch          :", torch.__version__^)
        echo     import torch_geometric
        echo     print^("torch_geometric:", torch_geometric.__version__^)
        echo     import torch_scatter
        echo     print^("torch_scatter  : ok"^)
        echo     import torch_sparse
        echo     print^("torch_sparse   : ok"^)
        echo     from rdkit import Chem
        echo     print^("rdkit          : ok"^)
        echo     import numpy, scipy, pandas, sklearn
        echo     import matplotlib, seaborn, yaml, tqdm
        echo     print^("all others     : ok"^)
        echo     print^(^)
        echo     print^("All imports succeeded."^)
        echo except ImportError as e:
        echo     print^("FAILED:", e^)
        echo     sys.exit^(1^)
    ) > "!CHECK_SCRIPT!"

    python "!CHECK_SCRIPT!"
    del "!CHECK_SCRIPT!"
)

echo.
echo Setup complete.
echo Activate with:  %VENV_DIR%\Scripts\activate.bat
echo Then verify:    setup.bat --check
echo.
endlocal
