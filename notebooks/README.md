# notebooks/

Jupyter notebooks for exploration, prototyping, and generating paper figures.
Notebooks are **not** the source of truth for any method — finalised code lives in `src/`.

## Naming Convention

```
{number}_{descriptor}.ipynb
```

## Current Notebooks

| Notebook | Status | Purpose |
|---|---|---|
| `00_pipeline_quickstart.ipynb` | ✅ done | Six sequential checks: imports → dataset loads → graph shapes → split file → DataLoaders → geometry ranges. First thing to run after `preprocess.py`. |
| `01_data_exploration.ipynb` | ✅ done | Dataset-wide statistics: feature dims, target distributions, graph size histograms, geometry distributions, atom-type breakdown. |
| `01b_single_molecule_debug.ipynb` | ✅ done | Deep-dive into one molecule (`MOL_IDX`): per-atom features, per-edge bond types and lengths, angle/torsion tables, 3D scatter plot, adjacency matrix. |
| `02_model_inspection.ipynb` | planned | Layer outputs, embedding spaces, attention weights |
| `03_explanation_demo.ipynb` | planned | Interactive attribution maps for specific molecules |
| `04_counterfactual_demo.ipynb` | planned | Walk through a counterfactual generation example end-to-end |
| `05_evaluation_summary.ipynb` | planned | Reproduce paper tables and figures from saved results |

## Typical Workflow

```bash
# 1. Preprocess data (run once)
python scripts/preprocess.py --config configs/preprocess_qm9.yaml

# 2. Open JupyterLab from the repo root
jupyter lab

# 3. Start here
notebooks/00_pipeline_quickstart.ipynb   # verify everything works
notebooks/01_data_exploration.ipynb      # understand the data
notebooks/01b_single_molecule_debug.ipynb  # inspect individual molecules
```

## Notes
- Clear output cells before committing (keeps diffs readable).
- Notebooks depend on `src/` — always launch JupyterLab from the **repo root**.
- `MOL_IDX` in `01b` can be changed to any integer in `[0, len(ds))`.
