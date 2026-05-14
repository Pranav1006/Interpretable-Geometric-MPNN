# notebooks/

Jupyter notebooks for exploration, prototyping, and generating paper figures.
Notebooks are **not** the source of truth for any method — finalized code lives in `src/`.

## Naming Convention

```
{number}_{descriptor}.ipynb
```

| Notebook | Purpose |
|---|---|
| `01_data_exploration.ipynb` | Dataset statistics, property distributions, 3D geometry sanity checks |
| `02_model_inspection.ipynb` | Layer outputs, embedding spaces, attention weights |
| `03_explanation_demo.ipynb` | Interactive attribution maps for specific molecules |
| `04_counterfactual_demo.ipynb` | Walk through a counterfactual generation example end-to-end |
| `05_evaluation_summary.ipynb` | Reproduce paper tables and figures from saved results |

## Notes
- Clear output cells before committing (keeps diffs readable).
- Notebooks depend on the `src/` package — run from the repo root or add it to `sys.path`.
