# src/evaluation/ — Evaluation Framework

## Purpose
Quantifies the quality of generated counterfactuals across multiple axes.

## Planned Components
- `evaluator.py`     — Main evaluation runner
- `metrics.py`       — Core metric implementations
- `benchmarks.py`    — Baseline comparisons
- `report.py`        — Summary table and figure generation

## Metrics
| Metric | Description |
|---|---|
| **Validity** | % of counterfactuals that are chemically valid |
| **Proximity** | Structural distance to the original molecule (e.g., graph edit distance, RMSD) |
| **Sparsity** | Number of atoms/bonds changed |
| **Diversity** | Variety across counterfactuals for the same input |
| **Prediction flip rate** | % that successfully cross the decision boundary |

## Design Notes
- Metrics output to `results/metrics/`
- Evaluation logs go to `logs/evaluation/`
