# src/explanation/ — Explanation Module

## Purpose
Identifies which atoms, bonds, and geometric features most influence the model's
prediction for a given molecule. Produces interpretable attribution maps.

## Planned Components
- `explainer.py`        — Main explanation interface
- `gradient_methods.py` — Gradient-based attribution (e.g., Integrated Gradients)
- `attention_maps.py`   — Attention-weight extraction (if attention is used)
- `subgraph_masks.py`   — Discrete subgraph selection methods

## Design Notes
- Should produce per-atom and per-edge importance scores
- Output format must be compatible with the counterfactual generator's perturbation targets
- Visualizations output to `visualizations/explanations/`
