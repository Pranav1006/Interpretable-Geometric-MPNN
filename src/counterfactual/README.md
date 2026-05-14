# src/counterfactual/ — Counterfactual Generator

## Purpose
Generates chemically valid molecules that are minimally different from an input molecule
but cross a prediction boundary (e.g., property flips from active to inactive).

## Planned Components
- `generator.py`        — Main counterfactual generation interface
- `perturbation.py`     — Geometric and structural perturbation strategies
- `validity.py`         — Chemical validity checks (valence, SMILES roundtrip, RDKit)
- `search.py`           — Search/optimization loop (gradient, beam, or evolutionary)

## Design Notes
- Minimize: prediction distance to boundary + structural distance to original
- Constraint: chemical validity must be enforced at each step
- Generated counterfactuals saved to `results/counterfactuals/`
- Failures and edge cases logged to `results/failure_modes/`
