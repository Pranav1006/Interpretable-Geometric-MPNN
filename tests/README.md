# tests/

Unit and integration tests for all four pillars.

## Structure

```
tests/
├── test_model.py           # Geometric MPNN forward pass, output shapes
├── test_explanation.py     # Attribution scores sum correctly, gradients flow
├── test_counterfactual.py  # Validity checks, proximity constraints
├── test_evaluation.py      # Metric computations on known examples
├── test_data.py            # Featurization, graph construction, splits
└── conftest.py             # Shared fixtures (small toy molecules, mock model)
```

## Running Tests

```bash
pytest tests/
```

## Guidelines
- Each module in `src/` should have corresponding tests here.
- Use small toy molecules (e.g., ethanol, benzene) as fixtures — not full datasets.
- Test edge cases: disconnected graphs, single-atom molecules, missing 3D coords.
