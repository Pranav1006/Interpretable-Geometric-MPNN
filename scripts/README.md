# scripts/

Entry-point scripts for running each stage of the pipeline.

## Scripts

| Script | Description |
|---|---|
| `train.py` | Train the Geometric MPNN |
| `evaluate.py` | Run the evaluation framework on a trained model |
| `generate_counterfactuals.py` | Run counterfactual generation on a test set |
| `explain.py` | Run the explanation module and save attribution maps |
| `preprocess.py` | Download and featurize raw datasets |

## Usage

```bash
# Preprocess data
python scripts/preprocess.py --config configs/train_geom_mpnn_baseline.yaml

# Train
python scripts/train.py --config configs/train_geom_mpnn_baseline.yaml

# Evaluate
python scripts/evaluate.py --config configs/eval_esol.yaml --checkpoint experiments/geom_mpnn_baseline_2025-06-01/checkpoints/best.pt

# Generate counterfactuals
python scripts/generate_counterfactuals.py --config configs/counterfactual_gradient_search.yaml --checkpoint experiments/geom_mpnn_baseline_2025-06-01/checkpoints/best.pt

# Explain
python scripts/explain.py --config configs/eval_esol.yaml --checkpoint experiments/geom_mpnn_baseline_2025-06-01/checkpoints/best.pt
```
