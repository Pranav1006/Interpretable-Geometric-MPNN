# scripts/

Entry-point scripts for running each stage of the pipeline.

## Scripts

| Script | Status | Description |
|---|---|---|
| `preprocess.py` | ✅ done | Download QM9, apply geometry transforms, generate 80/20 split |
| `clear_qm9_cache.py` | ✅ done | Delete raw/processed QM9 files to force a clean re-download |
| `train.py` | planned | Train the Geometric MPNN |
| `evaluate.py` | planned | Run the evaluation framework on a trained model |
| `generate_counterfactuals.py` | planned | Run counterfactual generation on a test set |
| `explain.py` | planned | Run the explanation module and save attribution maps |

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
