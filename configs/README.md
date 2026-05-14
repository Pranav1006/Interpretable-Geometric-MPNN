# configs/

YAML configuration files for training, evaluation, and counterfactual generation runs.

## File Naming

```
{stage}_{variant}.yaml
```

Examples:
- `train_geom_mpnn_baseline.yaml`
- `train_geom_mpnn_no_torsion.yaml`
- `eval_esol.yaml`
- `counterfactual_gradient_search.yaml`

## Config Structure (Training Example)

```yaml
experiment_name: geom_mpnn_baseline
seed: 42

data:
  dataset: esol            # esol | qm9 | lipophilicity | bace
  raw_dir: data/raw/
  processed_dir: data/processed/
  splits_dir: data/splits/
  target_property: logS

model:
  hidden_dim: 128
  num_layers: 4
  use_angles: true
  use_torsions: true
  dropout: 0.1

training:
  epochs: 200
  batch_size: 32
  lr: 1.0e-3
  weight_decay: 1.0e-5
  patience: 20            # early stopping

logging:
  log_dir: logs/training/
  checkpoint_dir: experiments/
  log_every_n_epochs: 5
```
