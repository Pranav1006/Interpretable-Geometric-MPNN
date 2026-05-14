# experiments/

Saved model checkpoints and the exact configuration used for each run.
Every experiment should be fully reproducible from what's stored here.

## Directory Convention

```
experiments/
└── {experiment_name}_{YYYY-MM-DD}/
    ├── config.yaml          # Full config snapshot (copy of configs/ file used)
    ├── checkpoints/
    │   ├── best.pt          # Best validation checkpoint
    │   └── last.pt          # Final epoch checkpoint
    └── notes.md             # Optional: short description of what this run was testing
```

## Naming Experiments
Use descriptive names that encode what's being varied:

```
geom_mpnn_baseline_2025-06-01
geom_mpnn_no_torsion_2025-06-03
cf_gen_gradient_esol_2025-06-10
```

## Notes
- Never delete old experiments — storage is cheap, reproducibility is not.
- If a run was a mistake or scratch test, note it in `notes.md` rather than deleting.
