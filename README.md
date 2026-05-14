# IG-MPNN: Interpretable Geometric Message Passing Neural Network

> **Paper:** *Counterfactual Explanations for Geometric Molecular Graph Neural Networks*

## Overview

IG-MPNN addresses the question:

> *What minimal structural or geometric changes to a molecule would significantly change a predicted molecular property?*

The framework is built on four pillars:

| Pillar | Description |
|---|---|
| **Geometric MPNN** | A 3D-aware message passing network that encodes bond lengths, angles, and torsions as geometric features |
| **Explanation Module** | Identifies which atoms, bonds, or geometric features most influence a prediction |
| **Counterfactual Generator** | Produces chemically valid, minimally perturbed molecules that cross a decision boundary |
| **Evaluation Framework** | Quantifies counterfactual quality: validity, proximity, sparsity, and diversity |

## Repository Structure

```
IG-MPNN/
├── src/                    # Core source code (four pillars)
│   ├── model/              # Geometric MPNN architecture
│   ├── explanation/        # Explanation module
│   ├── counterfactual/     # Counterfactual generator
│   ├── evaluation/         # Evaluation framework
│   ├── data/               # Dataset loading and featurization
│   └── utils/              # Shared utilities
│
├── configs/                # YAML configs for experiments
├── scripts/                # Training, evaluation, and generation scripts
├── notebooks/              # Exploratory analysis and demos
├── tests/                  # Unit and integration tests
│
├── data/                   # Molecular datasets
│   ├── raw/                # Downloaded datasets (QM9, ESOL, etc.)
│   ├── processed/          # Featurized PyTorch Geometric graphs
│   └── splits/             # Train/val/test split indices
│
├── experiments/            # Saved checkpoints and run configs
├── results/                # Output artifacts
│   ├── counterfactuals/    # Generated counterfactual molecules
│   ├── metrics/            # Evaluation scores and tables
│   └── failure_modes/      # Logged failure cases for analysis
│
├── visualizations/         # Figures and plots
│   ├── molecules/          # Original molecule renders
│   ├── counterfactuals/    # Counterfactual pair comparisons
│   ├── explanations/       # Attribution/saliency maps
│   └── training/           # Loss curves, learning dynamics
│
├── logs/                   # Runtime logs
│   ├── training/
│   ├── evaluation/
│   ├── counterfactual/
│   └── errors/
│
├── journal/                # Research journal — decisions, hypotheses, findings
├── paper/                  # Manuscript
│   ├── drafts/
│   └── figures/            # Publication-ready figures
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
git clone https://github.com/mucha/IG-MPNN.git
cd IG-MPNN
pip install -r requirements.txt
```

## Built With

- [PyTorch](https://pytorch.org/)
- [PyTorch Geometric](https://pyg.org/)
- [RDKit](https://www.rdkit.org/)

## Status

Active
