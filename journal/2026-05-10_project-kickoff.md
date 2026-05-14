# 2026-05-10 — Project Kickoff

## What I Did
Initialized the IG-MPNN repository. Established the full project structure for a
publishable paper: *Counterfactual Explanations for Geometric Molecular Graph Neural Networks*.

## Research Question
> What minimal structural or geometric changes to a molecule would significantly
> change a predicted molecular property?

## The Four Pillars
1. **Geometric MPNN** — 3D-aware message passing over molecular graphs
2. **Explanation Module** — Attribution over atoms, bonds, and geometric features
3. **Counterfactual Generator** — Minimally perturbed, chemically valid alternative molecules
4. **Evaluation Framework** — Validity, proximity, sparsity, diversity, flip rate

## Key Decisions
- PyTorch + PyTorch Geometric as the primary framework.
- 3D geometry encoded as invariant scalars (distances, angles, torsions) rather than raw coordinates.
- Chemical validity enforced via RDKit at every generation step.
- Counterfactuals, metrics, and failure modes tracked separately under `results/`.

## Open Questions
- [ ] Which molecular property datasets to prioritize? (QM9, ESOL, Lipophilicity, BACE?)
- [ ] Should geometry be learned end-to-end or use RDKit-generated conformers?
- [ ] What perturbation space for counterfactuals — discrete (bond/atom edits) or continuous (coordinate nudges)?
- [ ] Baselines to compare against?

## Next Steps
- [ ] Set up `requirements.txt` and confirm PyG install.
- [ ] Prototype data loading pipeline in `src/data/`.
- [ ] Sketch the MPNN layer design in `src/model/`.
