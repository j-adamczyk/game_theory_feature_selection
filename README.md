# Feature selection in chemoinformatics

A small project regarding usefulness of feature selection methods
for QSAR/QSPR. Descriptors used are known to contain many general
features that may be irrelevant to a particular task.

The pipeline tested is:
1. Molecular graph (from SMILES)
2. Descriptors, RDKit or Mordred
3. Feature selection
4. LightGBM

Project uses `uv` for setup and installing things. To reproduce
experiments, run `src/main.py`.

**Hypothesis:**
Removing the weakest features should improve the performance of
molecular property prediction models. In particular, methods based
on game theory, such as SHAP and SAGE, should be particularly robust.

**Descriptors:**
- RDKit 2D (>200 features)
- Mordred 2D (>1600 features)

**Feature selection methods:**
- none (baseline)
- correlation (threshold 0.9)
- filter methods
  - F-test
  - mutual information
- embedded methods:
  - LASSO
- wrapper methods:
  - Boruta
  - HSIC with LASSO
  - permutation feature importance
  - RFECV
- game theory based:
  - SHAP
  - SAGE
  - Shapley effects
  - sign-only SAGE
  - sign-only Shapley effects
  - missingness-aware SAGE (with LightGBM NaN handling)
  - missingness-aware Shapley effects (with LightGBM NaN handling)

Unless method can automatically derive the threshold to keep
the features, 20% of weakest ones were removed. Constant features
were always removed prior to any feature selection.

**Datasets and benchmarks:**
- MoleculeNet (classification & regression)
- TDC (classification & regression)
- MoleculeACE (regression)
- OpenADMET-ASAP Discovery (regression)
- OpenADMET-ExpansionRx (regression)

**Results:**
- CSVs are in `results_mordred` and `results_rdkit_desc` directories
- plots are in `plots` directory
- hypothesis was proven false, feature selection did not help
- LightGBM may be robust enough out-of-the-box
- methods based on game theory took very long time, particularly
  SAGE and Shapley effects, too long to be applicable in practice
