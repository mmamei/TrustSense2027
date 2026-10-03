# TrustSense 2027

Reproducible research repository for a six-page IEEE TrustSense 2027 paper on
reliability-aware multimodal fusion in mobile crowdsensing.

## Scope

The workshop paper studies trust at the **sensor-modality level**. It does not
introduce federated learning or persistent participant reputation; those are
reserved for a future journal extension.

The central question is whether per-sample reliability estimates can make
context recognition more robust when sensing modalities are naturally missing,
degraded, or corrupted.

## Dataset

The experiments use the public ExtraSensory feature-and-label release. Place
the official archive at:

`data/raw/ExtraSensory.per_uuid_features_labels.zip`

and extract its 60 per-user `.csv.gz` files under:

`data/raw/extrasensory/`

The raw data are intentionally excluded from Git.

## Installation

```bash
python -m pip install -e .
```

Install the optional development dependencies to run the tests:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Reproducing the experiments

After downloading and extracting ExtraSensory as described above, profile the
dataset and run the pilot experiments with:

```bash
python scripts/profile_dataset.py
python scripts/run_clean_pilot.py
python scripts/run_robust_pilot.py
```

The locked evaluation reported in the paper uses participant folds 1--4 as
independent test sets. Run each fold separately:

```bash
python scripts/run_experiment_fold.py --outer-fold 1 --locked
python scripts/run_experiment_fold.py --outer-fold 2 --locked
python scripts/run_experiment_fold.py --outer-fold 3 --locked
python scripts/run_experiment_fold.py --outer-fold 4 --locked
python scripts/summarize_locked_results.py
python scripts/plot_locked_results.py
```

### Figures 3 and 4: severity and unimodal experiments

The severity curves in Figure 3 and the unimodal baselines in Figure 4 require
the extended evaluation. Run it independently on each reported test fold, then
aggregate the results and create the figures:

```bash
python scripts/run_extended_evaluation.py --outer-fold 1
python scripts/run_extended_evaluation.py --outer-fold 2
python scripts/run_extended_evaluation.py --outer-fold 3
python scripts/run_extended_evaluation.py --outer-fold 4
python scripts/summarize_extended_results.py
python scripts/plot_extended_results.py
```

The plotting script writes both PDF and PNG versions:

- Figure 3: `paper/figures/severity_curves.pdf` and `.png`
- Figure 4: `paper/figures/unimodal_baselines.pdf` and `.png`

It also produces the supplementary `ablation_unimodal` and
`ablation_components` figures in the same directory.

Intermediate summaries and metrics are written below `results/`; publication
figures are written below `paper/figures/`. Both generated directories are
intentionally excluded from Git.

## Implemented methods

- Single-modality classifiers
- Availability-aware uniform late fusion
- Reliability-aware late fusion based on feature completeness, robust
  distributional familiarity, and cross-modal agreement

Evaluation is grouped by participant to prevent user leakage.
