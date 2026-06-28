# Manuscript Revision Results

Generated from the leakage-remediation repository results.

To regenerate:

```bash
export LEAKAGE_REMEDIATION_ROOT=/path/to/Leakage-Remediation
python generate_manuscript_revision_artifacts.py --outdir .
```

Python dependencies: `pandas`, `numpy`, `matplotlib`, and `lifelines`.

## Contents

- `tables/table3_classification_fold_internal_cv.md`
- `tables/table3_classification_fold_internal_cv_compact.csv`
- `tables/table3_classification_fold_internal_cv_detailed.csv`
- `tables/survival_primary_oos_multivariate.md`
- `tables/survival_primary_oos_multivariate.csv`
- `tables/survival_oos_multivariate_all.csv`
- `figures/fig5_classification_performance.png`
- `figures/fig5_classification_performance.pdf`
- `figures/fig7_km_oos_survival.png`
- `figures/fig7_km_oos_survival.pdf`

## Key Values

- H2C classification after strict fold-internal ComBat: RF AUC=0.981, F1=0.974; LogReg AUC=0.703, F1=0.948.
- H2C survival, out-of-sample Cox-beta with multivariate adjustment: HR/SD=1.226, Cox p=0.00266, C-index=0.767.
- H2C survival, out-of-sample PCA with multivariate adjustment: HR/SD=0.973, Cox p=0.684, C-index=0.727.

## Notes

- This folder contains only derived manuscript tables/figures and a generation script. It does not include raw expression matrices, patient-level score tables, model checkpoints, or controlled-access data.
- Source root used for this generation: `Leakage-Remediation`. See `source_manifest.json` for source file paths relative to `LEAKAGE_REMEDIATION_ROOT`.
- Classification source: `classification/results/leakage_free_combat/classification_summary_leakage_free_combat.csv`.
- Survival source: `survival/results/leakage_free/survival_priority1_results.csv` and `survival/results/leakage_free/survival_score_values.csv`.
- These outputs are figure/table artifacts only. They do not modify the manuscript text.
- The primary classification result is the strict batch-only, fold-internal ComBat run.
- Fig. 5 is RF-centered to preserve the manuscript's primary visual message; LogReg sensitivity is reported in Table III.
- Fig. 7 is a two-panel main-text figure focused on H2C vs Euclidean out-of-sample Cox-beta scoring; PCA and Random comparisons are retained in the survival table.
- The primary survival claim should use multivariate out-of-sample rows, not in-sample Cox-beta rows.
