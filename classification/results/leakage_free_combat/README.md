# Leakage Remediation: Classification

This folder contains the strict leakage-controlled classification results for the H2C/TDA biomarker project.

H2C is a 37-gene panel identified from the project's Topological Data Analysis (TDA) workflow. This audit checks whether the original tumor/normal classification result remains valid after removing feature-selection and preprocessing leakage from cross-validation.

## Scope

Included:

- fold-internal log-transform decisions;
- fold-internal ComBat fitting;
- held-out fold transformation using training-fold ComBat estimates;
- fold-internal TDA and Euclidean feature ranking;
- held-out tumor/normal classifier evaluation.

Not included in this folder:

- raw TCGA preprocessing files.

## Main Files

```text
README.md
classification_summary_leakage_free_combat.csv
classification_fold_metrics_combat.csv
classification_fold_features_combat.csv
classification_fold_audit_combat.csv
classification_audit_combat.json
fold_1_feature_ranks_combat.csv ... fold_5_feature_ranks_combat.csv
```

## Reproduction

Run from the `FindVar` repository root:

```bash
python phase4_biological_interpretation/leakage_free_classification_with_combat.py
```

The script expects the sibling `Data-preprocessing` repository to be available with the project preprocessing artifacts. Raw TCGA preprocessing inputs are required for this strict ComBat run but are intentionally not committed to GitHub.

## Main Result

| Feature set | Classifier | AUC mean | F1 mean | Accuracy mean |
| --- | ---: | ---: | ---: | ---: |
| Euclidean_top200_train | LogisticRegression | 0.998991 | 0.992740 | 0.986895 |
| Euclidean_top200_train | RandomForest | 0.998848 | 0.993233 | 0.987714 |
| TDA_top200_train | RandomForest | 0.998138 | 0.990543 | 0.982664 |
| TDA_top200_train | LogisticRegression | 0.981490 | 0.981966 | 0.967715 |
| H2C_train_rule_top37 | RandomForest | 0.981446 | 0.973847 | 0.951393 |
| H2C_train_rule_top37 | LogisticRegression | 0.702863 | 0.947700 | 0.900704 |

## Interpretation

H2C remains strong with RandomForest under strict leakage control, but it is classifier-sensitive and is not uniquely superior to Euclidean Top200.

The fold-internal ComBat run is batch-only by design. Held-out `Target` labels are unavailable in a valid prediction setup, so using them as ComBat covariates would leak labels.
