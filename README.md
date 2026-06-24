# Leakage Remediation

This repository contains leakage-controlled re-evaluation code and result tables for the H2C/TDA biomarker project.

H2C is a 37-gene panel identified from a Topological Data Analysis (TDA) workflow on TCGA-BRCA RNA-seq data. The analyses here test whether the original classification and survival findings remain valid after removing common sources of validation leakage.

## Repository Structure

```text
classification/
  code/
    leakage_free_classification.py
    leakage_free_classification_with_combat.py
  results/
    leakage_free/
    leakage_free_combat/

survival/
  code/
    leakage_free_survival.py
  data/
    brca_survival_cohort.tsv
  results/
    leakage_free/
```

## Classification Re-Evaluation

The strict classification audit is in:

```text
classification/results/leakage_free_combat/
```

It evaluates tumor/normal classification after:

- fitting ComBat within each training fold;
- applying held-out fold transformation using only training-fold ComBat estimates;
- recomputing TDA and Euclidean feature rankings inside each training fold;
- evaluating each held-out fold once.

Main result:

| Feature set | Classifier | AUC mean | F1 mean | Accuracy mean |
| --- | ---: | ---: | ---: | ---: |
| Euclidean_top200_train | LogisticRegression | 0.998991 | 0.992740 | 0.986895 |
| Euclidean_top200_train | RandomForest | 0.998848 | 0.993233 | 0.987714 |
| TDA_top200_train | RandomForest | 0.998138 | 0.990543 | 0.982664 |
| H2C_train_rule_top37 | RandomForest | 0.981446 | 0.973847 | 0.951393 |
| H2C_train_rule_top37 | LogisticRegression | 0.702863 | 0.947700 | 0.900704 |

Interpretation: H2C remains strong with RandomForest under strict leakage control, but it is classifier-sensitive and is not uniquely superior to Euclidean Top200.

## Survival Re-Evaluation

The survival audit is in:

```text
survival/results/leakage_free/
```

It compares H2C, Euclidean, and Random gene sets under the same scoring and modeling rules. The primary rows are the out-of-sample multivariate results.

Main result:

| Gene set | Score method | HR per SD | Cox p | Log-rank p | C-index |
| --- | ---: | ---: | ---: | ---: | ---: |
| H2C_37 | pca | 0.973121 | 0.684033 | 0.921137 | 0.727071 |
| H2C_37 | cox_beta | 1.226483 | 0.002658 | 0.000000456 | 0.766799 |
| Euclidean_top37 | pca | 1.062423 | 0.394671 | 0.012743 | 0.725779 |
| Euclidean_top37 | cox_beta | 1.127145 | 0.089836 | 0.001228 | 0.727601 |
| Random_37 | pca | 0.964446 | 0.603271 | 0.222353 | 0.735156 |
| Random_37 | cox_beta | 1.048946 | 0.548799 | 0.001177 | 0.733135 |

Interpretation: H2C retains adjusted prognostic signal under supervised out-of-sample Cox-beta scoring. PCA-based survival signal is not supported.

## Reproduction Notes

The committed result tables are self-contained. Re-running the scripts requires the original project repositories and preprocessing artifacts to be available as sibling directories:

```text
Data-preprocessing/
FindVar/
FindVar-Survival-Analysis/
Leakage-Remediation/
```

Raw TCGA preprocessing files and large model/data artifacts are not committed to this repository.

## Data Availability

The survival re-evaluation uses TCGA-BRCA survival and expression data from UCSC Xena. The strict classification ComBat run requires the original TCGA preprocessing input files used by the H2C/TDA project; those raw files are intentionally excluded from GitHub.
