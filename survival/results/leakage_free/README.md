# Leakage Remediation: Survival

This folder contains the fair and out-of-sample survival re-evaluation results for the H2C/TDA biomarker project.

H2C is a 37-gene panel identified from the project's Topological Data Analysis (TDA) workflow. This audit checks whether H2C survival associations remain valid when H2C, Euclidean, and Random gene sets are scored and evaluated under the same rules.

## Scope

Included:

- H2C, Euclidean, and Random gene-set comparison;
- same score methods across all gene sets;
- in-sample and out-of-sample Cox-beta scoring;
- multivariate Cox adjustment with age, stage, and treatment covariates when analyzable.

Not included in this folder:

- pan-cancer sex/cancer-type covariates, because this evaluation is BRCA-only.

## Main Files

```text
README.md
survival_priority1_results.csv
survival_score_values.csv
survival_effective_gene_sets.csv
survival_missing_genes.csv
survival_audit.json
```

## Reproduction

Run from the `FindVar-Survival-Analysis` repository checkout:

```bash
python phase7_survival_analysis/leakage_free_survival.py
```

The run uses TCGA-BRCA survival and expression data from UCSC Xena and writes the result files in this folder. Raw TCGA preprocessing inputs are not required for the survival re-evaluation.

## Main Out-of-Sample Multivariate Result

| Gene set | Score method | HR per SD | Cox p | Log-rank p | C-index |
| --- | ---: | ---: | ---: | ---: | ---: |
| H2C_37 | pca | 0.973121 | 0.684033 | 0.921137 | 0.727071 |
| H2C_37 | cox_beta | 1.226483 | 0.002658 | 0.000000456 | 0.766799 |
| Euclidean_top37 | pca | 1.062423 | 0.394671 | 0.012743 | 0.725779 |
| Euclidean_top37 | cox_beta | 1.127145 | 0.089836 | 0.001228 | 0.727601 |
| Random_37 | pca | 0.964446 | 0.603271 | 0.222353 | 0.735156 |
| Random_37 | cox_beta | 1.048946 | 0.548799 | 0.001177 | 0.733135 |

## Interpretation

H2C retains adjusted prognostic signal under supervised out-of-sample Cox-beta scoring. PCA-based survival signal is not supported.

The in-sample Cox-beta rows remain in `survival_priority1_results.csv` as circular comparators only and should not be used as the primary claim.
