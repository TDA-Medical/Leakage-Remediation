#!/usr/bin/env python3
"""Generate manuscript-ready tables and replacement figures from remediation results."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter


SOURCE_ROOT_ENV = "LEAKAGE_REMEDIATION_ROOT"

CLASSIFICATION_LABELS = {
    "Euclidean_top200_train": "Euclidean Top200",
    "TDA_top200_train": "TDA Top200",
    "H2C_train_rule_top37": "H2C (candidate, 37)",
    "Latent_32d_fold_combat": "Latent 32d",
}

CLASSIFIER_LABELS = {
    "RandomForest": "RF",
    "LogisticRegression": "LogReg",
}

SURVIVAL_LABELS = {
    "H2C_37": "H2C (37)",
    "Euclidean_top37": "Euclidean Top37",
    "Random_37": "Random 37",
}

SCORE_LABELS = {
    "pca": "PCA (OOS)",
    "cox_beta": "Cox-beta (OOS)",
}


def fmt_decimal(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.{digits}f}"


def fmt_p(value: float) -> str:
    if pd.isna(value):
        return ""
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines) + "\n"


def load_classification(source_root: Path) -> pd.DataFrame:
    path = (
        source_root
        / "classification"
        / "results"
        / "leakage_free_combat"
        / "classification_summary_leakage_free_combat.csv"
    )
    df = pd.read_csv(path)
    df["Feature set"] = df["feature_set"].map(CLASSIFICATION_LABELS)
    df["Classifier"] = df["classifier"].map(CLASSIFIER_LABELS)
    return df


def make_classification_tables(df: pd.DataFrame, tables_dir: Path) -> None:
    order = [
        ("Euclidean_top200_train", "RandomForest"),
        ("Euclidean_top200_train", "LogisticRegression"),
        ("TDA_top200_train", "RandomForest"),
        ("TDA_top200_train", "LogisticRegression"),
        ("H2C_train_rule_top37", "RandomForest"),
        ("H2C_train_rule_top37", "LogisticRegression"),
        ("Latent_32d_fold_combat", "LogisticRegression"),
        ("Latent_32d_fold_combat", "RandomForest"),
    ]
    keyed = df.set_index(["feature_set", "classifier"])
    detailed_rows = []
    for feature, classifier in order:
        row = keyed.loc[(feature, classifier)]
        detailed_rows.append(
            {
                "Feature set": row["Feature set"],
                "Classifier": row["Classifier"],
                "AUC mean": row["auc_mean"],
                "AUC SD": row["auc_std"],
                "F1 mean": row["f1_mean"],
                "F1 SD": row["f1_std"],
                "Accuracy mean": row["accuracy_mean"],
                "Accuracy SD": row["accuracy_std"],
                "n folds": int(row["n_folds"]),
                "n features": int(row["n_features_mean"]),
            }
        )
    detailed = pd.DataFrame(detailed_rows)
    detailed.to_csv(tables_dir / "table3_classification_fold_internal_cv_detailed.csv", index=False)

    compact_rows = []
    compact_specs = [
        ("Euclidean Top200", [("RandomForest", "RF"), ("LogisticRegression", "LogReg")]),
        ("TDA Top200", [("RandomForest", "RF"), ("LogisticRegression", "LogReg")]),
        ("H2C (candidate, 37)", [("RandomForest", "RF"), ("LogisticRegression", "LogReg")]),
        ("Latent 32d", [("LogisticRegression", "LogReg"), ("RandomForest", "RF")]),
    ]
    reverse_labels = {v: k for k, v in CLASSIFICATION_LABELS.items()}
    for label, classifiers in compact_specs:
        feature_key = reverse_labels[label]
        auc_values = []
        f1_values = []
        classifier_values = []
        for classifier_key, classifier_label in classifiers:
            row = keyed.loc[(feature_key, classifier_key)]
            classifier_values.append(classifier_label)
            auc_values.append(fmt_decimal(row["auc_mean"], 3))
            f1_values.append(fmt_decimal(row["f1_mean"], 3))
        compact_rows.append(
            {
                "Feature set": label,
                "Classifier": " / ".join(classifier_values),
                "AUC": " / ".join(auc_values),
                "F1": " / ".join(f1_values),
            }
        )
    compact = pd.DataFrame(compact_rows)
    compact.to_csv(tables_dir / "table3_classification_fold_internal_cv_compact.csv", index=False)
    (tables_dir / "table3_classification_fold_internal_cv.md").write_text(
        markdown_table(compact),
        encoding="utf-8",
    )


def plot_classification(df: pd.DataFrame, figures_dir: Path) -> None:
    plot_order = [
        "Euclidean_top200_train",
        "TDA_top200_train",
        "H2C_train_rule_top37",
        "Latent_32d_fold_combat",
    ]
    set_colors = {
        "Euclidean_top200_train": "#1F77B4",
        "TDA_top200_train": "#D62728",
        "H2C_train_rule_top37": "#2CA02C",
        "Latent_32d_fold_combat": "#7F8C8D",
    }
    keyed = df.set_index(["feature_set", "classifier"])

    fig, (ax_bar, ax_scatter) = plt.subplots(
        1,
        2,
        figsize=(9.2, 3.4),
        gridspec_kw={"width_ratios": [1.1, 1.0]},
        constrained_layout=True,
    )

    labels = []
    values = []
    errors = []
    colors = []
    for feature in plot_order:
        row = keyed.loc[(feature, "RandomForest")]
        labels.append(CLASSIFICATION_LABELS[feature])
        values.append(float(row["auc_mean"]))
        errors.append(float(row["auc_std"]))
        colors.append(set_colors[feature])

    y = np.arange(len(labels))
    ax_bar.barh(
        y,
        values,
        xerr=errors,
        color=colors,
        edgecolor="black",
        linewidth=0.4,
        error_kw={"elinewidth": 0.7, "capsize": 2, "capthick": 0.7},
    )
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(labels, fontsize=7)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0.80, 1.01)
    ax_bar.set_xlabel("AUC (5-fold CV)")
    ax_bar.set_title("(A) Classification: AUC\n(RandomForest, fold-internal CV)", fontsize=9, fontweight="bold")
    ax_bar.grid(axis="x", color="#D9D9D9", linewidth=0.6)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    for yi, value in zip(y, values):
        text_x = min(value + 0.006, 0.995)
        ha = "left" if value < 0.985 else "right"
        ax_bar.text(text_x, yi, f"{value:.3f}", va="center", ha=ha, fontsize=7)

    label_offsets = {
        "Euclidean_top200_train": (8, 12),
        "TDA_top200_train": (8, -12),
        "H2C_train_rule_top37": (8, 10),
        "Latent_32d_fold_combat": (8, 10),
    }
    markers = {
        "Euclidean_top200_train": "s",
        "TDA_top200_train": "o",
        "H2C_train_rule_top37": "D",
        "Latent_32d_fold_combat": "^",
    }
    for feature in plot_order:
        row = keyed.loc[(feature, "RandomForest")]
        x_value = float(row["n_features_mean"])
        y_value = float(row["auc_mean"])
        ax_scatter.scatter(
            x_value,
            y_value,
            s=44,
            marker=markers[feature],
            color=set_colors[feature],
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        short_feature = {
            "Euclidean_top200_train": "Euclidean",
            "TDA_top200_train": "TDA",
            "H2C_train_rule_top37": "H2C",
            "Latent_32d_fold_combat": "Latent",
        }[feature]
        xytext = label_offsets[feature]
        ax_scatter.annotate(
            short_feature,
            (x_value, y_value),
            xytext=xytext,
            textcoords="offset points",
            fontsize=6,
            ha="right" if xytext[0] < 0 else "left",
            va="center",
        )
    ax_scatter.set_xlim(20, 232)
    ax_scatter.set_ylim(0.80, 1.015)
    ax_scatter.set_xlabel("Number of Features")
    ax_scatter.set_ylabel("AUC (RandomForest, 5-fold CV)")
    ax_scatter.set_title("(B) Features vs Performance", fontsize=9, fontweight="bold")
    ax_scatter.grid(color="#D9D9D9", linewidth=0.6)
    ax_scatter.spines["top"].set_visible(False)
    ax_scatter.spines["right"].set_visible(False)

    fig.savefig(figures_dir / "fig5_classification_performance.png", dpi=300)
    fig.savefig(figures_dir / "fig5_classification_performance.pdf")
    plt.close(fig)


def load_survival(source_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result_path = source_root / "survival" / "results" / "leakage_free" / "survival_priority1_results.csv"
    score_path = source_root / "survival" / "results" / "leakage_free" / "survival_score_values.csv"
    cohort_path = source_root / "survival" / "data" / "brca_survival_cohort.tsv"
    return pd.read_csv(result_path), pd.read_csv(score_path), pd.read_csv(cohort_path, sep="\t")


def make_survival_tables(results: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    oos = results[
        (results["endpoint"] == "OS")
        & (results["model"] == "multivariate")
        & (results["score_fit_scope"] == "out_of_sample")
    ].copy()
    oos["Gene set"] = oos["gene_set"].map(SURVIVAL_LABELS)
    oos["Score"] = oos["score_method"].map(SCORE_LABELS)
    oos_table = oos[
        [
            "Gene set",
            "Score",
            "n",
            "events",
            "hr_per_sd",
            "ci_lower",
            "ci_upper",
            "cox_p",
            "logrank_p",
            "c_index",
        ]
    ].rename(
        columns={
            "n": "n",
            "events": "events",
            "hr_per_sd": "HR per SD",
            "ci_lower": "95% CI lower",
            "ci_upper": "95% CI upper",
            "cox_p": "Cox p",
            "logrank_p": "Log-rank p",
            "c_index": "C-index",
        }
    )
    oos_table.to_csv(tables_dir / "survival_oos_multivariate_all.csv", index=False)

    primary_specs = [
        ("H2C_37", "cox_beta"),
        ("H2C_37", "pca"),
        ("Euclidean_top37", "cox_beta"),
        ("Random_37", "cox_beta"),
    ]
    keyed = oos.set_index(["gene_set", "score_method"])
    primary_rows = []
    for gene_set, score_method in primary_specs:
        row = keyed.loc[(gene_set, score_method)]
        primary_rows.append(
            {
                "Gene set": SURVIVAL_LABELS[gene_set],
                "Score": SCORE_LABELS[score_method],
                "HR/SD": fmt_decimal(row["hr_per_sd"], 3),
                "Cox p": fmt_p(row["cox_p"]),
                "C-index": fmt_decimal(row["c_index"], 3),
            }
        )
    primary = pd.DataFrame(primary_rows)
    primary.to_csv(tables_dir / "survival_primary_oos_multivariate.csv", index=False)
    (tables_dir / "survival_primary_oos_multivariate.md").write_text(
        markdown_table(primary),
        encoding="utf-8",
    )
    return oos


def plot_km_panel(
    ax: plt.Axes,
    merged: pd.DataFrame,
    score_col: str,
    title: str,
    result_row: pd.Series,
) -> None:
    data = merged[["OS.time", "OS", score_col]].dropna().copy()
    median = data[score_col].median()
    data["risk_group"] = np.where(data[score_col] >= median, "High score", "Low score")

    kmf = KaplanMeierFitter()
    palette = {"Low score": "#117733", "High score": "#882255"}
    for group in ["Low score", "High score"]:
        subset = data[data["risk_group"] == group]
        n_events = int(subset["OS"].sum())
        label = f"{group} (n={len(subset)}, events={n_events})"
        kmf.fit(subset["OS.time"], event_observed=subset["OS"], label=label)
        kmf.plot_survival_function(
            ax=ax,
            ci_show=False,
            color=palette[group],
            linewidth=1.8,
        )

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Overall survival")
    ax.set_ylim(0.0, 1.04)
    ax.grid(color="#E0E0E0", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    note = (
        f"log-rank p={fmt_p(result_row['logrank_p'])}\n"
        f"adjusted Cox p={fmt_p(result_row['cox_p'])}, HR={result_row['hr_per_sd']:.2f}"
    )
    ax.text(
        0.98,
        0.08,
        note,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#BDBDBD", "lw": 0.5},
    )


def plot_survival(results: pd.DataFrame, scores: pd.DataFrame, cohort: pd.DataFrame, figures_dir: Path) -> None:
    merged = cohort.merge(scores, on="sample_xena", how="inner")
    merged["OS"] = pd.to_numeric(merged["OS"], errors="coerce")
    merged["OS.time"] = pd.to_numeric(merged["OS.time"], errors="coerce")

    oos_multi = results[
        (results["endpoint"] == "OS")
        & (results["model"] == "multivariate")
        & (results["score_fit_scope"] == "out_of_sample")
    ].set_index(["gene_set", "score_method"])

    panels = [
        ("H2C_37", "cox_beta", "H2C: Cox-beta score"),
        ("Euclidean_top37", "cox_beta", "Euclidean Top37: Cox-beta score"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5), constrained_layout=True)
    for ax, (gene_set, score_method, title) in zip(axes.flat, panels):
        score_col = f"{gene_set}__{score_method}__out_of_sample"
        plot_km_panel(ax, merged, score_col, title, oos_multi.loc[(gene_set, score_method)])

    fig.suptitle("Out-of-sample Kaplan-Meier survival curves (TCGA-BRCA)", fontsize=11)
    fig.savefig(figures_dir / "fig7_km_oos_survival.png", dpi=300)
    fig.savefig(figures_dir / "fig7_km_oos_survival.pdf")
    plt.close(fig)


def write_readme(
    outdir: Path,
    source_root: Path,
    classification_df: pd.DataFrame,
    survival_oos: pd.DataFrame,
) -> None:
    h2c_rf = classification_df[
        (classification_df["feature_set"] == "H2C_train_rule_top37")
        & (classification_df["classifier"] == "RandomForest")
    ].iloc[0]
    h2c_lr = classification_df[
        (classification_df["feature_set"] == "H2C_train_rule_top37")
        & (classification_df["classifier"] == "LogisticRegression")
    ].iloc[0]
    h2c_cox = survival_oos[
        (survival_oos["gene_set"] == "H2C_37")
        & (survival_oos["score_method"] == "cox_beta")
    ].iloc[0]
    h2c_pca = survival_oos[
        (survival_oos["gene_set"] == "H2C_37")
        & (survival_oos["score_method"] == "pca")
    ].iloc[0]

    text = f"""# Manuscript Revision Results

Generated from the leakage-remediation repository results.

To regenerate:

```bash
export {SOURCE_ROOT_ENV}=/path/to/Leakage-Remediation
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

- H2C classification after strict fold-internal ComBat: RF AUC={h2c_rf['auc_mean']:.3f}, F1={h2c_rf['f1_mean']:.3f}; LogReg AUC={h2c_lr['auc_mean']:.3f}, F1={h2c_lr['f1_mean']:.3f}.
- H2C survival, out-of-sample Cox-beta with multivariate adjustment: HR/SD={h2c_cox['hr_per_sd']:.3f}, Cox p={h2c_cox['cox_p']:.3g}, C-index={h2c_cox['c_index']:.3f}.
- H2C survival, out-of-sample PCA with multivariate adjustment: HR/SD={h2c_pca['hr_per_sd']:.3f}, Cox p={h2c_pca['cox_p']:.3g}, C-index={h2c_pca['c_index']:.3f}.

## Notes

- This folder contains only derived manuscript tables/figures and a generation script. It does not include raw expression matrices, patient-level score tables, model checkpoints, or controlled-access data.
- Source root used for this generation: `{source_root.name}`. See `source_manifest.json` for source file paths relative to `{SOURCE_ROOT_ENV}`.
- Classification source: `classification/results/leakage_free_combat/classification_summary_leakage_free_combat.csv`.
- Survival source: `survival/results/leakage_free/survival_priority1_results.csv` and `survival/results/leakage_free/survival_score_values.csv`.
- These outputs are figure/table artifacts only. They do not modify the manuscript text.
- The primary classification result is the strict batch-only, fold-internal ComBat run.
- Fig. 5 is RF-centered to preserve the manuscript's primary visual message; LogReg sensitivity is reported in Table III.
- Fig. 7 is a two-panel main-text figure focused on H2C vs Euclidean out-of-sample Cox-beta scoring; PCA and Random comparisons are retained in the survival table.
- The primary survival claim should use multivariate out-of-sample rows, not in-sample Cox-beta rows.
"""
    (outdir / "README.md").write_text(text, encoding="utf-8")


def write_manifest(outdir: Path, source_root: Path) -> None:
    manifest = {
        "source_root_env": SOURCE_ROOT_ENV,
        "source_root_note": f"Set {SOURCE_ROOT_ENV} to the local Leakage-Remediation repository root.",
        "classification_summary": "classification/results/leakage_free_combat/classification_summary_leakage_free_combat.csv",
        "classification_fold_metrics": "classification/results/leakage_free_combat/classification_fold_metrics_combat.csv",
        "survival_results": "survival/results/leakage_free/survival_priority1_results.csv",
        "survival_scores": "survival/results/leakage_free/survival_score_values.csv",
        "survival_cohort": "survival/data/brca_survival_cohort.tsv",
    }
    (outdir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help=f"Leakage-Remediation repository root. Defaults to ${SOURCE_ROOT_ENV}.",
    )
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    source_root = args.source_root
    if source_root is None:
        env_value = os.environ.get(SOURCE_ROOT_ENV)
        if not env_value:
            parser.error(f"--source-root is required unless ${SOURCE_ROOT_ENV} is set")
        source_root = Path(env_value)

    outdir = args.outdir
    tables_dir = outdir / "tables"
    figures_dir = outdir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    classification = load_classification(source_root)
    make_classification_tables(classification, tables_dir)
    plot_classification(classification, figures_dir)

    survival_results, survival_scores, survival_cohort = load_survival(source_root)
    survival_oos = make_survival_tables(survival_results, tables_dir)
    plot_survival(survival_results, survival_scores, survival_cohort, figures_dir)

    write_readme(outdir, source_root, classification, survival_oos)
    write_manifest(outdir, source_root)

    print(f"Wrote manuscript revision artifacts to {outdir}")


if __name__ == "__main__":
    main()
