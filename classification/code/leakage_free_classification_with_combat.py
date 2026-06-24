"""Leakage-free classification with fold-internal ComBat.

This is the stricter leakage-remediation runner. It starts from the original
TCGA preprocessing inputs, then performs label-free preprocessing inside each
fold:

1. fixed BRCA/sample filtering and fixed post-blacklist/model gene universe
2. train-fold-only log-transform decision
3. train-fold-only ComBat fitting with batch only
4. train-fold-only TDA/Euclidean feature selection
5. held-out fold evaluation

The original preprocessing used Target as a ComBat covariate. That is not
available for held-out samples in a valid prediction pipeline, so this runner
uses batch-only ComBat fitted on the training fold and applies those estimates
to the test fold.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from neuroCombat import neuroCombat
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from leakage_free_classification import (
    display_path,
    load_tae_model,
    rank_fold_features,
    select_feature_sets,
    summarize,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = REPO_ROOT.parent
DATA_ROOT = WORKSPACE / "Data-preprocessing"
DEFAULT_TUMOR = DATA_ROOT / "GSE62944_RAW" / "GSM1536837_06_01_15_TCGA_24.tumor_Rsubread_TPM.txt.gz"
DEFAULT_NORMAL = DATA_ROOT / "GSE62944_RAW" / "GSM1697009_06_01_15_TCGA_24.normal_Rsubread_TPM.txt.gz"
DEFAULT_CLINICAL = DATA_ROOT / "GSE62944_06_01_15_TCGA_24_548_Clinical_Variables_9264_Samples.txt.gz"
DEFAULT_CLEANED = DATA_ROOT / "data_preprocessing" / "cleaned_tcga_tpm_for_TAE.csv"
DEFAULT_MODEL = DATA_ROOT / "TAE" / "models" / "tae_dim32_cosine.pth"
DEFAULT_OUTDIR = REPO_ROOT / "classification" / "results" / "leakage_free_combat"


BRCA_TSS = {
    "A1", "A2", "A7", "A8", "AC", "AN", "AO", "AQ", "AR", "B6", "BH",
    "C8", "D8", "E2", "E9", "EW", "GM", "GI", "HN", "LD", "LL", "MS",
    "OL", "PE", "PL", "S3", "UL", "UU", "WT", "XX", "Z7",
}


def is_brca_sample(sample: str) -> bool:
    parts = str(sample).split("-")
    return len(parts) > 1 and parts[1] in BRCA_TSS


def load_model_gene_universe(cleaned_path: Path) -> list[str]:
    cols = pd.read_csv(cleaned_path, nrows=0).columns.tolist()
    return [c for c in cols if c not in {"Target", "Unnamed: 0"}]


def read_brca_tpm(path: Path, target: int, model_genes: list[str]) -> pd.DataFrame:
    header = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
    gene_col = header[0]
    sample_cols = [c for c in header[1:] if is_brca_sample(c)]
    usecols = [gene_col] + sample_cols
    dtype = {c: np.float32 for c in sample_cols}
    df = pd.read_csv(path, sep="\t", usecols=usecols, index_col=0, dtype=dtype).T
    missing = [g for g in model_genes if g not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing {len(missing)} model genes; first={missing[:5]}")
    df = df[model_genes].copy()
    df["Target"] = int(target)
    return df


def load_raw_brca(tumor_path: Path, normal_path: Path, clinical_path: Path, model_genes: list[str]) -> pd.DataFrame:
    tumor = read_brca_tpm(tumor_path, target=1, model_genes=model_genes)
    normal = read_brca_tpm(normal_path, target=0, model_genes=model_genes)
    df = pd.concat([tumor, normal], axis=0)

    clin = pd.read_csv(clinical_path, sep="\t", index_col=0, low_memory=False).T.iloc[2:]
    patient_to_tss = {
        idx[:12]: str(idx).split("-")[1]
        for idx in clin.index
        if isinstance(idx, str) and "-" in idx
    }
    df["TSS_Code"] = df.index.astype(str).str[:12].map(patient_to_tss)
    missing = df["TSS_Code"].isna()
    if missing.any():
        df.loc[missing, "TSS_Code"] = df.index[missing].astype(str).str.split("-").str[1]
    return df


def filter_batch_safe_samples(df: pd.DataFrame, min_batch_count: int = 3) -> pd.DataFrame:
    counts = df["TSS_Code"].value_counts()
    keep_batches = counts[counts >= min_batch_count].index
    return df[df["TSS_Code"].isin(keep_batches)].copy()


def make_batch_safe_folds(df: pd.DataFrame, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    fold_bins: list[list[int]] = [[] for _ in range(n_splits)]
    indexed = df.reset_index(drop=False)

    for (_batch, _target), sub in indexed.groupby(["TSS_Code", "Target"], sort=False):
        idx = sub.index.to_numpy()
        rng.shuffle(idx)
        for offset, row_idx in enumerate(idx):
            fold_bins[offset % n_splits].append(int(row_idx))

    all_idx = np.arange(len(indexed))
    folds = []
    for test_list in fold_bins:
        test_idx = np.array(sorted(test_list), dtype=int)
        train_mask = np.ones(len(indexed), dtype=bool)
        train_mask[test_idx] = False
        train_idx = all_idx[train_mask]

        train_batches = indexed.iloc[train_idx]["TSS_Code"].value_counts()
        test_batches = set(indexed.iloc[test_idx]["TSS_Code"])
        unsafe = [b for b in test_batches if train_batches.get(b, 0) < 2]
        if unsafe:
            raise ValueError(f"fold has test batches with <2 train samples: {unsafe[:10]}")
        folds.append((train_idx, test_idx))
    return folds


def train_log_transform_columns(x_train: pd.DataFrame) -> list[str]:
    values = x_train.to_numpy(dtype=np.float32)
    mean = values.mean(axis=0)
    diff = values - mean
    n = values.shape[0]
    std = np.sqrt((diff ** 2).sum(axis=0) / n)
    std[std == 0] = 1.0
    skew = ((diff ** 3).sum(axis=0) / n) / (std ** 3)
    kurt = ((diff ** 4).sum(axis=0) / n) / (std ** 4) - 3.0
    mask = (np.abs(skew) > 2.0) | (kurt > 10.0)
    return list(x_train.columns[mask])


def apply_train_defined_log_transform(x_train: pd.DataFrame, x_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    cols = train_log_transform_columns(x_train)
    train_out = x_train.copy()
    test_out = x_test.copy()
    if cols:
        train_out[cols] = np.log1p(train_out[cols])
        test_out[cols] = np.log1p(test_out[cols])
    return train_out, test_out, len(cols)


def combat_train_apply(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    batch_train: pd.Series,
    batch_test: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    covars = pd.DataFrame({"batch": batch_train.astype(str).to_numpy()})
    train_dat = x_train.to_numpy(dtype=np.float64).T
    test_dat = x_test.to_numpy(dtype=np.float64).T
    with contextlib.redirect_stdout(io.StringIO()):
        fitted = neuroCombat(dat=train_dat, covars=covars, batch_col="batch")
        transformed_test = combat_from_training(test_dat, batch_test.astype(str).to_numpy(), fitted["estimates"])
    train_corr = pd.DataFrame(fitted["data"].T, index=x_train.index, columns=x_train.columns)
    test_corr = pd.DataFrame(transformed_test.T, index=x_test.index, columns=x_test.columns)
    return train_corr, test_corr


def combat_from_training(dat: np.ndarray, batch: np.ndarray, estimates: dict[str, np.ndarray]) -> np.ndarray:
    batch = np.asarray(batch, dtype=str)
    old_levels = np.asarray(estimates["batches"], dtype=str)
    missing = np.setdiff1d(np.unique(batch), old_levels)
    if len(missing):
        raise ValueError(f"held-out batches are absent from training ComBat estimates: {missing.tolist()}")
    batch_idx = np.array([np.where(old_levels == b)[0][0] for b in batch], dtype=int)

    var_pooled = np.asarray(estimates["var.pooled"], dtype=float)
    if var_pooled.ndim == 1:
        var_pooled = var_pooled.reshape(-1, 1)
    stand_mean = np.asarray(estimates["stand.mean"], dtype=float)[:, 0]
    mod_mean = np.asarray(estimates["mod.mean"], dtype=float)
    stand_mean = stand_mean + mod_mean.mean(axis=1)
    stand_mean = np.tile(stand_mean.reshape(-1, 1), (1, dat.shape[1]))

    gamma = np.asarray(estimates["gamma.star"], dtype=float)[batch_idx, :].T
    delta = np.asarray(estimates["delta.star"], dtype=float)[batch_idx, :].T

    standardized = (dat - stand_mean) / np.sqrt(var_pooled)
    adjusted = (standardized - gamma) / np.sqrt(delta)
    return adjusted * np.sqrt(var_pooled) + stand_mean


def model_input_dim(model: torch.nn.Module) -> int:
    first = model.encoder[0]
    return int(first.in_features)


def encode_latent(model: torch.nn.Module, x: pd.DataFrame, batch_size: int = 256) -> pd.DataFrame:
    input_dim = model_input_dim(model)
    if x.shape[1] > input_dim:
        raise ValueError(f"expression has {x.shape[1]} features, model accepts {input_dim}")
    zs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            arr = x.iloc[start:start + batch_size].to_numpy(dtype=np.float32)
            if arr.shape[1] < input_dim:
                pad = np.zeros((arr.shape[0], input_dim - arr.shape[1]), dtype=np.float32)
                arr = np.hstack([arr, pad])
            tensor = torch.from_numpy(arr)
            _, z = model(tensor)
            zs.append(z.cpu().numpy())
    z_all = np.vstack(zs)
    return pd.DataFrame(z_all, index=x.index, columns=[f"z{i}" for i in range(z_all.shape[1])])


def evaluate(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    pred = (prob >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, prob)),
        "f1": float(f1_score(y_true, pred)),
        "accuracy": float(accuracy_score(y_true, pred)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tumor", type=Path, default=DEFAULT_TUMOR)
    parser.add_argument("--normal", type=Path, default=DEFAULT_NORMAL)
    parser.add_argument("--clinical", type=Path, default=DEFAULT_CLINICAL)
    parser.add_argument("--cleaned", type=Path, default=DEFAULT_CLEANED)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    model_genes = load_model_gene_universe(args.cleaned)
    raw = load_raw_brca(args.tumor, args.normal, args.clinical, model_genes)
    raw = filter_batch_safe_samples(raw, min_batch_count=3)

    model = load_tae_model(args.model, device=torch.device("cpu"))
    model.cpu().eval()

    classifiers = {
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, solver="liblinear", random_state=args.seed)),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=args.seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
    }

    folds = make_batch_safe_folds(raw, n_splits=args.folds, seed=args.seed)
    x_all = raw[model_genes]
    y_all = raw["Target"].astype(int).to_numpy()

    metric_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    fold_audit: list[dict[str, object]] = []

    for fold, (train_idx, test_idx) in enumerate(folds, start=1):
        train_samples = raw.index[train_idx]
        test_samples = raw.index[test_idx]
        x_train = x_all.loc[train_samples]
        x_test = x_all.loc[test_samples]
        y_train = raw.loc[train_samples, "Target"].astype(int).to_numpy()
        y_test = raw.loc[test_samples, "Target"].astype(int).to_numpy()

        x_train_log, x_test_log, n_log_cols = apply_train_defined_log_transform(x_train, x_test)
        train_var = x_train_log.var(axis=0)
        zero_var = train_var[train_var <= 1e-12].index.tolist()
        combat_cols = [c for c in x_train_log.columns if c not in set(zero_var)]
        x_train_combat_part, x_test_combat_part = combat_train_apply(
            x_train_log[combat_cols],
            x_test_log[combat_cols],
            raw.loc[train_samples, "TSS_Code"],
            raw.loc[test_samples, "TSS_Code"],
        )
        x_train_combat = x_train_log.copy()
        x_test_combat = x_test_log.copy()
        x_train_combat = x_train_combat.astype(np.float64)
        x_test_combat = x_test_combat.astype(np.float64)
        x_train_combat.loc[:, combat_cols] = x_train_combat_part[combat_cols]
        x_test_combat.loc[:, combat_cols] = x_test_combat_part[combat_cols]

        z_train = encode_latent(model, x_train_combat)
        z_test = encode_latent(model, x_test_combat)
        df_rank = rank_fold_features(x_train_combat, z_train, y_train, model=model, device=torch.device("cpu"))
        df_rank.to_csv(args.outdir / f"fold_{fold}_feature_ranks_combat.csv", index=False)

        feature_sets = select_feature_sets(df_rank)
        feature_sets["Latent_32d_fold_combat"] = list(z_train.columns)

        fold_audit.append(
            {
                "fold": fold,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "train_tumor": int(y_train.sum()),
                "test_tumor": int(y_test.sum()),
                "n_log_transform_genes_train_rule": int(n_log_cols),
                "n_zero_variance_genes_excluded_from_combat": int(len(zero_var)),
                "zero_variance_genes_json": json.dumps(zero_var),
                "n_train_batches": int(raw.loc[train_samples, "TSS_Code"].nunique()),
                "n_test_batches": int(raw.loc[test_samples, "TSS_Code"].nunique()),
            }
        )

        for feature_set, features in feature_sets.items():
            if feature_set == "Latent_32d_fold_combat":
                xtr = z_train[features].to_numpy(dtype=np.float32)
                xte = z_test[features].to_numpy(dtype=np.float32)
            else:
                xtr = x_train_combat[features].to_numpy(dtype=np.float32)
                xte = x_test_combat[features].to_numpy(dtype=np.float32)

            feature_rows.append(
                {
                    "fold": fold,
                    "feature_set": feature_set,
                    "n_features": len(features),
                    "genes_json": json.dumps(features, ensure_ascii=False),
                }
            )

            for clf_name, clf in classifiers.items():
                estimator = clone(clf)
                estimator.fit(xtr, y_train)
                prob = estimator.predict_proba(xte)[:, 1]
                metric_rows.append(
                    {
                        "fold": fold,
                        "feature_set": feature_set,
                        "classifier": clf_name,
                        "n_features": len(features),
                        **evaluate(y_test, prob),
                    }
                )

    df_metrics = pd.DataFrame(metric_rows)
    df_features = pd.DataFrame(feature_rows)
    df_summary = summarize(df_metrics)
    df_fold_audit = pd.DataFrame(fold_audit)

    df_metrics.to_csv(args.outdir / "classification_fold_metrics_combat.csv", index=False)
    df_features.to_csv(args.outdir / "classification_fold_features_combat.csv", index=False)
    df_summary.to_csv(args.outdir / "classification_summary_leakage_free_combat.csv", index=False)
    df_fold_audit.to_csv(args.outdir / "classification_fold_audit_combat.csv", index=False)

    audit = {
        "scope": "leakage-free classification with fold-internal ComBat",
        "raw_tumor": display_path(args.tumor),
        "raw_normal": display_path(args.normal),
        "raw_clinical": display_path(args.clinical),
        "model_gene_universe": display_path(args.cleaned),
        "n_samples_after_batch_filter": int(len(raw)),
        "class_counts": {
            "normal": int((raw["Target"] == 0).sum()),
            "tumor": int((raw["Target"] == 1).sum()),
        },
        "n_genes": int(len(model_genes)),
        "tae_model_input_dim": int(model_input_dim(model)),
        "tae_input_padding_zeros": int(max(0, model_input_dim(model) - len(model_genes))),
        "n_folds": int(args.folds),
        "combat_mode": "train-fold-only neuroCombat, batch-only covariate; held-out fold transformed with neuroCombatFromTraining",
        "why_no_target_covariate": (
            "Target is unavailable for true held-out prediction samples. Using held-out Target as a ComBat covariate "
            "would itself leak labels, so fold-internal ComBat is batch-only here."
        ),
        "leakage_controls": [
            "Log-transform genes are selected from training fold distribution only.",
            "ComBat estimates are fit on training fold only.",
            "Held-out fold is transformed with training ComBat estimates only.",
            "TDA and Euclidean feature ranks are recomputed after fold-internal ComBat using training samples only.",
            "Classifiers are fit on training fold features and evaluated once on held-out fold samples.",
        ],
    }
    (args.outdir / "classification_audit_combat.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print("\nLeakage-free classification with fold-internal ComBat")
    print(df_summary.sort_values("auc_mean", ascending=False).to_string(index=False))
    print(f"\nWrote results to {args.outdir}")


if __name__ == "__main__":
    main()
