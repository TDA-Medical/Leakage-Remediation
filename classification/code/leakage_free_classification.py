"""Leakage-free classification re-evaluation for H2C.

This script moves label-dependent feature selection inside the
cross-validation training fold. The default run uses the locally available
expression matrix, which is already ComBat-corrected by the original
preprocessing pipeline.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = REPO_ROOT.parent
FINDVAR_ROOT = WORKSPACE / "FindVar"
DEFAULT_EXPR = WORKSPACE / "Data-preprocessing" / "data_preprocessing" / "cleaned_tcga_tpm_for_TAE.csv"
DEFAULT_LATENT = WORKSPACE / "Data-preprocessing" / "TAE" / "results" / "woutSMOTE" / "latent_32d_cosine.csv"
DEFAULT_MODEL = WORKSPACE / "Data-preprocessing" / "TAE" / "models" / "tae_dim32_cosine.pth"
DEFAULT_ORIGINAL_GENE_IMPORTANCE = FINDVAR_ROOT / "phase3_gene_traceback" / "results" / "gene_importance_full.csv"
DEFAULT_ORIGINAL_RESULTS = FINDVAR_ROOT / "phase4_biological_interpretation" / "results" / "classification_results.csv"
DEFAULT_OUTDIR = REPO_ROOT / "classification" / "results" / "leakage_free"


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(WORKSPACE))
    except ValueError:
        return str(path)


class TopologicalAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, input_dim), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.decoder(z), z


@dataclass(frozen=True)
class FoldFeatures:
    fold: int
    feature_set: str
    genes: list[str]


def load_tae_model(path: Path, device: torch.device) -> TopologicalAutoencoder:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    input_dim = int(state["decoder.6.bias"].shape[0])
    model = TopologicalAutoencoder(input_dim=input_dim, latent_dim=32)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def cohen_d_by_column(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x0 = x[y == 0]
    x1 = x[y == 1]
    mean0 = np.nanmean(x0, axis=0)
    mean1 = np.nanmean(x1, axis=0)
    var0 = np.nanvar(x0, axis=0)
    var1 = np.nanvar(x1, axis=0)
    pooled = np.sqrt((var0 * len(x0) + var1 * len(x1)) / max(len(x0) + len(x1), 1))
    return np.abs(mean1 - mean0) / (pooled + 1e-12)


def welch_pvalues_and_tstats(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x0 = x[y == 0]
    x1 = x[y == 1]
    stat, pvalue = stats.ttest_ind(x1, x0, axis=0, equal_var=False, nan_policy="omit")
    stat = np.nan_to_num(stat, nan=0.0, posinf=0.0, neginf=0.0)
    pvalue = np.nan_to_num(pvalue, nan=1.0, posinf=1.0, neginf=1.0)
    return pvalue, stat


def decoder_jacobian(
    model: TopologicalAutoencoder,
    z_reference: np.ndarray,
    device: torch.device,
    epsilon: float = 0.01,
) -> np.ndarray:
    z_base = torch.as_tensor(z_reference, dtype=torch.float32, device=device).reshape(1, -1)
    with torch.no_grad():
        x_base = model.decoder(z_base).detach().cpu().numpy().ravel()
        jac = np.empty((x_base.shape[0], z_base.shape[1]), dtype=np.float32)
        for dim in range(z_base.shape[1]):
            z_plus = z_base.clone()
            z_plus[0, dim] += epsilon
            x_plus = model.decoder(z_plus).detach().cpu().numpy().ravel()
            jac[:, dim] = (x_plus - x_base) / epsilon
    return jac


def rank_fold_features(
    x_expr_train: pd.DataFrame,
    z_train: pd.DataFrame,
    y_train: np.ndarray,
    model: TopologicalAutoencoder,
    device: torch.device,
) -> pd.DataFrame:
    latent_cols = [c for c in z_train.columns if c.startswith("z")]
    z_values = z_train[latent_cols].to_numpy(dtype=np.float32)
    latent_d = cohen_d_by_column(z_values, y_train)
    z_ref = z_values[y_train == 1].mean(axis=0)
    jac = decoder_jacobian(model, z_ref, device=device)

    gene_names = list(x_expr_train.columns)
    if jac.shape[0] < len(gene_names):
        raise ValueError(f"model decoder output ({jac.shape[0]}) is smaller than expression genes ({len(gene_names)})")
    jac = jac[: len(gene_names), :]

    tda_importance = np.abs(jac).dot(latent_d)
    tda_order = np.argsort(-tda_importance)
    tda_rank = np.empty(len(gene_names), dtype=int)
    tda_rank[tda_order] = np.arange(1, len(gene_names) + 1)

    pvalue, tstat = welch_pvalues_and_tstats(x_expr_train.to_numpy(dtype=np.float32), y_train)
    euc_order = np.lexsort((-np.abs(tstat), pvalue))
    euc_rank = np.empty(len(gene_names), dtype=int)
    euc_rank[euc_order] = np.arange(1, len(gene_names) + 1)

    out = pd.DataFrame(
        {
            "gene": gene_names,
            "tda_importance": tda_importance,
            "tda_rank": tda_rank,
            "euclidean_t": tstat,
            "euclidean_p": pvalue,
            "euclidean_rank": euc_rank,
        }
    )
    return out.sort_values("tda_rank").reset_index(drop=True)


def select_feature_sets(df_rank: pd.DataFrame) -> dict[str, list[str]]:
    valid = df_rank[~df_rank["gene"].astype(str).str.startswith("unknown_")].copy()
    h2c_candidates = valid[(valid["tda_rank"] <= 200) & (valid["euclidean_p"] > 0.05)].sort_values("tda_rank")["gene"].tolist()
    h2c = h2c_candidates[:37]
    tda_top200 = valid.sort_values("tda_rank").head(200)["gene"].tolist()
    euclidean_top200 = valid.sort_values(["euclidean_p", "euclidean_rank"]).head(200)["gene"].tolist()
    return {
        "H2C_train_rule_top37": h2c,
        "TDA_top200_train": tda_top200,
        "Euclidean_top200_train": euclidean_top200,
    }


def evaluate_classifier(y_true: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "auc": float(roc_auc_score(y_true, prob)),
        "f1": float(f1_score(y_true, pred)),
        "accuracy": float(accuracy_score(y_true, pred)),
    }


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["auc", "f1", "accuracy", "n_features"]
    rows = []
    for keys, sub in df.groupby(["feature_set", "classifier"], sort=False):
        row = {"feature_set": keys[0], "classifier": keys[1], "n_folds": int(sub["fold"].nunique())}
        for col in metric_cols:
            row[f"{col}_mean"] = float(sub[col].mean())
            row[f"{col}_std"] = float(sub[col].std(ddof=1)) if len(sub) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["feature_set", "classifier"]).reset_index(drop=True)


def load_original_best(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if not {"gene_set", "classifier", "auc_mean"}.issubset(df.columns):
        return pd.DataFrame()
    idx = df.groupby("gene_set")["auc_mean"].idxmax()
    return df.loc[idx].sort_values("gene_set").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expr", type=Path, default=DEFAULT_EXPR)
    parser.add_argument("--latent", type=Path, default=DEFAULT_LATENT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--original-results", type=Path, default=DEFAULT_ORIGINAL_RESULTS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_expr = pd.read_csv(args.expr, index_col=0)
    if "Target" not in df_expr.columns:
        raise ValueError("expression matrix must include Target column")
    y = df_expr["Target"].astype(int).to_numpy()
    x_expr = df_expr.drop(columns=["Target"])

    df_latent = pd.read_csv(args.latent)
    latent_cols = [c for c in df_latent.columns if c.startswith("z")]
    if "Target" not in df_latent.columns or len(latent_cols) == 0:
        raise ValueError("latent matrix must include z* columns and Target")
    if len(df_latent) != len(df_expr):
        raise ValueError(f"sample mismatch: expr={len(df_expr)}, latent={len(df_latent)}")
    if not np.array_equal(y, df_latent["Target"].astype(int).to_numpy()):
        raise ValueError("Target labels differ between expression and latent matrices")

    model = load_tae_model(args.model, device=device)

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

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(x_expr, y), start=1):
        x_train = x_expr.iloc[train_idx]
        x_test = x_expr.iloc[test_idx]
        z_train = df_latent.iloc[train_idx][latent_cols]
        z_test = df_latent.iloc[test_idx][latent_cols]
        y_train = y[train_idx]
        y_test = y[test_idx]

        df_rank = rank_fold_features(x_train, z_train, y_train, model=model, device=device)
        df_rank.to_csv(args.outdir / f"fold_{fold}_feature_ranks.csv", index=False)
        feature_sets = select_feature_sets(df_rank)
        feature_sets["Latent_32d_fixed_unsupervised"] = latent_cols

        for feature_set, features in feature_sets.items():
            if not features:
                continue
            if feature_set == "Latent_32d_fixed_unsupervised":
                xtr = z_train[features].to_numpy(dtype=np.float32)
                xte = z_test[features].to_numpy(dtype=np.float32)
            else:
                xtr = x_train[features].to_numpy(dtype=np.float32)
                xte = x_test[features].to_numpy(dtype=np.float32)

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
                if hasattr(estimator, "predict_proba"):
                    prob = estimator.predict_proba(xte)[:, 1]
                else:
                    score = estimator.decision_function(xte)
                    prob = 1.0 / (1.0 + np.exp(-score))
                pred = (prob >= 0.5).astype(int)
                metrics = evaluate_classifier(y_test, prob, pred)
                fold_rows.append(
                    {
                        "fold": fold,
                        "feature_set": feature_set,
                        "classifier": clf_name,
                        "n_features": len(features),
                        **metrics,
                    }
                )

    df_fold = pd.DataFrame(fold_rows)
    df_feature = pd.DataFrame(feature_rows)
    df_summary = summarize(df_fold)
    df_original = load_original_best(args.original_results)

    df_fold.to_csv(args.outdir / "classification_fold_metrics.csv", index=False)
    df_feature.to_csv(args.outdir / "classification_fold_features.csv", index=False)
    df_summary.to_csv(args.outdir / "classification_summary_leakage_free.csv", index=False)
    if not df_original.empty:
        df_original.to_csv(args.outdir / "classification_original_best_per_set.csv", index=False)

    audit = {
        "scope": "leakage-free classification",
        "input_expression": display_path(args.expr),
        "input_latent": display_path(args.latent),
        "input_model": display_path(args.model),
        "device": str(device),
        "n_samples": int(len(y)),
        "class_counts": {"normal": int((y == 0).sum()), "tumor": int((y == 1).sum())},
        "n_expression_genes": int(x_expr.shape[1]),
        "n_folds": int(args.folds),
        "leakage_controls": [
            "TDA latent Cohen's d is recomputed on each training fold only.",
            "Decoder Jacobian reference point is recomputed from each training fold tumor mean only.",
            "Euclidean Welch t-test p-values/ranks are recomputed on each training fold only.",
            "Classifiers are fit on training fold features and evaluated once on held-out fold samples.",
            "Latent_32d baseline uses fixed unsupervised latent coordinates only.",
        ],
        "combat_status": (
            "The local available expression matrix is already ComBat-corrected by the original preprocessing pipeline. "
            "Raw tumor/normal/clinical files required to refit ComBat inside each fold were not present locally, "
            "so this run removes the core label-dependent feature-selection leakage but cannot refit ComBat."
        ),
    }
    (args.outdir / "classification_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print("\nLeakage-free classification summary")
    print(df_summary.sort_values("auc_mean", ascending=False).to_string(index=False))
    print(f"\nWrote results to {args.outdir}")


if __name__ == "__main__":
    main()
