"""Fair and out-of-sample survival re-evaluation for H2C.

* H2C, Euclidean, and Random gene sets are evaluated with the same score
  methods.
* Survival-supervised Cox-beta scores are evaluated both in-sample and
  out-of-sample; the out-of-sample score is fit on training folds only.
* Multivariate models retain available age/stage/treatment covariates.
  Sex and cancer type are intentionally excluded because this is a BRCA-only
  cohort and those variables do not provide a pan-cancer contrast here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = REPO_ROOT.parent
SURVIVAL_SOURCE = WORKSPACE / "FindVar-Survival-Analysis" / "phase7_survival_analysis"
if str(SURVIVAL_SOURCE) not in sys.path:
    sys.path.insert(0, str(SURVIVAL_SOURCE))

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xenaPython import xenaQuery as xq

from survival_analysis import (
    DEFAULT_EXPR_DATASET,
    DEFAULT_HOST,
    drop_all_nan_genes,
    fetch_expression,
    get_h2c_genes,
    load_gene_importance,
    parse_any_yes,
    stage_to_group,
    tcga_sample_to_xena_sample,
)


DEFAULT_GENE_IMPORTANCE = WORKSPACE / "FindVar" / "phase3_gene_traceback" / "results" / "gene_importance_full.csv"
DEFAULT_COHORT = REPO_ROOT / "survival" / "data" / "brca_survival_cohort.tsv"
DEFAULT_OUTDIR = REPO_ROOT / "survival" / "results" / "leakage_free"


CLINICAL_DATASET_CANDIDATES = [
    "TCGA.BRCA.sampleMap/BRCA_clinicalMatrix",
    "TCGA.BRCA.sampleMap/BRCA_survival",
    "TCGA.BRCA.sampleMap/clinicalMatrix",
]
OS_FIELD_CANDIDATES = ["OS_event_nature2012", "OS", "_OS", "overall_survival", "vital_status", "Vital_Status_nature2012"]
OS_TIME_FIELD_CANDIDATES = [
    "OS_Time_nature2012",
    "OS.time",
    "_OS.time",
    "overall_survival_time",
    "days_to_last_followup",
    "days_to_last_known_alive",
    "days_to_death",
]
AGE_FIELD_CANDIDATES = [
    "Age_at_Initial_Pathologic_Diagnosis_nature2012",
    "age_at_initial_pathologic_diagnosis",
    "age_at_diagnosis",
    "age_at_diagnosis.diagnoses",
    "age_at_index.demographic",
    "age",
]
STAGE_FIELD_CANDIDATES = [
    "Converted_Stage_nature2012",
    "AJCC_Stage_nature2012",
    "pathologic_stage",
    "ajcc_pathologic_stage",
    "ajcc_pathologic_stage.diagnoses",
    "clinical_stage",
    "stage_event_pathologic_stage",
]
TREATMENT_FIELD_CANDIDATES = [
    "additional_radiation_therapy",
    "additional_pharmaceutical_therapy",
    "treatment_or_therapy.treatments.diagnoses",
    "radiation_therapy",
    "pharmaceutical_therapy",
    "drug_treatment",
]


def first_present(fields: set[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in fields:
            return c
    return None


def fetch_matrix(host: str, dataset: str, samples: list[str], fields: list[str]) -> pd.DataFrame:
    values = xq.dataset_probe_values(host, dataset, samples, fields)
    if len(values) != len(fields):
        raise ValueError(f"Unexpected Xena response for {dataset}: {len(values)} rows for {len(fields)} fields")
    data = {field: values[i] for i, field in enumerate(fields)}
    return pd.DataFrame(data, index=samples)


def discover_clinical_dataset(host: str) -> tuple[str, dict[str, object]]:
    datasets = set(xq.datasets_list(host))
    candidates = [d for d in CLINICAL_DATASET_CANDIDATES if d in datasets]
    candidates += [
        d for d in sorted(datasets)
        if "TCGA.BRCA.sampleMap" in d and ("clinical" in d.lower() or "survival" in d.lower())
        and d not in candidates
    ]

    errors: list[str] = []
    for dataset in candidates:
        try:
            fields = set(xq.dataset_field(host, dataset))
        except Exception as exc:
            errors.append(f"{dataset}: {exc}")
            continue
        os_col = first_present(fields, OS_FIELD_CANDIDATES)
        time_col = first_present(fields, OS_TIME_FIELD_CANDIDATES)
        if os_col and time_col:
            selected = {
                "OS": os_col,
                "OS.time": time_col,
                "age": first_present(fields, AGE_FIELD_CANDIDATES),
                "stage": first_present(fields, STAGE_FIELD_CANDIDATES),
                "treatment": first_present(fields, TREATMENT_FIELD_CANDIDATES),
                "treatment_fields": [c for c in TREATMENT_FIELD_CANDIDATES if c in fields],
            }
            return dataset, selected
        errors.append(f"{dataset}: missing OS/OS.time compatible fields")
    raise ValueError("Could not discover a BRCA clinical survival dataset. Tried: " + "; ".join(errors))


def normalize_event(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() >= len(series) * 0.5:
        return numeric
    s = series.astype(str).str.lower()
    out = pd.Series(np.nan, index=series.index, dtype=float)
    out[s.str.contains("deceased|dead|yes|true|1", regex=True, na=False)] = 1.0
    out[s.str.contains("living|alive|no|false|0", regex=True, na=False)] = 0.0
    return out


def parse_binary_treatment(value: object) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric) and numeric in (0, 1):
        return float(numeric)
    return parse_any_yes(value)


def build_cohort_from_xena(host: str, expr_dataset: str, out_path: Path | None = None) -> tuple[pd.DataFrame, dict[str, object]]:
    clinical_dataset, selected = discover_clinical_dataset(host)
    clinical_samples = xq.dataset_samples(host, clinical_dataset)
    fields = [
        v
        for k, v in selected.items()
        if isinstance(v, str) and v is not None and k != "treatment"
    ]
    fields.extend(str(v) for v in selected.get("treatment_fields", []))
    fields = list(dict.fromkeys(fields))
    clinical = fetch_matrix(host, clinical_dataset, clinical_samples, fields)

    df = pd.DataFrame(index=clinical.index)
    df["sample"] = clinical.index.astype(str)
    df["patient_barcode"] = df["sample"].str[:12]
    df["OS"] = normalize_event(clinical[selected["OS"]])
    df["OS.time"] = pd.to_numeric(clinical[selected["OS.time"]], errors="coerce")
    if selected.get("age"):
        df["age_source"] = clinical[selected["age"]]
    if selected.get("stage"):
        df["stage_source"] = clinical[selected["stage"]]
    for treatment_field in selected.get("treatment_fields", []):
        df[f"treatment_source__{treatment_field}"] = clinical[str(treatment_field)]

    expr_samples = xq.dataset_samples(host, expr_dataset)
    expr_set = set(expr_samples)
    patient_to_tumor = {}
    for sample in expr_samples:
        parts = str(sample).split("-")
        if len(parts) >= 4 and parts[3].startswith("01"):
            patient_to_tumor.setdefault("-".join(parts[:3]), sample)

    mapped = []
    for sample in df["sample"].astype(str):
        candidates = [sample, tcga_sample_to_xena_sample(sample), patient_to_tumor.get(sample[:12])]
        mapped.append(next((c for c in candidates if c in expr_set), np.nan))
    df["sample_xena"] = mapped

    df = df.dropna(subset=["OS", "OS.time", "sample_xena"])
    df = df[df["OS.time"] > 0].copy()
    df = df.drop_duplicates(subset=["sample_xena"], keep="first")

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, sep="\t", index=False)

    audit = {
        "clinical_dataset": clinical_dataset,
        "selected_fields": selected,
        "n_clinical_samples": len(clinical_samples),
        "n_expression_samples": len(expr_samples),
        "n_survival_rows": int(len(df)),
        "events": int(df["OS"].sum()),
    }
    return df, audit


def load_or_build_cohort(path: Path, host: str, expr_dataset: str) -> tuple[pd.DataFrame, dict[str, object]]:
    if path.exists():
        df = pd.read_csv(path, sep="\t", low_memory=False)
        if "sample_xena" not in df.columns:
            df["sample_xena"] = df["sample"].astype(str).map(tcga_sample_to_xena_sample)
        df["OS"] = pd.to_numeric(df["OS"], errors="coerce")
        df["OS.time"] = pd.to_numeric(df["OS.time"], errors="coerce")
        df = df.dropna(subset=["OS", "OS.time", "sample_xena"])
        df = df[df["OS.time"] > 0].copy()
        event_rate = float(df["OS"].mean()) if len(df) else np.nan
        has_current_treatment_sources = any(c.startswith("treatment_source__") for c in df.columns)
        if len(df) >= 50 and 0.02 < event_rate < 0.98 and has_current_treatment_sources:
            return df, {"source": str(path), "n_survival_rows": int(len(df)), "events": int(df["OS"].sum())}
        # A cached cohort with all events usually means days_to_death was used
        # without censored follow-up time. Rebuild from authoritative Xena fields.
    return build_cohort_from_xena(host, expr_dataset, out_path=path)


def get_euclidean_top37(df_imp: pd.DataFrame) -> list[str]:
    df = df_imp[~df_imp["gene"].astype(str).str.startswith("unknown_")].copy()
    if "euclidean_rank" in df.columns:
        return df.sort_values("euclidean_rank")["gene"].astype(str).head(37).tolist()
    if "Abs_PB_Corr" in df.columns:
        return df.sort_values("Abs_PB_Corr", ascending=False)["gene"].astype(str).head(37).tolist()
    if "PB_Corr" in df.columns:
        return df.assign(abs_pb=df["PB_Corr"].abs()).sort_values("abs_pb", ascending=False)["gene"].astype(str).head(37).tolist()
    raise ValueError("gene importance file lacks Euclidean ranking/statistics")


def build_gene_sets(df_imp: pd.DataFrame, dataset_fields: set[str], seed: int) -> dict[str, list[str]]:
    h2c = get_h2c_genes(df_imp)
    if len(h2c) > 37:
        h2c = h2c[:37]
    euc = get_euclidean_top37(df_imp)
    candidates = [
        g for g in df_imp["gene"].astype(str)
        if not g.startswith("unknown_") and g in dataset_fields and g not in set(h2c) | set(euc)
    ]
    if len(candidates) < 37:
        raise ValueError(f"Not enough random candidate genes in expression dataset: {len(candidates)}")
    rng = np.random.default_rng(seed)
    random_genes = list(rng.choice(candidates, size=37, replace=False))
    return {"H2C_37": h2c, "Euclidean_top37": euc, "Random_37": random_genes}


def fetch_gene_set_expression(host: str, dataset: str, samples: list[str], gene_sets: dict[str, list[str]]) -> tuple[dict[str, pd.DataFrame], dict[str, list[str]]]:
    out: dict[str, pd.DataFrame] = {}
    missing: dict[str, list[str]] = {}
    for name, genes in gene_sets.items():
        expr = fetch_expression(host, dataset, samples, genes)
        expr, miss = drop_all_nan_genes(expr)
        keep = [g for g in genes if g in expr.columns]
        expr = expr[keep].replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
        out[name] = expr
        missing[name] = miss
    return out, missing


def score_pca_full(expr: pd.DataFrame) -> pd.Series:
    x = StandardScaler().fit_transform(expr.to_numpy(dtype=float))
    score = PCA(n_components=1, svd_solver="full", random_state=42).fit_transform(x).ravel()
    return pd.Series(score, index=expr.index, name="pca_full")


def score_pca_oof(expr: pd.DataFrame, events: pd.Series, folds: int, seed: int) -> pd.Series:
    score = pd.Series(np.nan, index=expr.index, name="pca_oof")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    x = expr.to_numpy(dtype=float)
    y_event = events.loc[expr.index].astype(int).to_numpy()
    for train_idx, test_idx in splitter.split(x, y_event):
        scaler = StandardScaler().fit(x[train_idx])
        x_train = scaler.transform(x[train_idx])
        x_test = scaler.transform(x[test_idx])
        pca = PCA(n_components=1, svd_solver="full", random_state=seed).fit(x_train)
        score.iloc[test_idx] = pca.transform(x_test).ravel()
    return score


def fit_cox_beta_score(train_x: np.ndarray, train_t: np.ndarray, train_e: np.ndarray, test_x: np.ndarray, penalizer: float) -> np.ndarray:
    scaler = StandardScaler().fit(train_x)
    x_train = scaler.transform(train_x)
    x_test = scaler.transform(test_x)
    cols = [f"g{i}" for i in range(x_train.shape[1])]
    df_train = pd.DataFrame(x_train, columns=cols)
    df_train["T"] = train_t
    df_train["E"] = train_e
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(df_train, duration_col="T", event_col="E")
    betas = cph.params_.reindex(cols).to_numpy(dtype=float)
    return x_test.dot(betas)


def score_cox_beta_full(expr: pd.DataFrame, durations: pd.Series, events: pd.Series, penalizer: float) -> pd.Series:
    values = fit_cox_beta_score(
        expr.to_numpy(dtype=float),
        durations.loc[expr.index].to_numpy(dtype=float),
        events.loc[expr.index].to_numpy(dtype=int),
        expr.to_numpy(dtype=float),
        penalizer,
    )
    return pd.Series(values, index=expr.index, name="cox_beta_full")


def score_cox_beta_oof(expr: pd.DataFrame, durations: pd.Series, events: pd.Series, folds: int, seed: int, penalizer: float) -> pd.Series:
    score = pd.Series(np.nan, index=expr.index, name="cox_beta_oof")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    x = expr.to_numpy(dtype=float)
    y_event = events.loc[expr.index].astype(int).to_numpy()
    t = durations.loc[expr.index].to_numpy(dtype=float)
    for train_idx, test_idx in splitter.split(x, y_event):
        score.iloc[test_idx] = fit_cox_beta_score(
            x[train_idx],
            t[train_idx],
            y_event[train_idx],
            x[test_idx],
            penalizer,
        )
    return score


def build_covariates(df: pd.DataFrame) -> pd.DataFrame:
    cov = pd.DataFrame(index=df.index)
    age_col = next((c for c in ["age_source", *AGE_FIELD_CANDIDATES] if c in df.columns), None)
    if age_col:
        age = pd.to_numeric(df[age_col], errors="coerce")
        if age.notna().sum() and float(age.dropna().median()) > 150:
            age = age / 365.25
        if age.isna().any():
            cov["age_missing"] = age.isna().astype(float)
        cov["age_years"] = age.fillna(age.median())

    stage_col = next((c for c in ["stage_source", *STAGE_FIELD_CANDIDATES] if c in df.columns), None)
    if stage_col:
        raw_stage = df[stage_col]
        stage_num = pd.to_numeric(raw_stage, errors="coerce")
        if stage_num.notna().sum() >= max(50, int(len(raw_stage) * 0.5)):
            if stage_num.isna().any():
                cov["stage_missing"] = stage_num.isna().astype(float)
            cov["stage_numeric"] = stage_num.fillna(stage_num.median())
        elif raw_stage.astype(str).str.contains("stage|i|v", case=False, regex=True, na=False).any():
            stage = raw_stage.map(stage_to_group)
            dummies = pd.get_dummies(stage.fillna("Unknown"), prefix="stage", drop_first=True, dtype=float)
            cov = pd.concat([cov, dummies], axis=1)
        else:
            stage = raw_stage.astype(str).replace({"nan": "Unknown", "NaN": "Unknown"})
            dummies = pd.get_dummies(stage.fillna("Unknown"), prefix="stage", drop_first=True, dtype=float)
            cov = pd.concat([cov, dummies], axis=1)

    treatment_cols = [
        c
        for c in df.columns
        if c == "treatment_source"
        or c.startswith("treatment_source__")
        or c in TREATMENT_FIELD_CANDIDATES
    ]
    if treatment_cols:
        parsed = pd.concat([df[c].map(parse_binary_treatment) for c in treatment_cols], axis=1)
        any_yes = parsed.max(axis=1, skipna=True)
        treatment = pd.Series("unknown", index=df.index)
        treatment[any_yes == 0.0] = "no"
        treatment[any_yes == 1.0] = "yes"
        dummies = pd.get_dummies(treatment, prefix="treatment", drop_first=True, dtype=float)
        cov = pd.concat([cov, dummies], axis=1)
    for col in list(cov.columns):
        values = pd.to_numeric(cov[col], errors="coerce")
        if values.nunique(dropna=True) <= 1 or float(values.var(ddof=0)) < 1e-10:
            cov = cov.drop(columns=[col])
    return cov


def fit_univariate(durations: pd.Series, events: pd.Series, score: pd.Series, penalizer: float) -> dict[str, object]:
    df = pd.DataFrame({"T": durations, "E": events, "score": score}).dropna()
    df["score_z"] = (df["score"] - df["score"].mean()) / (df["score"].std(ddof=0) + 1e-12)
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(df[["T", "E", "score_z"]], duration_col="T", event_col="E")
    row = cph.summary.loc["score_z"]
    return {
        "n": int(df.shape[0]),
        "events": int(df["E"].sum()),
        "hr_per_sd": float(np.exp(row["coef"])),
        "ci_lower": float(np.exp(row["coef lower 95%"])),
        "ci_upper": float(np.exp(row["coef upper 95%"])),
        "cox_p": float(row["p"]),
        "c_index": float(cph.concordance_index_),
    }


def fit_multivariate(durations: pd.Series, events: pd.Series, score: pd.Series, covariates: pd.DataFrame, penalizer: float) -> dict[str, object]:
    df = pd.DataFrame({"T": durations, "E": events, "score": score}, index=score.index)
    df = pd.concat([df, covariates], axis=1)
    df["score_z"] = (df["score"] - df["score"].mean()) / (df["score"].std(ddof=0) + 1e-12)
    cols = ["T", "E", "score_z"] + [c for c in covariates.columns if c in df.columns]
    model_df = df[cols].dropna()
    if model_df.shape[0] < 50:
        raise ValueError(f"Too few complete cases for multivariate Cox: {model_df.shape[0]}")
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(model_df, duration_col="T", event_col="E")
    row = cph.summary.loc["score_z"]
    return {
        "n": int(model_df.shape[0]),
        "events": int(model_df["E"].sum()),
        "hr_per_sd": float(np.exp(row["coef"])),
        "ci_lower": float(np.exp(row["coef lower 95%"])),
        "ci_upper": float(np.exp(row["coef upper 95%"])),
        "cox_p": float(row["p"]),
        "c_index": float(cph.concordance_index_),
        "covariates": json.dumps([c for c in model_df.columns if c not in {"T", "E", "score_z"}]),
    }


def logrank_p(durations: pd.Series, events: pd.Series, score: pd.Series) -> float:
    aligned = pd.DataFrame({"T": durations, "E": events, "score": score}).dropna()
    high = aligned["score"] >= float(aligned["score"].median())
    lr = logrank_test(
        aligned.loc[high, "T"],
        aligned.loc[~high, "T"],
        event_observed_A=aligned.loc[high, "E"],
        event_observed_B=aligned.loc[~high, "E"],
    )
    return float(lr.p_value)


def evaluate_score(
    gene_set: str,
    score_method: str,
    fit_scope: str,
    score: pd.Series,
    durations: pd.Series,
    events: pd.Series,
    covariates: pd.DataFrame,
    penalizer: float,
) -> list[dict[str, object]]:
    rows = []
    uni = fit_univariate(durations.loc[score.index], events.loc[score.index], score, penalizer=penalizer)
    rows.append(
        {
            "endpoint": "OS",
            "model": "univariate",
            "gene_set": gene_set,
            "score_method": score_method,
            "score_fit_scope": fit_scope,
            "logrank_p": logrank_p(durations.loc[score.index], events.loc[score.index], score),
            **uni,
        }
    )
    try:
        multi = fit_multivariate(
            durations.loc[score.index],
            events.loc[score.index],
            score,
            covariates.loc[score.index],
            penalizer=penalizer,
        )
        rows.append(
            {
                "endpoint": "OS",
                "model": "multivariate",
                "gene_set": gene_set,
                "score_method": score_method,
                "score_fit_scope": fit_scope,
                "logrank_p": logrank_p(durations.loc[score.index], events.loc[score.index], score),
                **multi,
            }
        )
    except Exception as exc:
        rows.append(
            {
                "endpoint": "OS",
                "model": "multivariate",
                "gene_set": gene_set,
                "score_method": score_method,
                "score_fit_scope": fit_scope,
                "logrank_p": np.nan,
                "n": np.nan,
                "events": np.nan,
                "hr_per_sd": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "cox_p": np.nan,
                "c_index": np.nan,
                "covariates": json.dumps([]),
                "error": str(exc),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--gene-importance", type=Path, default=DEFAULT_GENE_IMPORTANCE)
    parser.add_argument("--host", type=str, default=DEFAULT_HOST)
    parser.add_argument("--expr-dataset", type=str, default=DEFAULT_EXPR_DATASET)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cox-penalizer", type=float, default=0.1)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    df_imp = load_gene_importance(args.gene_importance)
    cohort, cohort_audit = load_or_build_cohort(args.cohort, args.host, args.expr_dataset)
    cohort = cohort.set_index("sample_xena", drop=False)

    dataset_fields = set(xq.dataset_field(args.host, args.expr_dataset))
    dataset_fields.discard("sampleID")
    gene_sets = build_gene_sets(df_imp, dataset_fields, seed=args.seed)
    samples = cohort.index.astype(str).tolist()
    expr_by_set, missing_by_set = fetch_gene_set_expression(args.host, args.expr_dataset, samples, gene_sets)

    common = None
    for expr in expr_by_set.values():
        common = expr.index if common is None else common.intersection(expr.index)
    if common is None or len(common) < 50:
        raise ValueError(f"Too few common expression samples across gene sets: {0 if common is None else len(common)}")

    cohort = cohort.loc[common].copy()
    durations = cohort["OS.time"].astype(float)
    events = cohort["OS"].astype(int)
    covariates = build_covariates(cohort)

    results = []
    score_values = pd.DataFrame(index=common)
    effective_gene_rows = []

    for gene_set, expr in expr_by_set.items():
        expr = expr.loc[common]
        effective_genes = list(expr.columns)
        for gene in effective_genes:
            effective_gene_rows.append({"gene_set": gene_set, "gene": gene})

        scores = {
            ("pca", "in_sample"): score_pca_full(expr),
            ("pca", "out_of_sample"): score_pca_oof(expr, events, folds=args.folds, seed=args.seed),
            ("cox_beta", "in_sample"): score_cox_beta_full(expr, durations, events, penalizer=args.cox_penalizer),
            ("cox_beta", "out_of_sample"): score_cox_beta_oof(
                expr,
                durations,
                events,
                folds=args.folds,
                seed=args.seed,
                penalizer=args.cox_penalizer,
            ),
        }

        for (method, scope), score in scores.items():
            score_values[f"{gene_set}__{method}__{scope}"] = score
            results.extend(
                evaluate_score(
                    gene_set=gene_set,
                    score_method=method,
                    fit_scope=scope,
                    score=score,
                    durations=durations,
                    events=events,
                    covariates=covariates,
                    penalizer=args.cox_penalizer,
                )
            )

    df_results = pd.DataFrame(results)
    df_results.to_csv(args.outdir / "survival_priority1_results.csv", index=False)
    score_values.to_csv(args.outdir / "survival_score_values.csv", index=True, index_label="sample_xena")
    pd.DataFrame(effective_gene_rows).to_csv(args.outdir / "survival_effective_gene_sets.csv", index=False)
    pd.DataFrame(
        [{"gene_set": k, "missing_genes_json": json.dumps(v)} for k, v in missing_by_set.items()]
    ).to_csv(args.outdir / "survival_missing_genes.csv", index=False)

    audit = {
        "scope": "fair and out-of-sample survival re-evaluation",
        "cohort": cohort_audit,
        "expr_dataset": args.expr_dataset,
        "n_common_samples": int(len(common)),
        "events": int(events.sum()),
        "gene_set_sizes": {k: int(len(v)) for k, v in gene_sets.items()},
        "effective_gene_set_sizes": {k: int(len(expr_by_set[k].columns)) for k in expr_by_set},
        "missing_genes": missing_by_set,
        "score_methods": [
            "pca/in_sample: PCA fit on all evaluation samples; no survival labels used.",
            "pca/out_of_sample: scaler and PCA fit on training folds, transformed held-out fold only.",
            "cox_beta/in_sample: Cox coefficients fit and evaluated on all samples; included as circular comparator for all gene sets.",
            "cox_beta/out_of_sample: Cox coefficients fit on training folds and evaluated on held-out folds.",
        ],
        "fairness_controls": [
            "No score method is selected by H2C performance.",
            "Every gene set is evaluated with the same score methods and fit scopes.",
            "Random_37 is sampled once with a fixed seed from expression-available non-H2C/non-Euclidean genes.",
            "Age/stage/treatment covariates are included when present; sex and cancer type are excluded for BRCA-only evaluation.",
        ],
        "covariate_columns": list(covariates.columns),
    }
    (args.outdir / "survival_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print("\nLeakage-remediation survival results")
    print(df_results[["model", "gene_set", "score_method", "score_fit_scope", "n", "events", "hr_per_sd", "cox_p", "logrank_p", "c_index"]].to_string(index=False))
    print(f"\nWrote results to {args.outdir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[leakage_free_survival] ERROR: {exc}", file=sys.stderr)
        raise
