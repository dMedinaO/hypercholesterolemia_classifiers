from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.calibration import CalibrationDisplay
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, StandardScaler
from sklearn.svm import SVC

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    from imblearn.combine import SMOTEENN
    from imblearn.under_sampling import RandomUnderSampler

    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path("/home/dmedina/Desktop/colabs/hypercholesterolemia_classifiers").resolve()

MODELLING_DIR = PROJECT_ROOT / "notebooks" / "modelling"
ANALYSIS_TABLES_DIR = MODELLING_DIR / "analysis_tables"
FINAL_SELECTION_DIR = MODELLING_DIR / "final_model_selection"
FINAL_SELECTION_TABLES_DIR = FINAL_SELECTION_DIR / "tables"
FINAL_SELECTION_MODELS_DIR = FINAL_SELECTION_DIR / "models"

DATA_VARIANTS_DIR = PROJECT_ROOT / "data" / "processed_variants"

RESULTS_SEED_LEVEL_PATH = ANALYSIS_TABLES_DIR / "results_seed_level.csv"
PIPELINE_GENERALIZATION_SUMMARY_PATH = FINAL_SELECTION_TABLES_DIR / "pipeline_generalization_summary.csv"
SELECTED_PIPELINES_PATH = FINAL_SELECTION_TABLES_DIR / "selected_pipelines.csv"

# =============================================================================
# CONFIG
# =============================================================================

TARGET_COL = "MSPH"
ID_COLS = ["row_id", "LocalID", "_sheet"]

PREDICTOR_COLS = [
    "Age",
    "Weight",
    "Height",
    "BMI_final",
    "Glycemia",
    "SBP_1T",
    "DBP_1T",
    "TC_1T",
    "TG_1T",
    "HDL_1T",
    "LDL_1T",
]

RANDOM_STATE = 13
N_BOOTSTRAPS = 500

sns.set(style="whitegrid")


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SelectionConfig:
    label: str
    dataset_variant: Optional[str] = None


@dataclass
class PipelineSelection:
    label: str
    dataset_variant: str
    config_name: str
    model_class: str
    scaler: str
    resampling: str
    pca: str
    mean_mcc: float
    std_mcc: float
    count: int
    generalization_score: float
    complexity_score: int


# =============================================================================
# HELPERS
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_pipeline_id(df: pd.DataFrame) -> pd.Series:
    return (
        df["dataset_variant"].astype(str)
        + " | " + df["config_name"].astype(str)
        + " | " + df["model_class"].astype(str)
        + " | " + df["scaler"].astype(str)
        + " | " + df["resampling"].astype(str)
        + " | " + df["pca"].astype(str)
    )


def normalize_variant_name(value: str) -> str:
    value = str(value).strip().upper()
    if value in {"STRICT", "COMPLETE_CASE", "STRICT_COMPLETE_CASE"}:
        return "STRICT"
    if value in {"IMPUTED", "IMPUTED_EXPLORATORY"}:
        return "IMPUTED"
    return value


def load_results_seed_level(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"results_seed_level.csv not found at: {path}")

    df = pd.read_csv(path)
    required = [
        "dataset_variant",
        "config_name",
        "model_class",
        "scaler",
        "resampling",
        "pca",
        "mcc",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in results_seed_level.csv: {missing}")

    for col in ["dataset_variant", "config_name", "model_class", "scaler", "resampling", "pca"]:
        df[col] = df[col].astype(str)

    df["dataset_variant"] = df["dataset_variant"].map(normalize_variant_name)
    df["pipeline_id"] = build_pipeline_id(df)
    return df


def compute_complexity_score(summary_df: pd.DataFrame) -> pd.Series:
    return (
        (summary_df["scaler"].astype(str).str.lower() != "none").astype(int)
        + (summary_df["resampling"].astype(str).str.lower() != "none").astype(int)
        + (summary_df["pca"].astype(str).str.lower() == "on").astype(int)
        + (~summary_df["model_class"].isin(["LogisticRegression", "LinearDiscriminantAnalysis"])).astype(int)
    )


def summarize_pipelines(results_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results_df.groupby(
            ["dataset_variant", "config_name", "model_class", "scaler", "resampling", "pca"],
            dropna=False,
            as_index=False,
        )["mcc"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    summary["std"] = summary["std"].fillna(0.0)
    summary["generalization_score"] = summary["mean"] - summary["std"]
    summary["complexity_score"] = compute_complexity_score(summary)

    summary = summary.sort_values(
        ["generalization_score", "mean", "std", "complexity_score"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    return summary


def select_best_pipeline(
    summary_df: pd.DataFrame,
    selection: SelectionConfig,
) -> PipelineSelection:
    df = summary_df.copy()

    if selection.dataset_variant is not None:
        target_variant = normalize_variant_name(selection.dataset_variant)
        df = df[df["dataset_variant"] == target_variant].copy()

    if df.empty:
        raise ValueError(f"No candidates found for selection: {selection}")

    top_mcc_threshold = df["mean"].quantile(0.90) if len(df) >= 10 else df["mean"].max()
    df = df[df["mean"] >= top_mcc_threshold].copy()

    if df.empty:
        raise ValueError(f"No top-MCC candidates found for selection: {selection}")

    df = df.sort_values(
        ["generalization_score", "mean", "std", "complexity_score"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    row = df.iloc[0]
    return PipelineSelection(
        label=selection.label,
        dataset_variant=str(row["dataset_variant"]),
        config_name=str(row["config_name"]),
        model_class=str(row["model_class"]),
        scaler=str(row["scaler"]),
        resampling=str(row["resampling"]),
        pca=str(row["pca"]),
        mean_mcc=float(row["mean"]),
        std_mcc=float(row["std"]),
        count=int(row["count"]),
        generalization_score=float(row["generalization_score"]),
        complexity_score=int(row["complexity_score"]),
    )


def find_dataset_path_for_variant(dataset_variant: str) -> Path:
    dataset_variant = normalize_variant_name(dataset_variant)

    if dataset_variant == "STRICT":
        candidates = sorted(DATA_VARIANTS_DIR.glob("hcs_strict_complete_case_*.csv"))
    elif dataset_variant == "IMPUTED":
        candidates = sorted(DATA_VARIANTS_DIR.glob("hcs_imputed_exploratory_*.csv"))
    else:
        raise ValueError(f"Unknown dataset variant: {dataset_variant}")

    if not candidates:
        raise FileNotFoundError(
            f"No dataset file found for variant '{dataset_variant}' in {DATA_VARIANTS_DIR}"
        )

    return candidates[-1]


def load_modeling_dataset_for_variant(dataset_variant: str) -> tuple[pd.DataFrame, Path]:
    dataset_path = find_dataset_path_for_variant(dataset_variant)
    df = pd.read_csv(dataset_path)

    required = [TARGET_COL] + PREDICTOR_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in modeling dataset {dataset_path.name}: {missing}")

    return df, dataset_path


def make_scaler(name: str):
    name = str(name).lower()
    if name == "none":
        return "passthrough"
    if name == "standard":
        return StandardScaler()
    if name == "robust":
        return RobustScaler()
    if name == "maxabs":
        return MaxAbsScaler()
    raise ValueError(f"Unknown scaler: {name}")


def make_resampler(name: str):
    if not IMBLEARN_AVAILABLE:
        if str(name).lower() != "none":
            raise ImportError("imblearn is required for non-none resampling.")
        return None

    name = str(name).lower()
    if name == "none":
        return None
    if name == "undersample":
        return RandomUnderSampler(random_state=RANDOM_STATE)
    if name == "smote":
        return SMOTE(random_state=RANDOM_STATE)
    if name == "smoteenn":
        return SMOTEENN(random_state=RANDOM_STATE)
    raise ValueError(f"Unknown resampling: {name}")


def make_model(name: str):
    name = str(name)
    if name == "LogisticRegression":
        return LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
    if name == "LinearDiscriminantAnalysis":
        return LinearDiscriminantAnalysis()
    if name == "QuadraticDiscriminantAnalysis":
        return QuadraticDiscriminantAnalysis()
    if name == "RandomForestClassifier":
        return RandomForestClassifier(
            n_estimators=500,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if name == "KNeighborsClassifier":
        return KNeighborsClassifier()
    if name in {"SVC", "SVC_linear"}:
        return SVC(kernel="linear", probability=True, random_state=RANDOM_STATE)
    if name == "GaussianNB":
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB()
    raise ValueError(f"Unknown model_class: {name}")


def build_pipeline(selection: PipelineSelection):
    numeric_transformer_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", make_scaler(selection.scaler)),
    ]
    numeric_transformer = Pipeline(numeric_transformer_steps)

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, PREDICTOR_COLS)],
        remainder="drop",
    )

    steps: list[tuple[str, Any]] = [("preprocessor", preprocessor)]

    if selection.pca.lower() == "on":
        steps.append(("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)))

    model = make_model(selection.model_class)
    resampler = make_resampler(selection.resampling)

    if resampler is not None:
        if not IMBLEARN_AVAILABLE:
            raise ImportError("imblearn is required for resampling pipelines.")
        return ImbPipeline([
            *steps,
            ("resampling", resampler),
            ("model", model),
        ])

    return Pipeline([
        *steps,
        ("model", model),
    ])


def get_eval_split(
    df: pd.DataFrame,
    config_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = df[TARGET_COL].astype(int)

    if config_name in {"config_kfold", "config_random80", "config_strat80", "config_loo"}:
        train_idx, test_idx = train_test_split(
            np.arange(len(df)),
            test_size=0.20,
            stratify=y,
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unknown config_name: {config_name}")

    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def get_pred_scores(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = np.asarray(scores, dtype=float)
        if np.allclose(scores.max(), scores.min()):
            return np.full_like(scores, 0.5, dtype=float)
        return (scores - scores.min()) / (scores.max() - scores.min())
    raise ValueError("Model has neither predict_proba nor decision_function.")


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstraps: int = 500,
) -> dict[str, float]:
    rng = np.random.default_rng(RANDOM_STATE)
    aucs = []

    for _ in range(n_bootstraps):
        idx = rng.integers(0, len(y_true), len(y_true))
        y_b = y_true[idx]
        s_b = y_score[idx]
        if len(np.unique(y_b)) < 2:
            continue
        aucs.append(roc_auc_score(y_b, s_b))

    if not aucs:
        return {"roc_auc_ci_low": np.nan, "roc_auc_ci_high": np.nan}

    return {
        "roc_auc_ci_low": float(np.percentile(aucs, 2.5)),
        "roc_auc_ci_high": float(np.percentile(aucs, 97.5)),
    }


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, Any], pd.DataFrame]:
    y_true = y_test.astype(int).to_numpy()
    y_score = get_pred_scores(model, X_test)
    y_pred = (y_score >= 0.5).astype(int)

    metrics = {
        "n_test": int(len(y_true)),
        "positive_rate_test": float(np.mean(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else np.nan,
        "pr_auc": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else np.nan,
        "brier": float(brier_score_loss(y_true, y_score)),
    }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics.update({
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan,
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan,
    })
    metrics.update(bootstrap_auc_ci(y_true, y_score, n_bootstraps=N_BOOTSTRAPS))

    pred_df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": y_score,
    })

    return metrics, pred_df


def plot_roc(y_true: np.ndarray, y_score: np.ndarray, out_path: Path, title: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_pr(y_true: np.ndarray, y_score: np.ndarray, out_path: Path, title: str) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f"PR-AUC = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(model, X_test: pd.DataFrame, y_test: pd.Series, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    CalibrationDisplay.from_estimator(model, X_test, y_test, n_bins=10, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def export_selected_pipeline(
    selection: PipelineSelection,
    full_df: pd.DataFrame,
    dataset_path: Path,
    export_root: Path,
) -> None:
    out_dir = export_root / selection.label
    ensure_dir(out_dir)

    train_df, test_df = get_eval_split(full_df, selection.config_name)

    X_train = train_df[PREDICTOR_COLS].copy()
    y_train = train_df[TARGET_COL].astype(int).copy()
    X_test = test_df[PREDICTOR_COLS].copy()
    y_test = test_df[TARGET_COL].astype(int).copy()

    pipeline = build_pipeline(selection)
    pipeline.fit(X_train, y_train)

    metrics, pred_df = evaluate_model(pipeline, X_test, y_test)
    y_true = pred_df["y_true"].to_numpy()
    y_pred = pred_df["y_pred"].to_numpy()
    y_score = pred_df["y_score"].to_numpy()

    joblib.dump(pipeline, out_dir / "pipeline.joblib")
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(out_dir / "metrics_summary.csv", index=False)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    with open(out_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    metadata = {
        "selection": asdict(selection),
        "dataset_path": str(dataset_path),
        "predictor_columns": PREDICTOR_COLS,
        "target_column": TARGET_COL,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "random_state": RANDOM_STATE,
        "project_root": str(PROJECT_ROOT),
        "notes": [
            "Final export split is a fixed stratified 80/20 split for artifact generation.",
            "Selection was based on robustness/generalization summary from results_seed_level.csv.",
            "For config_loo, the final artifact export still uses a stratified 80/20 split.",
        ],
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if len(np.unique(y_true)) > 1:
        plot_roc(y_true, y_score, out_dir / "roc_curve.png", f"ROC — {selection.label}")
        plot_pr(y_true, y_score, out_dir / "pr_curve.png", f"PR curve — {selection.label}")
        plot_calibration(
            pipeline,
            X_test,
            y_test,
            out_dir / "calibration_curve.png",
            f"Calibration — {selection.label}",
        )

    plot_confusion(
        y_true,
        y_pred,
        out_dir / "confusion_matrix.png",
        f"Confusion matrix — {selection.label}",
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ensure_dir(FINAL_SELECTION_DIR)
    ensure_dir(FINAL_SELECTION_TABLES_DIR)
    ensure_dir(FINAL_SELECTION_MODELS_DIR)

    results_df = load_results_seed_level(RESULTS_SEED_LEVEL_PATH)
    summary_df = summarize_pipelines(results_df)
    summary_df.to_csv(PIPELINE_GENERALIZATION_SUMMARY_PATH, index=False)

    selections = [
        SelectionConfig(label="best_overall", dataset_variant=None),
        SelectionConfig(label="best_strict", dataset_variant="STRICT"),
        SelectionConfig(label="best_imputed", dataset_variant="IMPUTED"),
    ]

    selected_rows = [select_best_pipeline(summary_df, s) for s in selections]
    selected_df = pd.DataFrame([asdict(x) for x in selected_rows])
    selected_df.to_csv(SELECTED_PIPELINES_PATH, index=False)

    for sel in selected_rows:
        print(f"Exporting: {sel.label} -> {sel}")
        full_df, dataset_path = load_modeling_dataset_for_variant(sel.dataset_variant)
        export_selected_pipeline(sel, full_df, dataset_path, FINAL_SELECTION_MODELS_DIR)

    print("Done.")
    print(f"Exports saved in: {FINAL_SELECTION_DIR}")


if __name__ == "__main__":
    main()