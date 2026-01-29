#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# sklearn
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    matthews_corrcoef,
)

# sklearn models
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.naive_bayes import GaussianNB

# optional: imblearn
try:
    from imblearn.pipeline import Pipeline as ImbPipeline  # type: ignore
    from imblearn.over_sampling import SMOTE  # type: ignore
    from imblearn.combine import SMOTEENN  # type: ignore
    from imblearn.under_sampling import RandomUnderSampler  # type: ignore

    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False
    ImbPipeline = None
    SMOTE = None
    SMOTEENN = None
    RandomUnderSampler = None

# optional: xgboost / lightgbm
def _optional_import_xgb():
    try:
        from xgboost import XGBClassifier  # type: ignore
        return XGBClassifier
    except Exception:
        return None

def _optional_import_lgbm():
    try:
        from lightgbm import LGBMClassifier  # type: ignore
        return LGBMClassifier
    except Exception:
        return None


# -------------------------
# Helpers: filesystem
# -------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def now_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def json_dumps_safe(x: Any) -> str:
    return json.dumps(x, sort_keys=True, default=str)

# -------------------------
# Helpers: grids
# -------------------------
def expand_param_grid(grid: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand a dict of lists into list of dict combinations."""
    if not grid:
        return [{}]
    keys = list(grid.keys())
    values: List[List[Any]] = []
    for k in keys:
        v = grid[k]
        values.append(v if isinstance(v, list) else [v])
    combos: List[Dict[str, Any]] = []
    for vals in product(*values):
        combos.append({keys[i]: vals[i] for i in range(len(keys))})
    return combos

# -------------------------
# Preprocessing builders
# -------------------------
def build_scaler(name: str):
    if name == "none":
        return None
    if name == "standard":
        return StandardScaler()
    if name == "robust":
        return RobustScaler()
    if name == "maxabs":
        return MaxAbsScaler()
    raise ValueError(f"Unknown scaler: {name}")

def build_pca(params: Dict[str, Any]) -> PCA:
    # n_components can be float (variance) or int
    return PCA(**params)

# -------------------------
# Resampling builders
# -------------------------
def build_resampler(strategy: str, params: Dict[str, Any], seed: int):
    if strategy == "none":
        return None
    if not IMBLEARN_AVAILABLE:
        raise RuntimeError("imblearn not available")

    if strategy == "undersample":
        # RandomUnderSampler supports sampling_strategy, replacement, random_state
        return RandomUnderSampler(random_state=seed, **params)

    if strategy == "smote":
        return SMOTE(random_state=seed, **params)

    if strategy == "smoteenn":
        # config uses smote_k_neighbors alias
        smote_k = params.get("smote_k_neighbors", 5)
        sampling_strategy = params.get("sampling_strategy", 1.0)
        sm = SMOTE(random_state=seed, k_neighbors=smote_k, sampling_strategy=sampling_strategy)
        # SMOTEENN signature varies slightly; keep it safe
        return SMOTEENN(random_state=seed, smote=sm)

    raise ValueError(f"Unknown resampling strategy: {strategy}")

# -------------------------
# Model builders
# -------------------------
MODEL_REGISTRY = {
    "LogisticRegression": LogisticRegression,
    "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis,
    "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis,
    "KNeighborsClassifier": KNeighborsClassifier,
    "SVC": SVC,
    "RandomForestClassifier": RandomForestClassifier,
    "ExtraTreesClassifier": ExtraTreesClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
    "GaussianNB": GaussianNB,
}

def build_estimator(model_key: str, model_cfg: Dict[str, Any], params: Dict[str, Any], seed: int) -> Optional[BaseEstimator]:
    # Determine class name (some keys like SVC_linear map to SVC)
    class_name = model_cfg.get("class_name", model_key)

    # Optional models
    if class_name == "XGBClassifier":
        XGB = _optional_import_xgb()
        if XGB is None:
            return None
        # Ensure deterministic seed
        if "random_state" not in params:
            params["random_state"] = seed
        return XGB(**{**model_cfg.get("init_params", {}), **params})

    if class_name == "LGBMClassifier":
        LGBM = _optional_import_lgbm()
        if LGBM is None:
            return None
        if "random_state" not in params:
            params["random_state"] = seed
        return LGBM(**{**model_cfg.get("init_params", {}), **params})

    if class_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown/unsupported model class_name={class_name} for key={model_key}")

    cls = MODEL_REGISTRY[class_name]
    init_params = dict(model_cfg.get("init_params", {}))

    # Set random_state if supported
    if "random_state" in cls().get_params().keys():
        init_params.setdefault("random_state", seed)

    return cls(**{**init_params, **params})

# -------------------------
# Metrics
# -------------------------
def prob_from_estimator(est: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1]
    if hasattr(est, "decision_function"):
        scores = est.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))
    raise ValueError("Estimator lacks predict_proba and decision_function")

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Dict[str, Any]:
    y_pred = (y_prob >= thr).astype(int)

    out: Dict[str, Any] = {}

    # AUC / PR-AUC undefined if only one class present in y_true
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
        out["pr_auc"] = average_precision_score(y_true, y_prob)
    else:
        out["roc_auc"] = np.nan
        out["pr_auc"] = np.nan

    out["brier"] = brier_score_loss(y_true, y_prob)

    # MCC undefined / unstable with single-class y_true; keep nan for consistency
    if len(np.unique(y_true)) > 1:
        out["mcc"] = matthews_corrcoef(y_true, y_pred)
    else:
        out["mcc"] = np.nan

    out["accuracy"] = accuracy_score(y_true, y_pred)
    out["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    out["precision"] = precision_score(y_true, y_pred, zero_division=0)
    out["recall"] = recall_score(y_true, y_pred, zero_division=0)
    out["f1"] = f1_score(y_true, y_pred, zero_division=0)

    # IMPORTANT: force 2x2 confusion matrix even if only one class present
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["tn"] = int(tn)
    out["fp"] = int(fp)
    out["fn"] = int(fn)
    out["tp"] = int(tp)

    return out
# -------------------------
# Validation iterators
# -------------------------
def iter_splits(
    X: pd.DataFrame,
    y: np.ndarray,
    strategy_name: str,
    strategy_params: Dict[str, Any],
    seed: Optional[int],
) -> Iterable[Tuple[np.ndarray, np.ndarray, str]]:
    n = len(y)

    if strategy_name == "loo":
        loo = LeaveOneOut()
        for i, (tr, te) in enumerate(loo.split(X, y), start=1):
            yield tr, te, f"loo_{i:04d}"
        return

    if strategy_name == "kfold":
        n_splits = int(strategy_params.get("n_splits", 5))
        shuffle = bool(strategy_params.get("shuffle", True))
        if seed is None:
            raise ValueError("kfold requires seed")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
        for i, (tr, te) in enumerate(skf.split(X, y), start=1):
            yield tr, te, f"kfold_{n_splits}_seed{seed}_{i:02d}"
        return

    if strategy_name == "random_split_80_20":
        if seed is None:
            raise ValueError("random_split requires seed")
        test_size = float(strategy_params.get("test_size", 0.2))
        idx = np.arange(n)
        tr, te = train_test_split(idx, test_size=test_size, shuffle=True, random_state=seed)
        yield tr, te, f"random80_20_seed{seed}"
        return

    if strategy_name == "stratified_split_80_20":
        if seed is None:
            raise ValueError("stratified_split requires seed")
        test_size = float(strategy_params.get("test_size", 0.2))
        idx = np.arange(n)
        tr, te = train_test_split(idx, test_size=test_size, shuffle=True, stratify=y, random_state=seed)
        yield tr, te, f"stratified80_20_seed{seed}"
        return

    raise ValueError(f"Unknown validation strategy: {strategy_name}")

# -------------------------
# Main runner
# -------------------------
@dataclass
class Combo:
    model_key: str
    model_class: str
    model_params: Dict[str, Any]
    imputer: str
    scaler: str
    pca: str
    pca_params: Dict[str, Any]
    resampling: str
    resampling_params: Dict[str, Any]
    val_strategy: str
    val_params: Dict[str, Any]
    seed: Optional[int]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV dataset path (STRICT complete-case recommended).")
    ap.add_argument("--config", required=True, help="JSON config path.")
    ap.add_argument("--outdir", required=True, help="Output directory.")
    ap.add_argument("--target", default=None, help="Override target column.")
    ap.add_argument("--predictors", default=None, help="Comma-separated predictors. If omitted, infer numeric cols minus target.")
    ap.add_argument("--id-cols", default="row_id,_sheet,LocalID", help="Comma-separated id cols to ignore for predictors inference.")
    ap.add_argument("--limit-combos", type=int, default=None, help="Debug: limit number of combos.")
    args = ap.parse_args()

    run_id = now_run_id()
    outdir = Path(args.outdir) / f"run_{run_id}"
    ensure_dir(outdir)

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    df = pd.read_csv(args.data)

    # Target + predictors
    target = args.target or cfg.get("target", "MSPH")
    id_cols = [c.strip() for c in args.id_cols.split(",") if c.strip()]

    if args.predictors:
        predictors = [c.strip() for c in args.predictors.split(",") if c.strip()]
    else:
        # infer numeric predictors
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        predictors = [c for c in numeric_cols if c not in id_cols and c != target]

    # ensure types
    df = df.copy()
    df[target] = pd.to_numeric(df[target], errors="coerce").astype(int)
    for c in predictors:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # drop rows with missing target or non-binary
    df = df[df[target].isin([0, 1])].reset_index(drop=True)

    X = df[predictors]
    y = df[target].to_numpy()

    # --- build grids from config ---
    # models
    model_entries = []
    for model_key, model_cfg in cfg.get("models", {}).items():
        if not model_cfg.get("enabled", True):
            continue
        class_name = model_cfg.get("class_name", model_key)
        model_grid = expand_param_grid(model_cfg.get("param_grid", {}))
        for mp in model_grid:
            model_entries.append((model_key, class_name, model_cfg, mp))

    # preprocessing
    prep = cfg.get("preprocessing", {})
    imp_strats = prep.get("imputation", {}).get("strategies", [{"name": "none"}])
    sc_strats = prep.get("scaling", {}).get("strategies", [{"name": "none"}])
    pca_block = prep.get("pca", {})
    pca_strats = pca_block.get("strategies", [{"name": "off"}])
    pca_constraints = pca_block.get("constraints", {})
    pca_apply_only_if_scaled = bool(pca_constraints.get("apply_only_if_scaled", True))
    pca_scalers_allowed = set(pca_constraints.get("scalers_allowed", ["standard", "robust", "maxabs"]))
    pca_forbidden_models = set(pca_constraints.get("do_not_apply_for_models", []))
    pca_behavior = pca_constraints.get("behavior_on_forbidden", "skip")

    # resampling
    res_block = cfg.get("resampling", {})
    res_strats = res_block.get("strategies", [{"name": "none"}])
    res_constraints = res_block.get("constraints", {})
    res_on_failure = res_constraints.get("on_failure", "fallback_to_none")

    # validation
    val = cfg.get("validation", {})
    val_strats = val.get("strategies", [])
    seed_grid = val.get("seed_grid", {})
    seeds = seed_grid.get("seeds", [])
    if not seeds and seed_grid.get("enabled", False):
        raise ValueError("seed_grid.seeds is empty; please fill it with your fixed 100 seeds.")

    reporting = val.get("reporting", {})
    thresholds = reporting.get("thresholds", {}).get("values", [0.5])
    store_preds = bool(reporting.get("store_fold_predictions", True))
    metrics_list = reporting.get("metrics", [])

    # --- build combinations ---
    combos: List[Combo] = []
    for (model_key, model_class, model_cfg, model_params) in model_entries:
        for imp in imp_strats:
            imp_name = imp["name"]
            for sc in sc_strats:
                sc_name = sc["name"]

                for pca_s in pca_strats:
                    pca_name = pca_s["name"]
                    pca_param_grid = expand_param_grid(pca_s.get("params_grid", {})) if pca_name == "on" else [{}]

                    for pca_params in pca_param_grid:
                        # PCA constraints
                        if pca_name == "on":
                            if pca_apply_only_if_scaled and sc_name == "none":
                                continue
                            if pca_apply_only_if_scaled and sc_name not in pca_scalers_allowed:
                                continue
                            # forbidden models (by class)
                            if model_class in pca_forbidden_models:
                                if pca_behavior == "skip":
                                    continue
                                # else force off (not implemented here)
                        # if pca off, ok always

                        for res in res_strats:
                            res_name = res["name"]
                            res_param_grid = expand_param_grid(res.get("params_grid", {})) if res_name != "none" else [{}]

                            for res_params in res_param_grid:
                                # if imblearn not installed, skip non-none resampling
                                if res_name != "none" and not IMBLEARN_AVAILABLE:
                                    continue

                                for vs in val_strats:
                                    vs_name = vs["name"]
                                    vs_param_grid = expand_param_grid(vs.get("params_grid", {}))

                                    use_seed_grid = bool(vs.get("use_seed_grid", False))
                                    seed_list: List[Optional[int]] = seeds if use_seed_grid else [None]

                                    for vs_params in vs_param_grid:
                                        for seed in seed_list:
                                            combos.append(
                                                Combo(
                                                    model_key=model_key,
                                                    model_class=model_class,
                                                    model_params=model_params,
                                                    imputer=imp_name,
                                                    scaler=sc_name,
                                                    pca=pca_name,
                                                    pca_params=pca_params,
                                                    resampling=res_name,
                                                    resampling_params=res_params,
                                                    val_strategy=vs_name,
                                                    val_params=vs_params,
                                                    seed=int(seed) if seed is not None else None,
                                                )
                                            )

    if args.limit_combos is not None:
        combos = combos[: args.limit_combos]

    print(f"[INFO] Dataset shape: {df.shape} | pos_rate={y.mean():.3f}")
    print(f"[INFO] Predictors: {len(predictors)}")
    print(f"[INFO] Total combos: {len(combos)}")
    print(f"[INFO] imblearn available: {IMBLEARN_AVAILABLE}")

    # --- run ---
    results_rows: List[Dict[str, Any]] = []
    pred_rows: List[Dict[str, Any]] = []

    for i, combo in enumerate(combos, start=1):
        # Build estimator
        est = build_estimator(combo.model_key, cfg["models"][combo.model_key], dict(combo.model_params), seed=combo.seed or 42)
        if est is None:
            # optional model not installed
            continue

        # Build preprocessing steps
        steps: List[Tuple[str, Any]] = []

        # Imputation
        use_imputer = (combo.imputer != "none")
        if use_imputer:
            steps.append(("imputer", SimpleImputer(strategy="median")))

        # Scaling
        scaler_obj = build_scaler(combo.scaler)
        if scaler_obj is not None:
            steps.append(("scaler", scaler_obj))

        # PCA
        if combo.pca == "on":
            steps.append(("pca", build_pca(combo.pca_params)))

        # Resampling
        use_resampling = (combo.resampling != "none")
        resampler_obj = None

        # Pipeline class
        PipelineClass = Pipeline

        if use_resampling:
            if not IMBLEARN_AVAILABLE:
                continue
            PipelineClass = ImbPipeline  # type: ignore

        # Validation splits
        # for train_test_split strategies, the "params_grid" carries stratify bool, but we map to correct iterator name
        vs_name = combo.val_strategy
        vs_params = dict(combo.val_params)

        # iterate splits
        try:
            split_iter = iter_splits(X, y, vs_name, vs_params, combo.seed)
        except Exception as e:
            print(f"[WARN] split iterator failed for combo {i}: {e}")
            continue

        split_count = 0
        for train_idx, test_idx, split_id in split_iter:
            split_count += 1
            X_train = X.iloc[train_idx].copy()
            y_train = y[train_idx]
            X_test = X.iloc[test_idx].copy()
            y_test = y[test_idx]

            # Build resampler (train-only; inside pipeline fit)
            if use_resampling:
                try:
                    resampler_obj = build_resampler(combo.resampling, combo.resampling_params, seed=(combo.seed or 42) + split_count)
                except Exception as e:
                    if res_on_failure == "fallback_to_none":
                        resampler_obj = None
                        PipelineClass = Pipeline
                    else:
                        # skip
                        continue

            # Build pipeline steps final
            steps_final = list(steps)
            if resampler_obj is not None:
                steps_final.append(("resample", resampler_obj))
            steps_final.append(("model", est))

            pipe = PipelineClass(steps_final)  # type: ignore

            # Fit / predict
            try:
                pipe.fit(X_train, y_train)
                y_prob = prob_from_estimator(pipe, X_test)
            except Exception as e:
                # fallback for resampling failure
                if use_resampling and res_on_failure == "fallback_to_none":
                    try:
                        # rebuild without resampling
                        steps_fb = list(steps) + [("model", est)]
                        pipe_fb = Pipeline(steps_fb)
                        pipe_fb.fit(X_train, y_train)
                        y_prob = prob_from_estimator(pipe_fb, X_test)
                    except Exception:
                        continue
                else:
                    continue

            for thr in thresholds:
                m = compute_metrics(y_test, y_prob, thr=float(thr))

                row = {
                    "run_id": run_id,
                    "combo_id": i,
                    "split_id": split_id,
                    "threshold": float(thr),
                    "val_strategy": combo.val_strategy,
                    "val_params": json_dumps_safe(combo.val_params),
                    "seed": combo.seed,
                    "model_key": combo.model_key,
                    "model_class": combo.model_class,
                    "model_params": json_dumps_safe(combo.model_params),
                    "imputer": combo.imputer,
                    "scaler": combo.scaler,
                    "pca": combo.pca,
                    "pca_params": json_dumps_safe(combo.pca_params),
                    "resampling": combo.resampling,
                    "resampling_params": json_dumps_safe(combo.resampling_params),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "pos_rate_test": float(np.mean(y_test)),
                }
                row.update(m)
                results_rows.append(row)

            if store_preds:
                for j in range(len(test_idx)):
                    pred_rows.append({
                        "run_id": run_id,
                        "combo_id": i,
                        "split_id": split_id,
                        "seed": combo.seed,
                        "val_strategy": combo.val_strategy,
                        "model_key": combo.model_key,
                        "imputer": combo.imputer,
                        "scaler": combo.scaler,
                        "pca": combo.pca,
                        "resampling": combo.resampling,
                        "y_true": int(y_test[j]),
                        "y_prob": float(y_prob[j]),
                    })

        if i % 50 == 0:
            print(f"[INFO] processed combos: {i}/{len(combos)} | rows={len(results_rows)}")

    # --- export ---
    results_df = pd.DataFrame(results_rows)
    results_path = outdir / "results.csv"
    results_df.to_csv(results_path, index=False)

    # summary
    group_cols = [
        "model_key", "model_class", "model_params",
        "imputer", "scaler", "pca", "pca_params",
        "resampling", "resampling_params",
        "val_strategy", "val_params", "threshold"
    ]

    agg_map = {m: ["mean", "std", "median"] for m in ["roc_auc", "pr_auc", "brier", "mcc", "balanced_accuracy", "f1"]}
    summary_df = results_df.groupby(group_cols).agg(agg_map).reset_index()
    # flatten columns
    summary_df.columns = ["__".join(c).strip("__") if isinstance(c, tuple) else c for c in summary_df.columns]

    summary_path = outdir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # predictions
    if store_preds and pred_rows:
        pred_df = pd.DataFrame(pred_rows)
        pred_path = outdir / "predictions.parquet"
        pred_df.to_parquet(pred_path, index=False)

    # run metadata
    meta = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "imblearn_available": IMBLEARN_AVAILABLE,
        "data_path": str(Path(args.data).resolve()),
        "config_path": str(Path(args.config).resolve()),
        "n_rows": int(len(df)),
        "pos_rate": float(np.mean(y)),
        "target": target,
        "predictors": predictors,
        "n_combos_attempted": int(len(combos)),
        "n_result_rows": int(len(results_df)),
        "outputs": {
            "results_csv": str(results_path),
            "summary_csv": str(summary_path),
            "predictions_parquet": str((outdir / "predictions.parquet")) if (store_preds and pred_rows) else None
        }
    }
    (outdir / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[DONE] results: {results_path}")
    print(f"[DONE] summary: {summary_path}")
    if store_preds and pred_rows:
        print(f"[DONE] predictions: {outdir / 'predictions.parquet'}")
    print(f"[DONE] metadata: {outdir / 'run_metadata.json'}")


if __name__ == "__main__":
    main()
