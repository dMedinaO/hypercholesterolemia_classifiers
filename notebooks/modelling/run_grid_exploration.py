#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.base import BaseEstimator, clone
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, StandardScaler

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# -------------------------
# Optional imblearn
# -------------------------
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
# Utils
# -------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def now_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def json_dumps_safe(x: Any) -> str:
    return json.dumps(x, sort_keys=True, default=str)


def expand_param_grid(grid: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expand a param_grid dict into a list of dicts.
    Accepts values as list or scalar.
    """
    if not grid:
        return [{}]
    keys = list(grid.keys())
    values: List[List[Any]] = []
    for k in keys:
        v = grid[k]
        values.append(v if isinstance(v, list) else [v])
    return [{keys[i]: vals[i] for i in range(len(keys))} for vals in product(*values)]


def shard_indices(n_items: int, task_id: int, n_tasks: int) -> np.ndarray:
    """Deterministic sharding: indices where idx % n_tasks == task_id."""
    idx = np.arange(n_items)
    return idx[idx % n_tasks == task_id]


# -------------------------
# Preprocessing
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
    return PCA(**params)


# -------------------------
# Resampling
# -------------------------
def build_resampler(strategy: str, params: Dict[str, Any], seed: int):
    if strategy == "none":
        return None
    if not IMBLEARN_AVAILABLE:
        raise RuntimeError("imblearn not available")

    if strategy == "undersample":
        return RandomUnderSampler(random_state=seed, **params)

    if strategy == "smote":
        return SMOTE(random_state=seed, **params)

    if strategy == "smoteenn":
        smote_k = params.get("smote_k_neighbors", 5)
        sampling_strategy = params.get("sampling_strategy", 1.0)
        sm = SMOTE(random_state=seed, k_neighbors=smote_k, sampling_strategy=sampling_strategy)
        return SMOTEENN(random_state=seed, smote=sm)

    raise ValueError(f"Unknown resampling strategy: {strategy}")


# -------------------------
# Models
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


def _override_n_jobs_if_possible(params: Dict[str, Any], n_jobs_model: Optional[int]) -> Dict[str, Any]:
    if n_jobs_model is None:
        return params
    if "n_jobs" in params:
        params["n_jobs"] = int(n_jobs_model)
    return params


def build_estimator(
    model_key: str,
    model_cfg: Dict[str, Any],
    params: Dict[str, Any],
    seed: int,
    n_jobs_model: Optional[int] = None,
) -> Optional[BaseEstimator]:
    """
    Build estimator from config entry.
    Supports sklearn models + optional XGB/LGBM if installed.

    IMPORTANT:
    - Forces n_jobs if supported to avoid oversubscription.
    - Sets random_state when supported.
    """
    class_name = model_cfg.get("class_name", model_key)

    # Optional XGB/LGBM
    if class_name == "XGBClassifier":
        XGB = _optional_import_xgb()
        if XGB is None:
            return None
        params = _override_n_jobs_if_possible(params, n_jobs_model)
        params.setdefault("random_state", seed)
        init_params = dict(model_cfg.get("init_params", {}))
        if n_jobs_model is not None:
            init_params["n_jobs"] = int(n_jobs_model)
        elif init_params.get("n_jobs", None) == -1:
            init_params["n_jobs"] = 1
        return XGB(**{**init_params, **params})

    if class_name == "LGBMClassifier":
        LGBM = _optional_import_lgbm()
        if LGBM is None:
            return None
        params = _override_n_jobs_if_possible(params, n_jobs_model)
        params.setdefault("random_state", seed)
        init_params = dict(model_cfg.get("init_params", {}))
        if n_jobs_model is not None:
            init_params["n_jobs"] = int(n_jobs_model)
        elif init_params.get("n_jobs", None) == -1:
            init_params["n_jobs"] = 1
        return LGBM(**{**init_params, **params})

    if class_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown/unsupported model class_name={class_name} for key={model_key}")

    cls = MODEL_REGISTRY[class_name]
    init_params = dict(model_cfg.get("init_params", {}))

    # random_state if supported
    try:
        base_params = cls().get_params()
    except Exception:
        base_params = {}

    if "random_state" in base_params:
        init_params.setdefault("random_state", seed)

    # n_jobs if supported (HARD FIX against n_jobs=-1)
    if "n_jobs" in base_params:
        if n_jobs_model is not None:
            init_params["n_jobs"] = int(n_jobs_model)
        else:
            if init_params.get("n_jobs", None) == -1:
                init_params["n_jobs"] = 1

    params = _override_n_jobs_if_possible(params, n_jobs_model)
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

    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
        out["pr_auc"] = average_precision_score(y_true, y_prob)
        out["mcc"] = matthews_corrcoef(y_true, y_pred)
    else:
        out["roc_auc"] = np.nan
        out["pr_auc"] = np.nan
        out["mcc"] = np.nan

    out["brier"] = brier_score_loss(y_true, y_prob)
    out["accuracy"] = accuracy_score(y_true, y_pred)
    out["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    out["precision"] = precision_score(y_true, y_pred, zero_division=0)
    out["recall"] = recall_score(y_true, y_pred, zero_division=0)
    out["f1"] = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["tn"] = int(tn)
    out["fp"] = int(fp)
    out["fn"] = int(fn)
    out["tp"] = int(tp)
    return out


# -------------------------
# Validation
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
# Combo specification
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


# -------------------------
# Streaming parquet writer
# -------------------------
class ParquetAppender:
    """
    Append rows to Parquet efficiently by writing multiple part files.

    This avoids reading the entire existing parquet for "append".
    At the end you can concatenate all parts.
    """
    def __init__(self, outdir: Path, prefix: str, task_id: int):
        self.outdir = outdir
        self.prefix = prefix
        self.task_id = task_id
        self.part_idx = 0
        ensure_dir(outdir)

    def write_part(self, rows: List[Dict[str, Any]]) -> Optional[Path]:
        if not rows:
            return None
        df = pd.DataFrame(rows)
        out = self.outdir / f"{self.prefix}_task{self.task_id:04d}_part{self.part_idx:04d}.parquet"
        df.to_parquet(out, index=False)  # requires pyarrow or fastparquet
        self.part_idx += 1
        return out


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Parallelizable grid exploration with SLURM array sharding.")

    ap.add_argument("--data", required=True, help="CSV dataset path.")
    ap.add_argument("--config", required=True, help="JSON config path.")
    ap.add_argument("--outdir", required=True, help="Output directory root.")
    ap.add_argument("--target", default=None, help="Override target column (else config or default MSPH).")
    ap.add_argument("--predictors", default=None, help="Comma-separated predictors list; else infer numeric cols.")
    ap.add_argument("--id-cols", default="row_id,_sheet,LocalID", help="Comma-separated ID columns to exclude.")
    ap.add_argument("--limit-combos", type=int, default=None, help="Limit number of combos (debug).")

    # Sharding (SLURM array)
    ap.add_argument("--task-id", type=int, default=0, help="0-based shard id (SLURM_ARRAY_TASK_ID).")
    ap.add_argument("--n-tasks", type=int, default=1, help="Total shards (SLURM_ARRAY_TASK_COUNT).")

    # Execution controls
    ap.add_argument("--n-jobs-model", type=int, default=1, help="Force n_jobs for models (prevents oversubscription).")
    ap.add_argument("--only-validation", default=None, help="Run only one validation strategy (e.g., kfold, loo).")
    ap.add_argument("--store-fold-predictions", action="store_true", help="Store fold predictions (can be huge).")

    # streaming / memory
    ap.add_argument("--write-every", type=int, default=2000, help="Flush buffers every N metric rows.")
    ap.add_argument("--pred-write-every", type=int, default=5000, help="Flush prediction rows every N rows.")

    args = ap.parse_args()

    run_id = now_run_id()
    outdir = Path(args.outdir) / f"run_{run_id}"
    ensure_dir(outdir)

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    df = pd.read_csv(args.data)

    # ---- target/predictors ----
    target = args.target or cfg.get("target", "MSPH")
    id_cols = [c.strip() for c in args.id_cols.split(",") if c.strip()]

    if args.predictors:
        predictors = [c.strip() for c in args.predictors.split(",") if c.strip()]
    else:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        predictors = [c for c in numeric_cols if c not in id_cols and c != target]

    # cast
    df = df.copy()
    df[target] = pd.to_numeric(df[target], errors="coerce").astype("Int64")
    for c in predictors:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[df[target].isin([0, 1])].reset_index(drop=True)
    df[target] = df[target].astype(int)

    X = df[predictors]
    y = df[target].to_numpy()

    print(f"[INFO] data={Path(args.data).name} shape={df.shape} pos_rate={y.mean():.3f}")
    print(f"[INFO] predictors={len(predictors)} target={target}")
    print(f"[INFO] task={args.task_id}/{args.n_tasks} n_jobs_model={args.n_jobs_model} store_preds={args.store_fold_predictions}")
    print(f"[INFO] imblearn_available={IMBLEARN_AVAILABLE}")

    # ---- build model entries ----
    model_entries: List[Tuple[str, str, Dict[str, Any], Dict[str, Any]]] = []
    for model_key, model_cfg in cfg.get("models", {}).items():
        if not model_cfg.get("enabled", True):
            continue
        class_name = model_cfg.get("class_name", model_key)
        grid = expand_param_grid(model_cfg.get("param_grid", {}))
        for mp in grid:
            model_entries.append((model_key, class_name, model_cfg, mp))

    # ---- preprocessing grids ----
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

    # ---- resampling ----
    res_block = cfg.get("resampling", {})
    res_strats = res_block.get("strategies", [{"name": "none"}])
    res_constraints = res_block.get("constraints", {})
    res_on_failure = res_constraints.get("on_failure", "fallback_to_none")

    # ---- validation ----
    val = cfg.get("validation", {})
    val_strats = val.get("strategies", [])
    if args.only_validation is not None:
        val_strats = [v for v in val_strats if v.get("name") == args.only_validation]
        if not val_strats:
            raise ValueError(f"--only-validation={args.only_validation} not found in config validation.strategies")

    seed_grid = val.get("seed_grid", {})
    seeds = seed_grid.get("seeds", [])
    if seed_grid.get("enabled", False) and not seeds:
        raise ValueError("seed_grid.enabled=true but seed_grid.seeds is empty")

    reporting = val.get("reporting", {})
    thresholds = reporting.get("thresholds", {}).get("values", [0.5])

    # ---- build combos ----
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
                        if pca_name == "on":
                            if pca_apply_only_if_scaled and sc_name == "none":
                                continue
                            if pca_apply_only_if_scaled and sc_name not in pca_scalers_allowed:
                                continue
                            if model_class in pca_forbidden_models and pca_behavior == "skip":
                                continue

                        for res in res_strats:
                            res_name = res["name"]
                            res_param_grid = expand_param_grid(res.get("params_grid", {})) if res_name != "none" else [{}]
                            for res_params in res_param_grid:
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
                                                    model_params=dict(model_params),
                                                    imputer=imp_name,
                                                    scaler=sc_name,
                                                    pca=pca_name,
                                                    pca_params=dict(pca_params),
                                                    resampling=res_name,
                                                    resampling_params=dict(res_params),
                                                    val_strategy=vs_name,
                                                    val_params=dict(vs_params),
                                                    seed=int(seed) if seed is not None else None,
                                                )
                                            )

    if args.limit_combos is not None:
        combos = combos[: args.limit_combos]

    all_n = len(combos)
    shard_idx = shard_indices(all_n, args.task_id, args.n_tasks)
    combos_shard = [combos[i] for i in shard_idx.tolist()]

    print(f"[INFO] total_combos={all_n} shard_size={len(combos_shard)} (task {args.task_id}/{args.n_tasks})")

    # ---- streaming writers ----
    results_writer = ParquetAppender(outdir, "results", args.task_id)
    preds_writer = ParquetAppender(outdir, "preds", args.task_id)

    results_buf: List[Dict[str, Any]] = []
    preds_buf: List[Dict[str, Any]] = []

    def flush_results():
        nonlocal results_buf
        p = results_writer.write_part(results_buf)
        results_buf = []
        if p is not None:
            print(f"[WRITE] {p.name}")

    def flush_preds():
        nonlocal preds_buf
        p = preds_writer.write_part(preds_buf)
        preds_buf = []
        if p is not None:
            print(f"[WRITE] {p.name}")

    # ---- run combos ----
    for local_i, combo in enumerate(combos_shard, start=1):
        global_combo_id = int(shard_idx[local_i - 1]) + 1  # 1-based id (stable)

        est = build_estimator(
            combo.model_key,
            cfg["models"][combo.model_key],
            dict(combo.model_params),
            seed=combo.seed or 42,
            n_jobs_model=args.n_jobs_model,
        )
        if est is None:
            continue

        # preprocessing steps
        base_steps: List[Tuple[str, Any]] = []
        if combo.imputer != "none":
            base_steps.append(("imputer", SimpleImputer(strategy="median")))

        sc_obj = build_scaler(combo.scaler)
        if sc_obj is not None:
            base_steps.append(("scaler", sc_obj))

        if combo.pca == "on":
            base_steps.append(("pca", build_pca(combo.pca_params)))

        use_resampling = (combo.resampling != "none")
        PipelineClass = Pipeline if not use_resampling else ImbPipeline  # type: ignore

        try:
            split_iter = iter_splits(X, y, combo.val_strategy, dict(combo.val_params), combo.seed)
        except Exception as e:
            print(f"[WARN] split iterator failed combo_id={global_combo_id}: {e}")
            continue

        split_count = 0
        for train_idx, test_idx, split_id in split_iter:
            split_count += 1

            X_train = X.iloc[train_idx].copy()
            y_train = y[train_idx]
            X_test = X.iloc[test_idx].copy()
            y_test = y[test_idx]

            resampler_obj = None
            if use_resampling:
                try:
                    resampler_obj = build_resampler(
                        combo.resampling,
                        combo.resampling_params,
                        seed=(combo.seed or 42) + split_count,
                    )
                except Exception:
                    if res_on_failure == "fallback_to_none":
                        resampler_obj = None
                        PipelineClass = Pipeline
                    else:
                        continue

            # clone estimator per split
            est_split = clone(est)

            steps_final = list(base_steps)
            if resampler_obj is not None:
                steps_final.append(("resample", resampler_obj))
            steps_final.append(("model", est_split))

            pipe = PipelineClass(steps_final)  # type: ignore

            try:
                pipe.fit(X_train, y_train)
                y_prob = prob_from_estimator(pipe, X_test)
            except Exception:
                # fallback without resampling if allowed
                if use_resampling and res_on_failure == "fallback_to_none":
                    try:
                        est_fb = clone(est)
                        pipe_fb = Pipeline(list(base_steps) + [("model", est_fb)])
                        pipe_fb.fit(X_train, y_train)
                        y_prob = prob_from_estimator(pipe_fb, X_test)
                    except Exception:
                        continue
                else:
                    continue

            # metrics per threshold
            for thr in thresholds:
                m = compute_metrics(y_test, y_prob, thr=float(thr))
                row = {
                    "run_id": run_id,
                    "combo_id": global_combo_id,
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
                results_buf.append(row)

            # optional predictions
            if args.store_fold_predictions:
                for j in range(len(test_idx)):
                    preds_buf.append(
                        {
                            "run_id": run_id,
                            "combo_id": global_combo_id,
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
                        }
                    )

            # flush buffers
            if len(results_buf) >= args.write_every:
                flush_results()
            if args.store_fold_predictions and len(preds_buf) >= args.pred_write_every:
                flush_preds()

        # combo cleanup
        gc.collect()

        if local_i % 10 == 0:
            print(f"[INFO] task={args.task_id} processed {local_i}/{len(combos_shard)} combos")

    # final flush
    flush_results()
    if args.store_fold_predictions:
        flush_preds()

    # metadata per task
    meta = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "task_id": args.task_id,
        "n_tasks": args.n_tasks,
        "n_jobs_model": args.n_jobs_model,
        "store_fold_predictions": bool(args.store_fold_predictions),
        "data_path": str(Path(args.data).resolve()),
        "config_path": str(Path(args.config).resolve()),
        "n_rows": int(len(df)),
        "pos_rate": float(np.mean(y)),
        "target": target,
        "n_predictors": int(len(predictors)),
        "only_validation": args.only_validation,
        "n_combos_total": int(all_n),
        "n_combos_shard": int(len(combos_shard)),
        "outputs": {"outdir": str(outdir)},
    }
    (outdir / f"run_metadata_task{args.task_id:04d}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[DONE] task={args.task_id} finished. outdir={outdir}")


if __name__ == "__main__":
    main()
