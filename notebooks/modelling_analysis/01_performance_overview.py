from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns

pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 200)
sns.set_context("talk")
plt.rcParams["figure.dpi"] = 140


METRIC_COLUMNS = [
    "roc_auc",
    "pr_auc",
    "mcc",
    "brier",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
]

COUNT_COLUMNS = ["tn", "fp", "fn", "tp", "n_train", "n_test", "pos_rate_test"]

GROUP_ID_COLUMNS = [
    "dataset_variant",
    "config_name",
    "run_id",
    "combo_id",
    "val_strategy",
    "seed",
    "threshold",
    "model_key",
    "model_class",
    "imputer",
    "scaler",
    "pca",
    "pca_params",
    "resampling",
    "resampling_params",
    "model_params",
    "task_id",
]

ANALYSIS_COLUMNS = [
    "run_id",
    "combo_id",
    "split_id",
    "threshold",
    "val_strategy",
    "seed",
    "model_key",
    "model_class",
    "model_params",
    "imputer",
    "scaler",
    "pca",
    "pca_params",
    "resampling",
    "resampling_params",
    "n_train",
    "n_test",
    "pos_rate_test",
    *METRIC_COLUMNS,
    *COUNT_COLUMNS[:4],
]

SEED_GROUP_COLS = [
    "dataset_variant",
    "config_name",
    "run_id",
    "combo_id",
    "val_strategy",
    "seed",
    "threshold",
    "model_key",
    "model_class",
    "model_params",
    "imputer",
    "scaler",
    "pca",
    "pca_params",
    "resampling",
    "resampling_params",
]

CONFIG_GROUP_COLS = [
    "dataset_variant",
    "config_name",
    "val_strategy",
    "threshold",
    "model_key",
    "model_class",
    "model_params",
    "imputer",
    "scaler",
    "pca",
    "pca_params",
    "resampling",
    "resampling_params",
]

DISPLAY_COLUMNS = [
    "dataset_variant",
    "config_name",
    "model_key",
    "imputer",
    "scaler",
    "pca",
    "pca_params",
    "resampling",
    "resampling_params",
    "n_seed_runs",
    "accuracy_mean",
    "accuracy_std",
    "precision_mean",
    "precision_std",
    "recall_mean",
    "recall_std",
    "f1_mean",
    "f1_std",
    "balanced_accuracy_mean",
    "balanced_accuracy_std",
    "mcc_mean",
    "mcc_std",
    "classical_composite_rank",
]


def find_repo_root(
    start: Path | None = None,
    repo_name: str = "hypercholesterolemia_classifiers",
) -> Path:
    candidates: list[Path] = []
    if start is not None:
        candidates.append(start.resolve())
    candidates.extend(
        [
            Path.cwd().resolve(),
            Path("/home/dmedina/Desktop/colabs/hypercholesterolemia_classifiers/"),
        ]
    )
    for candidate in candidates:
        if candidate.name == repo_name and candidate.exists():
            return candidate
        for parent in [candidate, *candidate.parents]:
            if parent.name == repo_name and parent.exists():
                return parent
    raise FileNotFoundError("Could not locate the repository root.")


def save_figure(fig: plt.Figure, figures_dir: Path, filename: str, dpi: int = 300) -> Path:
    out_path = figures_dir / filename
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return out_path


def parse_result_path(path: Path, runs_root: Path) -> dict:
    rel = path.relative_to(runs_root)
    task_match = re.search(r"results_task(\d+)_part(\d+)\.parquet$", path.name)
    return {
        "dataset_variant": rel.parts[0],
        "config_name": rel.parts[1],
        "run_id": rel.parts[2].replace("run_", ""),
        "task_id": int(task_match.group(1)) if task_match else None,
        "part_id": int(task_match.group(2)) if task_match else None,
        "file_path": str(path),
    }


def load_metadata_index(runs_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for meta_path in runs_root.glob("*/*/run_*/*.json"):
        if not meta_path.name.startswith("run_metadata_task"):
            continue
        rel = meta_path.relative_to(runs_root)
        dataset_variant, config_name, run_folder = rel.parts[:3]
        task_match = re.search(r"run_metadata_task(\d+)\.json$", meta_path.name)
        task_id = int(task_match.group(1)) if task_match else None
        with open(meta_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        rows.append(
            {
                "dataset_variant": dataset_variant,
                "config_name": config_name,
                "run_id": run_folder.replace("run_", ""),
                "task_id": task_id,
                "n_rows_dataset": data.get("n_rows"),
                "pos_rate_dataset": data.get("pos_rate"),
                "target": data.get("target"),
                "n_predictors": data.get("n_predictors"),
                "n_combos_total": data.get("n_combos_total"),
                "n_combos_shard": data.get("n_combos_shard"),
                "data_path": data.get("data_path"),
                "config_path": data.get("config_path"),
            }
        )
    meta = pd.DataFrame(rows)
    if not meta.empty:
        meta = meta.sort_values(
            ["dataset_variant", "config_name", "run_id", "task_id"]
        ).reset_index(drop=True)
    return meta


def read_result_file(path: Path, runs_root: Path, columns: list[str] | None = None) -> pd.DataFrame:
    parquet_file = pq.ParquetFile(path)
    available_columns = parquet_file.schema.names

    if columns is not None:
        requested_columns = columns
        columns = [col for col in columns if col in available_columns]
        missing = sorted(set(requested_columns) - set(columns))
        if missing:
            print(f"[warn] {path.name}: skipping missing columns {missing}")

    table = parquet_file.read(columns=columns)
    df = table.to_pandas()
    meta = parse_result_path(path, runs_root)
    for key, value in meta.items():
        df[key] = value
    return df


def compute_classical_metrics_from_counts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    tp = out["tp"].astype(float)
    tn = out["tn"].astype(float)
    fp = out["fp"].astype(float)
    fn = out["fn"].astype(float)

    out["accuracy"] = (tp + tn) / (tp + tn + fp + fn).replace(0, np.nan)
    tpr = tp / (tp + fn).replace(0, np.nan)
    tnr = tn / (tn + fp).replace(0, np.nan)
    out["balanced_accuracy"] = (tpr + tnr) / 2.0
    out["precision"] = tp / (tp + fp).replace(0, np.nan)
    out["recall"] = tpr
    out["f1"] = (2 * tp) / (2 * tp + fp + fn).replace(0, np.nan)

    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    out["mcc"] = ((tp * tn) - (fp * fn)) / pd.Series(mcc_den).replace(0, np.nan)
    return out


def plot_metric_distributions(
    df: pd.DataFrame,
    metrics: list[str],
    by: str = "dataset_variant",
    bins: int = 40,
    figsize: tuple[int, int] = (16, 12),
) -> plt.Figure:
    fig, axes = plt.subplots(math.ceil(len(metrics) / 2), 2, figsize=figsize)
    axes = np.array(axes).reshape(-1)
    for ax, metric in zip(axes, metrics):
        plot_df = df.loc[df[metric].notna()].copy()
        sns.histplot(
            data=plot_df,
            x=metric,
            hue=by,
            bins=bins,
            stat="density",
            common_norm=False,
            alpha=0.35,
            ax=ax,
        )
        ax.set_title(f"Distribution of {metric}")
        ax.grid(alpha=0.2)
    for ax in axes[len(metrics):]:
        ax.axis("off")
    plt.tight_layout()
    return fig


def main() -> None:
    repo_root = find_repo_root()
    runs_root = repo_root / "notebooks" / "modelling" / "runs"
    analysis_dir = repo_root / "notebooks" / "modelling" / "analysis_tables"
    figures_dir = repo_root / "notebooks" / "modelling" / "analysis_figures" / "01_performance_overview"

    analysis_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Repository root:", repo_root)
    print("Runs root:", runs_root)
    print("Analysis tables:", analysis_dir)
    print("Figures dir:", figures_dir)

    metadata_index = load_metadata_index(runs_root)
    print(f"Loaded metadata entries: {len(metadata_index):,}")

    result_files = sorted(runs_root.glob("*/*/run_*/*.parquet"))
    print(f"Detected {len(result_files):,} parquet result files.")
    if not result_files:
        raise FileNotFoundError(f"No parquet files found under {runs_root}")

    files_index = pd.DataFrame(parse_result_path(path, runs_root) for path in result_files)
    print("Files index preview:")
    print(files_index.head())

    sample_df = read_result_file(
        result_files[0],
        runs_root=runs_root,
        columns=GROUP_ID_COLUMNS[3:] + METRIC_COLUMNS + COUNT_COLUMNS,
    )
    print("Sample dataframe shape:", sample_df.shape)
    print(sample_df.head())

    raw_frames: list[pd.DataFrame] = []
    for idx, path in enumerate(result_files, start=1):
        df_part = read_result_file(path, runs_root=runs_root, columns=ANALYSIS_COLUMNS)
        raw_frames.append(df_part)
        if idx % 250 == 0 or idx == len(result_files):
            print(f"Loaded {idx:,}/{len(result_files):,} files")

    results_raw = pd.concat(raw_frames, ignore_index=True)
    del raw_frames

    results_raw = results_raw.merge(
        metadata_index,
        how="left",
        on=["dataset_variant", "config_name", "run_id", "task_id"],
        validate="many_to_one",
    )

    print("results_raw shape:", results_raw.shape)
    print(results_raw.head())

    mean_metric_agg = {metric: "mean" for metric in METRIC_COLUMNS}
    mean_metric_agg.update(
        {
            "n_train": "mean",
            "n_test": "mean",
            "pos_rate_test": "mean",
            "split_id": "nunique",
            "tn": "sum",
            "fp": "sum",
            "fn": "sum",
            "tp": "sum",
        }
    )

    results_seed_level_nonloo = (
        results_raw.query("config_name != 'config_loo'")
        .groupby(SEED_GROUP_COLS, dropna=False, as_index=False)
        .agg(mean_metric_agg)
        .rename(columns={"split_id": "n_evaluation_units"})
    )

    results_seed_level_loo = (
        results_raw.query("config_name == 'config_loo'")
        .groupby(SEED_GROUP_COLS, dropna=False, as_index=False)
        .agg(
            {
                "tn": "sum",
                "fp": "sum",
                "fn": "sum",
                "tp": "sum",
                "n_train": "mean",
                "n_test": "sum",
                "pos_rate_test": "mean",
                "split_id": "nunique",
            }
        )
        .rename(columns={"split_id": "n_evaluation_units"})
    )

    if not results_seed_level_loo.empty:
        results_seed_level_loo = compute_classical_metrics_from_counts(results_seed_level_loo)
        results_seed_level_loo["roc_auc"] = np.nan
        results_seed_level_loo["pr_auc"] = np.nan
        results_seed_level_loo["brier"] = np.nan
        results_seed_level_loo["loo_metric_source"] = "reconstructed_from_confusion_counts"
    else:
        results_seed_level_loo["loo_metric_source"] = pd.Series(dtype="object")

    results_seed_level_nonloo["loo_metric_source"] = "reported_fold_mean"

    results_seed_level = pd.concat(
        [results_seed_level_nonloo, results_seed_level_loo],
        ignore_index=True,
        sort=False,
    )

    preferred_cols = SEED_GROUP_COLS + [
        "n_evaluation_units",
        "n_train",
        "n_test",
        "pos_rate_test",
        "tn",
        "fp",
        "fn",
        "tp",
        *METRIC_COLUMNS,
        "loo_metric_source",
    ]
    results_seed_level = results_seed_level.loc[
        :, [c for c in preferred_cols if c in results_seed_level.columns]
    ]

    print("results_seed_level shape:", results_seed_level.shape)
    print(results_seed_level.head())

    run_summary = (
        results_raw[["dataset_variant", "config_name", "run_id", "task_id", "combo_id"]]
        .drop_duplicates()
        .groupby(["dataset_variant", "config_name", "run_id"], as_index=False)
        .agg(
            n_tasks=("task_id", "nunique"),
            n_unique_combos=("combo_id", "nunique"),
        )
        .merge(
            metadata_index.groupby(
                ["dataset_variant", "config_name", "run_id"], as_index=False
            ).agg(
                n_rows_dataset=("n_rows_dataset", "max"),
                pos_rate_dataset=("pos_rate_dataset", "max"),
                n_predictors=("n_predictors", "max"),
                target=("target", "first"),
            ),
            on=["dataset_variant", "config_name", "run_id"],
            how="left",
        )
        .sort_values(["dataset_variant", "config_name", "run_id"])
    )
    print("run_summary preview:")
    print(run_summary.head())

    fig = plot_metric_distributions(
        results_seed_level,
        metrics=["roc_auc", "pr_auc", "mcc", "brier", "balanced_accuracy", "f1"],
        by="dataset_variant",
        bins=50,
    )
    save_figure(fig, figures_dir, "metric_distributions_seed_level.png")
    plt.close(fig)

    metric_coverage = results_seed_level.groupby("config_name")[METRIC_COLUMNS].agg(
        lambda s: s.notna().sum()
    )
    metric_coverage_pct = results_seed_level.groupby("config_name")[METRIC_COLUMNS].agg(
        lambda s: s.notna().mean()
    )

    print("Metric coverage counts:")
    print(metric_coverage)
    print("Metric coverage fraction:")
    print(metric_coverage_pct)

    comparison_metrics = ["accuracy", "precision", "recall", "mcc", "balanced_accuracy", "f1"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    handles, labels = None, None
    for ax, metric in zip(axes.ravel(), comparison_metrics):
        plot_df = results_seed_level.loc[results_seed_level[metric].notna()].copy()
        sns.boxplot(
            data=plot_df,
            x="config_name",
            y=metric,
            hue="dataset_variant",
            ax=ax,
            showfliers=False,
        )
        ax.set_title(f"{metric} by validation setup")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.2)
        handles, labels = ax.get_legend_handles_labels()
        if ax.legend_ is not None:
            ax.legend_.remove()
    if handles and labels:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, figures_dir, "classical_metrics_by_validation_setup.png")
    plt.close(fig)

    metric_overview = (
        results_seed_level.groupby(["dataset_variant", "config_name"], as_index=False)
        .agg(
            n_config_seed_runs=("combo_id", "count"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            mcc_mean=("mcc", "mean"),
            mcc_std=("mcc", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            brier_mean=("brier", "mean"),
            brier_std=("brier", "std"),
        )
        .sort_values(["dataset_variant", "config_name"])
    )
    print("metric_overview preview:")
    print(metric_overview.head())

    results_config_level = (
        results_seed_level.groupby(CONFIG_GROUP_COLS, dropna=False, as_index=False)
        .agg(
            n_seed_runs=("seed", "nunique"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            mcc_mean=("mcc", "mean"),
            mcc_std=("mcc", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            brier_mean=("brier", "mean"),
            brier_std=("brier", "std"),
            mean_n_eval_units=("n_evaluation_units", "mean"),
        )
    )

    for metric in ["accuracy", "precision", "recall", "f1", "balanced_accuracy", "mcc"]:
        results_config_level[f"rank_{metric}"] = results_config_level[f"{metric}_mean"].rank(
            ascending=False,
            method="dense",
        )

    results_config_level["classical_composite_rank"] = (
        results_config_level["rank_accuracy"]
        + results_config_level["rank_precision"]
        + results_config_level["rank_recall"]
        + results_config_level["rank_f1"]
        + results_config_level["rank_balanced_accuracy"]
        + results_config_level["rank_mcc"]
    )

    top_configs = (
        results_config_level.sort_values(
            ["classical_composite_rank", "mcc_mean", "balanced_accuracy_mean", "f1_mean"],
            ascending=[True, False, False, False],
        ).reset_index(drop=True)
    )

    print("Top configurations:")
    print(top_configs.loc[:, DISPLAY_COLUMNS].head(20))

    top20 = top_configs.head(20).copy()
    top20["label"] = (
        top20["dataset_variant"]
        + " | "
        + top20["config_name"].str.replace("config_", "", regex=False)
        + " | "
        + top20["model_key"]
    )

    fig, axes = plt.subplots(1, 3, figsize=(22, 10))
    sns.barplot(data=top20.sort_values("mcc_mean", ascending=False), y="label", x="mcc_mean", ax=axes[0])
    axes[0].set_title("Top 20 by classical ranking: MCC")
    axes[0].grid(alpha=0.2)

    sns.barplot(
        data=top20.sort_values("balanced_accuracy_mean", ascending=False),
        y="label",
        x="balanced_accuracy_mean",
        ax=axes[1],
    )
    axes[1].set_title("Top 20 by classical ranking: balanced accuracy")
    axes[1].grid(alpha=0.2)

    sns.barplot(data=top20.sort_values("f1_mean", ascending=False), y="label", x="f1_mean", ax=axes[2])
    axes[2].set_title("Top 20 by classical ranking: F1")
    axes[2].grid(alpha=0.2)

    plt.tight_layout()
    save_figure(fig, figures_dir, "top20_classical_ranking.png")
    plt.close(fig)

    stability_df = results_config_level.copy()
    stability_df["performance_instability_ratio"] = (
        stability_df["mcc_std"] / stability_df["mcc_mean"].replace(0, np.nan)
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.scatterplot(
        data=stability_df,
        x="mcc_mean",
        y="mcc_std",
        hue="dataset_variant",
        style="config_name",
        alpha=0.75,
        ax=axes[0],
    )
    axes[0].set_title("MCC mean vs MCC standard deviation")
    axes[0].grid(alpha=0.2)

    sns.scatterplot(
        data=stability_df,
        x="balanced_accuracy_mean",
        y="balanced_accuracy_std",
        hue="dataset_variant",
        style="config_name",
        alpha=0.75,
        ax=axes[1],
    )
    axes[1].set_title("Balanced accuracy mean vs standard deviation")
    axes[1].grid(alpha=0.2)

    plt.tight_layout()
    save_figure(fig, figures_dir, "stability_classical_metrics.png")
    plt.close(fig)

    most_stable_high_perf = (
        stability_df.query("mcc_mean >= mcc_mean.quantile(0.90)")
        .sort_values(
            ["mcc_std", "balanced_accuracy_std", "f1_std", "mcc_mean"],
            ascending=[True, True, True, False],
        )
    )
    print("Most stable high-performance configurations:")
    print(most_stable_high_perf.loc[:, DISPLAY_COLUMNS].head(20))

    seed_variability_by_model = (
        results_seed_level.groupby(["dataset_variant", "config_name", "model_key"], as_index=False)
        .agg(
            mcc_mean=("mcc", "mean"),
            mcc_std=("mcc", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            n_rows=("combo_id", "count"),
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    sns.boxplot(
        data=seed_variability_by_model,
        y="model_key",
        x="mcc_std",
        hue="dataset_variant",
        ax=axes[0],
        showfliers=False,
    )
    axes[0].set_title("Seed-level variability in MCC by model")
    axes[0].grid(alpha=0.2)

    sns.boxplot(
        data=seed_variability_by_model,
        y="model_key",
        x="balanced_accuracy_std",
        hue="dataset_variant",
        ax=axes[1],
        showfliers=False,
    )
    axes[1].set_title("Seed-level variability in balanced accuracy by model")
    axes[1].grid(alpha=0.2)

    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        if ax.legend_ is not None:
            ax.legend_.remove()
    if handles and labels:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, figures_dir, "seed_variability_by_model_classical.png")
    plt.close(fig)

    results_seed_level.to_csv(analysis_dir / "results_seed_level.csv", index=False)
    results_config_level.to_csv(analysis_dir / "results_config_level.csv", index=False)
    run_summary.to_csv(analysis_dir / "run_summary.csv", index=False)
    metric_overview.to_csv(analysis_dir / "metric_overview.csv", index=False)
    metric_coverage.to_csv(analysis_dir / "metric_coverage_counts.csv")
    metric_coverage_pct.to_csv(analysis_dir / "metric_coverage_fraction.csv")

    print("Saved analysis tables to:", analysis_dir)
    print("Saved figures to:", figures_dir)
    print("Generated figures:")
    for path in sorted(figures_dir.glob("*.png")):
        print(" -", path.name)


if __name__ == "__main__":
    main()
