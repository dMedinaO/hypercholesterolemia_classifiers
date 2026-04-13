from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")
pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 200)


CLASSICAL_METRICS = ["accuracy", "precision", "recall", "f1", "balanced_accuracy", "mcc"]
PROB_METRICS = ["roc_auc", "pr_auc", "brier"]
FACTOR_COLUMNS = [
    "dataset_variant",
    "config_name",
    "val_strategy",
    "model_key",
    "model_class",
    "imputer",
    "scaler",
    "pca",
    "resampling",
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

    fallback = Path("/mnt/data/hyperchol_repo")
    if fallback.exists():
        return fallback
    raise FileNotFoundError("Could not locate the repository root.")


def save_figure(fig: plt.Figure, out_dir: Path, filename: str, dpi: int = 300) -> Path:
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def print_section(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def print_dataframe(df: pd.DataFrame, name: str, n: int = 20) -> None:
    print_section(name)
    if df.empty:
        print("<empty dataframe>")
        return
    print(df.head(n).to_string(index=False))
    if len(df) > n:
        print(f"\n... showing first {n} of {len(df)} rows")


def summarize_factor_effects(
    data: pd.DataFrame,
    factor: str,
    metrics: list[str],
    min_count: int = 1,
) -> pd.DataFrame:
    agg_dict: dict[str, tuple[str, str]] = {}
    for metric in metrics:
        agg_dict[f"{metric}_mean"] = (metric, "mean")
        agg_dict[f"{metric}_std"] = (metric, "std")
        agg_dict[f"{metric}_median"] = (metric, "median")
        agg_dict[f"{metric}_count"] = (metric, "count")

    summary = data.groupby(factor, dropna=False).agg(**agg_dict).reset_index()

    count_cols = [f"{metric}_count" for metric in metrics if f"{metric}_count" in summary.columns]
    if count_cols:
        keep_mask = summary[count_cols].max(axis=1) >= min_count
        summary = summary.loc[keep_mask].copy()

    return summary.sort_values(by=[f"{metrics[-1]}_mean"], ascending=False)


def plot_factor_boxplots(
    data: pd.DataFrame,
    factor: str,
    metrics: list[str],
    hue: str | None = None,
    order: list[str] | None = None,
    figsize: tuple[int, int] = (20, 12),
) -> plt.Figure:
    ncols = 2
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    handles = labels = None
    for ax, metric in zip(axes, metrics):
        plot_df = data.loc[data[metric].notna()].copy()
        sns.boxplot(
            data=plot_df,
            x=factor,
            y=metric,
            hue=hue,
            order=order,
            ax=ax,
            showfliers=False,
        )
        ax.set_title(f"{metric} by {factor}")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(alpha=0.2)
        if hue is not None:
            handles, labels = ax.get_legend_handles_labels()
            if ax.legend_ is not None:
                ax.legend_.remove()

    for ax in axes[len(metrics):]:
        ax.axis("off")

    if hue is not None and handles is not None:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.96] if hue is not None else None)
    return fig


def factor_ranking_table(
    data: pd.DataFrame,
    factor: str,
    primary_metric: str = "mcc",
    secondary_metrics: list[str] | None = None,
) -> pd.DataFrame:
    if secondary_metrics is None:
        secondary_metrics = ["balanced_accuracy", "f1", "recall", "precision", "accuracy"]

    metrics = [primary_metric] + [m for m in secondary_metrics if m in data.columns and m != primary_metric]
    summary = summarize_factor_effects(data, factor, metrics)

    if f"{primary_metric}_std" in summary.columns:
        summary = summary.sort_values(
            by=[f"{primary_metric}_mean", f"{primary_metric}_std"],
            ascending=[False, True],
        )

    summary.insert(0, "rank", range(1, len(summary) + 1))
    return summary


def interaction_heatmap(
    data: pd.DataFrame,
    row_factor: str,
    col_factor: str,
    metric: str = "mcc",
    aggfunc: str = "mean",
    figsize: tuple[int, int] = (10, 6),
    annot: bool = True,
    fmt: str = ".3f",
    cmap: str = "viridis",
) -> tuple[plt.Figure, pd.DataFrame]:
    plot_df = data.loc[data[metric].notna()].copy()
    pivot = plot_df.pivot_table(
        values=metric,
        index=row_factor,
        columns=col_factor,
        aggfunc=aggfunc,
    )
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(pivot, annot=annot, fmt=fmt, cmap=cmap, ax=ax)
    ax.set_title(f"{metric} ({aggfunc}) for {row_factor} × {col_factor}")
    plt.tight_layout()
    return fig, pivot


def main() -> None:
    repo_root = find_repo_root()
    analysis_dir = repo_root / "notebooks" / "modelling" / "analysis_tables"
    figures_dir = repo_root / "notebooks" / "modelling" / "analysis_figures" / "02_factor_effects"

    analysis_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Repository root:", repo_root)
    print("Analysis tables:", analysis_dir)
    print("Figures directory:", figures_dir)

    seed_level_path = analysis_dir / "results_seed_level.csv"
    config_level_path = analysis_dir / "results_config_level.csv"

    if not seed_level_path.exists():
        raise FileNotFoundError(
            f"Missing {seed_level_path}. Run Notebook 01 first so the consolidated tables are exported."
        )

    results_seed_level = pd.read_csv(seed_level_path)
    results_config_level = pd.read_csv(config_level_path) if config_level_path.exists() else None

    print("results_seed_level:", results_seed_level.shape)
    if results_config_level is not None:
        print("results_config_level:", results_config_level.shape)

    present_factor_columns = [c for c in FACTOR_COLUMNS if c in results_seed_level.columns]
    present_classical_metrics = [c for c in CLASSICAL_METRICS if c in results_seed_level.columns]
    present_prob_metrics = [c for c in PROB_METRICS if c in results_seed_level.columns]

    print("Available factor columns:", present_factor_columns)
    print("Available classical metrics:", present_classical_metrics)
    print("Available probabilistic metrics:", present_prob_metrics)

    df = results_seed_level.copy()
    for col in [
        "dataset_variant",
        "config_name",
        "val_strategy",
        "model_key",
        "model_class",
        "imputer",
        "scaler",
        "pca",
        "resampling",
    ]:
        if col in df.columns:
            df[col] = df[col].fillna("missing").astype(str)

    if "pca" in df.columns:
        df["pca"] = df["pca"].replace({"None": "none", "nan": "none", "False": "none"}).astype(str)
    if "resampling" in df.columns:
        df["resampling"] = df["resampling"].replace({"None": "none", "nan": "none"}).astype(str)
    if "scaler" in df.columns:
        df["scaler"] = df["scaler"].replace({"None": "none", "nan": "none"}).astype(str)

    dataset_order = sorted(df["dataset_variant"].dropna().unique().tolist()) if "dataset_variant" in df.columns else None
    config_order = sorted(df["config_name"].dropna().unique().tolist()) if "config_name" in df.columns else None

    if "config_name" in df.columns:
        coverage_by_config = (
            df.groupby("config_name")[present_classical_metrics + present_prob_metrics]
            .agg(lambda s: s.notna().sum())
            .sort_index()
        )
        print_dataframe(coverage_by_config.reset_index(), "Coverage by config")

    factor_cardinality = pd.DataFrame(
        {
            "factor": present_factor_columns,
            "n_unique_levels": [df[col].nunique(dropna=False) for col in present_factor_columns],
        }
    )
    print_dataframe(factor_cardinality, "Factor cardinality")

    factor_summaries: dict[str, pd.DataFrame] = {}
    for factor in ["dataset_variant", "config_name", "val_strategy", "model_key", "scaler", "resampling", "pca"]:
        if factor in df.columns and present_classical_metrics:
            factor_summaries[factor] = summarize_factor_effects(df, factor, present_classical_metrics)

    for factor, summary in factor_summaries.items():
        print_dataframe(summary, f"Factor summary: {factor}")

    if "dataset_variant" in df.columns and present_classical_metrics:
        fig = plot_factor_boxplots(
            df,
            factor="dataset_variant",
            metrics=present_classical_metrics,
            order=dataset_order,
            figsize=(20, 12),
        )
        save_figure(fig, figures_dir, "dataset_variant_effect_boxplots.png")

    if "config_name" in df.columns and "dataset_variant" in df.columns and present_classical_metrics:
        fig = plot_factor_boxplots(
            df,
            factor="config_name",
            metrics=present_classical_metrics,
            hue="dataset_variant",
            order=config_order,
            figsize=(22, 12),
        )
        save_figure(fig, figures_dir, "config_name_effect_boxplots.png")

    if "model_key" in df.columns and "dataset_variant" in df.columns and present_classical_metrics:
        model_order = (
            df.groupby("model_key")["mcc"].mean().sort_values(ascending=False).index.tolist()
            if "mcc" in df.columns
            else sorted(df["model_key"].unique().tolist())
        )
        fig = plot_factor_boxplots(
            df,
            factor="model_key",
            metrics=present_classical_metrics,
            hue="dataset_variant",
            order=model_order,
            figsize=(24, 14),
        )
        save_figure(fig, figures_dir, "model_key_effect_boxplots.png")

    if "scaler" in df.columns and "dataset_variant" in df.columns and present_classical_metrics:
        scaler_order = (
            df.groupby("scaler")["mcc"].mean().sort_values(ascending=False).index.tolist()
            if "mcc" in df.columns
            else sorted(df["scaler"].unique().tolist())
        )
        fig = plot_factor_boxplots(
            df,
            factor="scaler",
            metrics=present_classical_metrics,
            hue="dataset_variant",
            order=scaler_order,
            figsize=(22, 12),
        )
        save_figure(fig, figures_dir, "scaler_effect_boxplots.png")

    if "resampling" in df.columns and "dataset_variant" in df.columns and present_classical_metrics:
        resampling_order = (
            df.groupby("resampling")["mcc"].mean().sort_values(ascending=False).index.tolist()
            if "mcc" in df.columns
            else sorted(df["resampling"].unique().tolist())
        )
        fig = plot_factor_boxplots(
            df,
            factor="resampling",
            metrics=present_classical_metrics,
            hue="dataset_variant",
            order=resampling_order,
            figsize=(22, 12),
        )
        save_figure(fig, figures_dir, "resampling_effect_boxplots.png")

    if "pca" in df.columns and "dataset_variant" in df.columns and present_classical_metrics:
        pca_order = (
            df.groupby("pca")["mcc"].mean().sort_values(ascending=False).index.tolist()
            if "mcc" in df.columns
            else sorted(df["pca"].unique().tolist())
        )
        fig = plot_factor_boxplots(
            df,
            factor="pca",
            metrics=present_classical_metrics,
            hue="dataset_variant",
            order=pca_order,
            figsize=(18, 12),
        )
        save_figure(fig, figures_dir, "pca_effect_boxplots.png")

    ranking_tables: dict[str, pd.DataFrame] = {}
    for factor in ["dataset_variant", "config_name", "val_strategy", "model_key", "scaler", "resampling", "pca"]:
        if factor in df.columns and "mcc" in df.columns:
            ranking_tables[factor] = factor_ranking_table(df, factor=factor, primary_metric="mcc")
            print_dataframe(ranking_tables[factor], f"Ranking table: {factor}")

    interaction_specs = [
        ("dataset_variant", "scaler"),
        ("dataset_variant", "resampling"),
        ("dataset_variant", "pca"),
        ("model_key", "resampling"),
        ("model_key", "scaler"),
        ("scaler", "pca"),
        ("config_name", "model_key"),
    ]
    interaction_tables: dict[str, pd.DataFrame] = {}

    for row_factor, col_factor in interaction_specs:
        if row_factor in df.columns and col_factor in df.columns and "mcc" in df.columns:
            fig, pivot = interaction_heatmap(
                df,
                row_factor=row_factor,
                col_factor=col_factor,
                metric="mcc",
                aggfunc="mean",
                figsize=(max(18, int(1.2 * df[row_factor].nunique())), max(15, int(0.8 * df[col_factor].nunique()))),
            )
            interaction_tables[f"{row_factor}__{col_factor}"] = pivot
            save_figure(fig, figures_dir, f"heatmap_{row_factor}__{col_factor}__mcc_mean.png")

    for row_factor, col_factor in [("dataset_variant", "scaler"), ("model_key", "resampling"), ("config_name", "model_key")]:
        if row_factor in df.columns and col_factor in df.columns and "balanced_accuracy" in df.columns:
            fig, pivot = interaction_heatmap(
                df,
                row_factor=row_factor,
                col_factor=col_factor,
                metric="balanced_accuracy",
                aggfunc="mean",
                figsize=(max(18, int(1.2 * df[row_factor].nunique())), max(18, int(0.8 * df[col_factor].nunique()))),
                cmap="magma",
            )
            interaction_tables[f"{row_factor}__{col_factor}__balanced_accuracy"] = pivot
            save_figure(fig, figures_dir, f"heatmap_{row_factor}__{col_factor}__balanced_accuracy_mean.png")

    if results_config_level is not None:
        config_df = results_config_level.copy()
    else:
        group_cols = [
            c
            for c in [
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
            if c in df.columns
        ]
        agg_map = {
            "seed": "nunique",
            "accuracy": ["mean", "std"],
            "precision": ["mean", "std"],
            "recall": ["mean", "std"],
            "f1": ["mean", "std"],
            "balanced_accuracy": ["mean", "std"],
            "mcc": ["mean", "std"],
        }
        config_df = df.groupby(group_cols).agg(agg_map)
        config_df.columns = ["_".join([c for c in col if c]).rstrip("_") for col in config_df.columns]
        config_df = config_df.reset_index().rename(columns={"seed_nunique": "n_seed_runs"})

    fair_metric_cols = [
        c
        for c in [
            "accuracy_mean",
            "precision_mean",
            "recall_mean",
            "f1_mean",
            "balanced_accuracy_mean",
            "mcc_mean",
        ]
        if c in config_df.columns
    ]

    config_rank = config_df.copy()
    for metric_col in fair_metric_cols:
        rank_col = f"rank_{metric_col.replace('_mean', '')}"
        config_rank[rank_col] = config_rank[metric_col].rank(ascending=False, method="dense")

    rank_cols = [c for c in config_rank.columns if c.startswith("rank_")]
    if rank_cols:
        sort_cols = ["fair_composite_rank_mean"]
        sort_ascending = [True]
        config_rank["fair_composite_rank_mean"] = config_rank[rank_cols].mean(axis=1)
        if "mcc_mean" in config_rank.columns:
            sort_cols.append("mcc_mean")
            sort_ascending.append(False)
        config_rank = config_rank.sort_values(sort_cols, ascending=sort_ascending)

    top_columns = [
        c
        for c in [
            "dataset_variant",
            "config_name",
            "val_strategy",
            "model_key",
            "model_class",
            "imputer",
            "scaler",
            "pca",
            "resampling",
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
            "fair_composite_rank_mean",
        ]
        if c in config_rank.columns
    ]
    top_configs_fair = config_rank[top_columns].head(30)
    print_dataframe(top_configs_fair, "Top fair comparison configs", n=30)

    if "mcc_mean" in config_rank.columns and "balanced_accuracy_mean" in config_rank.columns:
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_df = config_rank.copy()
        sns.scatterplot(
            data=plot_df,
            x="balanced_accuracy_mean",
            y="mcc_mean",
            hue="dataset_variant" if "dataset_variant" in plot_df.columns else None,
            style="config_name" if "config_name" in plot_df.columns else None,
            s=120,
            ax=ax,
        )
        ax.set_title("Configuration landscape: balanced accuracy vs MCC")
        ax.grid(alpha=0.2)
        save_figure(fig, figures_dir, "config_landscape_balanced_accuracy_vs_mcc.png")

    if {"config_name", "roc_auc", "pr_auc", "brier"}.issubset(df.columns):
        nonloo_df = df.loc[df["config_name"] != "config_loo"].copy()
        if not nonloo_df.empty:
            metrics = [m for m in ["roc_auc", "pr_auc", "brier"] if m in nonloo_df.columns]
            fig = plot_factor_boxplots(
                nonloo_df,
                factor="config_name",
                metrics=metrics,
                hue="dataset_variant" if "dataset_variant" in nonloo_df.columns else None,
                figsize=(18, 10),
            )
            save_figure(fig, figures_dir, "nonloo_probabilistic_metrics_by_config.png")

    for factor, summary in factor_summaries.items():
        summary.to_csv(analysis_dir / f"factor_summary__{factor}.csv", index=False)
    for factor, summary in ranking_tables.items():
        summary.to_csv(analysis_dir / f"factor_ranking__{factor}.csv", index=False)
    for key, table in interaction_tables.items():
        safe_key = key.replace("/", "_")
        table.to_csv(analysis_dir / f"interaction__{safe_key}.csv")
    top_configs_fair.to_csv(analysis_dir / "top_configs_fair_comparison.csv", index=False)

    print("Saved factor summaries, rankings, interactions, and top fair-comparison configs.")
    print("Analysis directory:", analysis_dir)
    print("Figures directory:", figures_dir)
    print("Generated figures:")
    for fig_path in sorted(figures_dir.glob("*.png")):
        print(" -", fig_path.name)


if __name__ == "__main__":
    main()
