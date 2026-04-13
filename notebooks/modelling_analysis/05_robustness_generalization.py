#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Notebook 05: Robustness and Generalization Analysis

This script evaluates the robustness and generalization properties of the
explored predictive pipelines. It focuses on how performance changes across
validation strategies, dataset variants, and random seeds, and identifies
configurations that are not only strong but also stable and transferable
across experimental conditions.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set(style="whitegrid", context="talk")

TABLE_DIR = Path(
    "/home/dmedina/Desktop/colabs/hypercholesterolemia_classifiers/notebooks/modelling/analysis_tables"
)
FIG_DIR = Path(
    "/home/dmedina/Desktop/colabs/hypercholesterolemia_classifiers/notebooks/modelling/analysis_figures/05_robustness_generalization"
)


REQUIRED_COLS = [
    "dataset_variant",
    "config_name",
    "model_class",
    "scaler",
    "resampling",
    "pca",
    "seed",
    "mcc",
    "balanced_accuracy",
    "f1",
    "accuracy",
    "precision",
    "recall",
]

CATEGORICAL_COLS = [
    "dataset_variant",
    "config_name",
    "model_class",
    "scaler",
    "resampling",
    "pca",
]

METRIC_COLS = ["mcc", "balanced_accuracy", "f1", "accuracy", "precision", "recall"]

GROUP_COLS = [
    "dataset_variant",
    "config_name",
    "model_class",
    "scaler",
    "resampling",
    "pca",
]


def close_current_figure() -> None:
    """Close the current matplotlib figure."""
    plt.close()


def build_pipeline_id(df: pd.DataFrame) -> pd.Series:
    """Create a compact pipeline identifier."""
    return (
        df["dataset_variant"].astype(str)
        + " | "
        + df["config_name"].astype(str)
        + " | "
        + df["model_class"].astype(str)
        + " | "
        + df["scaler"].astype(str)
        + " | "
        + df["resampling"].astype(str)
        + " | "
        + df["pca"].astype(str)
    )


def load_and_validate_results(table_dir: Path) -> pd.DataFrame:
    """Load seed-level results and validate required columns."""
    df = pd.read_csv(table_dir / "results_seed_level.csv")
    print(df.shape)
    print(df.head())

    missing_required = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype(str)

    df["pipeline_id"] = build_pipeline_id(df)
    return df


def plot_boxplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    output_path: Path,
    figsize: tuple[int, int] = (11, 6),
    rotation: int = 30,
) -> None:
    """Save a seaborn boxplot."""
    plt.figure(figsize=figsize)
    sns.boxplot(data=df, x=x, y=y)
    plt.title(title)
    plt.xticks(rotation=rotation, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    close_current_figure()


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_validate_results(TABLE_DIR)

    # 1. Robustness across validation strategies
    validation_summary = df.groupby("config_name")[METRIC_COLS].agg(["mean", "std", "median", "count"])
    validation_summary.to_csv(TABLE_DIR / "validation_strategy_summary.csv")
    print("\nValidation summary:")
    print(validation_summary.head())

    plot_boxplot(
        df=df,
        x="config_name",
        y="mcc",
        title="MCC distribution across validation strategies",
        output_path=FIG_DIR / "mcc_by_validation_strategy.png",
        figsize=(11, 6),
        rotation=30,
    )

    plot_boxplot(
        df=df,
        x="config_name",
        y="balanced_accuracy",
        title="Balanced accuracy across validation strategies",
        output_path=FIG_DIR / "balanced_accuracy_by_validation_strategy.png",
        figsize=(11, 6),
        rotation=30,
    )

    validation_robustness = (
        df.groupby("config_name")["mcc"].agg(["mean", "std", "count"]).reset_index()
    )
    validation_robustness["cv_ratio"] = (
        validation_robustness["std"] / validation_robustness["mean"].replace(0, np.nan)
    )
    validation_robustness = validation_robustness.sort_values(["mean", "std"], ascending=[False, True])
    validation_robustness.to_csv(TABLE_DIR / "validation_robustness.csv", index=False)
    print("\nValidation robustness:")
    print(validation_robustness.head())

    # 2. Sensitivity to dataset variant
    dataset_summary = df.groupby("dataset_variant")[METRIC_COLS].agg(["mean", "std", "median", "count"])
    dataset_summary.to_csv(TABLE_DIR / "dataset_variant_summary.csv")
    print("\nDataset summary:")
    print(dataset_summary)

    plot_boxplot(
        df=df,
        x="dataset_variant",
        y="mcc",
        title="MCC across dataset variants",
        output_path=FIG_DIR / "mcc_by_dataset_variant.png",
        figsize=(9, 6),
        rotation=20,
    )

    plot_boxplot(
        df=df,
        x="dataset_variant",
        y="balanced_accuracy",
        title="Balanced accuracy across dataset variants",
        output_path=FIG_DIR / "balanced_accuracy_by_dataset_variant.png",
        figsize=(9, 6),
        rotation=20,
    )

    match_cols = ["config_name", "model_class", "scaler", "resampling", "pca", "seed"]
    matched = df.pivot_table(
        index=match_cols,
        columns="dataset_variant",
        values=METRIC_COLS,
        aggfunc="mean",
    )
    matched.columns = ["__".join(map(str, c)) for c in matched.columns]
    matched = matched.reset_index()
    print("\nMatched dataset comparison:")
    print(matched.head())

    dataset_variants = sorted(df["dataset_variant"].dropna().unique().tolist())
    dataset_deltas: pd.DataFrame | None = None

    if len(dataset_variants) == 2:
        d1, d2 = dataset_variants
        dataset_deltas = matched.copy()
        dataset_deltas["delta_mcc"] = dataset_deltas[f"mcc__{d2}"] - dataset_deltas[f"mcc__{d1}"]
        dataset_deltas["delta_balanced_accuracy"] = (
            dataset_deltas[f"balanced_accuracy__{d2}"] - dataset_deltas[f"balanced_accuracy__{d1}"]
        )
        dataset_deltas.to_csv(TABLE_DIR / "dataset_variant_deltas.csv", index=False)
        print("\nDataset deltas:")
        print(dataset_deltas[match_cols + ["delta_mcc", "delta_balanced_accuracy"]].head())
    else:
        print(
            "Dataset delta table skipped because the number of dataset variants is not equal to 2."
        )

    if dataset_deltas is not None:
        plt.figure(figsize=(10, 6))
        sns.histplot(dataset_deltas["delta_mcc"].dropna(), bins=30, kde=True)
        plt.axvline(0, linestyle="--")
        plt.title(
            f"Delta MCC between dataset variants ({dataset_variants[1]} - {dataset_variants[0]})"
        )
        plt.tight_layout()
        plt.savefig(FIG_DIR / "delta_mcc_dataset_variants.png", dpi=300, bbox_inches="tight")
        close_current_figure()

    # 3. Stability across random seeds
    seed_stability = (
        df.groupby(GROUP_COLS)["mcc"].agg(["mean", "std", "median", "min", "max", "count"]).reset_index()
    )
    seed_stability["range"] = seed_stability["max"] - seed_stability["min"]
    seed_stability["cv_ratio"] = seed_stability["std"] / seed_stability["mean"].replace(0, np.nan)
    seed_stability = seed_stability.sort_values(["mean", "std"], ascending=[False, True])
    seed_stability.to_csv(TABLE_DIR / "seed_stability_summary.csv", index=False)
    print("\nSeed stability:")
    print(seed_stability.head(20))

    stable_high_perf = seed_stability[
        (seed_stability["count"] >= 2)
        & (seed_stability["mean"] >= seed_stability["mean"].quantile(0.75))
    ].sort_values(["std", "mean"], ascending=[True, False])
    stable_high_perf.to_csv(TABLE_DIR / "stable_high_performing_configurations.csv", index=False)
    print("\nStable high-performing configurations:")
    print(stable_high_perf.head(20))

    top_pipelines = stable_high_perf.head(12).copy()
    if not top_pipelines.empty:
        top_ids = set(top_pipelines.apply(lambda r: " | ".join(str(r[c]) for c in GROUP_COLS), axis=1))
        plot_df = df[df["pipeline_id"].isin(top_ids)].copy()

        plt.figure(figsize=(14, 7))
        sns.boxplot(data=plot_df, x="pipeline_id", y="mcc")
        plt.title("MCC distribution across seeds for top stable pipelines")
        plt.xticks(rotation=75, ha="right")
        plt.tight_layout()
        plt.savefig(
            FIG_DIR / "mcc_across_seeds_top_stable_pipelines.png",
            dpi=300,
            bbox_inches="tight",
        )
        close_current_figure()
    else:
        print("No stable high-performing pipelines matched the current filters.")

    # 4. Generalization profile of top configurations
    generalization_profile = df.groupby(GROUP_COLS)[METRIC_COLS].agg(["mean", "std", "count"])
    generalization_profile.to_csv(TABLE_DIR / "generalization_profile.csv")
    print("\nGeneralization profile:")
    print(generalization_profile.head())

    gen_score = df.groupby(GROUP_COLS)["mcc"].agg(["mean", "std", "count"]).reset_index()
    gen_score["generalization_score"] = gen_score["mean"] - gen_score["std"].fillna(0)
    gen_score = gen_score.sort_values("generalization_score", ascending=False)
    gen_score.to_csv(TABLE_DIR / "generalization_score_ranking.csv", index=False)
    print("\nGeneralization score ranking:")
    print(gen_score.head(20))

    top_gen = gen_score.head(20).copy()
    top_gen["pipeline_label"] = build_pipeline_id(top_gen)

    plt.figure(figsize=(14, 7))
    sns.barplot(data=top_gen, x="pipeline_label", y="generalization_score")
    plt.title("Top pipelines by generalization score")
    plt.xticks(rotation=75, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "top_generalization_score_pipelines.png", dpi=300, bbox_inches="tight")
    close_current_figure()

    # 5. Cross-strategy consistency of model families
    model_family_summary = df.groupby("model_class")[METRIC_COLS].agg(["mean", "std", "median", "count"])
    model_family_summary.to_csv(TABLE_DIR / "model_family_summary.csv")
    print("\nModel family summary:")
    print(model_family_summary)

    plot_boxplot(
        df=df,
        x="model_class",
        y="mcc",
        title="MCC distribution across model families",
        output_path=FIG_DIR / "mcc_by_model_family.png",
        figsize=(11, 6),
        rotation=30,
    )

    heatmap_model_validation = df.pivot_table(
        index="model_class",
        columns="config_name",
        values="mcc",
        aggfunc="mean",
    )
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_model_validation, annot=True, fmt=".3f", cmap="viridis")
    plt.title("Mean MCC by model family and validation strategy")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "heatmap_model_family_by_validation.png", dpi=300, bbox_inches="tight")
    close_current_figure()
    heatmap_model_validation.to_csv(TABLE_DIR / "heatmap_model_family_by_validation.csv")

    heatmap_model_dataset = df.pivot_table(
        index="model_class",
        columns="dataset_variant",
        values="mcc",
        aggfunc="mean",
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_model_dataset, annot=True, fmt=".3f", cmap="magma")
    plt.title("Mean MCC by model family and dataset variant")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "heatmap_model_family_by_dataset.png", dpi=300, bbox_inches="tight")
    close_current_figure()
    heatmap_model_dataset.to_csv(TABLE_DIR / "heatmap_model_family_by_dataset.csv")

    # 6. Export compact summary tables for manuscript use
    compact_summary = gen_score.merge(
        seed_stability[
            [
                "dataset_variant",
                "config_name",
                "model_class",
                "scaler",
                "resampling",
                "pca",
                "range",
                "cv_ratio",
            ]
        ],
        on=["dataset_variant", "config_name", "model_class", "scaler", "resampling", "pca"],
        how="left",
    )
    compact_summary = compact_summary.sort_values(
        ["generalization_score", "mean", "std"],
        ascending=[False, False, True],
    )
    compact_summary.to_csv(TABLE_DIR / "robustness_generalization_compact_summary.csv", index=False)
    print("\nCompact summary:")
    print(compact_summary.head(25))

    print("\nDone.")
    print(f"Figures saved to: {FIG_DIR}")
    print(f"Tables saved to: {TABLE_DIR}")


if __name__ == "__main__":
    main()
