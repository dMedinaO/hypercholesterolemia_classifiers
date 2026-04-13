#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Notebook 04: Information Gain & Complexity Analysis

Convierte el análisis del notebook a un script Python ejecutable.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


sns.set(style="whitegrid")


DATA_PATH = Path(
    "/home/dmedina/Desktop/colabs/hypercholesterolemia_classifiers/notebooks/modelling/analysis_tables"
)
FIG_PATH = Path(
    "/home/dmedina/Desktop/colabs/hypercholesterolemia_classifiers/notebooks/modelling/analysis_figures/04_information_gain"
)


def compute_complexity(row: pd.Series) -> int:
    """
    Compute a simple pipeline complexity score.

    Parameters
    ----------
    row : pd.Series
        Row from the seed-level results table.

    Returns
    -------
    int
        Complexity score.
    """
    complexity = 0

    if row["scaler"] != "none":
        complexity += 1
    if row["resampling"] != "none":
        complexity += 1
    if row["pca"] != "none":
        complexity += 1
    if row["model_class"] not in ["LogisticRegression", "GaussianNB"]:
        complexity += 1

    return complexity


def save_complexity_vs_mcc(df: pd.DataFrame, fig_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="complexity_score", y="mcc")
    plt.title("Complexity vs MCC")
    plt.tight_layout()
    plt.savefig(fig_path / "complexity_vs_mcc.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_scaling_gain(df: pd.DataFrame, fig_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="scaler", y="delta_mcc_vs_baseline")
    plt.title("Scaling gain vs baseline")
    plt.tight_layout()
    plt.savefig(fig_path / "scaling_gain.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_saturation_curve(df: pd.DataFrame, fig_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x="complexity_score", y="mcc", estimator="mean", errorbar="sd")
    plt.title("Performance saturation curve")
    plt.tight_layout()
    plt.savefig(fig_path / "saturation_curve.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_top_complexity(df: pd.DataFrame, fig_path: Path) -> None:
    top_df = df.sort_values("mcc", ascending=False).head(50)

    plt.figure(figsize=(8, 5))
    sns.countplot(data=top_df, x="complexity_score")
    plt.title("Complexity distribution in top models")
    plt.tight_layout()
    plt.savefig(fig_path / "top_complexity.png", dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    FIG_PATH.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH / "results_seed_level.csv")

    print("\nLoaded dataframe:")
    print(df.head())
    print(f"\nShape: {df.shape}")

    df["complexity_score"] = df.apply(compute_complexity, axis=1)

    print("\nComplexity score counts:")
    print(df["complexity_score"].value_counts().sort_index())

    save_complexity_vs_mcc(df, FIG_PATH)

    complexity_stats = df.groupby("complexity_score")["mcc"].agg(["mean", "std", "count"])
    print("\nComplexity stats:")
    print(complexity_stats)

    complexity_stats.to_csv(DATA_PATH / "complexity_stats.csv", index=True)

    baseline = df[
        (df["model_class"] == "LogisticRegression")
        & (df["scaler"] == "none")
        & (df["resampling"] == "none")
        & (df["pca"] == "off")
    ]

    print("\nBaseline shape:")
    print(baseline.shape)

    baseline_mean = baseline["mcc"].mean()
    print("\nBaseline mean MCC:")
    print(baseline_mean)

    df["delta_mcc_vs_baseline"] = df["mcc"] - baseline_mean

    save_scaling_gain(df, FIG_PATH)
    save_saturation_curve(df, FIG_PATH)

    robustness = df.groupby("complexity_score")["mcc"].agg(["mean", "std"])
    robustness["stability_ratio"] = robustness["std"] / robustness["mean"]

    print("\nRobustness table:")
    print(robustness)

    robustness.to_csv(DATA_PATH / "complexity_robustness.csv", index=True)

    save_top_complexity(df, FIG_PATH)

    df.to_csv(DATA_PATH / "results_seed_level_with_complexity.csv", index=False)

    print("\nDone.")
    print(f"Figures saved to: {FIG_PATH}")
    print(f"Tables saved to: {DATA_PATH}")


if __name__ == "__main__":
    main()