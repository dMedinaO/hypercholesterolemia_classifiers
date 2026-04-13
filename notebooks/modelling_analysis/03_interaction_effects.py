"""Converted from 03_interaction_effects.ipynb.

This script analyzes pairwise interaction effects and combination-level behavior
across modelling strategies.
"""


# # 03 — Interaction Effects and Combination Synergies
#
# This notebook analyzes **pairwise interaction effects** and **combination-level behavior** across modeling strategies.  
# The main goal is to move beyond marginal effects and identify whether specific design choices reinforce or weaken each other under a fair comparison framework.
#
# This notebook assumes that **Notebook 01** exported the consolidated `results_seed_level.csv` table and that **Notebook 02** may already have generated factor-level summaries.  
# The analyses here focus on metrics that remain comparable across validation strategies, including reconstructed LOO metrics when applicable.
#
# ## Main objectives
#
# 1. Quantify pairwise interactions between major pipeline components.
# 2. Identify favorable and unfavorable combinations.
# 3. Evaluate whether some factors are only beneficial under specific contexts.
# 4. Produce figures and tables directly reusable in the Results section.


from __future__ import annotations

from pathlib import Path
import itertools
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 200)
sns.set_context("talk")
sns.set_style("whitegrid")


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path("/home/dmedina/Desktop/colabs/hypercholesterolemia_classifiers")
if not (PROJECT_ROOT / "notebooks").exists():
    candidates = [p for p in PROJECT_ROOT.rglob("notebooks/modelling") if p.is_dir()]
    if candidates:
        PROJECT_ROOT = candidates[0].parents[1]

MODELLING_DIR = PROJECT_ROOT / "notebooks" / "modelling"
TABLES_DIR = MODELLING_DIR / "analysis_tables"
FIG_DIR = MODELLING_DIR / "analysis_figures" / "03_interaction_effects"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = TABLES_DIR / "results_seed_level.csv"
RESULTS_SUMMARY_PATH = TABLES_DIR / "results_config_summary.csv"

print("PROJECT_ROOT:", PROJECT_ROOT)
print("RESULTS_PATH exists:", RESULTS_PATH.exists())
print("FIG_DIR:", FIG_DIR)


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
if not RESULTS_PATH.exists():
    raise FileNotFoundError(
        f"Could not find {RESULTS_PATH}. Run Notebook 01 first and ensure it exports results_seed_level.csv."
    )

df = pd.read_csv(RESULTS_PATH)
print(df.shape)
print(df.head())


# ---------------------------------------------------------------------
# Column harmonization helpers
# ---------------------------------------------------------------------
def find_first_existing(candidates: list[str], columns: list[str]) -> str | None:
    for c in candidates:
        if c in columns:
            return c
    return None

DATASET_COL = find_first_existing(
    ["dataset_variant", "dataset_name", "dataset_key", "dataset"], df.columns.tolist()
)
CONFIG_COL = find_first_existing(
    ["config_name", "validation_config", "config"], df.columns.tolist()
)
MODEL_COL = find_first_existing(
    ["model_key", "model_class", "model_name"], df.columns.tolist()
)
SCALER_COL = find_first_existing(
    ["scaler"], df.columns.tolist()
)
RESAMPLING_COL = find_first_existing(
    ["resampling"], df.columns.tolist()
)
PCA_COL = find_first_existing(
    ["pca"], df.columns.tolist()
)
SEED_COL = find_first_existing(
    ["seed"], df.columns.tolist()
)

required_context = [DATASET_COL, CONFIG_COL, MODEL_COL, SCALER_COL, RESAMPLING_COL, PCA_COL, SEED_COL]
missing_context = [c for c in required_context if c is None]
if missing_context:
    raise ValueError(f"Missing required context columns: {missing_context}")

METRIC_CANDIDATES = ["mcc", "balanced_accuracy", "f1", "accuracy", "precision", "recall"]
AVAILABLE_METRICS = [m for m in METRIC_CANDIDATES if m in df.columns]
if not AVAILABLE_METRICS:
    raise ValueError("No comparable metrics were found in the consolidated table.")

PRIMARY_METRIC = "mcc" if "mcc" in AVAILABLE_METRICS else AVAILABLE_METRICS[0]

print("Resolved columns:")
print({
    "DATASET_COL": DATASET_COL,
    "CONFIG_COL": CONFIG_COL,
    "MODEL_COL": MODEL_COL,
    "SCALER_COL": SCALER_COL,
    "RESAMPLING_COL": RESAMPLING_COL,
    "PCA_COL": PCA_COL,
    "SEED_COL": SEED_COL,
    "PRIMARY_METRIC": PRIMARY_METRIC,
    "AVAILABLE_METRICS": AVAILABLE_METRICS,
})


# ---------------------------------------------------------------------
# Basic cleaning and standardization
# ---------------------------------------------------------------------
analysis_df = df.copy()

for col in [DATASET_COL, CONFIG_COL, MODEL_COL, SCALER_COL, RESAMPLING_COL, PCA_COL]:
    analysis_df[col] = analysis_df[col].astype(str).fillna("missing")

analysis_df[PCA_COL] = (
    analysis_df[PCA_COL]
    .replace({"True": "pca_on", "False": "pca_off", "1": "pca_on", "0": "pca_off"})
    .fillna("unknown")
    .astype(str)
)

# Keep only rows with non-null primary metric for interaction analyses
analysis_df = analysis_df[analysis_df[PRIMARY_METRIC].notna()].copy()

print("Filtered analysis_df shape:", analysis_df.shape)
print(analysis_df[[DATASET_COL, CONFIG_COL, MODEL_COL, SCALER_COL, RESAMPLING_COL, PCA_COL, PRIMARY_METRIC]].head())


# ## Helper functions
#
# The following utilities generate summary tables, effect contrasts, and interaction heatmaps.  
# The focus is on **mean performance**, **variability**, and **sample support** per combination.


def save_current_figure(name: str) -> Path:
    outpath = FIG_DIR / f"{name}.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    return outpath


def summarize_pairwise_interaction(
    data: pd.DataFrame,
    row_factor: str,
    col_factor: str,
    metric: str,
) -> pd.DataFrame:
    grouped = (
        data.groupby([row_factor, col_factor], dropna=False)[metric]
        .agg(["mean", "std", "count", "median"])
        .reset_index()
        .sort_values(["mean", "count"], ascending=[False, False])
    )
    return grouped


def build_heatmap_table(
    data: pd.DataFrame,
    row_factor: str,
    col_factor: str,
    metric: str,
    aggfunc: str = "mean",
) -> pd.DataFrame:
    return data.pivot_table(
        values=metric,
        index=row_factor,
        columns=col_factor,
        aggfunc=aggfunc,
        dropna=False,
    )


def contrast_within_context(
    data: pd.DataFrame,
    context_col: str,
    factor_col: str,
    metric: str,
) -> pd.DataFrame:
    rows = []
    for context_value, subdf in data.groupby(context_col):
        stats = (
            subdf.groupby(factor_col)[metric]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values("mean", ascending=False)
        )
        if stats.shape[0] < 2:
            continue
        best = stats.iloc[0]
        worst = stats.iloc[-1]
        rows.append(
            {
                context_col: context_value,
                "best_level": best[factor_col],
                "best_mean": best["mean"],
                "best_std": best["std"],
                "best_count": best["count"],
                "worst_level": worst[factor_col],
                "worst_mean": worst["mean"],
                "worst_std": worst["std"],
                "worst_count": worst["count"],
                "delta_best_worst": best["mean"] - worst["mean"],
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("delta_best_worst", ascending=False)


def plot_heatmap(
    table: pd.DataFrame,
    title: str,
    filename: str,
    center: float | None = None,
    annot: bool = True,
    fmt: str = ".3f",
):
    plt.figure(figsize=(max(15, table.shape[1] * 1.2), max(15, table.shape[0] * 0.8)))
    sns.heatmap(table, annot=annot, fmt=fmt, cmap="viridis", center=center)
    plt.title(title)
    plt.xlabel(table.columns.name if table.columns.name else "")
    plt.ylabel(table.index.name if table.index.name else "")
    out = save_current_figure(filename)
    plt.close()
    print("Saved:", out)


# ## 1. Global interaction landscape
#
# We begin by defining the pairwise interactions of highest practical interest.  
# These comparisons target the combinations most likely to influence downstream conclusions and reviewer-facing methodological claims.


PAIRWISE_FACTORS = [
    (DATASET_COL, SCALER_COL),
    (DATASET_COL, RESAMPLING_COL),
    (DATASET_COL, PCA_COL),
    (MODEL_COL, SCALER_COL),
    (MODEL_COL, RESAMPLING_COL),
    (MODEL_COL, PCA_COL),
    (SCALER_COL, RESAMPLING_COL),
    (SCALER_COL, PCA_COL),
    (RESAMPLING_COL, PCA_COL),
    (CONFIG_COL, MODEL_COL),
]

pairwise_tables = {}
for row_factor, col_factor in PAIRWISE_FACTORS:
    tbl = summarize_pairwise_interaction(analysis_df, row_factor, col_factor, PRIMARY_METRIC)
    key = f"{row_factor}__{col_factor}"
    pairwise_tables[key] = tbl
    out_csv = TABLES_DIR / f"03_pairwise_summary_{key}.csv"
    tbl.to_csv(out_csv, index=False)

print(f"Generated {len(pairwise_tables)} pairwise summary tables.")
list(pairwise_tables.keys())[:5]


# Preview one pairwise summary table
example_key = list(pairwise_tables.keys())[0]
print("Example:", example_key)
pairwise_tables[example_key].head(15)


# ## 2. Mean-performance heatmaps for key pairwise interactions
#
# These heatmaps summarize average performance for each combination, making it easier to detect synergy, incompatibility, and context-specific effects.


KEY_HEATMAPS = [
    (DATASET_COL, SCALER_COL),
    (DATASET_COL, RESAMPLING_COL),
    (MODEL_COL, SCALER_COL),
    (MODEL_COL, RESAMPLING_COL),
    (SCALER_COL, PCA_COL),
    (CONFIG_COL, MODEL_COL),
]

for row_factor, col_factor in KEY_HEATMAPS:
    heat = build_heatmap_table(analysis_df, row_factor, col_factor, PRIMARY_METRIC, aggfunc="mean")
    plot_heatmap(
        heat,
        title=f"Mean {PRIMARY_METRIC.upper()} — {row_factor} × {col_factor}",
        filename=f"heatmap_mean_{PRIMARY_METRIC}_{row_factor}_x_{col_factor}",
        center=None,
        annot=True,
        fmt=".3f",
    )


# ## 3. Support heatmaps
#
# Mean values can be misleading when some combinations are represented by very few runs.  
# To contextualize performance patterns, we also visualize the number of observations supporting each interaction cell.


for row_factor, col_factor in KEY_HEATMAPS:
    heat_count = build_heatmap_table(analysis_df, row_factor, col_factor, PRIMARY_METRIC, aggfunc="count")
    plot_heatmap(
        heat_count,
        title=f"Support count — {row_factor} × {col_factor}",
        filename=f"heatmap_count_{row_factor}_x_{col_factor}",
        center=None,
        annot=True,
        fmt=".0f",
    )


# ## 4. Variability heatmaps
#
# To distinguish robust combinations from unstable ones, we examine the within-cell standard deviation for the primary metric.


for row_factor, col_factor in KEY_HEATMAPS:
    heat_std = build_heatmap_table(analysis_df, row_factor, col_factor, PRIMARY_METRIC, aggfunc="std")
    plot_heatmap(
        heat_std,
        title=f"Standard deviation of {PRIMARY_METRIC.upper()} — {row_factor} × {col_factor}",
        filename=f"heatmap_std_{PRIMARY_METRIC}_{row_factor}_x_{col_factor}",
        center=None,
        annot=True,
        fmt=".3f",
    )


# ## 5. Context-specific contrasts
#
# The next analyses ask whether a factor behaves differently depending on context.  
# For example, a resampling strategy may look favorable overall, but only under a subset of models or only for one dataset representation.


contrast_specs = [
    (MODEL_COL, RESAMPLING_COL),
    (MODEL_COL, SCALER_COL),
    (DATASET_COL, RESAMPLING_COL),
    (DATASET_COL, SCALER_COL),
    (CONFIG_COL, MODEL_COL),
]

contrast_results = {}
for context_col, factor_col in contrast_specs:
    out = contrast_within_context(analysis_df, context_col, factor_col, PRIMARY_METRIC)
    key = f"{context_col}__{factor_col}"
    contrast_results[key] = out
    out_csv = TABLES_DIR / f"03_context_contrast_{key}.csv"
    out.to_csv(out_csv, index=False)

for key, out in contrast_results.items():
    print("\n", "=" * 80)
    print(key)
print(out.head(15))


# ## 6. Combination ranking under fair comparison
#
# Here we rank **full combinations** using the primary metric while preserving comparable evaluation settings.  
# This helps identify which full pipeline designs dominate not only on average, but also in terms of stability and support.


combination_cols = [DATASET_COL, CONFIG_COL, MODEL_COL, SCALER_COL, RESAMPLING_COL, PCA_COL]

combo_summary = (
    analysis_df.groupby(combination_cols, dropna=False)[AVAILABLE_METRICS]
    .agg(["mean", "std", "count", "median"])
)

combo_summary.columns = ["__".join(col).strip("_") for col in combo_summary.columns]
combo_summary = combo_summary.reset_index()

primary_mean_col = f"{PRIMARY_METRIC}__mean"
primary_std_col = f"{PRIMARY_METRIC}__std"
primary_count_col = f"{PRIMARY_METRIC}__count"

combo_summary = combo_summary.sort_values(
    by=[primary_mean_col, primary_count_col],
    ascending=[False, False],
).reset_index(drop=True)

combo_summary.to_csv(TABLES_DIR / "03_combination_ranking.csv", index=False)
combo_summary.head(25)


# Plot top combinations by primary metric
top_n = 20
plot_df = combo_summary.head(top_n).copy()
plot_df["combo_label"] = (
    plot_df[DATASET_COL].astype(str) + " | " +
    plot_df[CONFIG_COL].astype(str) + " | " +
    plot_df[MODEL_COL].astype(str) + " | " +
    plot_df[SCALER_COL].astype(str) + " | " +
    plot_df[RESAMPLING_COL].astype(str) + " | " +
    plot_df[PCA_COL].astype(str)
)

plt.figure(figsize=(14, max(8, top_n * 0.45)))
sns.barplot(data=plot_df, y="combo_label", x=primary_mean_col)
plt.title(f"Top {top_n} combinations by mean {PRIMARY_METRIC.upper()}")
plt.xlabel(f"Mean {PRIMARY_METRIC.upper()}")
plt.ylabel("Combination")
out = save_current_figure(f"top_{top_n}_combinations_by_{PRIMARY_METRIC}")
plt.close()
print("Saved:", out)


# ## 7. Dominance of factor levels within top-performing combinations
#
# This section evaluates whether some factor levels are overrepresented among the strongest combinations, which can help justify downstream model selection decisions.


top_fraction = 0.20
k = max(1, int(np.ceil(combo_summary.shape[0] * top_fraction)))
top_combo_df = combo_summary.head(k).copy()

dominance_tables = {}
for factor in [DATASET_COL, CONFIG_COL, MODEL_COL, SCALER_COL, RESAMPLING_COL, PCA_COL]:
    tbl = (
        top_combo_df[factor]
        .value_counts(dropna=False)
        .rename_axis(factor)
        .reset_index(name="count_top")
    )
    tbl["fraction_top"] = tbl["count_top"] / tbl["count_top"].sum()

    global_counts = (
        combo_summary[factor]
        .value_counts(dropna=False)
        .rename_axis(factor)
        .reset_index(name="count_global")
    )
    tbl = tbl.merge(global_counts, on=factor, how="left")
    tbl["fraction_global"] = tbl["count_global"] / tbl["count_global"].sum()
    tbl["enrichment_top_vs_global"] = tbl["fraction_top"] / tbl["fraction_global"]
    tbl = tbl.sort_values("enrichment_top_vs_global", ascending=False)

    dominance_tables[factor] = tbl
    tbl.to_csv(TABLES_DIR / f"03_top_combination_dominance_{factor}.csv", index=False)

for factor, tbl in dominance_tables.items():
    print("\n", "=" * 80)
    print(factor)
print(tbl.head(15))


# ## 8. Interaction analyses across multiple metrics
#
# To ensure that conclusions are not driven by a single metric, we repeat the main interaction summaries across all metrics that remain comparable under the current evaluation framework.


metric_interaction_panels = [
    (MODEL_COL, RESAMPLING_COL),
    (SCALER_COL, PCA_COL),
]

for metric in AVAILABLE_METRICS:
    for row_factor, col_factor in metric_interaction_panels:
        panel = build_heatmap_table(analysis_df, row_factor, col_factor, metric, aggfunc="mean")
        plot_heatmap(
            panel,
            title=f"Mean {metric.upper()} — {row_factor} × {col_factor}",
            filename=f"heatmap_mean_{metric}_{row_factor}_x_{col_factor}",
            center=None,
            annot=True,
            fmt=".3f",
        )


# ## 9. Export concise results for downstream reporting
#
# The following compact tables summarize the most directly reportable findings from this notebook.


report_tables = {}

# Best pairwise cell for each interaction
best_cells = []
for key, tbl in pairwise_tables.items():
    if tbl.empty:
        continue
    row_factor, col_factor = key.split("__")
    top = tbl.sort_values(["mean", "count"], ascending=[False, False]).iloc[0].to_dict()
    top["row_factor"] = row_factor
    top["col_factor"] = col_factor
    best_cells.append(top)

best_cells_df = pd.DataFrame(best_cells).sort_values(["mean", "count"], ascending=[False, False])
best_cells_df.to_csv(TABLES_DIR / "03_best_pairwise_cells.csv", index=False)
report_tables["best_pairwise_cells"] = best_cells_df

# Most context-sensitive contrasts
all_contrasts = []
for key, tbl in contrast_results.items():
    if tbl.empty:
        continue
    tbl = tbl.copy()
    tbl["contrast_key"] = key
    all_contrasts.append(tbl)

if all_contrasts:
    all_contrasts_df = pd.concat(all_contrasts, ignore_index=True)
    all_contrasts_df = all_contrasts_df.sort_values("delta_best_worst", ascending=False)
    all_contrasts_df.to_csv(TABLES_DIR / "03_all_context_contrasts.csv", index=False)
    report_tables["all_context_contrasts"] = all_contrasts_df

for name, tbl in report_tables.items():
    print("\n", "=" * 80)
    print(name)
print(tbl.head(20))


# ## 10. Interpretation guide
#
# When writing the Results section, the outputs of this notebook can support statements such as the following:
#
# - Certain preprocessing decisions are only beneficial under specific model families.
# - Some interactions show strong mean performance but poor support or high instability, suggesting caution.
# - Top-performing combinations are enriched for a subset of factor levels, indicating non-random dominance.
# - The effect of dataset representation, resampling, and dimensionality reduction is conditional rather than universal.
#
# These conclusions should be cross-checked with the global performance overview from Notebook 01 and the marginal factor analyses from Notebook 02.
