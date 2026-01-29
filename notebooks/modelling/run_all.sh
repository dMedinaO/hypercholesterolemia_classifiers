#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# EDIT THESE PATHS
# -----------------------------
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="run_grid_exploration.py"

# Datasets (edit <RUN_TAG>)
DATA_STRICT="../../data/processed_variants/hcs_strict_complete_case_20260129_114259.csv"
DATA_IMPUTED="../../data/processed_variants/hcs_imputed_exploratory_20260129_114259.csv"

# Master config (already contains 30 seeds)
CONFIG_MASTER="../../configs/config_grid_models_classification.json"

# Output + logs
RUNS_DIR="runs"
LOGS_DIR="logs"
CONFIGS_DIR="../../configs/variants"

# Which datasets to run
RUN_STRICT=1
RUN_IMPUTED=1

# Optional: parallel execution file
CMD_FILE="commands_$(date +'%Y%m%d_%H%M%S').txt"

# -----------------------------
# Checks
# -----------------------------
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "[ERROR] python not found"; exit 1; }
command -v jq >/dev/null 2>&1 || { echo "[ERROR] jq not found (needed to create config variants)"; exit 1; }

mkdir -p "$RUNS_DIR" "$LOGS_DIR" "$CONFIGS_DIR"
: > "$CMD_FILE"

# -----------------------------
# Create config variants (one validation strategy per file)
# -----------------------------
make_config_variant () {
  local variant_name="$1"
  local strategy_name="$2"
  local out_cfg="${CONFIGS_DIR}/${variant_name}.json"

  jq --arg STRAT "$strategy_name" '
    .validation.strategies = (.validation.strategies | map(select(.name == $STRAT)))
  ' "$CONFIG_MASTER" > "$out_cfg"

  echo "$out_cfg"
}

CFG_KFOLD="$(make_config_variant "config_kfold" "kfold")"
CFG_LOO="$(make_config_variant "config_loo" "loo")"
CFG_RAND="$(make_config_variant "config_random80" "random_split_80_20")"
CFG_STRAT="$(make_config_variant "config_strat80" "stratified_split_80_20")"

CONFIGS=("$CFG_KFOLD" "$CFG_LOO" "$CFG_RAND" "$CFG_STRAT")

# -----------------------------
# Add runs to command file
# -----------------------------
add_run () {
  local data_path="$1"
  local cfg_path="$2"
  local tag="$3"

  local cfg_base
  cfg_base="$(basename "$cfg_path" .json)"

  local outdir="${RUNS_DIR}/${tag}/${cfg_base}"
  local logfile="${LOGS_DIR}/${tag}_${cfg_base}_$(date +'%Y%m%d_%H%M%S').log"

  mkdir -p "$outdir" "$LOGS_DIR"

  echo "$PYTHON_BIN $SCRIPT --data \"$data_path\" --config \"$cfg_path\" --outdir \"$outdir\" > \"$logfile\" 2>&1" >> "$CMD_FILE"
}

for cfg in "${CONFIGS[@]}"; do
  if [[ "$RUN_STRICT" -eq 1 ]]; then
    add_run "$DATA_STRICT" "$cfg" "STRICT"
  fi
  if [[ "$RUN_IMPUTED" -eq 1 ]]; then
    add_run "$DATA_IMPUTED" "$cfg" "IMPUTED"
  fi
done

echo "[OK] Created: $CMD_FILE"
echo "[OK] Runs: $(wc -l < "$CMD_FILE")"
echo
echo "Execute sequentially:"
echo "  bash $CMD_FILE"
echo
echo "Execute in parallel (if you have GNU parallel):"
echo "  parallel -j 4 < $CMD_FILE"
