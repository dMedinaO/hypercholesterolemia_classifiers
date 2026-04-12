#!/usr/bin/env bash
#SBATCH --job-name=hcs_grid
#SBATCH --output=/home/dmedina/run_cluster/%x_%A_%a.out
#SBATCH --error=/home/dmedina/run_cluster/%x_%A_%a.err
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-159

source ~/.bashrc
conda activate ml_models

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ---------------------------
# Threading control
# ---------------------------
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# ---------------------------
# Paths
# ---------------------------
SCRIPT="/home/dmedina/hypercholesterolemia_classifiers/notebooks/modelling/run_grid_exploration.py"

DATA_STRICT="/home/dmedina/hypercholesterolemia_classifiers/data/processed_variants/hcs_strict_complete_case_20260129_114259.csv"
DATA_IMPUTED="/home/dmedina/hypercholesterolemia_classifiers/data/processed_variants/hcs_imputed_exploratory_20260129_114259.csv"

CFG_KFOLD="/home/dmedina/hypercholesterolemia_classifiers/configs/variants/config_kfold.json"
CFG_LOO="/home/dmedina/hypercholesterolemia_classifiers/configs/variants/config_loo.json"
CFG_RAND="/home/dmedina/hypercholesterolemia_classifiers/configs/variants/config_random80.json"
CFG_STRAT="/home/dmedina/hypercholesterolemia_classifiers/configs/variants/config_strat80.json"

RUNS_ROOT="/home/dmedina/hypercholesterolemia_classifiers/notebooks/modelling/runs"
LOGS_ROOT="/home/dmedina/hypercholesterolemia_classifiers/notebooks/modelling/logs"

mkdir -p "$LOGS_ROOT"

# ---------------------------
# Experiment list (8)
# ---------------------------
COMMANDS=(
  "STRICT  $DATA_STRICT  $CFG_KFOLD  $RUNS_ROOT/STRICT/config_kfold"
  "IMPUTED $DATA_IMPUTED $CFG_KFOLD  $RUNS_ROOT/IMPUTED/config_kfold"
  "STRICT  $DATA_STRICT  $CFG_LOO    $RUNS_ROOT/STRICT/config_loo"
  "IMPUTED $DATA_IMPUTED $CFG_LOO    $RUNS_ROOT/IMPUTED/config_loo"
  "STRICT  $DATA_STRICT  $CFG_RAND   $RUNS_ROOT/STRICT/config_random80"
  "IMPUTED $DATA_IMPUTED $CFG_RAND   $RUNS_ROOT/IMPUTED/config_random80"
  "STRICT  $DATA_STRICT  $CFG_STRAT  $RUNS_ROOT/STRICT/config_strat80"
  "IMPUTED $DATA_IMPUTED $CFG_STRAT  $RUNS_ROOT/IMPUTED/config_strat80"
)

# ---------------------------
# Sharding settings
# ---------------------------
N_SHARDS=20  # tune: 10–50 depending on cluster capacity

EXP_ID=$(( SLURM_ARRAY_TASK_ID / N_SHARDS ))   # 0..7
SHARD_ID=$(( SLURM_ARRAY_TASK_ID % N_SHARDS )) # 0..(N_SHARDS-1)

read -r TAG DATA CFG OUTDIR <<< "${COMMANDS[$EXP_ID]}"

CFG_BASE=$(basename "$CFG" .json)
LOGFILE="$LOGS_ROOT/${TAG}_${CFG_BASE}_job${SLURM_JOB_ID}_exp${EXP_ID}_shard${SHARD_ID}.log"

echo "[INFO] job=${SLURM_JOB_ID} array_task=${SLURM_ARRAY_TASK_ID}"
echo "[INFO] exp_id=${EXP_ID}/7 shard=${SHARD_ID}/${N_SHARDS}"
echo "[INFO] TAG=$TAG CFG=$CFG_BASE"
echo "[INFO] DATA=$DATA"
echo "[INFO] OUTDIR=$OUTDIR"
echo "[INFO] LOG=$LOGFILE"
echo "[INFO] CPUS=${SLURM_CPUS_PER_TASK} MEM=32G"

python "$SCRIPT" --data "$DATA" --config "$CFG" --outdir "$OUTDIR" --task-id "$SHARD_ID" --n-tasks "$N_SHARDS" --n-jobs-model "${SLURM_CPUS_PER_TASK}" --write-every 2000 > "$LOGFILE" 2>&1
