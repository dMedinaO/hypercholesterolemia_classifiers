#!/usr/bin/env bash
set -euo pipefail

# Number of parallel jobs
N_JOBS=4

# Check GNU parallel
command -v parallel >/dev/null 2>&1 || {
  echo "[ERROR] GNU parallel not found. Install with:"
  echo "  sudo apt install -y parallel"
  exit 1
}

# Find latest commands file
CMD_FILE="$(ls -t commands_*.txt 2>/dev/null | head -n 1)"

if [[ -z "$CMD_FILE" ]]; then
  echo "[ERROR] No commands_*.txt found. Run ./run_all.sh first."
  exit 1
fi

echo "[INFO] Using command file: $CMD_FILE"
echo "[INFO] Running with parallel -j $N_JOBS"
echo "------------------------------------------"

parallel -j "$N_JOBS" < "$CMD_FILE"

echo "------------------------------------------"
echo "[DONE] All jobs completed."
