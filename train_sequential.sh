#!/usr/bin/env bash
# Sequential pipeline: convert → norm stats → train, for two datasets back-to-back.
#
# Usage:
#   ./train_sequential.sh
#
# Runs:
#   1. Convert + norm stats + train  ur5_dataset_20260402_assembly  (pi05_ur5_assembly)
#   2. Convert + norm stats + train  ur5_dataset_20260402_pnpa      (pi05_ur5_pnpa)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export HF_LEROBOT_HOME="$SCRIPT_DIR/dataset/for_training"

log() { echo ""; echo "════════════════════════════════════════"; echo " $*"; echo "════════════════════════════════════════"; }

# ── Job definitions ───────────────────────────────────────────────────────────
declare -a RAW_DIRS=("dataset/processed/trimmed/20260402_assembly"
                     "dataset/processed/trimmed/20260402_pnpa")
declare -a REPO_IDS=("ur5_dataset_20260402_assembly"
                     "ur5_dataset_20260402_pnpa")
declare -a CONFIGS=("pi05_ur5_assembly"
                    "pi05_ur5_pnpa")
declare -a EXP_NAMES=("ur5_assembly_v1"
                      "ur5_pnpa_v1")
FPS=20

# ── Run each job ──────────────────────────────────────────────────────────────
for i in 0 1; do
    RAW_DIR="${RAW_DIRS[$i]}"
    REPO_ID="${REPO_IDS[$i]}"
    CONFIG="${CONFIGS[$i]}"
    EXP_NAME="${EXP_NAMES[$i]}"

    log "JOB $((i+1))/2: $CONFIG  ($REPO_ID)"

    # Step 1: Convert
    log "[${EXP_NAME}] Step 1/3 — Converting HDF5 → LeRobot..."
    uv run examples/ur5/convert_ur5_data_to_lerobot.py \
        --raw-dir "$RAW_DIR" \
        --repo-id "$REPO_ID" \
        --fps "$FPS"

    # Step 2: Norm stats
    log "[${EXP_NAME}] Step 2/3 — Computing norm stats..."
    uv run scripts/compute_norm_stats.py --config-name "$CONFIG"

    # Step 3: Train
    log "[${EXP_NAME}] Step 3/3 — Training..."
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
    uv run scripts/train.py "$CONFIG" \
        --exp-name "$EXP_NAME" \
        --overwrite

    log "JOB $((i+1))/2 DONE: $EXP_NAME"
done

log "ALL DONE."
