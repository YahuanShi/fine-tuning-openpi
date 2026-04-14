#!/usr/bin/env bash
# UR5 fine-tuning pipeline: convert → norm stats → train
#
# Usage:
#   ./examples/ur5/train_pipeline.sh --raw-dir dataset/2_processed/no_front/<DATE> --repo-id ur5_dataset_<DATE> --exp-name ur5_pick_place_<VERSION>
#
# All flags:
#   --raw-dir    DIR    Directory containing episode_*.hdf5 files  (required)
#   --repo-id    ID     Dataset name for LeRobot output             (required)
#   --exp-name   NAME   Training experiment name                    (required)
#   --fps        N      Recording frequency, default 10
#   --config     NAME   Training config name, default pi05_ur5
#   --skip-convert      Skip conversion step (dataset already converted)
#   --skip-stats        Skip norm stats step (stats already computed)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ── Defaults ─────────────────────────────────────────────────────────────────
RAW_DIR=""
REPO_ID=""
EXP_NAME=""
FPS=10
CONFIG_NAME="pi05_ur5"
SKIP_CONVERT=0
SKIP_STATS=0

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --raw-dir)    RAW_DIR="$2";    shift 2 ;;
        --repo-id)    REPO_ID="$2";    shift 2 ;;
        --exp-name)   EXP_NAME="$2";   shift 2 ;;
        --fps)        FPS="$2";        shift 2 ;;
        --config)     CONFIG_NAME="$2"; shift 2 ;;
        --skip-convert) SKIP_CONVERT=1; shift ;;
        --skip-stats)   SKIP_STATS=1;   shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Validate required args ────────────────────────────────────────────────────
if [[ -z "$REPO_ID" || -z "$EXP_NAME" ]]; then
    echo "Error: --repo-id and --exp-name are required."
    echo "Usage: $0 --raw-dir <dir> --repo-id <id> --exp-name <name>"
    exit 1
fi
if [[ $SKIP_CONVERT -eq 0 && -z "$RAW_DIR" ]]; then
    echo "Error: --raw-dir is required unless --skip-convert is set."
    exit 1
fi

export HF_LEROBOT_HOME="$PROJECT_ROOT/dataset/for_training"

echo "========================================================"
echo " UR5 Fine-Tuning Pipeline"
echo "========================================================"
echo "  raw-dir   : ${RAW_DIR:-[skipped]}"
echo "  repo-id   : $REPO_ID"
echo "  exp-name  : $EXP_NAME"
echo "  fps       : $FPS"
echo "  config    : $CONFIG_NAME"
echo "========================================================"

# ── Step 1: Convert ───────────────────────────────────────────────────────────
if [[ $SKIP_CONVERT -eq 0 ]]; then
    echo ""
    echo "[1/3] Converting HDF5 → LeRobot dataset..."
    uv run examples/ur5/convert_ur5_data_to_lerobot.py \
        --raw-dir "$RAW_DIR" \
        --repo-id "$REPO_ID"
    echo "[1/3] Done."
else
    echo "[1/3] Skipped (--skip-convert)."
fi

# ── Step 2: Norm stats ────────────────────────────────────────────────────────
if [[ $SKIP_STATS -eq 0 ]]; then
    echo ""
    echo "[2/3] Computing norm stats..."
    uv run scripts/compute_norm_stats.py --config-name "$CONFIG_NAME"
    echo "[2/3] Done."
else
    echo "[2/3] Skipped (--skip-stats)."
fi

# ── Step 3: Train ─────────────────────────────────────────────────────────────
echo ""
echo "[3/3] Starting training (exp: $EXP_NAME)..."
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
uv run scripts/train.py "$CONFIG_NAME" \
    --exp-name "$EXP_NAME" \
    --overwrite
