#!/usr/bin/env bash
# One-click training launcher.
#
# Usage:
#   ./scripts/train_oneclick.sh <repo_id> <exp_name> [extra train.py args...]
#
# Example:
#   ./scripts/train_oneclick.sh ur5_dataset_20260415 ur5_pick_place_v5 --overwrite
#
# First run: bootstraps Docker + NVIDIA toolkit if missing, builds image (~10 min),
# then starts training. Subsequent runs skip bootstrap and reuse the cached image.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

ENV_FILE="scripts/docker/.env.train"
ENV_EXAMPLE="scripts/docker/.env.train.example"
COMPOSE_FILE="scripts/docker/compose.train.yml"
IMAGE_TAG="openpi_trainer"
CONFIG_NAME="pi05_ur5"

# ── Args ────────────────────────────────────────────────────────────────────
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <repo_id> <exp_name> [extra train.py args...]" >&2
    echo "Example: $0 ur5_dataset_20260415 ur5_pick_place_v5 --overwrite" >&2
    exit 1
fi
REPO_ID="$1"
EXP_NAME="$2"
shift 2
EXTRA_ARGS=("$@")

# ── 1. Host bootstrap (only if needed) ──────────────────────────────────────
if ! command -v docker &>/dev/null || ! command -v nvidia-ctk &>/dev/null; then
    echo "[oneclick] Host missing Docker or nvidia-ctk — running bootstrap_host.sh"
    bash scripts/docker/bootstrap_host.sh
fi

# ── 2. Env file ─────────────────────────────────────────────────────────────
if [[ ! -f "$ENV_FILE" ]]; then
    echo "[oneclick] $ENV_FILE not found." >&2
    echo "             cp $ENV_EXAMPLE $ENV_FILE  and fill in WANDB_API_KEY / HF_TOKEN." >&2
    exit 1
fi

# ── 3. Build image once ─────────────────────────────────────────────────────
if ! docker image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
    echo "[oneclick] Image '$IMAGE_TAG' not found — building (first run, ~10 min)..."
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" build
fi

# ── 4. Run training ─────────────────────────────────────────────────────────
echo "[oneclick] Training: repo_id=$REPO_ID exp_name=$EXP_NAME config=$CONFIG_NAME"
UR5_REPO_ID="$REPO_ID" docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" run --rm \
    -e UR5_REPO_ID="$REPO_ID" \
    trainer \
    uv run scripts/train.py "$CONFIG_NAME" --exp-name "$EXP_NAME" "${EXTRA_ARGS[@]}"
