#!/usr/bin/env bash
# Serve a Pi0.5 checkpoint, auto-detecting the correct config.
#
# For new checkpoints: reads repo_id from assets/metadata.json (written at training time).
# For old checkpoints: falls back to assets directory name inference.
#
# Usage:
#   ./examples/ur5/serve.sh <checkpoint_dir>
#
# Example:
#   ./examples/ur5/serve.sh checkpoints/pi05_ur5/ur5_pnpa_v2/19999

set -euo pipefail

CKPT_DIR="${1:-}"
if [[ -z "$CKPT_DIR" ]]; then
    echo "Usage: $0 <checkpoint_dir>"
    echo "Example: $0 checkpoints/pi05_ur5/ur5_pnpa_v2/19999"
    exit 1
fi

ASSETS_DIR="$CKPT_DIR/assets"
if [[ ! -d "$ASSETS_DIR" ]]; then
    echo "Error: assets directory not found: $ASSETS_DIR"
    exit 1
fi

# Prefer metadata.json (written by training since the fix)
METADATA="$ASSETS_DIR/metadata.json"
if [[ -f "$METADATA" ]]; then
    REPO_ID=$(python3 -c "import json; print(json.load(open('$METADATA'))['repo_id'])")
else
    # Fallback: infer from assets directory name
    REPO_ID=$(ls "$ASSETS_DIR" | head -1)
fi

if [[ -z "$REPO_ID" ]]; then
    echo "Error: cannot determine repo_id from $ASSETS_DIR"
    exit 1
fi

# Map repo_id → config name
case "$REPO_ID" in
    ur5_dataset_20260402_assembly)  CONFIG="pi05_ur5_assembly" ;;
    ur5_dataset_20260402_pnpa)      CONFIG="pi05_ur5_pnpa" ;;
    ur5_dataset_20260402)
        # Old checkpoints trained before separate configs existed; infer from path
        if [[ "$CKPT_DIR" == *pnpa* ]]; then
            CONFIG="pi05_ur5_pnpa"
            ALIAS="ur5_dataset_20260402_pnpa"
        elif [[ "$CKPT_DIR" == *assembly* ]]; then
            CONFIG="pi05_ur5_assembly"
            ALIAS="ur5_dataset_20260402_assembly"
        else
            CONFIG="pi05_ur5"
            ALIAS=""
        fi
        if [[ -n "${ALIAS:-}" && ! -d "$ASSETS_DIR/$ALIAS" ]]; then
            echo "Creating norm stats alias: $ASSETS_DIR/$ALIAS"
            cp -r "$ASSETS_DIR/$REPO_ID" "$ASSETS_DIR/$ALIAS"
        fi
        REPO_ID="${ALIAS:-$REPO_ID}"
        ;;
    ur5_dataset_*)  CONFIG="pi05_ur5" ;;
    *)
        echo "Error: unknown repo_id '$REPO_ID', cannot auto-detect config."
        echo "Available assets: $(ls "$ASSETS_DIR")"
        exit 1
        ;;
esac

# Ensure norm stats exist under the resolved repo_id
NORM_FILE="$ASSETS_DIR/$REPO_ID/norm_stats.json"
if [[ ! -f "$NORM_FILE" ]]; then
    echo "Error: norm stats not found: $NORM_FILE"
    exit 1
fi

echo "========================================"
echo " Serving checkpoint"
echo "  dir    : $CKPT_DIR"
echo "  config : $CONFIG"
echo "  repo_id: $REPO_ID"
echo "========================================"

uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config "$CONFIG" \
    --policy.dir "$CKPT_DIR"
