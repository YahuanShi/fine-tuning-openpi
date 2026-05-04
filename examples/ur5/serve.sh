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

# Read metadata.json: prefer explicit config + repo_id fields
METADATA="$ASSETS_DIR/metadata.json"
CONFIG=""
REPO_ID=""

if [[ -f "$METADATA" ]]; then
    CONFIG=$(python3 -c "import json; d=json.load(open('$METADATA')); print(d.get('config',''))" 2>/dev/null || true)
    REPO_ID=$(python3 -c "import json; d=json.load(open('$METADATA')); print(d.get('repo_id',''))" 2>/dev/null || true)
fi

# Fallback: infer repo_id from assets directory name
if [[ -z "$REPO_ID" ]]; then
    REPO_ID=$(ls "$ASSETS_DIR" | grep -v '^metadata' | head -1)
fi

if [[ -z "$REPO_ID" ]]; then
    echo "Error: cannot determine repo_id from $ASSETS_DIR"
    exit 1
fi

# If config not set via metadata, map repo_id → config
if [[ -z "$CONFIG" ]]; then
    case "$REPO_ID" in
        ur5_dataset_20260402_assembly)  CONFIG="pi05_ur5_assembly" ;;
        ur5_dataset_20260402_pnpa)      CONFIG="pi05_ur5_pnpa" ;;
        ur5_dataset_20260402)
            # Legacy: infer from experiment name in path
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
fi

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

export UR5_REPO_ID="$REPO_ID"

uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config "$CONFIG" \
    --policy.dir "$CKPT_DIR"
