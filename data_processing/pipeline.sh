#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# pipeline.sh — Full data-processing pipeline for UR5 episodes
#
# Steps:
#   1. check_dataset      — quality report
#   2. visualize_episode  — interactive review / delete bad episodes
#   3. drop_front_camera  — remove front_image_1 stream
#   4. trim_episodes      — cut start/end frames via cuts.json
#   5. smooth_episodes    — Savitzky-Golay trajectory smoothing
#   → training_dataset/   (final output)
#
# Usage:
#   ./pipeline.sh <raw_dataset_dir> [options]
#
# Options:
#   --cuts   FILE   cuts.json from visualize_trajectory.py  (default: cuts.json)
#   --trim   N      global fallback: frames to cut each end (default: 0)
#   --window N      SG filter window, must be odd           (default: 15)
#   --poly   N      SG polynomial order                     (default: 3)
#   --out    DIR    final output directory                   (default: training_dataset)
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── defaults ──────────────────────────────────────────────────────────────────
RAW_DIR=""
CUTS_FILE="cuts.json"
GLOBAL_TRIM=0
WINDOW=15
POLY=3
OUT_DIR="training_dataset"

# ── parse args ────────────────────────────────────────────────────────────────
usage() {
    grep '^#' "$0" | sed 's/^# \{0,1\}//' | head -20
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuts)   CUTS_FILE="$2";   shift 2 ;;
        --trim)   GLOBAL_TRIM="$2"; shift 2 ;;
        --window) WINDOW="$2";      shift 2 ;;
        --poly)   POLY="$2";        shift 2 ;;
        --out)    OUT_DIR="$2";     shift 2 ;;
        -h|--help) usage ;;
        -*) echo "Unknown option: $1"; usage ;;
        *)  RAW_DIR="$1"; shift ;;
    esac
done

[[ -z "$RAW_DIR" ]] && { echo "ERROR: raw_dataset_dir is required."; usage; }
[[ -d "$RAW_DIR" ]] || { echo "ERROR: '$RAW_DIR' is not a directory."; exit 1; }

RAW_DIR="${RAW_DIR%/}"
STAGE_NO_FRONT="${RAW_DIR}_no_front"
STAGE_TRIMMED="${RAW_DIR}_trimmed"

# ── helper ────────────────────────────────────────────────────────────────────
banner() { echo; echo "════════════════════════════════════════════════════════"; echo " $1"; echo "════════════════════════════════════════════════════════"; }
confirm() {
    echo
    read -rp ">>> $1  Continue? [y/N] " ans
    [[ "$ans" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
}

# ── Step 1: quality check ─────────────────────────────────────────────────────
banner "Step 1 / 5 — Quality check"
python3 "$SCRIPT_DIR/check_dataset.py" "$RAW_DIR" || true
confirm "Review the report above."

# ── Step 2: visual review (interactive) ──────────────────────────────────────
banner "Step 2 / 5 — Visual review   (D+Y to delete bad episodes, Q when done)"
python3 "$SCRIPT_DIR/visualize_episode.py" "$RAW_DIR" || true
confirm "Visual review complete."

# ── Step 3: drop front camera ─────────────────────────────────────────────────
banner "Step 3 / 5 — Drop front camera  →  $STAGE_NO_FRONT"
python3 "$SCRIPT_DIR/drop_front_camera.py" "$RAW_DIR" "$STAGE_NO_FRONT"

# ── Step 4: trim episodes ─────────────────────────────────────────────────────
banner "Step 4 / 5 — Trim episodes  →  $STAGE_TRIMMED"
TRIM_ARGS=("$STAGE_NO_FRONT" --output "$STAGE_TRIMMED")
[[ -f "$CUTS_FILE" ]] && TRIM_ARGS+=(--cuts "$CUTS_FILE")
[[ "$GLOBAL_TRIM" -gt 0 ]] && TRIM_ARGS+=(--trim "$GLOBAL_TRIM")
python3 "$SCRIPT_DIR/trim_episodes.py" "${TRIM_ARGS[@]}"

# ── Step 5: smooth trajectories ───────────────────────────────────────────────
banner "Step 5 / 5 — Smooth trajectories  →  $OUT_DIR"
python3 "$SCRIPT_DIR/smooth_episodes.py" "$STAGE_TRIMMED" \
    --output "$OUT_DIR" --window "$WINDOW" --poly "$POLY"

# ── done ──────────────────────────────────────────────────────────────────────
echo
echo "✓  Pipeline complete."
echo "   Final dataset  : $OUT_DIR"
echo "   Intermediates  : $STAGE_NO_FRONT  |  $STAGE_TRIMMED  (safe to delete)"
