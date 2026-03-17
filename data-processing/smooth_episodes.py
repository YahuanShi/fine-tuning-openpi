#!/usr/bin/env python3
"""
Apply Savitzky-Golay smoothing to qpos and action trajectories in HDF5 episodes.

The gripper dimension (last column) is intentionally left unsmoothed to
preserve discrete open/close transitions.

Usage:
    python3 smooth_episodes.py path/to/dataset_dir --output smoothed/
    python3 smooth_episodes.py path/to/dataset_dir --output smoothed/ --window 9 --poly 2
"""

import argparse
import glob
import os
import sys

import h5py
import numpy as np
from scipy.signal import savgol_filter


DEFAULT_WINDOW = 5   # must be odd
DEFAULT_POLY   = 2


def smooth_array(arr: np.ndarray, window: int, poly: int) -> np.ndarray:
    """Savitzky-Golay column-wise; last column (gripper) is left unchanged."""
    if len(arr) < window:
        return arr.copy()
    out = arr.copy()
    for d in range(arr.shape[1] - 1):
        out[:, d] = savgol_filter(arr[:, d], window_length=window, polyorder=poly)
    return out


def smooth_episode(src: str, dst: str, window: int, poly: int) -> None:
    with h5py.File(src, "r") as f:
        qpos     = f["observations/qpos"][:]
        action   = f["action"][:]
        exterior = f["observations/images/exterior_image_1_left"][:]
        wrist    = f["observations/images/wrist_image_left"][:]

    with h5py.File(dst, "w") as f:
        f.create_dataset("observations/qpos",   data=smooth_array(qpos,   window, poly),
                         compression="gzip")
        f.create_dataset("action",               data=smooth_array(action, window, poly),
                         compression="gzip")
        f.create_dataset("observations/images/exterior_image_1_left",
                         data=exterior, compression="gzip")
        f.create_dataset("observations/images/wrist_image_left",
                         data=wrist,    compression="gzip")


def main():
    parser = argparse.ArgumentParser(
        description="Smooth qpos and action with Savitzky-Golay filter.")
    parser.add_argument("dataset_dir",
                        help="Directory containing episode_*.hdf5 files")
    parser.add_argument("--output", "-o", required=True,
                        help="Output directory for smoothed files")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                        help=f"Filter window length (odd, default {DEFAULT_WINDOW})")
    parser.add_argument("--poly",   type=int, default=DEFAULT_POLY,
                        help=f"Polynomial order (default {DEFAULT_POLY})")
    args = parser.parse_args()

    if args.window % 2 == 0:
        sys.exit("ERROR: --window must be odd")
    if args.poly >= args.window:
        sys.exit("ERROR: --poly must be less than --window")

    files = sorted(glob.glob(os.path.join(args.dataset_dir, "*.hdf5")))
    if not files:
        sys.exit(f"No .hdf5 files found in: {args.dataset_dir}")

    os.makedirs(args.output, exist_ok=True)
    print(f"Smoothing {len(files)} episode(s)  "
          f"(window={args.window}, poly={args.poly})  →  {args.output}")

    for path in files:
        name = os.path.basename(path)
        dst  = os.path.join(args.output, name)
        smooth_episode(path, dst, args.window, args.poly)
        print(f"  WROTE {dst}")

    print("Done.")


if __name__ == "__main__":
    main()
