#!/usr/bin/env python3
"""
Cut frames from the start and/or end of each episode using a cuts JSON file
produced by visualize_trajectory.py.

The cuts file maps episode filenames to start/end frame counts:
    {
      "episode_0.hdf5": {"start": 12, "end": 8},
      "episode_1.hdf5": {"start": 5,  "end": 0},
      ...
    }

Episodes not listed in the cuts file fall back to the global --trim value.

Usage:
    # Apply per-episode cuts from JSON
    python3 trim_episodes.py path/to/dataset_dir --cuts cuts.json --output trimmed/

    # Global fallback trim (same N frames from both ends for every episode)
    python3 trim_episodes.py path/to/dataset_dir --trim 5 --output trimmed/

    # Both: per-episode cuts take priority; --trim used for unlisted episodes
    python3 trim_episodes.py path/to/dataset_dir --cuts cuts.json --trim 5 --output trimmed/

    # Dry-run: print what would be cut without writing
    python3 trim_episodes.py path/to/dataset_dir --cuts cuts.json
"""

import argparse
import glob
import json
import os
import shutil
import sys

import h5py


def load_cuts(path: str) -> dict:
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def resolve_cut(ep_name: str, cuts: dict, global_trim: int) -> tuple[int, int]:
    if ep_name in cuts:
        c = cuts[ep_name]
        return c.get("start", 0), c.get("end", 0)
    return global_trim, global_trim


def trim_episode(src: str, dst: str, trim_start: int, trim_end: int) -> int:
    """Write trimmed copy of src to dst. Returns number of kept frames."""
    with h5py.File(src, "r") as f:
        qpos = f["observations/qpos"][:]
        action = f["action"][:]
        exterior = f["observations/images/exterior_image_1_left"][:]
        wrist = f["observations/images/wrist_image_left"][:]
        imgs = f["observations/images"]
        front = imgs["front_image_1"][:] if "front_image_1" in imgs else None
        attrs = dict(f.attrs)

    T = len(qpos)
    end_idx = T - trim_end if trim_end > 0 else T
    sl = slice(trim_start, end_idx)

    with h5py.File(dst, "w") as f:
        for k, v in attrs.items():
            f.attrs[k] = v
        f.create_dataset("observations/qpos", data=qpos[sl], compression="gzip")
        f.create_dataset("action", data=action[sl], compression="gzip")
        f.create_dataset("observations/images/exterior_image_1_left", data=exterior[sl], compression="gzip")
        f.create_dataset("observations/images/wrist_image_left", data=wrist[sl], compression="gzip")
        if front is not None:
            f.create_dataset("observations/images/front_image_1", data=front[sl], compression="gzip")
    return end_idx - trim_start


def main():
    parser = argparse.ArgumentParser(description="Trim episode start/end frames.")
    parser.add_argument("dataset_dir", help="Directory containing episode_*.hdf5 files")
    parser.add_argument("--output", "-o", default=None, help="Output directory (omit for dry-run)")
    parser.add_argument("--cuts", default=None, metavar="FILE", help="JSON cuts file from visualize_trajectory.py")
    parser.add_argument(
        "--trim",
        type=int,
        default=0,
        help="Global fallback: frames to cut from each end for episodes not listed in --cuts (default 0)",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.dataset_dir, "*.hdf5")))
    if not files:
        sys.exit(f"No .hdf5 files found in: {args.dataset_dir}")

    cuts = load_cuts(args.cuts)
    if args.cuts and not os.path.exists(args.cuts):
        print(f"WARNING: cuts file not found ({args.cuts}), using --trim={args.trim}")

    print(f"{'Episode':<30}  {'T_orig':>7}  {'cut_start':>9}  {'cut_end':>7}  {'T_kept':>7}")
    print("─" * 68)

    plan = []
    for path in files:
        name = os.path.basename(path)
        with h5py.File(path, "r") as f:
            T = f["observations/qpos"].shape[0]
        cs, ce = resolve_cut(name, cuts, args.trim)
        t_kept = T - cs - ce
        plan.append((path, name, T, cs, ce, t_kept))
        flag = "  ⚠ kept < 10" if t_kept < 10 else ""
        print(f"  {name:<28}  {T:>7}  {cs:>9}  {ce:>7}  {t_kept:>7}{flag}")

    print("─" * 68)

    if args.output is None:
        print("\nDry-run — pass --output <dir> to write files.")
        return

    os.makedirs(args.output, exist_ok=True)
    for path, name, _T, cs, ce, _t_kept in plan:
        dst = os.path.join(args.output, name)
        if cs == 0 and ce == 0:
            shutil.copy2(path, dst)
        else:
            trim_episode(path, dst, cs, ce)
        print(f"  WROTE {dst}")

    print(f"\nDone. {len(plan)} episodes → {args.output}")


if __name__ == "__main__":
    main()
