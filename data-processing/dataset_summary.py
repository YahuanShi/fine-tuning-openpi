#!/usr/bin/env python3
"""
Print a quick summary table + joint-trajectory overview for all episodes.

Usage:
    python dataset_summary.py <dataset_dir>
    python dataset_summary.py dataset/ur5_dataset_20260313
"""

import argparse
import glob
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt


def summarize(dataset_dir):
    paths = sorted(glob.glob(os.path.join(dataset_dir, "episode_*.hdf5")))
    if not paths:
        print(f"No episode_*.hdf5 files found in {dataset_dir}")
        return

    rows = []
    all_qpos   = []
    all_action = []

    print(f"\n{'Episode':<20} {'Steps':>6} {'qpos min':>10} {'qpos max':>10} "
          f"{'act min':>10} {'act max':>10}  Size")
    print("-" * 80)

    for p in paths:
        name = os.path.basename(p)
        size_mb = os.path.getsize(p) / 1e6
        with h5py.File(p, "r") as f:
            steps  = f["action"].shape[0]
            qpos   = f["observations/qpos"][:]
            action = f["action"][:]
        print(f"{name:<20} {steps:>6} {qpos.min():>10.3f} {qpos.max():>10.3f} "
              f"{action.min():>10.3f} {action.max():>10.3f}  {size_mb:.1f} MB")
        all_qpos.append(qpos)
        all_action.append(action)
        rows.append((name, steps, qpos, action))

    total_steps = sum(r[1] for r in rows)
    print("-" * 80)
    print(f"{'TOTAL':<20} {total_steps:>6} episodes={len(rows)}\n")

    # plot all trajectories overlaid
    n_joints = rows[0][2].shape[1]
    fig, axes = plt.subplots(n_joints, 2, figsize=(16, 2.2 * n_joints), sharex=False)
    fig.suptitle(f"All episodes — {os.path.basename(dataset_dir)}", fontsize=13)

    colors = plt.cm.tab10.colors
    for ep_i, (name, steps, qpos, action) in enumerate(rows):
        c = colors[ep_i % len(colors)]
        t = np.arange(steps)
        for j in range(n_joints):
            axes[j, 0].plot(t, qpos[:, j],   color=c, alpha=0.7, linewidth=0.8, label=name if j == 0 else "")
            axes[j, 1].plot(t, action[:, j],  color=c, alpha=0.7, linewidth=0.8)

    for j in range(n_joints):
        axes[j, 0].set_ylabel(f"j{j}", fontsize=9)
        axes[j, 0].grid(True, alpha=0.25)
        axes[j, 1].grid(True, alpha=0.25)

    axes[0, 0].set_title("qpos (rad)")
    axes[0, 1].set_title("action (rad)")
    axes[0, 0].legend(fontsize=7, loc="upper right", ncol=3)
    axes[-1, 0].set_xlabel("timestep")
    axes[-1, 1].set_xlabel("timestep")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Summarize all episodes in a dataset directory.")
    parser.add_argument("dataset_dir", help="Directory containing episode_*.hdf5 files")
    args = parser.parse_args()
    summarize(args.dataset_dir)


if __name__ == "__main__":
    main()
