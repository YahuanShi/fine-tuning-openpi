#!/usr/bin/env python3
"""
Interactive dataset browser — navigate all episodes with arrow keys.

Usage:
    python browse_dataset.py <dataset_dir> [--fps 15]
    python browse_dataset.py dataset/ur5_dataset_20260313
"""

import argparse
import glob
import os
import sys
import h5py
import cv2
import numpy as np


def load_episode(path):
    with h5py.File(path, "r") as f:
        return {
            "action":   f["action"][:],
            "qpos":     f["observations/qpos"][:],
            "exterior": f["observations/images/exterior_image_1_left"][:],
            "wrist":    f["observations/images/wrist_image_left"][:],
        }


def build_canvas(exterior_rgb, wrist_rgb, qpos_t, qmin, qrange, t, T,
                 ep_idx, n_eps, ep_name, paused):
    H, W = exterior_rgb.shape[:2]
    frame_w = W * 2 + 8
    bar_h = 60

    ext_bgr   = cv2.cvtColor(exterior_rgb, cv2.COLOR_RGB2BGR)
    wrist_bgr = cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR)
    sep = np.zeros((H, 8, 3), dtype=np.uint8)
    frame = np.concatenate([ext_bgr, sep, wrist_bgr], axis=1)

    # joint bar
    bar = np.zeros((bar_h, frame_w, 3), dtype=np.uint8)
    n_joints = len(qpos_t)
    slot_w = frame_w // n_joints
    for j in range(n_joints):
        norm  = float(np.clip((qpos_t[j] - qmin[j]) / qrange[j], 0, 1))
        filled = int(norm * (slot_w - 6))
        x0 = j * slot_w + 3
        cv2.rectangle(bar, (x0, 10), (x0 + slot_w - 6, bar_h - 10), (60, 60, 60), -1)
        cv2.rectangle(bar, (x0, 10), (x0 + filled,     bar_h - 10), (0, 200, 100), -1)
        cv2.putText(bar, f"j{j}", (x0 + 2, bar_h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

    canvas = np.concatenate([frame, bar], axis=0)

    # overlays
    status = "PAUSED" if paused else "PLAYING"
    ep_label = f"ep {ep_idx+1}/{n_eps}: {os.path.basename(ep_name)}"
    cv2.putText(canvas, ep_label, (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(canvas, f"t={t}/{T-1}  [{status}]", (6, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(canvas, "exterior", (6, H - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
    cv2.putText(canvas, "wrist", (W + 14, H - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)

    # progress bar along bottom of image area
    prog = int((t / max(T - 1, 1)) * frame_w)
    cv2.rectangle(canvas, (0, H - 4), (prog, H), (0, 160, 255), -1)

    return canvas


def browse(episode_paths, fps):
    ep_idx = 0
    t = 0
    paused = False
    data = None

    delay = max(1, int(1000 / fps))
    WIN = "Dataset Browser  (LEFT/RIGHT=episode  SPACE=pause  Q=quit)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    while True:
        # load episode on first visit or when index changes
        if data is None:
            path = episode_paths[ep_idx]
            print(f"Loading {path} ...")
            data = load_episode(path)
            T = len(data["qpos"])
            qmin  = data["qpos"].min(axis=0)
            qrange = data["qpos"].max(axis=0) - qmin
            qrange = np.where(qrange > 1e-6, qrange, 1.0)
            t = 0

        canvas = build_canvas(
            data["exterior"][t], data["wrist"][t],
            data["qpos"][t], qmin, qrange,
            t, T, ep_idx, len(episode_paths),
            episode_paths[ep_idx], paused,
        )
        cv2.imshow(WIN, canvas)

        key = cv2.waitKey(1 if not paused else 50) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == 81 or key == ord("a"):   # left arrow or 'a'  → prev episode
            ep_idx = (ep_idx - 1) % len(episode_paths)
            data = None
        elif key == 83 or key == ord("d"):   # right arrow or 'd' → next episode
            ep_idx = (ep_idx + 1) % len(episode_paths)
            data = None
        elif key == 84 or key == ord("s"):   # down arrow or 's'  → step back
            t = max(0, t - 1)
            paused = True
        elif key == 82 or key == ord("w"):   # up arrow or 'w'    → step forward
            t = min(T - 1, t + 1)
            paused = True
        elif key == ord("r"):                # restart current episode
            t = 0

        if not paused:
            t += 1
            if t >= T:
                # auto-advance to next episode
                ep_idx = (ep_idx + 1) % len(episode_paths)
                data = None

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Browse all episodes in a dataset directory.")
    parser.add_argument("dataset_dir", help="Directory containing episode_*.hdf5 files")
    parser.add_argument("--fps", type=int, default=15, help="Playback FPS (default 15)")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dataset_dir, "episode_*.hdf5")))
    if not paths:
        print(f"No episode_*.hdf5 files found in {args.dataset_dir}")
        sys.exit(1)

    print(f"Found {len(paths)} episodes.")
    print("Controls: SPACE=pause  LEFT/RIGHT (or A/D)=prev/next episode")
    print("          UP/DOWN (or W/S)=step frame   R=restart   Q=quit\n")

    browse(paths, fps=args.fps)


if __name__ == "__main__":
    main()
