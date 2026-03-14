#!/usr/bin/env python3
"""
Visualize a single episode from the UR5 HDF5 dataset.

Usage:
    python visualize_episode.py <path/to/episode.hdf5> [--fps 30] [--save]
    python visualize_episode.py dataset/ur5_dataset_20260313/episode_0.hdf5
"""

import argparse
import sys
import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


JOINT_NAMES = [f"joint_{i}" for i in range(7)]


def load_episode(path):
    with h5py.File(path, "r") as f:
        data = {
            "action": f["action"][:],
            "qpos": f["observations/qpos"][:],
            "exterior": f["observations/images/exterior_image_1_left"][:],
            "wrist": f["observations/images/wrist_image_left"][:],
        }
    return data


def play_video(data, fps: int, save_path: str | None = None):
    """Show side-by-side camera streams with a joint-position bar overlay."""
    exterior = data["exterior"]   # (T, H, W, 3) RGB
    wrist = data["wrist"]         # (T, H, W, 3) RGB
    qpos = data["qpos"]           # (T, 7)
    action = data["action"]       # (T, 7)
    T = len(exterior)

    H, W = exterior.shape[1], exterior.shape[2]
    frame_w = W * 2 + 8          # two frames side by side with gap
    bar_h = 60
    frame_h = H + bar_h

    delay = max(1, int(1000 / fps))

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (frame_w, frame_h))

    qmin, qmax = qpos.min(axis=0), qpos.max(axis=0)
    qrange = np.where((qmax - qmin) > 1e-6, qmax - qmin, 1.0)

    for t in range(T):
        ext_bgr = cv2.cvtColor(exterior[t], cv2.COLOR_RGB2BGR)
        wrist_bgr = cv2.cvtColor(wrist[t], cv2.COLOR_RGB2BGR)

        # side-by-side with a thin separator
        sep = np.zeros((H, 8, 3), dtype=np.uint8)
        frame = np.concatenate([ext_bgr, sep, wrist_bgr], axis=1)

        # bottom bar: joint position bars
        bar = np.zeros((bar_h, frame_w, 3), dtype=np.uint8)
        n_joints = qpos.shape[1]
        slot_w = frame_w // n_joints
        for j in range(n_joints):
            norm = (qpos[t, j] - qmin[j]) / qrange[j]
            filled = int(norm * (slot_w - 6))
            x0 = j * slot_w + 3
            cv2.rectangle(bar, (x0, 10), (x0 + slot_w - 6, bar_h - 10), (60, 60, 60), -1)
            cv2.rectangle(bar, (x0, 10), (x0 + filled, bar_h - 10), (0, 200, 100), -1)
            cv2.putText(bar, f"j{j}", (x0 + 2, bar_h - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        canvas = np.concatenate([frame, bar], axis=0)

        # overlay: step counter + camera labels
        cv2.putText(canvas, f"t={t}/{T-1}", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
        cv2.putText(canvas, "exterior", (6, H - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(canvas, "wrist", (W + 14, H - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow("Episode Viewer  (q=quit  SPACE=pause)", canvas)
        if writer:
            writer.write(canvas)

        key = cv2.waitKey(delay) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            cv2.waitKey(0)   # pause until next SPACE

    cv2.destroyAllWindows()
    if writer:
        writer.release()
        print(f"Saved video to {save_path}")


def plot_trajectories(data, title: str):
    """Plot qpos and action trajectories for all joints."""
    qpos = data["qpos"]
    action = data["action"]
    T = len(qpos)
    t = np.arange(T)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(title, fontsize=13)

    for j in range(qpos.shape[1]):
        axes[0].plot(t, qpos[:, j], label=f"j{j}")
    axes[0].set_ylabel("qpos (rad)")
    axes[0].legend(loc="upper right", ncol=4, fontsize=8)
    axes[0].grid(True, alpha=0.3)

    for j in range(action.shape[1]):
        axes[1].plot(t, action[:, j], label=f"j{j}")
    axes[1].set_ylabel("action (rad)")
    axes[1].set_xlabel("timestep")
    axes[1].legend(loc="upper right", ncol=4, fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def print_summary(data, path: str):
    qpos = data["qpos"]
    action = data["action"]
    print(f"\n{'='*50}")
    print(f"  Episode : {path}")
    print(f"  Steps   : {len(qpos)}")
    print(f"  Cameras : exterior {data['exterior'].shape[1:]}, wrist {data['wrist'].shape[1:]}")
    print(f"  qpos    : min={qpos.min():.3f}  max={qpos.max():.3f}")
    print(f"  action  : min={action.min():.3f}  max={action.max():.3f}")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize a UR5 episode HDF5 file.")
    parser.add_argument("episode", help="Path to episode .hdf5 file")
    parser.add_argument("--fps", type=int, default=15, help="Playback speed (default 15)")
    parser.add_argument("--save", metavar="OUT.mp4", default=None,
                        help="Save video instead of (or in addition to) displaying")
    parser.add_argument("--plot", action="store_true",
                        help="Show joint trajectory plots instead of video")
    args = parser.parse_args()

    print(f"Loading {args.episode} ...")
    data = load_episode(args.episode)
    print_summary(data, args.episode)

    if args.plot:
        plot_trajectories(data, title=args.episode)
    else:
        play_video(data, fps=args.fps, save_path=args.save)


if __name__ == "__main__":
    main()
