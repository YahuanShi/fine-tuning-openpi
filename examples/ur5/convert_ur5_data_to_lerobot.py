"""
Convert UR5 + Weiss CRG 30-050 teleoperation HDF5 episodes to LeRobot format.

Source format (written by teleoperation/data_collection/episode_recorder.py):

    episode_N.hdf5
    ├── attrs: sim, prompt, task, hz, n_steps, timestamp
    ├── observations/
    │   ├── qpos    (T, 7) float64  [joints_degx6, gripper 0=closed/1=open]
    │   └── images/
    │       ├── exterior_image_1_left  (T, 224, 224, 3) uint8 RGB
    │       └── wrist_image_left       (T, 224, 224, 3) uint8 RGB
    └── action      (T, 7) float64  [joints_degx6, gripper 0=closed/1=open]

Target LeRobot dataset features (matching the UR5 policy transforms in
examples/ur5/README.md):

    base_rgb     (224, 224, 3) uint8    exterior camera RGB
    wrist_rgb    (224, 224, 3) uint8    wrist camera RGB
    joints       (6,)          float32  joint angles in radians
    gripper      (1,)          float32  gripper [0.0=open, 1.0=closed]
    actions      (7,)          float32  [joints_radx6, gripper_pi05]

Conversions applied here:
    joints_rad   = deg2rad(qpos[:, :6])
    gripper_pi05 = 1.0 - qpos[:, 6]      ← invert convention
    action_rad   = deg2rad(action[:, :6])
    act_grip     = 1.0 - action[:, 6]    ← invert convention

Output is saved to openpi/dataset/<repo-id>/ (next to the raw HDF5 files).

Usage:
    # Convert today's recording (defaults auto-match episode_recorder.py paths):
    uv run examples/ur5/convert_ur5_data_to_lerobot.py

    # Specify a different raw-dir or dataset name explicitly:
    uv run examples/ur5/convert_ur5_data_to_lerobot.py \\
        --raw-dir openpi/dataset/ur5_dataset_20241201 \\
        --repo-id ur5_dataset_20241201
"""

import argparse
import datetime
import os
from pathlib import Path
import shutil

# ── Redirect LeRobot storage to openpi/dataset/ ──────────────────────────────
# Must be done BEFORE importing lerobot (HF_LEROBOT_HOME is read at import time).
_OPENPI_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
_TODAY = datetime.datetime.now(tz=datetime.UTC).date().strftime("%Y%m%d")
_DEFAULT_REPO_ID = f"ur5_dataset_{_TODAY}"

os.environ.setdefault("HF_LEROBOT_HOME", str(_OPENPI_ROOT / "dataset"))

import h5py  # noqa: E402
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME  # noqa: E402
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
import numpy as np  # noqa: E402

# Recording frequency used by episode_recorder.py (change if you used --hz)
DEFAULT_FPS = 15


def build_parser() -> argparse.ArgumentParser:
    _default_raw_dir = _OPENPI_ROOT / "dataset" / _DEFAULT_REPO_ID
    p = argparse.ArgumentParser(
        description="Convert UR5 HDF5 episodes to LeRobot format for pi0.5 training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=_default_raw_dir,
        help="Directory containing episode_N.hdf5 files",
    )
    p.add_argument(
        "--repo-id",
        type=str,
        default=_DEFAULT_REPO_ID,
        help="Output dataset name (stored under openpi/dataset/<repo-id>/)",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help="Recording frequency (must match --hz used during data collection)",
    )
    return p


def convert(raw_dir: Path, repo_id: str, fps: int) -> None:
    hdf5_files = sorted(raw_dir.glob("episode_*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No episode_*.hdf5 files found in {raw_dir}")
    print(f"Found {len(hdf5_files)} episode(s) in {raw_dir}")

    # ── Clean up existing dataset ─────────────────────────────────────────────
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)

    # ── Create empty LeRobot dataset ─────────────────────────────────────────
    # Feature names match what LeRobotUR5DataConfig's RepackTransform expects
    # (see examples/ur5/README.md).
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="ur5e",
        fps=fps,
        features={
            "base_rgb": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_rgb": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            # Stored separately so UR5Inputs can concatenate them.
            "joints": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["joints"],
            },
            "gripper": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper"],
            },
            # Concatenated [joints_radx6, gripper_pi05x1]
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=4,
        image_writer_processes=2,
    )

    # ── Process each episode ─────────────────────────────────────────────────
    for ep_path in hdf5_files:
        print(f"  Processing {ep_path.name} …")
        with h5py.File(ep_path, "r") as ep:
            # ── Read raw data ─────────────────────────────────────────────
            qpos_deg = ep["/observations/qpos"][:]  # (T, 7) float64
            action_deg = ep["/action"][:]  # (T, 7) float64
            imgs_ext = ep["/observations/images/exterior_image_1_left"][:]  # (T,224,224,3)
            imgs_wrist = ep["/observations/images/wrist_image_left"][:]  # (T,224,224,3)
            prompt = str(ep.attrs.get("prompt", ""))

        n_steps = qpos_deg.shape[0]
        if n_steps == 0:
            print(f"    Skipping empty episode {ep_path.name}")
            continue

        # ── Unit/convention conversions ───────────────────────────────────
        # Joint angles: degrees → radians
        joints_rad = np.deg2rad(qpos_deg[:, :6]).astype(np.float32)  # (n_steps, 6)

        # Gripper: episode_recorder stores 0=closed/1=open
        #          pi0.5 norm stats expect            0=open /1=closed
        gripper_pi05 = (1.0 - qpos_deg[:, 6:7]).astype(np.float32)  # (n_steps, 1)

        act_joints = np.deg2rad(action_deg[:, :6]).astype(np.float32)  # (n_steps, 6)
        act_gripper = (1.0 - action_deg[:, 6:7]).astype(np.float32)  # (n_steps, 1)
        actions = np.concatenate([act_joints, act_gripper], axis=1)  # (n_steps, 7)

        # ── Add frames ────────────────────────────────────────────────────
        for t in range(n_steps):
            dataset.add_frame(
                {
                    "base_rgb": imgs_ext[t],  # uint8 (224, 224, 3)
                    "wrist_rgb": imgs_wrist[t],  # uint8 (224, 224, 3)
                    "joints": joints_rad[t],  # float32 (6,)
                    "gripper": gripper_pi05[t],  # float32 (1,)
                    "actions": actions[t],  # float32 (7,)
                }
            )

        # task= is stored per-episode and becomes the LeRobot "task" / "prompt" field
        dataset.save_episode(task=prompt)
        print(f"    Saved {n_steps} steps, prompt: '{prompt[:60]}'")

    # ── Finalise ─────────────────────────────────────────────────────────────
    dataset.consolidate(run_compute_stats=True)
    print(f"\nDataset saved to {output_path}")


if __name__ == "__main__":
    args = build_parser().parse_args()
    convert(
        raw_dir=args.raw_dir,
        repo_id=args.repo_id,
        fps=args.fps,
    )
