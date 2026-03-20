#!/usr/bin/env python3
"""
Copy all HDF5 episodes from src to dst, dropping the front_image_1 camera stream.

Usage:
    python3 drop_front_camera.py path/to/raw_dir path/to/no_front_dir
"""

import argparse
import glob
import os
import sys

import h5py


def drop_front(src: str, dst: str) -> None:
    with h5py.File(src, "r") as f:
        qpos     = f["observations/qpos"][:]
        action   = f["action"][:]
        exterior = f["observations/images/exterior_image_1_left"][:]
        wrist    = f["observations/images/wrist_image_left"][:]
        attrs    = dict(f.attrs)

    with h5py.File(dst, "w") as f:
        for k, v in attrs.items():
            f.attrs[k] = v
        f.create_dataset("observations/qpos",   data=qpos,     compression="gzip")
        f.create_dataset("action",               data=action,   compression="gzip")
        f.create_dataset("observations/images/exterior_image_1_left",
                         data=exterior, compression="gzip")
        f.create_dataset("observations/images/wrist_image_left",
                         data=wrist,    compression="gzip")


def main():
    parser = argparse.ArgumentParser(
        description="Copy episodes to a new directory, dropping front_image_1.")
    parser.add_argument("src", help="Source directory with episode_*.hdf5 files")
    parser.add_argument("dst", help="Destination directory")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.src, "*.hdf5")))
    if not files:
        sys.exit(f"No .hdf5 files found in: {args.src}")

    os.makedirs(args.dst, exist_ok=True)
    print(f"Dropping front camera from {len(files)} episode(s)  →  {args.dst}")
    for path in files:
        name = os.path.basename(path)
        drop_front(path, os.path.join(args.dst, name))
        print(f"  WROTE {name}")
    print("Done.")


if __name__ == "__main__":
    main()
