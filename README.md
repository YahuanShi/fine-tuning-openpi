# UR5 Pi0.5 Fine-Tuning

Fine-tuning [π₀.₅](https://www.physicalintelligence.company/blog/pi05) on a UR5e robot arm for pick-and-place tasks.
Forked from [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi).

---

## Hardware

| Component | Spec |
|-----------|------|
| Robot arm | Universal Robots UR5e |
| Gripper | Weiss Robotics CRG 30-050 (`/dev/ttyACM0`) |
| Exterior camera | Intel RealSense D415 (serial `105422061000`) |
| Wrist camera | Intel RealSense D405 (serial `352122273671`) |
| GPU | NVIDIA RTX 6000 48GB |
| Teleoperation | Uarm master arm → UR5 follower (ROS 2 Humble) |

---

## Task Configs

| Config | Dataset | Task | Latest Checkpoint |
|--------|---------|------|-------------------|
| `pi05_ur5` | `UR5_REPO_ID` env var | pick-and-place (general) | `ur5_pick_place_20260415/19999` |
| `pi05_ur5_assembly` | `ur5_dataset_20260402_assembly` | assembly | `ur5_assembly_v1/19999` |
| `pi05_ur5_pnpa` | `ur5_dataset_20260402_pnpa` | pick-and-place-and-arrange | `ur5_pnpa_v2/19999` |

### Checkpoint History

| Checkpoint | Trained On | Steps | Notes |
|------------|-----------|-------|-------|
| `ur5_pick_place_v3/19999` | `ur5_dataset_20260323` | 19999 | early |
| `ur5_pick_place_v4/19999` | `ur5_dataset_20260331` | 19999 | early |
| `ur5_pick_place_assembly_v1/19999` | `ur5_dataset_20260402` | 19999 | mixed tasks |
| `ur5_pnpa_v2/19999` | `ur5_dataset_20260402` | 19999 | pnpa only |
| `ur5_pick_place_20260415/19999` | `ur5_dataset_20260415` | 19999 | latest pick-and-place |

---

## Installation

```bash
git clone --recurse-submodules <this-repo>
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

---

## Full Pipeline

### 1. Data Collection (Teleoperation)

See [teleoperation/README.md](teleoperation/README.md) for setup and usage.

```bash
cd teleoperation
bash uarm/scripts/UR5/run_ur5_nodes.sh
```

Episodes are saved as `episode_N.hdf5` under `dataset/raw/`.

### 2. Data Processing

```bash
bash data_processing/pipeline/pipeline.sh --input dataset/raw/<DATE>
# Output: dataset/processed/trimmed/<DATE>/
```

### 3. Train (convert → norm stats → train)

One-command pipeline:

```bash
./examples/ur5/train_pipeline.sh \
    --raw-dir dataset/processed/trimmed/<DATE> \
    --repo-id ur5_dataset_<DATE> \
    --exp-name ur5_pick_place_<VERSION>
```

Optional flags: `--skip-convert`, `--skip-stats`, `--resume`, `--fps 20`, `--config pi05_ur5`

Or run steps manually — see [examples/ur5/README.md](examples/ur5/README.md).

### 4. Serve Policy

```bash
./examples/ur5/serve.sh checkpoints/pi05_ur5/<EXP_NAME>/<STEP>
```

`serve.sh` reads `assets/metadata.json` to resolve the correct config automatically.
First inference takes ~60s (JAX JIT). Subsequent calls: ~300–500ms.

### 5. Run Inference on Robot

```bash
PYTHONPATH=. uv run examples/ur5/main.py \
    --prompt "pick yellow cube and place it into red box"
```

Policy server (Step 4) must be running first. Both cannot share the GPU.

---

## Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `UR5_REPO_ID` | Dataset name for `pi05_ur5` config — **required** | `ur5_dataset_20260415` |
| `HF_LEROBOT_HOME` | LeRobot dataset root | `$(pwd)/dataset/for_training` |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | JAX GPU memory fraction | `0.95` |

`train_pipeline.sh` sets all three automatically.

---

## Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Collection Hz | 20 Hz | ~18 Hz also acceptable |
| Training Hz (`--fps`) | 20 Hz | match collection |
| Inference Hz | 10 Hz | least-noticeable pause at `action_horizon=10` |
| `action_horizon` | 10 | Pi0.5 hard limit, do not change |
| `batch_size` | 32 | max stable on RTX 6000 48GB (~41GB VRAM) |
| `ema_decay` | None | EMA adds ~12.5GB VRAM, causes OOM |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | 0.95 | required for training |
| Gripper convention | 0=open / 1=closed | inverted from raw HDF5 (0=closed / 1=open) |
| Joint angles | radians in model | HDF5 stores degrees; converted in convert script |

---

## Directory Structure

```
fine-tuning-openpi/
├── examples/ur5/           # UR5 scripts: convert, env, main, serve.sh, train_pipeline.sh
├── src/openpi/             # Core model library (JAX)
├── scripts/                # train.py, serve_policy.py, compute_norm_stats.py
├── data_processing/        # Data processing pipeline (git submodule)
├── teleoperation/          # ROS 2 teleoperation system (git submodule)
├── packages/openpi-client/ # Lightweight inference client package
├── dataset/
│   ├── raw/                # Original HDF5 recordings (never overwrite)
│   ├── processed/          # Trimmed/smoothed episodes
│   └── for_training/       # LeRobot-converted datasets (HF_LEROBOT_HOME)
├── checkpoints/            # Training checkpoints (gitignored)
├── assets/                 # Norm stats: assets/<config>/<repo_id>/norm_stats.json
└── docs/                   # Additional documentation
```

---

## Documentation

- [examples/ur5/README.md](examples/ur5/README.md) — detailed pipeline reference
- [teleoperation/README.md](teleoperation/README.md) — teleoperation setup
- [DEVELOPMENT.md](DEVELOPMENT.md) — project development conventions and known pitfalls
- [docs/norm_stats.md](docs/norm_stats.md) — norm stats reload guide
- [docs/remote_inference.md](docs/remote_inference.md) — remote inference setup
