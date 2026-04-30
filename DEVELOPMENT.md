# Development Guide — UR5 Pi0.5 Fine-Tuning

Project conventions and rules. For operation commands and parameters, see `examples/ur5/README.md`.

---

## Environment Variables (always required)

| Variable | Purpose | Example |
|----------|---------|---------|
| `UR5_REPO_ID` | Dataset name for `pi05_ur5` config — **mandatory**, no default | `ur5_dataset_20260415` |
| `HF_LEROBOT_HOME` | LeRobot dataset storage root | `$(pwd)/dataset/for_training` |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | JAX GPU memory fraction for training | `0.95` |

`train_pipeline.sh` sets all three automatically. When calling scripts directly, prefix every command:

```bash
UR5_REPO_ID=ur5_dataset_<DATE> \
HF_LEROBOT_HOME=$(pwd)/dataset/for_training \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
uv run scripts/train.py pi05_ur5 ...
```

---

## Data Conventions

- **Joint angles**: HDF5 stores degrees; model uses radians. Conversion happens in `convert_ur5_data_to_lerobot.py`. Never apply `deg2rad` twice.
- **Gripper convention**: HDF5 raw = `0=closed / 1=open`; Pi0.5 model = `0=open / 1=closed`. Conversion `gripper_pi05 = 1 - gripper_raw` is applied in the convert script. Do not re-invert elsewhere.
- **dtype**: All robot state and actions use `float32` throughout `env.py` and inference code. Never use `float64`.
- **Image size**: 224×224 uint8 RGB (BGR→RGB conversion done in `env.py`).

---

## Directory Structure Rules

```
dataset/
  raw/              ← original HDF5 recordings (never overwrite)
  processed/        ← output of data_processing pipeline
    trimmed/<DATE>/
  for_training/     ← HF_LEROBOT_HOME, LeRobot-converted datasets
checkpoints/pi05_ur5/<EXP_NAME>/<STEP>/
  assets/
    <repo_id>/norm_stats.json   ← repo_id = what checkpoint was trained on
    metadata.json               ← {"repo_id": "..."} written at training time
assets/pi05_ur5/<repo_id>/norm_stats.json  ← global norm stats
```

**Never use the same path for `--raw-dir` and LeRobot output** — convert script deletes the output directory before writing.

---

## Config Rules

- `pi05_ur5` reads `repo_id` from `os.environ["UR5_REPO_ID"]` — never hardcode a date in `config.py` for this config.
- `pi05_ur5_assembly` and `pi05_ur5_pnpa` have hardcoded `repo_id` (they rarely change).
- Do not change `action_horizon` — Pi0.5 hard limit of 10.
- Do not enable `ema_decay` — adds ~12.5GB VRAM, causes OOM on RTX 6000.

---

## Training Rules

- `--overwrite` is required to replace an existing converted dataset. Without it, convert exits safely.
- Norm stats must be recomputed whenever `UR5_REPO_ID` changes.
- Use `--overwrite` to start fresh training, `--resume` to continue a checkpoint.
- Each saved checkpoint writes `assets/metadata.json` with `{"repo_id": "..."}`.

---

## Inference Rules

- **Always use `serve.sh`** — it reads `metadata.json` and resolves the correct config automatically:
  ```bash
  ./examples/ur5/serve.sh checkpoints/pi05_ur5/<EXP_NAME>/<STEP>
  ```
- Never manually guess which config to pass to `serve_policy.py`. Check `assets/metadata.json` or `ls checkpoints/.../assets/` first.
- For old checkpoints (no `metadata.json`): `serve.sh` falls back to path-name inference.

---

## Known Pitfalls

### Norm stats mismatch
**Symptom:** `FileNotFoundError: Norm stats file not found at: .../assets/<REPO_ID>/norm_stats.json`
**Fix:** Use `serve.sh` (auto-handled). Manual: `cp -r assets/<OLD_ID> assets/<NEW_ID>` inside the checkpoint directory.

### OOM on training start
**Cause:** Previous process still holding GPU memory.
**Fix:** `nvidia-smi` → find PID → `kill -9 <PID>`

### Shell interprets `&` in exp-name
**Fix:** Use underscores only: `ur5_pick_place_and_arrange_v1`

### servoJ failure causes position drift
**Status:** Handled — `env.py` re-reads actual joint angles from RTDE on exception.

### Camera frame drop crashes episode
**Status:** Handled — `env.py` retries up to 3 times with 500ms timeout each.
