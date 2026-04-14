# UR5 Pi0.5 Fine-Tuning Pipeline

Complete pipeline for fine-tuning Pi0.5 on a UR5 pick-and-place task.

**Hardware:**
- UR5e robot arm
- Weiss CRG gripper (serial `/dev/ttyACM0`)
- RealSense D415 exterior camera (serial `105422061000`)
- RealSense D405 wrist camera (serial `352122273671`)
- GPU: NVIDIA RTX 6000 48GB

**Directory structure:**
```
dataset/
  raw/          ← original HDF5 recordings
  processed/    ← trimmed/smoothed HDF5 episodes
  for_training/ ← converted LeRobot datasets (HF_LEROBOT_HOME)
```

---

## Step 1 — Convert Dataset

Raw HDF5 episodes → LeRobot format.

```bash
HF_LEROBOT_HOME=$(pwd)/dataset/for_training uv run examples/ur5/convert_ur5_data_to_lerobot.py \
    --raw-dir dataset/processed/<SUBDIR>/<DATE> \
    --repo-id ur5_dataset_<DATE>
```

- `--fps` is optional: auto-detected from the `hz` attribute in each HDF5 file
- Output is written to `dataset/for_training/ur5_dataset_<DATE>/`
- Raw HDF5 and output **must not share the same directory** — the script deletes the output dir before writing

**HDF5 format expected:**
```
episode_N.hdf5
  attrs:  hz, prompt
  /observations/qpos         (T, 7)   joints in degrees [0:6] + gripper [6] in {0,1}
  /observations/images/exterior_image_1_left  (T, H, W, 3)  uint8
  /observations/images/wrist_image_left       (T, H, W, 3)  uint8
  /action                    (T, 7)   same layout as qpos
```

The script converts joints from degrees to radians and inverts the gripper convention (`gripper_pi05 = 1 - gripper_raw`) to match Pi0.5 (0=open, 1=closed).

---

## Step 2 — Update Training Config

Edit `src/openpi/training/config.py`, find the relevant TrainConfig and set `repo_id` to the new dataset:

```python
TrainConfig(
    name="pi05_ur5",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ),
    data=LeRobotUR5DataConfig(
        repo_id="ur5_dataset_<DATE>",          # <-- update this
        base_config=DataConfig(prompt_from_task=True),
    ),
    batch_size=32,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=1_000,
        peak_lr=5e-5,
        decay_steps=20_000,
        decay_lr=1e-6,
    ),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    ema_decay=None,
    freeze_filter=pi0_config.Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ).get_freeze_filter(),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    num_train_steps=20_000,
    save_interval=2000,
    keep_period=5000,
)
```

**Notes:**
- LoRA is required — full fine-tuning exceeds 48GB on RTX 6000
- `ema_decay=None` — EMA adds ~12.5GB and causes OOM
- `batch_size=32` — stable at ~41GB VRAM
- `action_horizon=10` — model max, works for both 10Hz and 20Hz datasets

---

## Step 3 — Compute Norm Stats

```bash
HF_LEROBOT_HOME=$(pwd)/dataset/for_training uv run scripts/compute_norm_stats.py --config-name pi05_ur5
```

Output is written to `assets/pi05_ur5/ur5_dataset_<DATE>/norm_stats.json`.

---

## Step 4 — Train

```bash
HF_LEROBOT_HOME=$(pwd)/dataset/for_training XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
uv run scripts/train.py pi05_ur5 \
    --exp-name ur5_pick_place_<VERSION> \
    --overwrite
```

To resume an interrupted training:
```bash
HF_LEROBOT_HOME=$(pwd)/dataset/for_training XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
uv run scripts/train.py pi05_ur5 \
    --exp-name ur5_pick_place_<VERSION> \
    --resume
```

Checkpoints are saved to `checkpoints/pi05_ur5/ur5_pick_place_<VERSION>/<STEP>/`.

**Convergence indicators:**
- `loss` stabilises around `0.0006–0.002`
- `grad_norm` stabilises around `0.01–0.05`
- Typically converges at 20k–33k steps (~8–12 hours on RTX 6000)

**One-command pipeline (convert + norm stats + train):**
```bash
./examples/ur5/train_pipeline.sh \
    --raw-dir dataset/processed/<SUBDIR>/<DATE> \
    --repo-id ur5_dataset_<DATE> \
    --exp-name ur5_pick_place_<VERSION>
```

---

## Step 5 — Serve Policy

Start the policy server (Terminal 1):

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config pi05_ur5 \
    --policy.dir checkpoints/pi05_ur5/ur5_pick_place_<VERSION>/<STEP>
```

First inference takes ~60s (JAX JIT compilation). Subsequent inferences take ~300–500ms.

**Config ↔ Checkpoint mapping** (always verify with `ls checkpoints/.../assets/`):
| Checkpoint path | Config |
|---|---|
| `checkpoints/pi05_ur5/...` | `pi05_ur5` |
| `checkpoints/pi05_ur5_assembly/...` | `pi05_ur5_assembly` |
| `checkpoints/pi05_ur5_pnpa/...` | `pi05_ur5_pnpa` |

**If norm stats mismatch error occurs**, copy correct stats into checkpoint:
```bash
cp -r checkpoints/pi05_ur5/<EXP>/<STEP>/assets/<ORIGINAL_DATASET> \
       checkpoints/pi05_ur5/<EXP>/<STEP>/assets/<CURRENT_REPO_ID>
```

---

## Step 6 — Run Inference on Robot

Start inference client (Terminal 2):

```bash
PYTHONPATH=. uv run examples/ur5/main.py \
    --prompt "pick yellow cube and place it into red box"
```

Key parameters in `main.py`:
| Parameter | Value | Notes |
|---|---|---|
| `action_horizon` | `10` | Max actions per inference call (model limit) |
| `num_episodes` | `10` | Number of consecutive task cycles |
| `max_episode_steps` | `400` | ~40s at 10Hz — increase for longer tasks |
| `control_hz` | `10.0` | Must match training data frequency |

**Note:** Policy server must be running before starting inference. Shut down the server before starting training — both processes cannot share the GPU.

---

## Hardware Constants (env.py)

| Constant | Value | Description |
|---|---|---|
| `UR5_IP` | `10.0.0.1` | UR5 RTDE IP address |
| `GRIPPER_PORT` | `/dev/ttyACM0` | Weiss CRG serial port |
| `HOME_DEG` | `[45, -20, -140, -40, -270, 0]` | Home joint angles (degrees) |
| `MAX_JOINT_VEL` | `0.8 rad/s` | Safety velocity clamp |
| `SERVO_J_LOOKAHEAD` | `0.2 s` | servoJ smoothing |
| `CAM_SERIAL_BASE` | `105422061000` | D415 exterior camera |
| `CAM_SERIAL_WRIST` | `352122273671` | D405 wrist camera |

---

## Known Limitations

- **Motion stops every ~1s:** Synchronous inference — robot waits for next action chunk. Irreducible without async prefetching.
- **No validation during training:** Only train loss and grad_norm are logged. Evaluate by running inference on hardware.
- **Norm stats must match checkpoint:** Always ensure `config.repo_id` matches the dataset the checkpoint was trained on.
- **10Hz recommended:** Even for 20Hz-collected datasets, 10Hz inference gives smoother motion (less frequent re-query pauses).
