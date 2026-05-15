# Deploy Training on a New Server

One-click recipe to fine-tune Pi0.5 on a GPU server. Training-only — no inference,
no submodules, no robot drivers.

All training runs inside a Docker container — no Python or uv installation needed on the host.

> **Full step-by-step guide (Chinese)**: [new_server_training_setup.md](new_server_training_setup.md)

---

## Prerequisites

| Item | Required | How to check |
|------|----------|--------------|
| NVIDIA driver | ≥ 535 (CUDA 12.x capable) | `nvidia-smi` |
| Disk space | ≥ 200 GB free (dataset + image + caches + checkpoints) | `df -h` |
| Network | outbound to docker.io, ghcr.io, huggingface.co, wandb.ai | — |

**Not required on the host** (Docker + nvidia-toolkit auto-installed by `bootstrap_host.sh`;
everything else — Python, uv, FFmpeg, all Python deps — lives inside the container):

---

## Step 0 — Prepare credentials (browser, any machine)

### WandB API key
1. Log in at https://wandb.ai → avatar → **User Settings** → **API keys** → **Reveal**
2. Copy the 40-character key

### HuggingFace token
1. Log in at https://huggingface.co → avatar → **Settings** → **Access Tokens** → **New token**
2. Type: `Read`, enable **Public gated repos**
3. Visit https://huggingface.co/google/paligemma2-3b-pt-224 and click **Agree and access repository**
   (must be done before the token can download the base weights)

---

## Step 1 — Transfer dataset to target server

```bash
# On the TARGET server — create the destination directory first.
mkdir -p /mnt/data/openpi/dataset/Task_Graph_Dataset
mkdir -p /mnt/data/openpi/checkpoints

# On your LOCAL machine — rsync the raw HDF5 episodes.
rsync -avhP --partial \
    /path/to/Task_Graph_Dataset/ \
    <user>@<target-server>:/mnt/data/openpi/dataset/Task_Graph_Dataset/

# On the TARGET server — verify; should print 405.
ls /mnt/data/openpi/dataset/Task_Graph_Dataset/*.hdf5 | wc -l
```

---

## Step 2 — One-time host + repo setup (target server)

```bash
# Clone repo (NO submodules needed for training).
git clone https://github.com/YahuanShi/fine-tuning-openpi.git
cd fine-tuning-openpi

# Install Docker + NVIDIA Container Toolkit (idempotent, Ubuntu 22.04).
bash scripts/docker/bootstrap_host.sh

# REQUIRED: log out and back in so the docker group takes effect.
exit   # then SSH back in

cd ~/fine-tuning-openpi   # CWD resets on reconnect — must cd back

# Verify GPU access from Docker.
docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi

# Create env file from template.
cp scripts/docker/.env.train.example scripts/docker/.env.train
$EDITOR scripts/docker/.env.train
```

Fill in `.env.train` — all fields that matter for training:

```bash
# ── Required ──────────────────────────────────────────────────────────────
UR5_REPO_ID=Task_Graph_V1          # pick a name; must match --repo-id in Step 4
WANDB_API_KEY=<your-40-char-key>
HF_TOKEN=<your-hf-token>

# ── Paths (adjust to where you put the data in Step 1) ────────────────────
DATASET_PATH=/mnt/data/openpi/dataset
CHECKPOINTS_PATH=/mnt/data/openpi/checkpoints

# ── GPU tuning ────────────────────────────────────────────────────────────
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95   # use 0.90 for 24 GB GPUs
```

---

## Step 3 — Build image (one-time, ~10–15 min)

```bash
docker compose -f scripts/docker/compose.train.yml \
               --env-file scripts/docker/.env.train \
               build
```

The image bundles: CUDA 12.2 + cuDNN 8 · FFmpeg 7 (compiled from source) ·
Python 3.11 · uv · all Python deps from `uv.lock`. Nothing needs to be installed on the host.

---

## Step 4 — Convert + norm stats + train (all-in-one)

`train_pipeline.sh` runs all three steps in sequence: convert → norm stats → train.
It runs **inside the Docker container** so uv, Python and all deps are available.

> **Critical**: SSH disconnection kills the foreground process without any error message,
> leaving a partial dataset. Always wrap in `tmux` or `nohup`.

**Option A — tmux** (recommended; lets you watch live output):
> If `tmux` is not installed: `sudo apt install -y tmux`

```bash
tmux new -s train
docker compose -f scripts/docker/compose.train.yml \
               --env-file scripts/docker/.env.train \
               run --rm trainer \
               ./examples/ur5/train_pipeline.sh \
                   --raw-dir dataset/Task_Graph_Dataset \
                   --repo-id Task_Graph_V1 \
                   --exp-name ur5_task_graph_v1 \
                   --batch-size 64
# Ctrl+B D to detach safely; tmux attach -t train to return
```

**Option B — nohup** (fire-and-forget; safe to close terminal immediately):

```bash
nohup docker compose -f scripts/docker/compose.train.yml \
               --env-file scripts/docker/.env.train \
               run --rm -T trainer \
               ./examples/ur5/train_pipeline.sh \
                   --raw-dir dataset/Task_Graph_Dataset \
                   --repo-id Task_Graph_V1 \
                   --exp-name ur5_task_graph_v1 \
                   --batch-size 64 \
    > /tmp/pipeline.log 2>&1 &
echo "PID: $!"
```

Monitor progress:

```bash
tail -f /tmp/pipeline.log
grep -c "Saved.*steps" /tmp/pipeline.log
```

Time estimates: conversion ~20–40 min · norm stats ~5 min · training ~hours.

**Skip flags** (if steps already completed):

| Flag | Effect |
|------|--------|
| `--skip-convert` | skip HDF5 → LeRobot conversion |
| `--skip-stats` | skip norm stats computation |
| `--resume` | resume training from last checkpoint |

Path note: `--raw-dir dataset/Task_Graph_Dataset` is the path **inside the container**
(`/app/dataset` is bind-mounted from `$DATASET_PATH` set in `.env.train`).

First ~60 s of training: JAX JIT compilation. Training loop starts at `step 0`.

---

## What's inside the image

| Layer | Pinned by |
|-------|-----------|
| `nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04` (digest-pinned) | `train.Dockerfile` |
| FFmpeg 7.0.2 from source (shared libs, required by LeRobot) | `train.Dockerfile` |
| Python 3.11.9 + uv 0.5.1 | `train.Dockerfile` |
| All Python deps (JAX[cuda12]==0.5.3, flax, lerobot@rev, transformers==4.53.2 …) | `uv.lock` |

`rlds` group (TensorFlow / DROID) and `transformers_replace` patch excluded —
JAX UR5 training does not need them.

---

## Monitoring

```bash
watch -n 1 nvidia-smi                       # GPU utilization + VRAM on host
tail -f /tmp/pipeline.log                   # pipeline progress (if using nohup)
# WandB dashboard: https://wandb.ai/<entity>/openpi
```

Expected VRAM during training: ~41 GB (RTX 6000 48 GB, batch 32).

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Conversion/training stops mid-way, no error | SSH disconnected — use `tmux` or `nohup` (see Step 4) |
| `docker: Cannot connect to daemon` | Log out + back in (docker group not yet active) |
| `nvidia-container-cli: initialization error` | `sudo systemctl restart docker` |
| `No episode_*.hdf5 files found` | `DATASET_PATH` in `.env.train` wrong — check `docker compose run --rm trainer ls /app/dataset/Task_Graph_Dataset` |
| OOM at training start | Kill other GPU processes: `nvidia-smi` → `kill -9 <PID>` |
| `Norm stats file not found` | Re-run with `--skip-convert` to redo norm stats only |
| HuggingFace 401 on base weights | Accept license at huggingface.co model page; check `HF_TOKEN` in `.env.train` |
| WandB 401 | Check `WANDB_API_KEY` in `.env.train` (no trailing spaces) |
| Build context upload is huge | Confirm `.dockerignore` excludes `dataset/`, `checkpoints/`, `wandb/` |
| Non-Ubuntu host | Replace install scripts in `bootstrap_host.sh` with distro equivalents |

---

## Files

| Path | Purpose |
|------|---------|
| [scripts/docker/train.Dockerfile](../scripts/docker/train.Dockerfile) | Training image definition |
| [scripts/docker/compose.train.yml](../scripts/docker/compose.train.yml) | GPU + volume + env orchestration |
| [scripts/docker/entrypoint.sh](../scripts/docker/entrypoint.sh) | Editable install on container start |
| [scripts/docker/bootstrap_host.sh](../scripts/docker/bootstrap_host.sh) | Idempotent host setup (Docker + nvidia-toolkit) |
| [scripts/docker/.env.train.example](../scripts/docker/.env.train.example) | Env-var template |
| [scripts/train_oneclick.sh](../scripts/train_oneclick.sh) | Shortcut launcher (train only, no conversion) |
| [docs/new_server_training_setup.md](new_server_training_setup.md) | Full step-by-step guide (Chinese) |
