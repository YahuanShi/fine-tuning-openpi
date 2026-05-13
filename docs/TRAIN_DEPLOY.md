# Deploy Training on a New Server

One-click recipe to fine-tune Pi0.5 on a fresh GPU server. Training-only — no inference,
no submodules, no robot drivers.

---

## Prerequisites on target server

| Item | Required version | How to check |
|------|------------------|--------------|
| NVIDIA driver | ≥ 535 (CUDA 12.x capable) | `nvidia-smi` |
| Disk space | ~50 GB free (image + caches + checkpoints) | `df -h ~` |
| Network | outbound to docker.io, ghcr.io, huggingface.co, wandb.ai | — |

**Not required** (auto-installed by `bootstrap_host.sh` on Ubuntu 22.04):
Docker engine, NVIDIA Container Toolkit, FFmpeg, Python, build tools.

---

## One-time setup

```bash
# 1. Clone repo (NO submodules needed for training).
git clone https://github.com/YahuanShi/fine-tuning-openpi.git
cd fine-tuning-openpi

# 2. Create env file from template.
cp scripts/docker/.env.train.example scripts/docker/.env.train
$EDITOR scripts/docker/.env.train     # fill in WANDB_API_KEY, HF_TOKEN

# 3. (Optional) Mount dataset / checkpoints from a fast disk by editing .env.train:
#       DATASET_PATH=/mnt/nvme/openpi_dataset
#       CHECKPOINTS_PATH=/mnt/nvme/openpi_checkpoints

# 4. Stage your dataset under $DATASET_PATH (or ./dataset by default).
#    Layout: <DATASET_PATH>/raw/<DATE>/episode_*.hdf5  or already-converted lerobot/.
```

---

## Train (one command)

```bash
./scripts/train_oneclick.sh <repo_id> <exp_name> [extra train.py args]
```

Examples:

```bash
# Fresh training run.
./scripts/train_oneclick.sh ur5_dataset_20260415 ur5_pick_place_v5 --overwrite

# Resume from last checkpoint.
./scripts/train_oneclick.sh ur5_dataset_20260415 ur5_pick_place_v5 --resume
```

What happens on first run:

1. Detects missing Docker / `nvidia-ctk`; auto-installs via `bootstrap_host.sh` (Ubuntu 22.04).
2. Builds `openpi_trainer` image (~10 min): CUDA 12.2 base + FFmpeg 7 + Python 3.11 + `uv sync --frozen`.
3. Launches `scripts/train.py` inside the container with GPU passthrough.

Subsequent runs: skip steps 1–2, jump straight to training (cold start ~30 s).

---

## What's inside the image

| Layer | Versions pinned by |
|-------|---------------------|
| `nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04` (digest pin) | `train.Dockerfile` |
| FFmpeg 7.0.2 from source (shared libs) | `train.Dockerfile` |
| Python 3.11.9 + `uv 0.5.1` | `train.Dockerfile` |
| All Python deps (JAX[cuda12]==0.5.3, flax, lerobot@<rev>, transformers==4.53.2, ...) | `uv.lock` |

`rlds` group (TensorFlow / DROID) and `transformers_replace` patch are intentionally
excluded — JAX UR5 training does not need them.

---

## Manual data-prep steps (if dataset not yet converted)

The one-click script runs `train.py` only. If you need conversion + norm stats first,
shell into the image:

```bash
docker compose -f scripts/docker/compose.train.yml --env-file scripts/docker/.env.train \
    run --rm trainer bash

# Inside the container:
bash examples/ur5/train_pipeline.sh \
    --raw-dir dataset/raw/<DATE> \
    --repo-id ur5_dataset_<DATE> \
    --exp-name ur5_pick_place_v5
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `docker: Cannot connect to the Docker daemon` after install | Log out + back in (the user was added to `docker` group). |
| `nvidia-container-cli: initialization error` | `sudo systemctl restart docker`. |
| OOM at training start | Another process is holding the GPU: `nvidia-smi`, then `kill -9 <PID>`. |
| `Norm stats file not found` | Conversion / norm-stats step was skipped — run `train_pipeline.sh` first. |
| Build context upload is huge | Confirm `.dockerignore` excludes `dataset/`, `checkpoints/`, `wandb/`. |
| Non-Ubuntu host | Replace `install_docker_ubuntu22.sh` / `install_nvidia_container_toolkit.sh` calls in `bootstrap_host.sh` with your distro's equivalents. |

---

## Files

| Path | Purpose |
|------|---------|
| [scripts/docker/train.Dockerfile](../scripts/docker/train.Dockerfile) | Training image definition |
| [scripts/docker/compose.train.yml](../scripts/docker/compose.train.yml) | GPU + volume + env orchestration |
| [scripts/docker/entrypoint.sh](../scripts/docker/entrypoint.sh) | Editable install on container start |
| [scripts/docker/bootstrap_host.sh](../scripts/docker/bootstrap_host.sh) | Idempotent host setup |
| [scripts/docker/.env.train.example](../scripts/docker/.env.train.example) | Env-var template |
| [scripts/train_oneclick.sh](../scripts/train_oneclick.sh) | Top-level launcher |
