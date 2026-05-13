#!/bin/bash
# Idempotent host bootstrap: Docker engine + NVIDIA Container Toolkit.
# Safe to re-run. Targets Ubuntu 22.04 (delegates to existing install scripts).
#
# Pre-requisites this script does NOT handle:
#   - NVIDIA kernel driver (host-specific; ask your admin or `ubuntu-drivers autoinstall`).
#   - Non-Ubuntu hosts (RHEL/Rocky/etc. — adapt install_docker_*.sh accordingly).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. NVIDIA driver presence (required, but not auto-installed here).
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Install NVIDIA driver first." >&2
    echo "       Ubuntu: sudo ubuntu-drivers autoinstall && reboot" >&2
    exit 1
fi
echo "[bootstrap] NVIDIA driver OK: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"

# 2. Docker engine.
if ! command -v docker &>/dev/null; then
    echo "[bootstrap] Installing Docker engine..."
    bash "$SCRIPT_DIR/install_docker_ubuntu22.sh"
    echo "[bootstrap] Docker installed. You may need to log out/in for the 'docker' group to take effect."
else
    echo "[bootstrap] Docker OK: $(docker --version)"
fi

# 3. NVIDIA Container Toolkit.
if ! command -v nvidia-ctk &>/dev/null; then
    echo "[bootstrap] Installing NVIDIA Container Toolkit..."
    bash "$SCRIPT_DIR/install_nvidia_container_toolkit.sh"
else
    echo "[bootstrap] NVIDIA Container Toolkit OK: $(nvidia-ctk --version | head -1)"
fi

# 4. Smoke test: can Docker see the GPU?
echo "[bootstrap] Smoke test: docker run --gpus all ..."
if docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
    echo "[bootstrap] GPU access from Docker: OK"
else
    echo "[bootstrap] WARN: GPU smoke test failed. Likely fixes:"
    echo "             - log out / log back in (docker group)"
    echo "             - sudo systemctl restart docker"
fi

echo "[bootstrap] Done."
