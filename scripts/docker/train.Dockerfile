# Training image for openpi UR5 fine-tuning (JAX path).
#
# Build (from project root):
#   docker build . -t openpi_trainer -f scripts/docker/train.Dockerfile
#
# Run via compose (recommended) — see scripts/docker/compose.train.yml.
#
# Differences from serve_policy.Dockerfile:
#   - Includes dev group (pytest/ruff/matplotlib/pynvml) — useful during training.
#   - Skips rlds group (TensorFlow / DROID — not needed for UR5).
#   - Skips transformers_replace patch (PyTorch-only; UR5 training uses JAX).
#   - No CMD: launched via entrypoint + compose `command:`.

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

WORKDIR /app

# System deps: git-lfs for LeRobot, build chain for FFmpeg + occasional source wheels.
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    linux-headers-generic \
    build-essential \
    clang \
    pkg-config \
    cython3 \
    wget \
    yasm \
    nasm \
    && rm -rf /var/lib/apt/lists/*

# FFmpeg 7.0.2 from source — LeRobot's video decoding needs shared libs ≥ 7.
RUN cd /tmp && \
    wget -q https://ffmpeg.org/releases/ffmpeg-7.0.2.tar.xz && \
    tar xf ffmpeg-7.0.2.tar.xz && \
    cd ffmpeg-7.0.2 && \
    ./configure \
        --enable-shared \
        --disable-static \
        --enable-gpl \
        --disable-doc \
        --disable-ffplay \
        --disable-x86asm && \
    make -j"$(nproc)" && \
    make install && \
    ldconfig && \
    cd / && \
    rm -rf /tmp/ffmpeg-7.0.2*

ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/.venv
ENV PATH="/.venv/bin:${PATH}"

# Resolve dependencies from the lockfile into a venv outside /app
# (project source itself is bind-mounted at runtime; entrypoint runs `uv pip install -e .`).
RUN uv venv --python 3.11.9 $UV_PROJECT_ENVIRONMENT
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/pyproject.toml,target=packages/openpi-client/pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/src,target=packages/openpi-client/src \
    GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-install-project

# Default training tunables (overridable by compose env).
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

ENTRYPOINT ["/app/scripts/docker/entrypoint.sh"]
CMD ["/bin/bash"]
