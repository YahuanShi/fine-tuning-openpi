#!/bin/bash
# Container entrypoint: install project editable (deps already in image) then exec command.
set -e

# Idempotent, <1s. Deps already resolved during image build; this only wires up the
# editable .pth so imports from src/openpi work.
uv pip install -e . --no-deps -q 2>/dev/null || true

exec "$@"
