# Push Guide — fine-tuning-openpi

Instruction file for Claude to assist with git operations in this repo.

---

## Repo Structure

This is the parent repo. Two directories are **git submodules** with their own remotes:

| Path | Remote |
|------|--------|
| `data_processing/` | `git@github.com:YahuanShi/data_processing.git` |
| `teleoperation/` | `git@github.com:YahuanShi/teleoperation.git` |

---

## Push Workflow

### 1. Check all changes
```bash
git status
git diff --stat
# Check submodules too:
cd data_processing && git status && cd ..
cd teleoperation && git status && cd ..
```

### 2. Ruff check (Python files only — never run ruff on .sh files)
```bash
uv run ruff check <changed_python_files>
uv run ruff format --check <changed_python_files>
# Auto-fix if needed:
uv run ruff format <file>
uv run ruff check --fix --unsafe-fixes <file>
```

### 3. Push submodules first, then parent

**If `data_processing` has changes:**
```bash
cd data_processing
# If HEAD is detached: git stash → git checkout main → git stash pop
git add <files>
git commit -m "<message>"
git push
cd ..
```

**If `teleoperation` has changes:**
```bash
cd teleoperation
# Same detached HEAD handling as above
git add <files>
git commit -m "<message>"
git push
cd ..
```

**Then update parent repo:**
```bash
git add <files> [data_processing] [teleoperation]
git commit -m "<message>"
git push
```

---

## Rules

- **Never** add `Co-Authored-By:` trailer to commits
- **Never** run `ruff` on `.sh` files — shell syntax errors are expected and harmless
- When a submodule is in detached HEAD state, switch to `main` before committing:
  ```bash
  git stash && git checkout main && git stash pop
  # If conflict: git checkout --theirs <file> && git add <file> && git stash drop
  ```
- Always commit submodule changes **before** updating the parent pointer
- `.vscode/extensions.json` — do not commit unless user explicitly asks
- `CLAUDE.md`, `PUSH_GUIDE.md`, `UR5_Pi05_Manual.md`, `examples/ur5/serve.sh` — project files, commit normally with parent repo

---

## Project Structure

```
fine-tuning-openpi/
├── dataset/
│   ├── raw/                        # original HDF5 recordings from episode_recorder.py
│   ├── processed/                  # pipeline outputs
│   │   ├── no_front/<date>/
│   │   ├── smoothed/<date>/
│   │   ├── trimmed/<date>/         # input to convert_ur5_data_to_lerobot.py
│   │   └── resized/<date>/
│   └── for_training/               # LeRobot-format datasets (HF_LEROBOT_HOME)
│       └── ur5_dataset_<date>/
├── checkpoints/pi05_ur5/<exp>/     # training checkpoints
├── assets/                         # norm stats (compute_norm_stats output)
├── data_processing/                # submodule: data pipeline scripts
└── teleoperation/                  # submodule: recording + robot scripts
```

### Key environment variables

| Variable | Value | Used by |
|----------|-------|---------|
| `HF_LEROBOT_HOME` | `$(pwd)/dataset/for_training` | LeRobot, convert script, train_pipeline.sh |
| `UR5_REPO_ID` | e.g. `ur5_dataset_20260415` | `src/openpi/training/config.py` (pi05_ur5 repo_id) |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | `0.95` | training script |

---

## Claude Collaboration Rules

- **Report before push**: Claude must summarize all changes and wait for explicit user confirmation before committing or pushing anything.
- **Long-running commands** (training, data conversion, serving): Claude explains the command and the user runs it. Claude never launches these autonomously.
- **Scope discipline**: Claude must not modify files unrelated to the task. If a file seems relevant but wasn't mentioned, ask first.
- **No unsolicited cleanup**: Do not refactor, rename, or reorganize code unless explicitly asked.
- **Submodule awareness**: Always check submodule status before reporting "nothing to push".

---

## Common Ruff Per-File-Ignores (already in pyproject.toml)

| File pattern | Ignored rules | Reason |
|---|---|---|
| `data-processing/*.py`, `data_processing/*.py` | N806, N803, FBT, E402, UP031 | T/W/H conventions, cv2 API |
| `teleoperation/data_collection/episode_recorder.py` | N806, FBT001, FBT003 | dimension conventions, cv2 |
| `examples/ur5/record_demo.py` | FBT001 | intentional bool arg |
| `teleoperation/uarm/scripts/UR5/test_ur5.py` | B905 | test script |

---

## CI (pre-commit)

Runs on every push. Uses ruff `v0.11.12` (pinned in `.pre-commit-config.yaml`).
Local ruff version must match — check with `uv run ruff --version`.

Hooks: `uv-lock` → `ruff` (lint + fix) → `ruff-format`

If CI fails with "files were modified by hook", run locally and re-push:
```bash
uv run ruff format .
uv run ruff check --fix .
git add -u && git commit -m "Apply ruff fixes" && git push
```
