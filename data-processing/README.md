# data-processing

Scripts for inspecting, visualizing, and cleaning UR5 episode datasets (HDF5 format)
before conversion to LeRobot for pi0.5 fine-tuning.

## Dataset format

Each `episode_N.hdf5` contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `action` | `(T, 7)` | Joint commands (rad) |
| `observations/qpos` | `(T, 7)` | Joint positions (rad) |
| `observations/images/exterior_image_1_left` | `(T, 224, 224, 3)` | Exterior camera (RGB) |
| `observations/images/wrist_image_left` | `(T, 224, 224, 3)` | Wrist camera (RGB) |
| `observations/images/front_image_1` | `(T, 224, 224, 3)` | Front camera (RGB) — **optional**, present only in 3-camera recordings |

All scripts auto-detect whether `front_image_1` is present and adapt accordingly.

---

## Scripts

### `01_check_dataset.py` — quality report

Scans all episodes and prints a report of data issues. No files are written.

```bash
python3 01_check_dataset.py path/to/dataset_dir
python3 01_check_dataset.py path/to/dataset_dir --spike-thresh 0.10
```

| Flag | Meaning |
|------|---------|
| `cameras` | Number of camera streams detected (informational) |
| `static_action` | `action` never changes — likely a recording bug |
| `frozen_gripper` | Gripper dimension is constant for the whole episode |
| `qpos_eq_action` | `qpos ≈ action` — recorder may be duplicating state |
| `short` | Fewer than `--min-steps` timesteps |
| `spikes` | Joint step exceeds `--spike-thresh` rad (warning, not failure) |

Exit code is `1` if any structural issues are found, `0` otherwise.

---

### `02_visualize_episode.py` — video playback viewer

Plays back all camera streams side-by-side (2 or 3 cameras, auto-detected) with
per-joint trajectory strips and a scrubbing progress bar. Supports multi-episode navigation.

```bash
python3 02_visualize_episode.py path/to/dataset_dir
python3 02_visualize_episode.py path/to/episode_0.hdf5 --fps 30 --scale 2.0
```

| Key | Action |
|-----|--------|
| `SPACE` | Pause / resume |
| `←` / `→` | Step ±5 frames |
| `↑` / `↓` | Previous / next episode |
| `F` | Toggle 2× speed |
| `R` | Restart |
| `D` | Arm delete (pauses and shows red confirmation banner) |
| `Y` | Confirm delete — removes file from disk, advances to next episode |
| any other key | Cancel delete |
| `Q` | Quit |

Mouse drag on the progress bar or trajectory strips to scrub.

---

### `03_drop_front_camera.py` — remove front camera stream

Copies all episodes to a new directory, dropping `front_image_1` so downstream
tools only see the exterior and wrist cameras.

```bash
python3 03_drop_front_camera.py path/to/raw_dir path/to/no_front_dir
```

---

### `04_smooth_episodes.py` — trajectory smoothing

Applies Savitzky-Golay smoothing to `qpos` and `action` trajectories.
The gripper dimension is left unsmoothed to preserve open/close transitions.

```bash
python3 04_smooth_episodes.py path/to/dataset_dir --output smoothed/
python3 04_smooth_episodes.py path/to/dataset_dir --output smoothed/ --window 9 --poly 2
```

---

### `05_trim_episodes.py` — cut start/end frames

**Interactive mode** (default) — loops through each episode, shows the frame
count, and asks you to type the keep range:

```bash
python3 05_trim_episodes.py path/to/dataset_dir --output trimmed/
```

```
  episode_0.hdf5  [250 frames]
  Keep range (start end, negative ok, Enter = keep all): 10 -8
  → keeps frames 10-242  (233 frames)

  episode_1.hdf5  [180 frames]
  Keep range (start end, negative ok, Enter = keep all):
  → keeps all 180 frames
```

- `10 230`  — keep frames 10 to 230 (inclusive)
- `10 -5`   — keep frames 10 to T−5 (negative counts from end)
- Enter      — keep all frames unchanged

**Batch mode** — apply the same range to every episode:

```bash
python3 05_trim_episodes.py path/to/dataset_dir --start 10 --end -8 --output trimmed/
```

**Dry-run** — preview without writing (omit `--output`):

```bash
python3 05_trim_episodes.py path/to/dataset_dir --start 10 --end -8
```

---

### `06_visualize_trajectory.py` — original vs processed trajectory comparison

Overlays original and processed (trimmed + smoothed) joint trajectories side-by-side
in 7 vertical subplots — one per joint. Used as the **final verification step**.

```bash
python3 06_visualize_trajectory.py original_dir/ training_dataset/
python3 06_visualize_trajectory.py original_dir/ training_dataset/ --no-norm
```

`--no-norm` plots absolute frame numbers instead of a normalised 0–1 x-axis.

| Key | Action |
|-----|--------|
| `↑` / `←` / `P` | Previous episode |
| `↓` / `→` / `N` | Next episode |
| `S` | Save figure as PNG |
| `Q` / `Escape` | Quit |

---

### `pipeline.sh` — full automated pipeline

Runs all five processing steps in sequence with interactive pauses.
See **Recommended workflow** below.

```bash
./pipeline.sh path/to/raw_dir --cuts cuts.json --out training_dataset
```

---

## Recommended workflow

### Automated pipeline (recommended)

```bash
./pipeline.sh <raw_dir> --out training_dataset
```

`pipeline.sh` runs all five steps in order, pausing after each interactive step
for confirmation before proceeding:

| Step | Script | What happens |
|------|--------|--------------|
| 1 | `01_check_dataset.py` | Quality report printed; user reviews |
| 2 | `02_visualize_episode.py` | Interactive playback; delete bad episodes with D+Y |
| 3 | `03_drop_front_camera.py` | Copies episodes to `<raw_dir>_no_front/`, removes `front_image_1` |
| 4 | `04_trim_episodes.py` | Interactive trim → `<raw_dir>_trimmed/` |
| 5 | `05_smooth_episodes.py` | SG smoothing → `training_dataset/` |

**Pipeline options:**

```bash
./pipeline.sh <raw_dir> \
    --trim   5           \   # global fallback trim per end (default: 0, interactive)
    --window 15          \   # SG window, must be odd (default: 15)
    --poly   3           \   # SG poly order (default: 3)
    --out    training_dataset
```

### Manual step-by-step

```
1. python3 01_check_dataset.py        <raw_dir>                     # quality report
2. python3 02_visualize_episode.py    <raw_dir>                     # review; D+Y to delete bad
3. python3 03_drop_front_camera.py    <raw_dir>  no_front/          # remove front camera
4. python3 04_trim_episodes.py        no_front/  --output trimmed/  # trim start/end frames
5. python3 05_smooth_episodes.py      trimmed/   --output training_dataset/  # smooth
6. python3 06_visualize_trajectory.py <raw_dir>  training_dataset/  # verify result
7. python3 convert_ur5_data_to_lerobot.py training_dataset/         # convert & train
```

---

## Dependencies

```
h5py  opencv-python  numpy  matplotlib  scipy
```

```bash
pip install h5py opencv-python numpy matplotlib scipy
```
