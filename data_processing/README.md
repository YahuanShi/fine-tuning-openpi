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

**Dry-run** — preview the plan without writing anything (omit `--output`):

```bash
python3 05_trim_episodes.py path/to/dataset_dir --start 10 --end -8
```

---

### `06_visualize_trajectory.py` — original vs processed trajectory comparison

Overlays original and processed (trimmed + smoothed) joint trajectories in 7
vertical subplots — one per joint. Used as the **final verification step**
after the pipeline completes.

```bash
python3 06_visualize_trajectory.py original_dir/ training_dataset/
python3 06_visualize_trajectory.py original_dir/ training_dataset/ --no-norm
```

`--no-norm` plots absolute frame numbers instead of a normalised 0–1 x-axis
(useful when original and processed have similar lengths).

| Key | Action |
|-----|--------|
| `↑` / `←` / `P` | Previous episode |
| `↓` / `→` / `N` | Next episode |
| `S` | Save figure as PNG |
| `Q` / `Escape` | Quit |

---

### `pipeline.sh` — full automated pipeline