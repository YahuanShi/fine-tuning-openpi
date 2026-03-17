# data-processing

Scripts for inspecting, visualizing, and cleaning UR5 episode datasets (HDF5 format)
before conversion to LeRobot for pi0.5 fine-tuning.

## Dataset format

Each `episode_N.hdf5` contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `action` | `(T, 7)` | Joint commands (rad) |
| `observations/qpos` | `(T, 7)` | Joint positions (rad) |
| `observations/images/exterior_image_1_left` | `(T, 224, 224, 3)` | External camera (RGB) |
| `observations/images/wrist_image_left` | `(T, 224, 224, 3)` | Wrist camera (RGB) |

---

## Scripts

### `visualize_episode.py` — video playback viewer

Plays back both camera streams side-by-side with per-joint trajectory strips
and a scrubbing progress bar. Supports multi-episode navigation.

```bash
python3 visualize_episode.py path/to/dataset_dir
python3 visualize_episode.py path/to/episode_0.hdf5 --fps 30 --scale 2.0
```

| Key | Action |
|-----|--------|
| `SPACE` | Pause / resume |
| `←` / `→` | Step ±10 frames |
| `↑` / `↓` | Previous / next episode |
| `F` | Toggle 2× speed |
| `R` | Restart |
| `Q` | Quit |

Mouse drag on the progress bar or trajectory strips to scrub.

---

### `visualize_trajectory.py` — joint trajectory viewer + cut editor

Shows raw vs Savitzky-Golay smoothed joint trajectories (7 vertical subplots).
Lets you mark per-episode start/end cut points interactively and save them
to a `cuts.json` file for use with `trim_episodes.py`.

```bash
python3 visualize_trajectory.py path/to/dataset_dir
python3 visualize_trajectory.py path/to/dataset_dir --cuts cuts.json
python3 visualize_trajectory.py path/to/dataset_dir --no-smooth
python3 visualize_trajectory.py path/to/dataset_dir --window 9 --poly 2
```

| Key | Action |
|-----|--------|
| `]` / `[` | Move **start** cut ±1 frame |
| `}` / `{` | Move **start** cut ±10 frames |
| `.` / `,` | Move **end** cut ±1 frame |
| `>` / `<` | Move **end** cut ±10 frames |
| `←` / `→` | Previous / next episode (auto-saves current cuts) |
| `W` | Write all cuts to JSON |
| `S` | Save figure as PNG |
| `Q` | Quit |

Red shading shows the frames that will be removed.

---

### `check_dataset.py` — quality report

Scans all episodes and prints a report of data issues. No files are written.

```bash
python3 check_dataset.py path/to/dataset_dir
python3 check_dataset.py path/to/dataset_dir --spike-thresh 0.10
```

| Flag | Meaning |
|------|---------|
| `static_action` | `action` never changes — likely a recording bug |
| `frozen_gripper` | Gripper dimension is constant for the whole episode |
| `qpos_eq_action` | `qpos ≈ action` — recorder may be duplicating state |
| `short` | Fewer than `--min-steps` timesteps |
| `spikes` | Joint step exceeds `--spike-thresh` rad (warning, not failure) |

Exit code is `1` if any structural issues are found, `0` otherwise.

---

### `trim_episodes.py` — cut start/end frames

Applies per-episode frame cuts from `cuts.json` (produced by
`visualize_trajectory.py`) and writes trimmed copies. Prints a dry-run
table when `--output` is omitted.

```bash
# Dry-run: show what would be cut
python3 trim_episodes.py path/to/dataset_dir --cuts cuts.json

# Apply cuts
python3 trim_episodes.py path/to/dataset_dir --cuts cuts.json --output trimmed/

# Global fallback: same trim for all episodes not in cuts.json
python3 trim_episodes.py path/to/dataset_dir --cuts cuts.json --trim 5 --output trimmed/
```

---

### `smooth_episodes.py` — trajectory smoothing

Applies Savitzky-Golay smoothing to `qpos` and `action` trajectories.
The gripper dimension is left unsmoothed to preserve open/close transitions.

```bash
python3 smooth_episodes.py path/to/dataset_dir --output smoothed/
python3 smooth_episodes.py path/to/dataset_dir --output smoothed/ --window 9 --poly 2
```

---

## Recommended workflow

```
1. python3 check_dataset.py        <dir>              # inspect data quality
2. python3 visualize_episode.py    <dir>              # watch video playback
3. python3 visualize_trajectory.py <dir> \
           --cuts cuts.json                           # mark cuts, press W to save
4. python3 trim_episodes.py        <dir> \
           --cuts cuts.json --output trimmed/         # apply cuts
5. python3 smooth_episodes.py      trimmed/ \
           --output clean/                            # smooth trajectories
6. python3 convert_ur5_data_to_lerobot.py clean/      # then train
```

---

## Dependencies

```
h5py  opencv-python  numpy  matplotlib  scipy
```

```bash
pip install h5py opencv-python numpy matplotlib scipy
```
