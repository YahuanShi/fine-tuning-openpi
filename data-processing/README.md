# data-processing

Scripts for inspecting and visualizing UR5 episode datasets stored as HDF5 files.

## Dataset format

Each `episode_N.hdf5` file contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `action` | `(T, 7)` | Joint commands (rad) |
| `observations/qpos` | `(T, 7)` | Joint positions (rad) |
| `observations/images/exterior_image_1_left` | `(T, 224, 224, 3)` | External camera (RGB) |
| `observations/images/wrist_image_left` | `(T, 224, 224, 3)` | Wrist camera (RGB) |

## Scripts

### `visualize_episode.py` — single episode viewer

Plays back side-by-side camera streams with a live joint-position bar at the bottom.

```bash
# Basic playback
python visualize_episode.py path/to/episode_0.hdf5

# Adjust speed
python visualize_episode.py path/to/episode_0.hdf5 --fps 30

# Save to video file
python visualize_episode.py path/to/episode_0.hdf5 --save episode_0.mp4

# Plot joint trajectories instead of video
python visualize_episode.py path/to/episode_0.hdf5 --plot
```

**Controls:** `SPACE` pause/resume · `Q` quit

---

### `browse_dataset.py` — interactive multi-episode browser

Loads all episodes in a directory and lets you navigate between them interactively.

```bash
python browse_dataset.py path/to/dataset_dir
python browse_dataset.py path/to/dataset_dir --fps 20
```

**Controls:**

| Key | Action |
|-----|--------|
| `SPACE` | Pause / resume |
| `←` / `A` | Previous episode |
| `→` / `D` | Next episode |
| `↑` / `W` | Step one frame forward |
| `↓` / `S` | Step one frame backward |
| `R` | Restart current episode |
| `Q` | Quit |

Auto-advances to the next episode when playback reaches the end.

---

### `dataset_summary.py` — statistics and trajectory overview

Prints a summary table for every episode and shows all joint trajectories overlaid in a single plot.

```bash
python dataset_summary.py path/to/dataset_dir
```

**Output:**
- Console table: episode name, step count, qpos/action min/max, file size
- Matplotlib figure: qpos and action per joint, all episodes overlaid

## Dependencies

```
h5py
opencv-python
numpy
matplotlib
```

Install with:

```bash
pip install h5py opencv-python numpy matplotlib
```
