#!/usr/bin/env python3
"""
Interactive joint-trajectory viewer with before/after smoothing and cut-point editor.

7 vertically stacked subplots — one per joint.  Each subplot overlays:
  • faint line   : raw qpos
  • solid line   : Savitzky-Golay smoothed qpos
  • red dots     : spike frames (step > --spike-thresh)
  • red shade    : frames that will be cut from start / end

Cut-point editing
─────────────────
  [  /  ]        move start-cut marker  left / right  by 1 frame
  {  /  }        move start-cut marker  left / right  by 10 frames
  ,  /  .        move end-cut marker    left / right  by 1 frame
  <  /  >        move end-cut marker    left / right  by 10 frames
  W              write (save) all cut decisions to --cuts file

Navigate episodes
─────────────────
  Right / D / N  next episode
  Left  / A / P  previous episode

Other
─────
  S              save current figure as PNG
  Q / Escape     quit

Usage:
    python plot_trajectories.py path/to/dataset_dir
    python plot_trajectories.py path/to/dataset_dir --cuts cuts.json
    python plot_trajectories.py path/to/dataset_dir --no-smooth
    python plot_trajectories.py path/to/dataset_dir --window 9 --poly 2
"""

import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import savgol_filter


# ─── constants ────────────────────────────────────────────────────────────────
JOINT_COLORS = [
    "#EF5350",  # J0 – red
    "#42A5F5",  # J1 – blue
    "#66BB6A",  # J2 – green
    "#FFA726",  # J3 – orange
    "#AB47BC",  # J4 – purple
    "#26C6DA",  # J5 – cyan
    "#8D6E63",  # J6 – gripper / brown
]
JOINT_LABELS  = ["J0", "J1", "J2", "J3", "J4", "J5", "Gripper"]
CUT_COLOR     = "#FF1744"   # red shading for cut regions
CUT_ALPHA     = 0.18
DEFAULT_SPIKE = 0.15        # rad


# ─── helpers ──────────────────────────────────────────────────────────────────

def find_episodes(path: str) -> tuple[list[str], int]:
    if os.path.isfile(path):
        parent = os.path.dirname(os.path.abspath(path))
        files  = sorted(glob.glob(os.path.join(parent, "*.hdf5")))
        start  = next((i for i, f in enumerate(files)
                       if os.path.abspath(path) == f), 0)
        return files or [os.path.abspath(path)], start
    files = sorted(glob.glob(os.path.join(path, "*.hdf5")))
    if not files:
        sys.exit(f"No .hdf5 files found in: {path}")
    return files, 0


def load_qpos(path: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return f["observations/qpos"][:]


def apply_smooth(arr: np.ndarray, window: int, poly: int) -> np.ndarray:
    if len(arr) < window:
        return arr.copy()
    out = arr.copy()
    for d in range(arr.shape[1] - 1):   # skip gripper
        out[:, d] = savgol_filter(arr[:, d], window_length=window, polyorder=poly)
    return out


def load_cuts(path: str) -> dict:
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_cuts(cuts: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(cuts, f, indent=2)
    print(f"Cuts saved → {path}")


# ─── figure builder ───────────────────────────────────────────────────────────

def build_figure(n_joints: int):
    fig = plt.figure(figsize=(13, 14))
    fig.patch.set_facecolor("#0d0d0d")

    gs = gridspec.GridSpec(n_joints, 1, hspace=0.06,
                           left=0.08, right=0.97, top=0.93, bottom=0.04)
    axes         = []
    lines_raw    = []
    lines_smooth = []
    scatters     = []
    cut_spans    = []   # list of (start_span, end_span) per axis

    for j in range(n_joints):
        ax = fig.add_subplot(gs[j])
        ax.set_facecolor("#111111")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2a2a")
        ax.tick_params(colors="#777777", labelsize=7)
        ax.grid(axis="y", color="#1e1e1e", linewidth=0.6)

        col = JOINT_COLORS[j]
        lr, = ax.plot([], [], color=col, alpha=0.22, linewidth=0.9, label="raw")
        ls, = ax.plot([], [], color=col, alpha=0.95, linewidth=1.5, label="smoothed")
        sc   = ax.scatter([], [], color="#FF5252", s=14, zorder=5)

        # cut shading (axvspan placeholders — will be replaced each redraw)
        sp_start = ax.axvspan(0, 0, color=CUT_COLOR, alpha=0, zorder=3)
        sp_end   = ax.axvspan(0, 0, color=CUT_COLOR, alpha=0, zorder=3)

        ax.set_ylabel(JOINT_LABELS[j], color=col, fontsize=9,
                      rotation=0, labelpad=30, va="center")
        if j < n_joints - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("frame", color="#777777", fontsize=8)
        if j == 0:
            ax.legend(loc="upper right", fontsize=7,
                      facecolor="#1a1a1a", edgecolor="#3a3a3a",
                      labelcolor="#cccccc", framealpha=0.85)

        axes.append(ax)
        lines_raw.append(lr)
        lines_smooth.append(ls)
        scatters.append(sc)
        cut_spans.append((sp_start, sp_end))

    return fig, axes, lines_raw, lines_smooth, scatters, cut_spans


# ─── draw ─────────────────────────────────────────────────────────────────────

def redraw(fig, axes, lines_raw, lines_smooth, scatters, cut_spans,
           episodes, state, window, poly, do_smooth, spike_thresh):
    ep_idx   = state["ep_idx"]
    path     = episodes[ep_idx]
    ep_name  = os.path.basename(path)
    raw      = load_qpos(path)
    T        = len(raw)
    t        = np.arange(T)
    sm       = apply_smooth(raw, window, poly) if do_smooth else raw

    cut_start = state["cut_start"]   # frames to cut from front
    cut_end   = state["cut_end"]     # frames to cut from back

    for j, ax in enumerate(axes):
        raw_vals = raw[:, j]
        sm_vals  = sm[:, j]

        lines_raw[j].set_data(t, raw_vals)
        lines_smooth[j].set_data(t, sm_vals)
        lines_smooth[j].set_visible(do_smooth)

        # spikes
        scatters[j].remove()
        deltas    = np.abs(np.diff(raw_vals))
        spike_idx = np.where(deltas > spike_thresh)[0]
        scatters[j] = ax.scatter(
            spike_idx, raw_vals[spike_idx],
            color="#FF5252", s=14, zorder=5,
        )

        ax.set_xlim(0, max(T - 1, 1))
        vmin, vmax = raw_vals.min(), raw_vals.max()
        margin = max((vmax - vmin) * 0.08, 0.05)
        ax.set_ylim(vmin - margin, vmax + margin)

        # cut shading
        sp_s, sp_e = cut_spans[j]
        sp_s.remove()
        sp_e.remove()
        cut_spans[j] = (
            ax.axvspan(-0.5, cut_start - 0.5,
                       color=CUT_COLOR, alpha=CUT_ALPHA if cut_start > 0 else 0,
                       zorder=3),
            ax.axvspan(T - cut_end - 0.5, T - 0.5,
                       color=CUT_COLOR, alpha=CUT_ALPHA if cut_end > 0 else 0,
                       zorder=3),
        )

    # title with cut info
    n_kept  = T - cut_start - cut_end
    cut_tag = f"  cut start={cut_start}  end={cut_end}  kept={n_kept}"
    nav_tag = f"  [{ep_idx + 1}/{len(episodes)}]  ←/→ episode"
    sm_tag  = f"  w={window},p={poly}" if do_smooth else "  raw"
    fig.suptitle(f"{ep_name}{sm_tag}{cut_tag}{nav_tag}",
                 color="#dddddd", fontsize=9, fontweight="bold")
    fig.canvas.draw_idle()

    return scatters, cut_spans


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Interactive joint-trajectory viewer with cut editor.")
    parser.add_argument("path",
                        help="HDF5 file or directory of episodes")
    parser.add_argument("--cuts",       default="cuts.json",
                        help="JSON file to load/save per-episode cut decisions "
                             "(default: cuts.json)")
    parser.add_argument("--window",     type=int,   default=5,
                        help="Savitzky-Golay window (odd, default 5)")
    parser.add_argument("--poly",       type=int,   default=2,
                        help="Savitzky-Golay poly order (default 2)")
    parser.add_argument("--no-smooth",  action="store_true",
                        help="Show raw trajectory only")
    parser.add_argument("--spike-thresh", type=float, default=DEFAULT_SPIKE,
                        help=f"Spike step threshold in rad (default {DEFAULT_SPIKE})")
    args = parser.parse_args()

    if args.window % 2 == 0:
        sys.exit("ERROR: --window must be odd")

    episodes, ep_idx = find_episodes(args.path)
    do_smooth = not args.no_smooth
    cuts      = load_cuts(args.cuts)   # {ep_name: {"start": N, "end": M}}

    def get_cuts(name):
        c = cuts.get(name, {})
        return c.get("start", 0), c.get("end", 0)

    def set_cuts(name, start, end):
        cuts[name] = {"start": int(start), "end": int(end)}

    def ep_len(path):
        with h5py.File(path, "r") as f:
            return f["observations/qpos"].shape[0]

    init_s, init_e = get_cuts(os.path.basename(episodes[ep_idx]))
    state = {
        "ep_idx":    ep_idx,
        "cut_start": init_s,
        "cut_end":   init_e,
    }

    print(f"Found {len(episodes)} episode(s).  Cuts file: {args.cuts}")
    print("  [ / ]       move start cut  ±1     { / }   ±10")
    print("  , / .       move end cut    ±1     < / >   ±10")
    print("  W           save all cuts to JSON")
    print("  ←/→ D/A     prev/next episode   S  save PNG   Q  quit")

    fig, axes, lines_raw, lines_smooth, scatters, cut_spans = build_figure(7)

    state["scatters"], state["cut_spans"] = redraw(
        fig, axes, lines_raw, lines_smooth, scatters, cut_spans,
        episodes, state, args.window, args.poly, do_smooth, args.spike_thresh
    )

    def on_key(event):
        key     = event.key
        ep_name = os.path.basename(episodes[state["ep_idx"]])
        T       = ep_len(episodes[state["ep_idx"]])
        cs      = state["cut_start"]
        ce      = state["cut_end"]
        changed = True

        # ── start-cut controls ──
        if   key == "[":         cs = max(0, cs - 1)
        elif key == "]":         cs = min(T - ce - 1, cs + 1)
        elif key == "{":         cs = max(0, cs - 10)
        elif key == "}":         cs = min(T - ce - 1, cs + 10)
        # ── end-cut controls ──
        elif key == ",":         ce = max(0, ce - 1)
        elif key == ".":         ce = min(T - cs - 1, ce + 1)
        elif key == "<":         ce = max(0, ce - 10)
        elif key == ">":         ce = min(T - cs - 1, ce + 10)
        # ── navigation ──
        elif key in ("right", "d", "n"):
            set_cuts(ep_name, cs, ce)
            state["ep_idx"] = (state["ep_idx"] + 1) % len(episodes)
            new_name = os.path.basename(episodes[state["ep_idx"]])
            state["cut_start"], state["cut_end"] = get_cuts(new_name)
        elif key in ("left", "a", "p"):
            set_cuts(ep_name, cs, ce)
            state["ep_idx"] = (state["ep_idx"] - 1) % len(episodes)
            new_name = os.path.basename(episodes[state["ep_idx"]])
            state["cut_start"], state["cut_end"] = get_cuts(new_name)
        # ── save / export ──
        elif key == "w":
            set_cuts(ep_name, cs, ce)
            save_cuts(cuts, args.cuts)
            changed = False
        elif key == "s":
            out = os.path.splitext(ep_name)[0] + "_traj.png"
            fig.savefig(out, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"Figure saved → {out}")
            changed = False
        elif key in ("q", "escape"):
            plt.close(fig)
            return
        else:
            return

        if key not in ("right", "d", "n", "left", "a", "p"):
            state["cut_start"] = cs
            state["cut_end"]   = ce

        if changed:
            state["scatters"], state["cut_spans"] = redraw(
                fig, axes, lines_raw, lines_smooth,
                state["scatters"], state["cut_spans"],
                episodes, state, args.window, args.poly, do_smooth,
                args.spike_thresh
            )

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


if __name__ == "__main__":
    main()
