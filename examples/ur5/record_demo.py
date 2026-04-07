"""
Demo recorder — captures the front D415 camera during UR5 inference.

Camera layout:
    SERIAL_1 = "105422061000"  # exterior D415  → base_0_rgb   (used by inference)
    SERIAL_2 = "352122273671"  # wrist D405     → left_wrist_0_rgb (used by inference)
    SERIAL_3 = "104122061227"  # front D415 USB2.1 → NOT used by inference ← records here

Launch this in a separate terminal before (or during) inference:

    uv run examples/ur5/record_demo.py
    uv run examples/ur5/record_demo.py --realsense-serial 104122061227  # explicit serial

    # Fall back to a USB/UVC webcam instead
    uv run examples/ur5/record_demo.py --no-realsense --camera-index 0

    # Override output directory and target FPS
    uv run examples/ur5/record_demo.py --out-dir /tmp/demos --fps 30

    # Burn the OSD overlay into the saved video
    uv run examples/ur5/record_demo.py --burn-overlay

Press Ctrl+C to stop. The video is saved automatically on exit.

Notes:
    - Default mode streams the front D415 (serial 104122061227) via pyrealsense2.
    - Use --no-realsense to fall back to a USB/UVC webcam (cv2.VideoCapture).
    - Output filename: YYYYMMDD_HHMMSS.mp4  (or .avi when mp4v unavailable)
    - OSD overlay is always shown in the preview window.
      Use --burn-overlay to also write it into the saved video file.
"""

from collections import deque
import dataclasses
import datetime
import logging
from pathlib import Path
import signal
import time

import cv2
import numpy as np
import tyro

log = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    # ── Camera source ──────────────────────────────────────────────────────
    # Use RealSense front D415 (default). Pass --no-realsense to use a USB webcam.
    realsense: bool = True
    # Front D415 serial — not used by inference, safe to stream concurrently.
    realsense_serial: str = "104122061227"
    # RealSense stream resolution.
    # Front D415 USB 2.1 supported bgr8 modes: 640x480@30, 1280x720@15, 1920x1080@8.
    realsense_width: int = 640
    realsense_height: int = 480
    # USB/UVC webcam index (only used when --no-realsense)
    camera_index: int = 0

    # ── Recording settings ─────────────────────────────────────────────────
    fps: float = 30.0
    out_dir: str = "demos"
    # Show a live preview window (disable for headless servers)
    preview: bool = True
    # Burn the OSD overlay into the saved video (default: save clean frames)
    burn_overlay: bool = False
    # Print connected RealSense devices and supported color modes, then exit
    list_devices: bool = False


# ── Camera backends ────────────────────────────────────────────────────────


class UVCCamera:
    def __init__(self, index: int, fps: float):
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {index}")
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        # Prefer higher resolution if the device supports it
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info(f"[Camera] UVC index={index}  resolution={w}x{h}  fps={fps}")

    @property
    def frame_size(self) -> tuple[int, int]:
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    def read_bgr(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self):
        self._cap.release()


def list_realsense_devices() -> None:
    """Print all connected RealSense devices and their supported color stream modes."""
    import pyrealsense2 as rs

    ctx = rs.context()
    devices = ctx.query_devices()
    if not devices:
        print("No RealSense devices found.")
        return
    for dev in devices:
        serial = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
        usb = dev.get_info(rs.camera_info.usb_type_descriptor)
        print(f"\n  Device: {name}  serial={serial}  USB={usb}")
        for sensor in dev.query_sensors():
            for profile in sensor.get_stream_profiles():
                vp = profile.as_video_stream_profile()
                if profile.stream_type() == rs.stream.color and profile.format() == rs.format.bgr8:
                    print(f"    color bgr8  {vp.width()}x{vp.height()}@{vp.fps()}fps")


class RealSenseCamera:
    def __init__(self, serial: str, width: int, height: int, fps: float):
        import pyrealsense2 as rs  # lazy import — not required for UVC mode

        # Verify the device is present before trying to start
        ctx = rs.context()
        found = [d.get_info(rs.camera_info.serial_number) for d in ctx.query_devices()]
        if serial not in found:
            raise RuntimeError(
                f"RealSense serial {serial!r} not found. Connected devices: {found}\n"
                "Run with --list-devices to see available serials and supported modes."
            )

        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, int(fps))
        self._pipeline.start(cfg)
        log.info(f"[Camera] RealSense serial={serial}  {width}x{height}@{fps}fps")
        self._size = (width, height)

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._size

    def read_bgr(self) -> np.ndarray | None:
        frames = self._pipeline.wait_for_frames(timeout_ms=2000)
        color = frames.get_color_frame()
        if not color:
            return None
        return np.asanyarray(color.get_data())

    def close(self):
        self._pipeline.stop()


# ── OSD overlay ───────────────────────────────────────────────────────────

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_BAR_H = 36  # px height of each status bar
_ALPHA = 0.55  # translucency of the dark bar background


def _draw_overlay(
    frame: np.ndarray,
    elapsed: float,
    frame_count: int,
    actual_fps: float,
    out_path: Path,
    rec_visible: bool,  # blinking state
) -> np.ndarray:
    """Return a copy of *frame* with a top and bottom OSD bar."""
    vis = frame.copy()
    h, w = vis.shape[:2]

    # ── helper: draw semi-transparent filled rect ─────────────────────────
    def _bar(y0: int, y1: int) -> None:
        roi = vis[y0:y1, 0:w]
        dark = np.zeros_like(roi)
        cv2.addWeighted(dark, _ALPHA, roi, 1 - _ALPHA, 0, roi)
        vis[y0:y1, 0:w] = roi

    _bar(0, _BAR_H)  # top bar
    _bar(h - _BAR_H, h)  # bottom bar

    # ── top-left: blinking REC dot + label ───────────────────────────────
    cx, cy = 16, _BAR_H // 2
    if rec_visible:
        cv2.circle(vis, (cx, cy), 8, (0, 0, 220), -1)
    cv2.putText(vis, "REC", (cx + 14, cy + 5), _FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # ── top-center: elapsed time HH:MM:SS ────────────────────────────────
    h_e, m_e, s_e = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
    elapsed_str = f"{h_e:02d}:{m_e:02d}:{s_e:02d}"
    (tw, _), _ = cv2.getTextSize(elapsed_str, _FONT, 0.6, 1)
    cv2.putText(vis, elapsed_str, (w // 2 - tw // 2, cy + 6), _FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # ── top-right: live FPS ───────────────────────────────────────────────
    fps_str = f"{actual_fps:.1f} fps"
    (fw, _), _ = cv2.getTextSize(fps_str, _FONT, 0.55, 1)
    cv2.putText(vis, fps_str, (w - fw - 10, cy + 5), _FONT, 0.55, (200, 255, 200), 1, cv2.LINE_AA)

    # ── bottom-left: wall-clock timestamp ────────────────────────────────
    ts = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(vis, ts, (10, h - _BAR_H + 24), _FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # ── bottom-right: frame count + filename ─────────────────────────────
    info = f"#{frame_count:06d}  {out_path.name}"
    (iw, _), _ = cv2.getTextSize(info, _FONT, 0.45, 1)
    cv2.putText(vis, info, (w - iw - 10, h - _BAR_H + 24), _FONT, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    return vis


# ── Video writer helpers ───────────────────────────────────────────────────


def _make_writer(path: Path, frame_size: tuple[int, int], fps: float) -> tuple[cv2.VideoWriter, Path]:
    """Try H264 MP4 first, fall back to XVID AVI."""
    fourcc_mp4 = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc_mp4, fps, frame_size)
    if writer.isOpened():
        return writer, path
    # Fallback
    avi_path = path.with_suffix(".avi")
    fourcc_avi = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(avi_path), fourcc_avi, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter at {avi_path}")
    log.warning(f"[Recorder] mp4v unavailable, using XVID AVI: {avi_path}")
    return writer, avi_path


# ── Main ───────────────────────────────────────────────────────────────────


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.list_devices:
        list_realsense_devices()
        return

    # ── Open camera ──────────────────────────────────────────────────────
    if args.realsense:
        camera = RealSenseCamera(
            serial=args.realsense_serial,
            width=args.realsense_width,
            height=args.realsense_height,
            fps=args.fps,
        )
    else:
        camera = UVCCamera(index=args.camera_index, fps=args.fps)

    # ── Prepare output file ───────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{timestamp}.mp4"
    writer, out_path = _make_writer(out_path, camera.frame_size, args.fps)
    log.info(f"[Recorder] Recording to {out_path}")

    # ── Graceful shutdown on Ctrl+C ───────────────────────────────────────
    running = True

    def _stop(_sig, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    # ── Capture loop ──────────────────────────────────────────────────────
    frame_count = 0
    t_start = time.monotonic()
    period = 1.0 / args.fps
    # Rolling window of recent frame timestamps for live FPS measurement
    ts_window: deque[float] = deque(maxlen=30)

    try:
        while running:
            t0 = time.monotonic()

            frame = camera.read_bgr()
            if frame is None:
                log.warning("[Recorder] Dropped frame")
            else:
                frame_count += 1
                ts_window.append(t0)

                # Rolling actual FPS
                actual_fps = (len(ts_window) - 1) / (ts_window[-1] - ts_window[0]) if len(ts_window) >= 2 else 0.0

                elapsed = t0 - t_start
                rec_visible = int(elapsed) % 2 == 0  # blink every second

                if args.burn_overlay:
                    to_save = _draw_overlay(frame, elapsed, frame_count, actual_fps, out_path, rec_visible)
                else:
                    to_save = frame
                writer.write(to_save)

                if args.preview:
                    vis = _draw_overlay(frame, elapsed, frame_count, actual_fps, out_path, rec_visible)
                    cv2.imshow("Demo Recording  (press q to stop)", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            # Pace to target FPS
            elapsed_loop = time.monotonic() - t0
            sleep = period - elapsed_loop
            if sleep > 0:
                time.sleep(sleep)

    finally:
        writer.release()
        camera.close()
        if args.preview:
            cv2.destroyAllWindows()

        duration = time.monotonic() - t_start
        summary_fps = frame_count / duration if duration > 0 else 0
        log.info(f"[Recorder] Saved {frame_count} frames ({duration:.1f}s, {summary_fps:.1f} fps actual) → {out_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
