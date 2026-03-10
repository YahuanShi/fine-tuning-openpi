"""
UR5 real-robot environment for Pi0/Pi0.5 inference.

Hardware:
    - UR5 @ 192.168.1.100 (RTDE)
    - Weiss CRG 30-050 gripper @ /dev/ttyACM0
    - RealSense D415 exterior  (serial 105422060444)
    - RealSense D405 wrist     (serial 352122273671)

Observation dict (keys consumed by UR5Inputs):
    base_rgb   (224, 224, 3) uint8  RGB exterior image
    wrist_rgb  (224, 224, 3) uint8  RGB wrist image
    joints     (6,) float32         current joint angles in radians
    gripper    (1,) float32         0 = open, 1 = closed
    prompt     str                  task instruction

Action dict (produced by UR5Outputs after AbsoluteActions):
    actions    (7,) float32         [6 joint angles rad (absolute), gripper 0/1]
"""

import logging
import threading
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import rtde_control
import rtde_receive
import serial
from openpi_client.runtime import environment as _environment
from typing_extensions import override

log = logging.getLogger(__name__)

# ══════════════════════════════ Configuration ══════════════════════════════

UR5_IP            = "192.168.1.100"
GRIPPER_PORT      = "/dev/ttyACM0"
GRIPPER_BAUDRATE  = 115200
GRIPPER_MAX_MM    = 30.0
GRIPPER_OPEN_THRESH_MM = 5.0   # below this → considered closed

CAM_SERIAL_BASE   = "105422060444"   # D415 exterior
CAM_SERIAL_WRIST  = "352122273671"   # D405 wrist
IMAGE_SIZE        = 224

HOME_DEG  = [0.0, -90.0, 90.0, -90.0, -90.0, 0.0]
HOME_RAD  = np.radians(HOME_DEG)

SERVO_J_TIME      = 0.016   # s per step (~60 Hz)
SERVO_J_LOOKAHEAD = 0.08
SERVO_J_GAIN      = 300
MAX_JOINT_VEL     = 1.0     # rad/s safety clamp


# ══════════════════════════════ Gripper ══════════════════════════════


class WeissCRGGripper:
    """Minimal driver for Weiss CRG 30-050 (binary open / close)."""

    PDIN_PER_MM = 163.17

    def __init__(self, port: str = GRIPPER_PORT, baudrate: int = GRIPPER_BAUDRATE):
        self._lock       = threading.Lock()
        self._ser        = serial.Serial(port=port, baudrate=baudrate, timeout=2.0)
        self._position_mm = 0.0
        self._closed_pdin = 150
        log.info(f"[Gripper] Opened {port} @ {baudrate} baud")

    def _parse_pdin(self, line: str):
        try:
            data = line[7:].split("]")[0].split(",")
            raw = (int(data[0], 16) << 8) | int(data[1], 16)
            self._position_mm = max(0.0, (raw - self._closed_pdin) / self.PDIN_PER_MM)
        except Exception:
            pass

    def _send(self, cmd: str) -> str:
        with self._lock:
            try:
                self._ser.reset_input_buffer()
                self._ser.write((cmd + "\n").encode("ascii"))
                for _ in range(30):
                    resp = self._ser.readline().decode("ascii", errors="ignore").strip()
                    if resp.startswith("@PDIN"):
                        self._parse_pdin(resp)
                        continue
                    return resp
            except Exception as e:
                log.warning(f"[Gripper] Serial error '{cmd}': {e}")
        return ""

    def _wait_motion(self, timeout: float = 5.0):
        start, prev, stable, moved = None, None, 0, False
        t0 = time.monotonic()
        with self._lock:
            saved, self._ser.timeout = self._ser.timeout, 0.1
        try:
            while time.monotonic() - t0 < timeout:
                with self._lock:
                    line = self._ser.readline().decode("ascii", errors="ignore").strip()
                if not line.startswith("@PDIN"):
                    continue
                try:
                    data = line[7:].split("]")[0].split(",")
                    v = (int(data[0], 16) << 8) | int(data[1], 16)
                except Exception:
                    continue
                if start is None:
                    start = v
                if not moved and abs(v - start) >= 200:
                    moved = True
                if moved:
                    stable = (stable + 1) if prev is not None and abs(v - prev) <= 5 else 0
                    if stable >= 15:
                        self._position_mm = max(0.0, (v - self._closed_pdin) / self.PDIN_PER_MM)
                        return
                prev = v
        finally:
            with self._lock:
                self._ser.timeout = saved

    def home(self):
        log.info("[Gripper] Homing — closing to calibrate...")
        self._send("PDOUT=[00,00]")
        time.sleep(0.1)
        self._send("PDOUT=[03,00]")
        self._wait_motion(5.0)
        self._send("PDOUT=[00,00]")
        time.sleep(0.1)
        self._send("PDOUT=[07,00]")
        self._wait_motion(5.0)
        log.info("[Gripper] Homing done, gripper open.")

    def move_to_pos(self, width_mm: float):
        if width_mm > GRIPPER_OPEN_THRESH_MM:
            self._send("PDOUT=[00,00]")
            self._send("PDOUT=[07,00]")
        else:
            self._send("PDOUT=[00,00]")
            self._send("PDOUT=[03,00]")

    def get_width(self) -> float:
        with self._lock:
            saved, self._ser.timeout = self._ser.timeout, 0.3
            try:
                for _ in range(20):
                    line = self._ser.readline().decode("ascii", errors="ignore").strip()
                    if line.startswith("@PDIN"):
                        self._parse_pdin(line)
                        break
            except Exception:
                pass
            finally:
                self._ser.timeout = saved
        return self._position_mm

    def close(self):
        self._send("PDOUT=[00,00]")
        with self._lock:
            if self._ser and self._ser.is_open:
                self._ser.close()
        log.info("[Gripper] Port closed.")


# ══════════════════════════════ Image helpers ══════════════════════════════


def _center_crop_resize(bgr: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Center-crop to square then resize; convert BGR → RGB."""
    h, w = bgr.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    crop = bgr[y0:y0 + s, x0:x0 + s]
    resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def _start_pipeline(serial_num: str) -> rs.pipeline:
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial_num)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(cfg)
    log.info(f"[Camera] Pipeline started — serial {serial_num}")
    return pipeline


# ══════════════════════════════ UR5 Environment ══════════════════════════════


class UR5Environment(_environment.Environment):
    """
    Real UR5 environment for Pi0/Pi0.5 policy inference.

    Usage::

        env = UR5Environment(prompt="pick up the cube")
        # env.reset() moves robot to home and opens gripper
        # env.get_observation() returns current state + images
        # env.apply_action({"actions": np.array(...)}) sends servoJ + gripper cmd
    """

    def __init__(
        self,
        prompt: str = "perform the task",
        control_hz: float = 10.0,
    ):
        self._prompt      = prompt
        self._control_hz  = control_hz
        self._dt          = 1.0 / control_hz

        # ── UR5 RTDE ─────────────────────────────────────────────
        log.info(f"[UR5] Connecting RTDE to {UR5_IP}...")
        self._rtde_r = rtde_receive.RTDEReceiveInterface(UR5_IP)
        self._rtde_c = rtde_control.RTDEControlInterface(UR5_IP)
        log.info("[UR5] RTDE connected.")

        # ── Gripper ───────────────────────────────────────────────
        log.info(f"[UR5] Connecting gripper on {GRIPPER_PORT}...")
        self._gripper = WeissCRGGripper(GRIPPER_PORT, GRIPPER_BAUDRATE)
        self._last_gripper_open: bool | None = None

        # ── Cameras ───────────────────────────────────────────────
        log.info("[UR5] Starting cameras...")
        self._pipe_base  = _start_pipeline(CAM_SERIAL_BASE)
        self._pipe_wrist = _start_pipeline(CAM_SERIAL_WRIST)

        # ── state ─────────────────────────────────────────────────
        self._last_cmd_rad = HOME_RAD.copy()

    # ── Environment interface ──────────────────────────────────────────────

    @override
    def reset(self) -> None:
        """Move UR5 to home, home the gripper."""
        log.info("[UR5] Resetting to home position...")
        self._rtde_c.moveJ(HOME_RAD.tolist(), speed=0.5, acceleration=0.5)
        self._last_cmd_rad = HOME_RAD.copy()
        self._gripper.home()
        self._last_gripper_open = True
        log.info("[UR5] Reset complete.")

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        # Joint state
        q_rad   = np.array(self._rtde_r.getActualQ(), dtype=np.float32)
        width   = self._gripper.get_width()
        gripper = np.array([0.0 if width > GRIPPER_OPEN_THRESH_MM else 1.0], dtype=np.float32)

        # Images
        base_bgr  = self._grab_frame(self._pipe_base)
        wrist_bgr = self._grab_frame(self._pipe_wrist)
        base_rgb  = _center_crop_resize(base_bgr)
        wrist_rgb = _center_crop_resize(wrist_bgr)

        return {
            "joints":    q_rad,
            "gripper":   gripper,
            "base_rgb":  base_rgb,
            "wrist_rgb": wrist_rgb,
            "prompt":    self._prompt,
        }

    @override
    def apply_action(self, action: dict) -> None:
        acts = np.asarray(action["actions"], dtype=np.float64)

        # ── joints: velocity-limited servoJ ──────────────────────
        target_rad = acts[:6]
        max_step   = MAX_JOINT_VEL * self._dt
        delta      = target_rad - self._last_cmd_rad
        cmd_rad    = self._last_cmd_rad + np.clip(delta, -max_step, max_step)
        self._last_cmd_rad = cmd_rad.copy()

        try:
            self._rtde_c.servoJ(
                cmd_rad.tolist(),
                velocity=0,
                acceleration=0,
                time=SERVO_J_TIME,
                lookahead_time=SERVO_J_LOOKAHEAD,
                gain=SERVO_J_GAIN,
            )
        except Exception as e:
            log.warning(f"[UR5] servoJ error: {e}")

        # ── gripper: binary open / close ─────────────────────────
        # action[6]: 0 = open, 1 = closed
        want_open = float(acts[6]) < 0.5
        if want_open != self._last_gripper_open:
            width_mm = GRIPPER_MAX_MM if want_open else 0.0
            self._gripper.move_to_pos(width_mm)
            self._last_gripper_open = want_open
            log.info(f"[UR5] Gripper {'opening' if want_open else 'closing'}")

    # ── cleanup ───────────────────────────────────────────────────────────

    def close(self) -> None:
        """Stop RTDE streaming and release hardware."""
        try:
            self._rtde_c.servoStop()
            self._rtde_c.stopScript()
        except Exception:
            pass
        self._gripper.close()
        try:
            self._pipe_base.stop()
            self._pipe_wrist.stop()
        except Exception:
            pass
        log.info("[UR5] Environment closed.")

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _grab_frame(pipeline: rs.pipeline) -> np.ndarray:
        frames = pipeline.wait_for_frames(timeout_ms=2000)
        color  = frames.get_color_frame()
        if not color:
            raise RuntimeError("No color frame from camera")
        return np.asanyarray(color.get_data())
