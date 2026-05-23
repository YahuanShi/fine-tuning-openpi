"""
UR5 real-robot environment for Pi0/Pi0.5 inference.

Hardware:
    - UR5 @ 10.0.0.1 (RTDE)
    - Weiss CRG 30-050 gripper @ /dev/ttyACM0
    - RealSense D415 exterior  (serial 105422061000)
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

import contextlib
import logging
import threading
import time

import cv2
import numpy as np
from openpi_client.runtime import environment as _environment
import pyrealsense2 as rs
import rtde_control
import rtde_receive
import serial
from typing_extensions import override

log = logging.getLogger(__name__)

# ══════════════════════════════ Configuration ══════════════════════════════

UR5_IP = "10.0.0.1"
GRIPPER_PORT = "/dev/ttyACM0"
GRIPPER_BAUDRATE = 9600
GRIPPER_MAX_MM = 50
GRIPPER_OPEN_THRESH_MM = 5.0  # below this → considered closed

CAM_SERIAL_BASE = "105422061000"  # D415 exterior
CAM_SERIAL_WRIST = "352122273671"  # D405 wrist
IMAGE_SIZE = 224

HOME_DEG = [45.0, -20.0, -140.0, -40.0, -270.0, 0.0]
HOME_RAD = np.radians(HOME_DEG)

SERVO_J_TIME = 0.1  # s per step (must match 1/CONTROL_HZ)
SERVO_J_LOOKAHEAD = 0.2  # s look-ahead — servoJ's primary smoothing knob; higher = smoother
SERVO_J_GAIN = 200
MAX_JOINT_VEL = 0.8  # rad/s safety clamp — training data peaks at ~1.03 rad/s


# ══════════════════════════════ Gripper ══════════════════════════════


class WeissCRGGripper:
    """
    Driver for Weiss IEG76 / CRG gripper via DC-IOLink USB adapter.

    Protocol ref: https://github.com/ipa320/weiss_gripper_ieg76
    @PDIN=[B0,B1,B2,B3]: pos_mm = ((B0<<8)|B1)/100, B3 = status flags
    PDOUT=[02,00] open  |  PDOUT=[03,00] close  |  PDOUT=[07,00] reference
    """

    FLAG_OPEN = 1
    FLAG_CLOSED = 2
    FLAG_HOLDING = 3
    FLAG_FAULT = 4

    def __init__(self, port: str = GRIPPER_PORT, baudrate: int = GRIPPER_BAUDRATE):
        self._lock = threading.Lock()
        self._ser = serial.Serial(port=port, baudrate=baudrate, timeout=0.2)
        self._position_mm = 0.0
        self._flags = 0
        log.info(f"[Gripper] Opened {port} @ {baudrate} baud")
        self._initialise()

    # ── Internal helpers ──────────────────────────────────────────────────

    def _send(self, cmd: str, wait: float = 0.3) -> None:
        with self._lock:
            try:
                self._ser.reset_input_buffer()
                self._ser.write((cmd + "\n").encode("ascii"))
            except Exception as e:
                log.warning(f"[Gripper] Serial error '{cmd}': {e}")
        time.sleep(wait)

    def _parse_pdin(self, line: str) -> bool:
        try:
            inner = line[7:].split("]")[0]
            parts = [int(x, 16) for x in inner.split(",")]
            self._position_mm = ((parts[0] << 8) | parts[1]) / 100.0
            self._flags = parts[3] if len(parts) >= 4 else 0
            return True
        except Exception:
            return False

    def _read_pdin(self, timeout: float = 1.0) -> bool:
        t0 = time.monotonic()
        with self._lock:
            saved, self._ser.timeout = self._ser.timeout, 0.15
        try:
            while time.monotonic() - t0 < timeout:
                with self._lock:
                    line = self._ser.readline().decode("ascii", errors="ignore").strip()
                if line.startswith("@PDIN=[") and self._parse_pdin(line):
                    return True
        finally:
            with self._lock:
                self._ser.timeout = saved
        return False

    def _wait_flag(self, flag_bit: int, timeout: float = 6.0) -> bool:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self._read_pdin(0.5) and (self._flags & (1 << flag_bit)):
                return True
        return False

    def _set_positions(self, open_mm: float = GRIPPER_MAX_MM, close_mm: float = 0.5) -> None:
        def enc(mm):
            v = int(mm * 100)
            return f"[{(v >> 8) & 0xFF:02x},{v & 0xFF:02x}]"

        self._send(f"SETPARAM(96, 2, {enc(open_mm)})", 0.3)
        self._send(f"SETPARAM(96, 1, {enc(close_mm)})", 0.3)
        self._send("SETPARAM(96, 3, [64])", 0.3)

    def _initialise(self) -> None:
        """Mandatory startup sequence before any PDOUT command."""
        for cmd in ["ID?", "ID?", "FALLBACK(1)", "MODE?", "RESTART()", "OPERATE()"]:
            self._send(cmd, 0.5)
        self._send("PDOUT=[00,00]", 0.5)
        log.info("[Gripper] Initialisation complete.")

    # ── Public API ────────────────────────────────────────────────────────

    def home(self) -> None:
        """Reference cycle: closes then opens fully."""
        log.info("[Gripper] Homing (reference cycle)...")
        self._set_positions(GRIPPER_MAX_MM, 0.5)
        self._send("PDOUT=[07,00]", 0.2)
        self._wait_flag(self.FLAG_OPEN, timeout=10.0)
        log.info("[Gripper] Home complete — gripper open.")

    def move_to_pos(self, width_mm: float) -> None:
        """Open (width_mm > threshold) or close."""
        self._set_positions(GRIPPER_MAX_MM, 0.5)
        if width_mm > GRIPPER_OPEN_THRESH_MM:
            self._send("PDOUT=[02,00]", 0.2)  # open
        else:
            self._send("PDOUT=[03,00]", 0.2)  # close

    def get_width(self) -> float:
        self._read_pdin(timeout=0.08)
        return self._position_mm

    def close(self) -> None:
        """Shutdown: deactivate and close serial port."""
        self._send("PDOUT=[00,00]", 0.3)
        self._send("FALLBACK(1)", 0.3)
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
    crop = bgr[y0 : y0 + s, x0 : x0 + s]
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
        self._prompt = prompt
        self._control_hz = control_hz
        self._dt = 1.0 / control_hz

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
        self._pipe_base = _start_pipeline(CAM_SERIAL_BASE)
        self._pipe_wrist = _start_pipeline(CAM_SERIAL_WRIST)

        # ── state ─────────────────────────────────────────────────
        self._last_cmd_rad = HOME_RAD.copy()
        self._is_first_reset = True

    # ── Environment interface ──────────────────────────────────────────────

    @override
    def reset(self) -> None:
        with contextlib.suppress(Exception):
            self._rtde_c.servoStop()
        time.sleep(0.2)
        if self._is_first_reset:
            log.info("[UR5] First reset — moving to home position...")
            self._rtde_c.moveJ(HOME_RAD.tolist(), speed=0.5, acceleration=0.5)
            self._last_cmd_rad = HOME_RAD.copy()
            self._gripper.home()
            self._is_first_reset = False
        else:
            log.info("[UR5] In-place reset — disabled for comparison.")
            return
        self._last_gripper_open = True
        log.info("[UR5] Ready.")

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        # Joint state
        q_rad = np.array(self._rtde_r.getActualQ(), dtype=np.float32)
        width = self._gripper.get_width()
        gripper = np.array([0.0 if width > GRIPPER_OPEN_THRESH_MM else 1.0], dtype=np.float32)

        # Images
        base_bgr = self._grab_frame(self._pipe_base)
        wrist_bgr = self._grab_frame(self._pipe_wrist)
        base_rgb = _center_crop_resize(base_bgr)
        wrist_rgb = _center_crop_resize(wrist_bgr)

        return {
            "joints": q_rad,
            "gripper": gripper,
            "base_rgb": base_rgb,
            "wrist_rgb": wrist_rgb,
            "prompt": self._prompt,
        }

    @override
    def apply_action(self, action: dict) -> None:
        acts = np.asarray(action["actions"], dtype=np.float32)

        # ── joints: velocity-limited servoJ ──────────────────────
        target_rad = acts[:6]
        max_step = MAX_JOINT_VEL * self._dt
        delta = target_rad - self._last_cmd_rad
        cmd_rad = self._last_cmd_rad + np.clip(delta, -max_step, max_step)
        self._last_cmd_rad = cmd_rad.copy()

        try:
            self._rtde_c.servoJ(
                cmd_rad.tolist(),
                0,
                0,
                SERVO_J_TIME,
                SERVO_J_LOOKAHEAD,
                SERVO_J_GAIN,
            )
        except Exception as e:
            log.warning(f"[UR5] servoJ error: {e}")
            self._last_cmd_rad = np.array(self._rtde_r.getActualQ(), dtype=np.float32)[:6]

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
    def _grab_frame(pipeline: rs.pipeline, retries: int = 3) -> np.ndarray:
        for attempt in range(retries):
            frames = pipeline.wait_for_frames(timeout_ms=500)
            color = frames.get_color_frame()
            if color:
                return np.asanyarray(color.get_data())
            log.warning(f"[Camera] No color frame, retry {attempt + 1}/{retries}")
        raise RuntimeError("Camera failed to return a frame after retries")
