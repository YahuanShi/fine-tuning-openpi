"""
UR5 inference entry point.

Starts the Pi0/Pi0.5 policy WebSocket client and runs it on the real UR5 robot.

Prerequisites:
    1. Start the policy server on the training machine:
           uv run scripts/serve_policy.py --env-name pi0_ur5
       (or pi05_ur5 / pi0_ur5_lora depending on which checkpoint you trained)

    2. Ensure all hardware is connected:
           UR5 @ 10.0.0.1 (Ethernet)
           Weiss CRG gripper @ /dev/ttyACM0
           RealSense D415 exterior (serial 105422061000)
           RealSense D405 wrist    (serial 352122273671)

Usage:
    uv run examples/ur5/main.py --prompt "pick up the cube"
    uv run examples/ur5/main.py --host 192.168.1.X --prompt "place the block on the plate"
"""

import dataclasses
import logging

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _ws_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

from examples.ur5 import env as _env


@dataclasses.dataclass
class Args:
    # Policy server address
    host: str = "0.0.0.0"
    port: int = 8000

    # Task prompt sent to the model at every step
    prompt: str = "perform the task"

    # How many actions from each chunk to execute before re-querying the policy.
    # Lower = more responsive but more inference calls. Pi0 action_horizon = 50.
    action_horizon: int = 10

    # Episode settings
    num_episodes: int = 100
    max_episode_steps: int = 400  # ~40 s at 10 Hz

    # Robot control rate (Hz). Must be ≤ your policy server throughput.
    control_hz: float = 10.0


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)

    # ── Connect to policy server ─────────────────────────────────────────────
    log = logging.getLogger(__name__)
    log.info(f"Connecting to policy server at {args.host}:{args.port} ...")
    ws_policy = _ws_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    log.info(f"Server metadata: {ws_policy.get_server_metadata()}")

    # ── Build robot environment ──────────────────────────────────────────────
    environment = _env.UR5Environment(
        prompt=args.prompt,
        control_hz=args.control_hz,
    )

    # ── Assemble runtime ─────────────────────────────────────────────────────
    runtime = _runtime.Runtime(
        environment=environment,
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=ws_policy,
                action_horizon=args.action_horizon,
            )
        ),
        subscribers=[],
        max_hz=args.control_hz,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    try:
        runtime.run()
    finally:
        environment.close()


if __name__ == "__main__":
    tyro.cli(main)
