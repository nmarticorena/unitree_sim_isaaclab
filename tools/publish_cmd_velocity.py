#!/usr/bin/env python3
"""Publish whole-body velocity commands for the simulator DDS bridge."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from dataclasses import dataclass


DEFAULT_TOPIC = "rt/run_command/cmd"
DEFAULT_HEIGHT = 0.8


@dataclass(frozen=True)
class CmdVelocity:
    x: float
    y: float
    yaw: float
    height: float

    def as_sim_command(self) -> list[float]:
        """Return the payload consumed by DDSRLActionProvider."""
        return [self.x, self.y, self.yaw, self.height]


def _bounded(value: float, lower: float, upper: float, name: str) -> float:
    if value < lower or value > upper:
        raise argparse.ArgumentTypeError(f"{name} must be in [{lower}, {upper}], got {value}")
    return value


def _positive(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Publish [x_vel, y_vel, yaw_vel, height] to the whole-body simulator command topic. "
            "Only Wholebody tasks consume this command."
        )
    )
    parser.add_argument("--x", type=float, default=0.0, help="Forward velocity in m/s.")
    parser.add_argument("--y", type=float, default=0.0, help="Lateral velocity in m/s.")
    parser.add_argument("--yaw", type=float, default=0.0, help="Yaw velocity in rad/s.")
    parser.add_argument("--height", type=float, default=DEFAULT_HEIGHT, help="Commanded stand height.")
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Seconds to publish for. Use 0 for one message or a negative value to publish until Ctrl+C.",
    )
    parser.add_argument("--rate", type=_positive, default=50.0, help="Publish rate in Hz.")
    parser.add_argument("--domain-id", type=int, default=1, help="DDS domain id.")
    parser.add_argument(
        "--network-interface",
        default=None,
        help="Optional DDS network interface, for example eth0. Omit for auto selection.",
    )
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help="DDS topic to publish.")
    parser.add_argument(
        "--no-stop-on-exit",
        action="store_true",
        help="Do not send zero velocity before exiting.",
    )
    return parser.parse_args()


def validate_command(args: argparse.Namespace) -> CmdVelocity:
    return CmdVelocity(
        x=_bounded(args.x, -0.6, 1.0, "x"),
        y=_bounded(args.y, -0.5, 0.5, "y"),
        yaw=_bounded(args.yaw, -1.57, 1.57, "yaw"),
        height=_bounded(args.height, 0.0, 1.0, "height"),
    )


def init_publisher(domain_id: int, network_interface: str | None, topic: str):
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
    from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

    if network_interface:
        ChannelFactoryInitialize(domain_id, network_interface)
    else:
        ChannelFactoryInitialize(domain_id)

    publisher = ChannelPublisher(topic, String_)
    publisher.Init()
    return publisher, String_


def publish_command(publisher, string_type, command: CmdVelocity, timeout: float = 0.1) -> bool:
    message = string_type(data=str(command.as_sim_command()))
    return bool(publisher.Write(message, timeout))


def main() -> int:
    args = parse_args()
    try:
        command = validate_command(args)
    except argparse.ArgumentTypeError as exc:
        raise SystemExit(f"error: {exc}") from exc
    stop_command = CmdVelocity(0.0, 0.0, 0.0, command.height)
    publisher, string_type = init_publisher(args.domain_id, args.network_interface, args.topic)

    interrupted = False

    def _handle_signal(_signum, _frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    period = 1.0 / args.rate
    deadline = None if args.duration < 0 else time.monotonic() + args.duration

    print(
        f"Publishing cmd_velocity {command.as_sim_command()} to {args.topic} "
        f"at {args.rate:g} Hz on DDS domain {args.domain_id}"
    )
    if args.duration < 0:
        print("Publishing until Ctrl+C")

    count = 0
    try:
        if args.duration == 0:
            publish_command(publisher, string_type, command)
            count += 1
        else:
            while not interrupted and (deadline is None or time.monotonic() < deadline):
                publish_command(publisher, string_type, command)
                count += 1
                time.sleep(period)
    finally:
        if not args.no_stop_on_exit:
            for _ in range(5):
                publish_command(publisher, string_type, stop_command)
                time.sleep(period)
        publisher.Close()

    print(f"Published {count} command messages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
