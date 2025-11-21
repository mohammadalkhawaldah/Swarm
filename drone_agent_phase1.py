"""
Single-drone MAVSDK control script for Phase 1 of the decentralized swarm project.

This module connects to a MAVSDK server, waits for readiness, arms, takes off,
flies to a target waypoint, optionally loiters, lands, and disarms.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
from dataclasses import dataclass
from typing import Optional

from mavsdk import System
from mavsdk.telemetry import Position


LOGGER = logging.getLogger("drone_agent_phase1")


@dataclass
class MissionConfig:
    """Configuration parameters for a single run of the mission."""

    mavsdk_host: str
    mavsdk_port: int
    target_lat: float
    target_lon: float
    target_alt: float
    takeoff_alt: float
    yaw_deg: float
    loiter_seconds: float
    connection_timeout: float
    readiness_timeout: float


async def connect_and_prepare(config: MissionConfig) -> System:
    """
    Create a MAVSDK System, connect to the server, and wait for readiness.

    Args:
        config: Mission configuration.

    Returns:
        A connected MAVSDK System instance ready for mission execution.

    Raises:
        RuntimeError: If connection or readiness checks fail.
    """

    address = f"{config.mavsdk_host}:{config.mavsdk_port}"
    drone = System(mavsdk_server_address=config.mavsdk_host, port=config.mavsdk_port)
    LOGGER.info("Connecting to MAVSDK server at %s", address)
    try:
        await asyncio.wait_for(drone.connect(), timeout=config.connection_timeout)
    except asyncio.TimeoutError as exc:
        raise RuntimeError("Timeout while connecting to MAVSDK server") from exc

    LOGGER.info("MAVSDK server connection established, waiting for vehicle discovery...")
    await _wait_for_connection(drone, timeout=config.readiness_timeout)
    await _wait_for_global_position(drone, timeout=config.readiness_timeout)
    LOGGER.info("Vehicle discovered and global position is valid.")

    return drone


async def _wait_for_connection(drone: System, timeout: float) -> None:
    """Block until a vehicle reports it is connected or raise on timeout."""

    async def _monitor() -> None:
        async for state in drone.core.connection_state():
            if state.is_connected:
                LOGGER.info("Vehicle discovered.")
                return

    try:
        await asyncio.wait_for(_monitor(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise RuntimeError("Timeout waiting for vehicle discovery") from exc


async def _wait_for_global_position(drone: System, timeout: float) -> None:
    """Wait until the autopilot reports a valid global position."""

    async def _monitor() -> None:
        async for health in drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                LOGGER.info("Global position and home position reports are OK.")
                return

    try:
        await asyncio.wait_for(_monitor(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise RuntimeError("Timeout waiting for global position lock") from exc


async def run_mission(config: MissionConfig) -> None:
    """
    Execute the single-drone mission according to the provided configuration.

    Args:
        config: Mission configuration parameters.
    """

    drone = await connect_and_prepare(config)

    LOGGER.info("Setting takeoff altitude to %.1f m", config.takeoff_alt)
    await drone.action.set_takeoff_altitude(config.takeoff_alt)

    LOGGER.info("Arming...")
    await drone.action.arm()

    LOGGER.info("Taking off...")
    await drone.action.takeoff()
    await _wait_until_altitude(drone, config.takeoff_alt * 0.95)

    LOGGER.info(
        "Flying to waypoint lat=%.7f lon=%.7f alt=%.1f m",
        config.target_lat,
        config.target_lon,
        config.target_alt,
    )
    await drone.action.goto_location(
        config.target_lat,
        config.target_lon,
        config.target_alt,
        config.yaw_deg,
    )

    await _wait_until_at_position(
        drone,
        target_lat=config.target_lat,
        target_lon=config.target_lon,
        target_alt=config.target_alt,
        horiz_threshold_m=5.0,
        vert_threshold_m=2.0,
    )
    LOGGER.info("Reached target waypoint.")

    if config.loiter_seconds > 0:
        LOGGER.info(
            "Loitering for %.1f seconds at target waypoint before landing.",
            config.loiter_seconds,
        )
        await asyncio.sleep(config.loiter_seconds)

    LOGGER.info("Initiating landing.")
    await drone.action.land()

    await _wait_until_not_in_air(drone)
    LOGGER.info("Landed successfully, disarming.")
    await drone.action.disarm()

    LOGGER.info("Mission completed.")


async def _wait_until_altitude(drone: System, target_alt: float, tolerance_m: float = 1.0) -> None:
    """Wait until the drone's relative altitude is above the target."""

    async for position in drone.telemetry.position():
        if position.relative_altitude_m >= target_alt - tolerance_m:
            LOGGER.info(
                "Reached target relative altitude (current: %.1f m)",
                position.relative_altitude_m,
            )
            return


async def _wait_until_at_position(
    drone: System,
    target_lat: float,
    target_lon: float,
    target_alt: float,
    horiz_threshold_m: float,
    vert_threshold_m: float,
) -> None:
    """Wait until the drone is within specified thresholds of a target position."""

    async for position in drone.telemetry.position():
        if _horizontal_distance_m(position, target_lat, target_lon) <= horiz_threshold_m and abs(
            position.absolute_altitude_m - target_alt
        ) <= vert_threshold_m:
            return
        await asyncio.sleep(0.5)


async def _wait_until_not_in_air(drone: System) -> None:
    """Block until the telemetry reports the drone is no longer airborne."""

    async for in_air in drone.telemetry.in_air():
        if not in_air:
            return
        await asyncio.sleep(0.5)


def _horizontal_distance_m(position: Position, target_lat: float, target_lon: float) -> float:
    """Compute the planar distance between the current position and a target lat/lon."""

    current_lat = position.latitude_deg
    current_lon = position.longitude_deg

    # Use a simple equirectangular approximation, sufficient for small distances.
    lat_rad = _deg_to_rad(current_lat)
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * abs(math.cos(lat_rad))

    delta_lat = (current_lat - target_lat) * meters_per_deg_lat
    delta_lon = (current_lon - target_lon) * meters_per_deg_lon
    return (delta_lat**2 + delta_lon**2) ** 0.5


def _deg_to_rad(value: float) -> float:
    """Convert degrees to radians."""

    return value * (math.pi / 180.0)


def _parse_address(value: str, default_port: int = 50051) -> tuple[str, int]:
    """Split MAVSDK address string into host and port components."""

    if ":" not in value:
        return value, default_port

    host, _, port_str = value.rpartition(":")
    if not host:
        raise ValueError(f"Invalid MAVSDK address '{value}'. Expected format host:port.")
    try:
        port = int(port_str)
    except ValueError as exc:
        raise ValueError(f"Invalid port in MAVSDK address '{value}'.") from exc
    return host, port


def parse_args(argv: Optional[list[str]] = None) -> MissionConfig:
    """Parse command-line arguments into a MissionConfig object."""

    parser = argparse.ArgumentParser(description="Phase 1 drone agent using MAVSDK.")
    parser.add_argument(
        "--mavsdk-address",
        default="127.0.0.1:50051",
        help="Address of the MAVSDK server in host:port format.",
    )
    parser.add_argument("--target-lat", type=float, required=True, help="Target latitude in degrees.")
    parser.add_argument("--target-lon", type=float, required=True, help="Target longitude in degrees.")
    parser.add_argument(
        "--target-alt",
        type=float,
        required=True,
        help="Target altitude AMSL in meters (used for goto).",
    )
    parser.add_argument(
        "--takeoff-alt",
        type=float,
        default=30.0,
        help="Desired takeoff altitude AGL in meters (default: 30).",
    )
    parser.add_argument(
        "--yaw-deg",
        type=float,
        default=float("nan"),
        help="Yaw angle at waypoint in degrees. NaN keeps current yaw.",
    )
    parser.add_argument(
        "--loiter-seconds",
        type=float,
        default=10.0,
        help="Time to loiter at the waypoint before landing. Set to 0 to land immediately.",
    )
    parser.add_argument(
        "--connection-timeout",
        type=float,
        default=30.0,
        help="Max seconds to wait while connecting to MAVSDK server.",
    )
    parser.add_argument(
        "--readiness-timeout",
        type=float,
        default=60.0,
        help="Max seconds to wait for vehicle discovery and global position lock.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )

    args = parser.parse_args(argv)

    try:
        mavsdk_host, mavsdk_port = _parse_address(args.mavsdk_address)
    except ValueError as exc:
        parser.error(str(exc))

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    return MissionConfig(
        mavsdk_host=mavsdk_host,
        mavsdk_port=mavsdk_port,
        target_lat=args.target_lat,
        target_lon=args.target_lon,
        target_alt=args.target_alt,
        takeoff_alt=args.takeoff_alt,
        yaw_deg=args.yaw_deg,
        loiter_seconds=args.loiter_seconds,
        connection_timeout=args.connection_timeout,
        readiness_timeout=args.readiness_timeout,
    )


async def async_main(argv: Optional[list[str]] = None) -> None:
    """Async entrypoint with error handling."""

    config = parse_args(argv)
    try:
        await run_mission(config)
    except Exception:
        LOGGER.exception("Mission failed due to an unexpected error.")
        raise


def main() -> None:
    """Synchronous entrypoint wrapper for asyncio."""

    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        LOGGER.warning("Mission interrupted by user.")


if __name__ == "__main__":
    main()

# Example usage:
# python drone_agent_phase1.py --mavsdk-address 127.0.0.1:50040 --target-lat 47.3977 --target-lon 8.5456 --target-alt 35
