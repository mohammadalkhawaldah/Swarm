"""
Phase 2 multi-drone MAVSDK agent script.

Each instance of this script controls a single ArduPilot SITL vehicle via its
own MAVSDK server. The operator launches the script in multiple terminals
with different command-line arguments (drone ID, MAVSDK address, waypoint, etc.).

Mission flow per drone:
    connect → wait for readiness → optional takeoff delay → arm → takeoff
    → fly to waypoint → loiter → land → disarm
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


BASE_LOGGER = logging.getLogger("drone_agent_phase2")


class DroneLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that prefixes every message with the drone ID."""

    def process(self, msg, kwargs):
        prefix = f"[Drone {self.extra['drone_id']}] "
        return f"{prefix}{msg}", kwargs


@dataclass
class MissionConfig:
    """Runtime configuration for a single drone."""

    drone_id: int
    mavsdk_host: str
    mavsdk_port: int
    takeoff_alt_agl: float
    target_lat: float
    target_lon: float
    target_alt_agl: float
    yaw_deg: float
    loiter_seconds: float
    connection_timeout: float
    readiness_timeout: float
    takeoff_delay: float


async def connect_and_prepare(config: MissionConfig, logger: logging.LoggerAdapter) -> tuple[System, float]:
    """
    Connect to MAVSDK server, wait for discovery, and obtain home altitude.

    Returns:
        (System instance, home altitude AMSL)
    """

    address = f"{config.mavsdk_host}:{config.mavsdk_port}"
    drone = System(mavsdk_server_address=config.mavsdk_host, port=config.mavsdk_port)
    logger.info("Connecting to MAVSDK server at %s", address)
    try:
        await asyncio.wait_for(drone.connect(), timeout=config.connection_timeout)
    except asyncio.TimeoutError as exc:
        raise RuntimeError("Timeout while connecting to MAVSDK server") from exc

    logger.info("Connection established, waiting for vehicle discovery...")
    await _wait_for_connection(drone, logger, timeout=config.readiness_timeout)
    await _wait_for_global_position(drone, logger, timeout=config.readiness_timeout)
    home_alt = await _get_home_altitude(drone, timeout=config.readiness_timeout)
    logger.info("Vehicle ready. Home altitude: %.1f m AMSL", home_alt)
    return drone, home_alt


async def run_mission(config: MissionConfig) -> None:
    """Execute the mission flow for a single drone."""

    logger = _create_logger(config.drone_id)
    drone, home_alt = await connect_and_prepare(config, logger)

    if config.takeoff_delay > 0:
        logger.info("Waiting %.1f seconds before takeoff (deconfliction delay).", config.takeoff_delay)
        await asyncio.sleep(config.takeoff_delay)

    logger.info("Setting takeoff altitude to %.1f m AGL", config.takeoff_alt_agl)
    await drone.action.set_takeoff_altitude(config.takeoff_alt_agl)

    logger.info("Arming…")
    await drone.action.arm()

    logger.info("Taking off…")
    await drone.action.takeoff()
    await _wait_until_altitude(drone, config.takeoff_alt_agl * 0.95, logger)

    target_alt_amsl = home_alt + config.target_alt_agl
    logger.info(
        "Flying to waypoint lat=%.7f lon=%.7f alt=%.1f m AMSL (%.1f m AGL)",
        config.target_lat,
        config.target_lon,
        target_alt_amsl,
        config.target_alt_agl,
    )
    await drone.action.goto_location(
        config.target_lat,
        config.target_lon,
        target_alt_amsl,
        config.yaw_deg,
    )

    await _wait_until_at_position(
        drone,
        target_lat=config.target_lat,
        target_lon=config.target_lon,
        target_alt_amsl=target_alt_amsl,
        horiz_threshold_m=5.0,
        vert_threshold_m=2.0,
        logger=logger,
    )
    logger.info("Reached target waypoint.")

    if config.loiter_seconds > 0:
        logger.info("Loitering for %.1f seconds.", config.loiter_seconds)
        await asyncio.sleep(config.loiter_seconds)

    logger.info("Initiating landing.")
    await drone.action.land()
    await _wait_until_not_in_air(drone, logger)

    logger.info("Disarming.")
    await drone.action.disarm()
    logger.info("Mission completed.")


async def _wait_for_connection(drone: System, logger: logging.LoggerAdapter, timeout: float) -> None:
    """Wait for the first connection_state event indicating a vehicle is connected."""

    async def _monitor() -> None:
        async for state in drone.core.connection_state():
            if state.is_connected:
                logger.info("Vehicle discovered.")
                return

    try:
        await asyncio.wait_for(_monitor(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise RuntimeError("Timeout waiting for vehicle discovery") from exc


async def _wait_for_global_position(drone: System, logger: logging.LoggerAdapter, timeout: float) -> None:
    """Wait until global and home positions are healthy."""

    async def _monitor() -> None:
        async for health in drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                logger.info("Global and home position reports are OK.")
                return

    try:
        await asyncio.wait_for(_monitor(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise RuntimeError("Timeout waiting for global position lock") from exc


async def _get_home_altitude(drone: System, timeout: float) -> float:
    """Obtain the home position altitude AMSL."""

    async def _monitor() -> float:
        async for home in drone.telemetry.home():
            return home.absolute_altitude_m
        raise RuntimeError("Home telemetry stream finished unexpectedly.")

    try:
        return await asyncio.wait_for(_monitor(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise RuntimeError("Timeout waiting for home position data") from exc


async def _wait_until_altitude(drone: System, target_alt: float, logger: logging.LoggerAdapter, tolerance_m: float = 1.0) -> None:
    """Wait until the drone's relative altitude is near the target."""

    async for position in drone.telemetry.position():
        if position.relative_altitude_m >= target_alt - tolerance_m:
            logger.info("Reached relative altitude %.1f m.", position.relative_altitude_m)
            return
        await asyncio.sleep(0.5)


async def _wait_until_at_position(
    drone: System,
    target_lat: float,
    target_lon: float,
    target_alt_amsl: float,
    horiz_threshold_m: float,
    vert_threshold_m: float,
    logger: logging.LoggerAdapter,
) -> None:
    """Wait until the drone is within specified horizontal and vertical thresholds of the waypoint."""

    async for position in drone.telemetry.position():
        horizontal = _horizontal_distance_m(position, target_lat, target_lon)
        vertical = abs(position.absolute_altitude_m - target_alt_amsl)
        if horizontal <= horiz_threshold_m and vertical <= vert_threshold_m:
            return
        logger.debug(
            "Approaching waypoint: horizontal %.1f m, vertical %.1f m (thresholds %.1f/%.1f).",
            horizontal,
            vertical,
            horiz_threshold_m,
            vert_threshold_m,
        )
        await asyncio.sleep(0.5)


async def _wait_until_not_in_air(drone: System, logger: logging.LoggerAdapter) -> None:
    """Wait until the telemetry reports the vehicle is no longer airborne."""

    async for in_air in drone.telemetry.in_air():
        if not in_air:
            logger.info("Confirmed on ground.")
            return
        await asyncio.sleep(0.5)


def _horizontal_distance_m(position: Position, target_lat: float, target_lon: float) -> float:
    """Approximate planar distance between current position and target lat/lon."""

    current_lat = position.latitude_deg
    current_lon = position.longitude_deg

    # Simple equirectangular approximation, sufficient for short ranges.
    lat_rad = math.radians(current_lat)
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * abs(math.cos(lat_rad))

    delta_lat = (current_lat - target_lat) * meters_per_deg_lat
    delta_lon = (current_lon - target_lon) * meters_per_deg_lon
    return math.hypot(delta_lat, delta_lon)


def _parse_address(value: str, default_port: int = 50051) -> tuple[str, int]:
    """Split MAVSDK address string into host and port."""

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


def _create_logger(drone_id: int) -> DroneLoggerAdapter:
    """Return a LoggerAdapter that prefixes messages with the drone ID."""

    return DroneLoggerAdapter(BASE_LOGGER, {"drone_id": drone_id})


def parse_args(argv: Optional[list[str]] = None) -> MissionConfig:
    """Parse command-line arguments into a MissionConfig."""

    parser = argparse.ArgumentParser(description="Phase 2 multi-drone agent using MAVSDK.")
    parser.add_argument("--drone-id", type=int, required=True, help="Identifier of this drone (1-5).")
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
        help="Target altitude AGL in meters (relative to home).",
    )
    parser.add_argument(
        "--takeoff-alt",
        type=float,
        default=30.0,
        help="Takeoff altitude AGL in meters (default: 30).",
    )
    parser.add_argument(
        "--yaw-deg",
        type=float,
        default=float("nan"),
        help="Desired yaw angle when reaching the waypoint (NaN keeps current yaw).",
    )
    parser.add_argument(
        "--loiter-seconds",
        type=float,
        default=15.0,
        help="Loiter duration at waypoint before landing (default: 15 seconds).",
    )
    parser.add_argument(
        "--takeoff-delay",
        type=float,
        default=None,
        help="Optional delay before takeoff in seconds. Defaults to drone_id * 2.",
    )
    parser.add_argument(
        "--connection-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait while connecting to MAVSDK server.",
    )
    parser.add_argument(
        "--readiness-timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for vehicle discovery and GPS readiness.",
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
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    takeoff_delay = args.takeoff_delay if args.takeoff_delay is not None else max(0.0, args.drone_id * 2.0)

    return MissionConfig(
        drone_id=args.drone_id,
        mavsdk_host=mavsdk_host,
        mavsdk_port=mavsdk_port,
        takeoff_alt_agl=args.takeoff_alt,
        target_lat=args.target_lat,
        target_lon=args.target_lon,
        target_alt_agl=args.target_alt,
        yaw_deg=args.yaw_deg,
        loiter_seconds=args.loiter_seconds,
        connection_timeout=args.connection_timeout,
        readiness_timeout=args.readiness_timeout,
        takeoff_delay=takeoff_delay,
    )


async def async_main(argv: Optional[list[str]] = None) -> None:
    """Async entrypoint with error handling."""

    config = parse_args(argv)
    logger = _create_logger(config.drone_id)
    try:
        await run_mission(config)
    except Exception:
        logger.exception("Mission failed due to an unexpected error.")
        raise


def main() -> None:
    """Synchronous entrypoint to run the asyncio program."""

    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        BASE_LOGGER.warning("Mission interrupted by user.")


if __name__ == "__main__":
    main()

# Example usage (assuming SITL + MAVSDK servers described in the docs are running):
#
# Drone 1 (UAV1):
# python drone_agent_phase2.py --drone-id 1 --mavsdk-address 127.0.0.1:50040 --target-lat <LAT> --target-lon <LON> --target-alt 30
#
# Drone 2 (UAV2):
# python drone_agent_phase2.py --drone-id 2 --mavsdk-address 127.0.0.1:50041 --target-lat <LAT> --target-lon <LON> --target-alt 30
#
# Drone 3 (UAV3):
# python drone_agent_phase2.py --drone-id 3 --mavsdk-address 127.0.0.1:50042 --target-lat <LAT> --target-lon <LON> --target-alt 30
#
# Drone 4 (UAV4):
# python drone_agent_phase2.py --drone-id 4 --mavsdk-address 127.0.0.1:50043 --target-lat <LAT> --target-lon <LON> --target-alt 30
#
# Drone 5 (UAV5):
# python drone_agent_phase2.py --drone-id 5 --mavsdk-address 127.0.0.1:50044 --target-lat <LAT> --target-lon <LON> --target-alt 30
