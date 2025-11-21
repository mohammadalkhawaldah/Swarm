"""
Phase 4 mesh-enabled MAVSDK agent with V-shape formation.

Each instance of this script controls one SITL drone via MAVSDK while also
participating in a lightweight UDP-based swarm data exchange. The flight plan
matches Phase 3 (leader + V-shaped followers), but now each drone broadcasts
its state and logs a periodic swarm snapshot.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from mavsdk import System
from mavsdk.telemetry import Position


BASE_LOGGER = logging.getLogger("drone_agent_phase4_mesh")
SWARM_BASE_PORT = 61000


class DroneLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that prefixes every message with the drone ID."""

    def process(self, msg, kwargs):
        prefix = f"[Drone {self.extra['drone_id']}] "
        return f"{prefix}{msg}", kwargs


@dataclass
class MissionConfig:
    """Runtime configuration for a single drone agent."""

    drone_id: int
    swarm_size: int
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


@dataclass
class HomePosition:
    """Home position tuple with latitude, longitude, and altitude AMSL."""

    latitude: float
    longitude: float
    altitude_m: float


@dataclass
class DroneState:
    """State broadcast structure for each drone in the swarm."""

    drone_id: int
    lat: float
    lon: float
    alt_amsl: float
    heading_deg: float
    timestamp: float
    received_time: float


class SwarmReceiverProtocol(asyncio.DatagramProtocol):
    """Datagram protocol that forwards received packets to SwarmComm."""

    def __init__(self, handler):
        self._handler = handler

    def datagram_received(self, data, addr):
        self._handler(data, addr)


class SwarmComm:
    """Simple UDP-based swarm communication helper."""

    def __init__(
        self,
        drone_id: int,
        swarm_size: int,
        logger: DroneLoggerAdapter,
        broadcast_interval: float = 0.5,
        summary_interval: float = 5.0,
    ):
        self.drone_id = drone_id
        self.swarm_size = swarm_size
        self.logger = logger
        self.broadcast_interval = broadcast_interval
        self.summary_interval = summary_interval
        self.swarm_state: Dict[int, DroneState] = {}
        self._loop = asyncio.get_running_loop()
        self._listen_transport: Optional[asyncio.DatagramTransport] = None
        self._broadcast_task: Optional[asyncio.Task] = None
        self._summary_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._current_state: Optional[DroneState] = None

    async def start(self) -> None:
        """Create the UDP listener and spawn broadcaster/summary tasks."""

        listen_port = SWARM_BASE_PORT + self.drone_id
        self.logger.info("Starting swarm listener on UDP port %d.", listen_port)
        self._listen_transport, _ = await self._loop.create_datagram_endpoint(
            lambda: SwarmReceiverProtocol(self._handle_datagram),
            local_addr=("127.0.0.1", listen_port),
        )

        self._broadcast_task = asyncio.create_task(self._broadcast_loop())
        self._summary_task = asyncio.create_task(self._summary_loop())

    async def stop(self) -> None:
        """Stop all communication tasks and close sockets."""

        self.logger.info("Stopping swarm communication tasks.")
        self._stop_event.set()

        for task in (self._broadcast_task, self._summary_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        if self._listen_transport:
            self._listen_transport.close()
            self._listen_transport = None

    def update_local_state(self, state: DroneState) -> None:
        """Update the latest state for this drone (used for broadcasting)."""

        self._current_state = state
        self.swarm_state[self.drone_id] = state

    def get_swarm_snapshot(self) -> Dict[int, DroneState]:
        """Return a shallow copy of the swarm state dictionary."""

        return dict(self.swarm_state)

    def _handle_datagram(self, data: bytes, addr: Tuple[str, int]) -> None:
        """Parse incoming JSON packets and update swarm_state."""

        try:
            payload = json.loads(data.decode("utf-8"))
            sender_id = int(payload["drone_id"])
            if sender_id == self.drone_id:
                return
            state = DroneState(
                drone_id=sender_id,
                lat=float(payload["lat"]),
                lon=float(payload["lon"]),
                alt_amsl=float(payload["alt_amsl"]),
                heading_deg=float(payload["heading_deg"]),
                timestamp=float(payload["timestamp"]),
                received_time=time.time(),
            )
            self.swarm_state[sender_id] = state
        except Exception as exc:
            self.logger.debug("Failed to parse swarm packet from %s: %s", addr, exc)

    async def _broadcast_loop(self) -> None:
        """Periodically send the current state to all peers."""

        try:
            while not self._stop_event.is_set():
                state = self._current_state
                if state and self._listen_transport:
                    message = json.dumps(
                        {
                            "drone_id": state.drone_id,
                            "lat": state.lat,
                            "lon": state.lon,
                            "alt_amsl": state.alt_amsl,
                            "heading_deg": state.heading_deg,
                            "timestamp": state.timestamp,
                        }
                    ).encode("utf-8")
                    for peer_id in range(1, self.swarm_size + 1):
                        if peer_id == self.drone_id:
                            continue
                        peer_port = SWARM_BASE_PORT + peer_id
                        self._listen_transport.sendto(message, ("127.0.0.1", peer_port))
                await asyncio.sleep(self.broadcast_interval)
        except asyncio.CancelledError:
            pass

    async def _summary_loop(self) -> None:
        """Log swarm snapshots at the configured interval."""

        try:
            while not self._stop_event.is_set():
                await asyncio.sleep(self.summary_interval)
                self._log_summary()
        except asyncio.CancelledError:
            pass

    def _log_summary(self) -> None:
        """Format and log the known swarm state."""

        snapshot = self.get_swarm_snapshot()
        now = time.time()
        if not snapshot:
            self.logger.info("Swarm state: no peers heard yet.")
            return

        summary_lines = ["Swarm state snapshot:"]
        for drone_id in sorted(snapshot):
            state = snapshot[drone_id]
            age = now - state.received_time
            summary_lines.append(
                f"  - Drone {drone_id}: last seen {age:.1f}s ago at "
                f"lat={state.lat:.6f}, lon={state.lon:.6f}, alt={state.alt_amsl:.1f}m, heading={state.heading_deg:.1f}"
            )
        self.logger.info("\n".join(summary_lines))


async def connect_and_prepare(config: MissionConfig, logger: DroneLoggerAdapter) -> tuple[System, HomePosition]:
    """
    Connect to MAVSDK server, wait for readiness, and fetch the home position.
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
    home_pos = await _get_home_position(drone, timeout=config.readiness_timeout)
    logger.info(
        "Vehicle ready. Home lat=%.7f lon=%.7f alt=%.1f m AMSL",
        home_pos.latitude,
        home_pos.longitude,
        home_pos.altitude_m,
    )
    return drone, home_pos


async def start_local_state_updates(drone: System, comm: SwarmComm, logger: DroneLoggerAdapter) -> list[asyncio.Task]:
    """
    Spawn telemetry listeners that keep the latest local state updated for broadcasting.
    """

    tasks: list[asyncio.Task] = []
    heading_holder = {"value": 0.0}

    async def heading_worker():
        try:
            async for heading in drone.telemetry.heading():
                heading_holder["value"] = heading.heading_deg
        except asyncio.CancelledError:
            pass

    async def position_worker():
        try:
            async for position in drone.telemetry.position():
                state = DroneState(
                    drone_id=comm.drone_id,
                    lat=position.latitude_deg,
                    lon=position.longitude_deg,
                    alt_amsl=position.absolute_altitude_m,
                    heading_deg=heading_holder["value"],
                    timestamp=time.time(),
                    received_time=time.time(),
                )
                comm.update_local_state(state)
        except asyncio.CancelledError:
            pass

    tasks.append(asyncio.create_task(heading_worker()))
    tasks.append(asyncio.create_task(position_worker()))
    return tasks


async def run_mission(config: MissionConfig) -> None:
    """Execute the V-formation mission with swarm communication."""

    logger = _create_logger(config.drone_id)
    drone, home_pos = await connect_and_prepare(config, logger)

    comm = SwarmComm(config.drone_id, config.swarm_size, logger)
    await comm.start()
    telem_tasks = await start_local_state_updates(drone, comm, logger)

    try:
        if config.takeoff_delay > 0:
            logger.info("Waiting %.1f seconds before takeoff (deconfliction delay).", config.takeoff_delay)
            await asyncio.sleep(config.takeoff_delay)

        logger.info("Setting takeoff altitude to %.1f m AGL.", config.takeoff_alt_agl)
        await drone.action.set_takeoff_altitude(config.takeoff_alt_agl)

        logger.info("Arming…")
        await drone.action.arm()

        logger.info("Taking off…")
        await drone.action.takeoff()
        await _wait_until_altitude(drone, config.takeoff_alt_agl * 0.95, logger)

        forward_offset, lateral_offset = compute_v_formation_offset(config.drone_id)
        assigned_lat, assigned_lon = apply_offset_to_target(
            home_lat=home_pos.latitude,
            home_lon=home_pos.longitude,
            target_lat=config.target_lat,
            target_lon=config.target_lon,
            forward_offset_m=forward_offset,
            lateral_offset_m=lateral_offset,
        )

        target_alt_amsl = home_pos.altitude_m + config.target_alt_agl
        logger.info(
            "Nominal target lat=%.7f lon=%.7f; assigned V-waypoint lat=%.7f lon=%.7f; forward %.1f m, lateral %.1f m.",
            config.target_lat,
            config.target_lon,
            assigned_lat,
            assigned_lon,
            forward_offset,
            lateral_offset,
        )

        await drone.action.goto_location(assigned_lat, assigned_lon, target_alt_amsl, config.yaw_deg)

        await _wait_until_at_position(
            drone,
            target_lat=assigned_lat,
            target_lon=assigned_lon,
            target_alt_amsl=target_alt_amsl,
            horiz_threshold_m=5.0,
            vert_threshold_m=2.0,
            logger=logger,
        )
        logger.info("Reached assigned V-waypoint.")

        if config.loiter_seconds > 0:
            logger.info("Loitering for %.1f seconds.", config.loiter_seconds)
            await asyncio.sleep(config.loiter_seconds)

        logger.info("Initiating landing.")
        await drone.action.land()
        await _wait_until_not_in_air(drone, logger)

        logger.info("Disarming.")
        await drone.action.disarm()
        logger.info("Mission completed.")
        logger.info("Final swarm snapshot: %s", comm.get_swarm_snapshot())
    finally:
        for task in telem_tasks:
            task.cancel()
        if telem_tasks:
            await asyncio.gather(*telem_tasks, return_exceptions=True)
        await comm.stop()


def compute_v_formation_offset(drone_id: int) -> Tuple[float, float]:
    """Return (forward_m, lateral_m) offsets for each drone ID."""

    spacing = 30.0  # meters
    mapping = {
        1: (0.0, 0.0),
        2: (-spacing, -spacing),
        3: (-2 * spacing, -2 * spacing),
        4: (-spacing, spacing),
        5: (-2 * spacing, 2 * spacing),
    }
    try:
        return mapping[drone_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported drone_id '{drone_id}' for formation offsets.") from exc


def apply_offset_to_target(
    home_lat: float,
    home_lon: float,
    target_lat: float,
    target_lon: float,
    forward_offset_m: float,
    lateral_offset_m: float,
) -> Tuple[float, float]:
    """Convert formation offsets into latitude/longitude adjustments."""

    meters_per_deg_lat = 111_320.0
    lat_rad = math.radians(home_lat)
    meters_per_deg_lon = 111_320.0 * math.cos(lat_rad)
    if abs(meters_per_deg_lon) < 1e-6:
        meters_per_deg_lon = 1e-6

    north_to_target = (target_lat - home_lat) * meters_per_deg_lat
    east_to_target = (target_lon - home_lon) * meters_per_deg_lon
    heading = math.atan2(east_to_target, north_to_target) if (north_to_target or east_to_target) else 0.0

    forward_north = math.cos(heading)
    forward_east = math.sin(heading)
    right_north = -math.sin(heading)
    right_east = math.cos(heading)

    offset_north = forward_offset_m * forward_north + lateral_offset_m * right_north
    offset_east = forward_offset_m * forward_east + lateral_offset_m * right_east

    delta_lat = offset_north / meters_per_deg_lat
    delta_lon = offset_east / meters_per_deg_lon

    return target_lat + delta_lat, target_lon + delta_lon


async def _wait_for_connection(drone: System, logger: DroneLoggerAdapter, timeout: float) -> None:
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


async def _wait_for_global_position(drone: System, logger: DroneLoggerAdapter, timeout: float) -> None:
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


async def _get_home_position(drone: System, timeout: float) -> HomePosition:
    """Obtain the home position (latitude, longitude, altitude AMSL)."""

    async def _monitor() -> HomePosition:
        async for home in drone.telemetry.home():
            return HomePosition(
                latitude=home.latitude_deg,
                longitude=home.longitude_deg,
                altitude_m=home.absolute_altitude_m,
            )
        raise RuntimeError("Home telemetry stream finished unexpectedly.")

    try:
        return await asyncio.wait_for(_monitor(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise RuntimeError("Timeout waiting for home position data") from exc


async def _wait_until_altitude(
    drone: System,
    target_alt: float,
    logger: DroneLoggerAdapter,
    tolerance_m: float = 1.0,
) -> None:
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
    logger: DroneLoggerAdapter,
) -> None:
    """Wait until the drone is within specified thresholds of the waypoint."""

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


async def _wait_until_not_in_air(drone: System, logger: DroneLoggerAdapter) -> None:
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

    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(current_lat))
    if abs(meters_per_deg_lon) < 1e-6:
        meters_per_deg_lon = 1e-6

    delta_lat = (current_lat - target_lat) * meters_per_deg_lat
    delta_lon = (current_lon - target_lon) * meters_per_deg_lon
    return math.hypot(delta_lat, delta_lon)


def _parse_address(value: str, default_port: int = 50051) -> tuple[str, int]:
    """Split MAVSDK address string into (host, port)."""

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
    """Return a logger adapter that prefixes log lines with the drone ID."""

    return DroneLoggerAdapter(BASE_LOGGER, {"drone_id": drone_id})


def parse_args(argv: Optional[list[str]] = None) -> MissionConfig:
    """Parse command-line arguments into a MissionConfig."""

    parser = argparse.ArgumentParser(description="Phase 4 mesh-enabled drone agent using MAVSDK.")
    parser.add_argument("--drone-id", type=int, required=True, help="Identifier of this drone (1-5).")
    parser.add_argument(
        "--swarm-size",
        type=int,
        default=5,
        help="Total number of drones participating in the swarm (default: 5).",
    )
    parser.add_argument(
        "--mavsdk-address",
        default="127.0.0.1:50051",
        help="Address of the MAVSDK server in host:port format.",
    )
    parser.add_argument("--target-lat", type=float, required=True, help="Nominal target latitude in degrees.")
    parser.add_argument("--target-lon", type=float, required=True, help="Nominal target longitude in degrees.")
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
        swarm_size=args.swarm_size,
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
    """Synchronous entrypoint for running the asyncio workflow."""

    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        BASE_LOGGER.warning("Mission interrupted by user.")


if __name__ == "__main__":
    main()

# Example usage (after starting all SITL + MAVSDK instances described above):
#
# Drone 1 (leader):
# python drone_agent_phase4_mesh.py --drone-id 1 --mavsdk-address 127.0.0.1:50040 --target-lat <LAT> --target-lon <LON> --target-alt 30
#
# Drone 2:
# python drone_agent_phase4_mesh.py --drone-id 2 --mavsdk-address 127.0.0.1:50041 --target-lat <LAT> --target-lon <LON> --target-alt 30
#
# Drone 3:
# python drone_agent_phase4_mesh.py --drone-id 3 --mavsdk-address 127.0.0.1:50042 --target-lat <LAT> --target-lon <LON> --target-alt 30
#
# Drone 4:
# python drone_agent_phase4_mesh.py --drone-id 4 --mavsdk-address 127.0.0.1:50043 --target-lat <LAT> --target-lon <LON> --target-alt 30
#
# Drone 5:
# python drone_agent_phase4_mesh.py --drone-id 5 --mavsdk-address 127.0.0.1:50044 --target-lat <LAT> --target-lon <LON> --target-alt 30
