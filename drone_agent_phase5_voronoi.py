"""
Phase 5 Voronoi-enabled MAVSDK agent with mesh communication.

Each instance of this script controls a single SITL drone. After forming a
V-shape, the swarm exchanges positions over a lightweight mesh. Once the drones
reach their staging waypoints, each computes a bounded Voronoi partition around
the mission target and executes one of multiple search patterns within its cell.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import socket
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from mavsdk import System
from mavsdk.telemetry import Position
from scipy.spatial import Voronoi
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union


BASE_LOGGER = logging.getLogger("drone_agent_phase5_voronoi")


class DroneLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that prefixes messages with the drone ID."""

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
    initial_loiter_seconds: float
    loiter_after_pattern: float
    voronoi_loiter_seconds: float
    swarm_broadcast_port_base: int
    swarm_age_threshold: float
    voronoi_wait_seconds: float
    search_radius: float
    search_mode: int
    coverage_spacing: float
    spiral_step: float
    spiral_theta_step: float
    coverage_batch_size: int
    coverage_report_interval: float
    connection_timeout: float
    readiness_timeout: float
    takeoff_delay: float
    vis_collector_host: str
    vis_collector_port: int
    cell_size: float
    grid_origin: Tuple[float, float] = (0.0, 0.0)
    coordinator_port: int = 61000
    coordinator_host: str = "127.0.0.1"
    cruise_speed: float = 15.0


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
    x_local: float = 0.0
    y_local: float = 0.0


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
        base_port: int,
        logger: DroneLoggerAdapter,
        broadcast_interval: float = 0.5,
        summary_interval: float = 5.0,
        assignment_callback: Optional[callable] = None,
        coordinator_host: str = "127.0.0.1",
        coordinator_port: int = 61000,
    ):
        self.drone_id = drone_id
        self.swarm_size = swarm_size
        self.base_port = base_port
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
        self._assignment_callback = assignment_callback
        self._coord_host = coordinator_host
        self._coord_port = coordinator_port

    async def start(self) -> None:
        """Create the UDP listener and spawn broadcaster/summary tasks."""

        listen_port = self.base_port + self.drone_id
        self.logger.info("Starting swarm listener on UDP port %d.", listen_port)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except OSError:
                pass
        sock.bind(("127.0.0.1", listen_port))
        self._listen_transport, _ = await self._loop.create_datagram_endpoint(
            lambda: SwarmReceiverProtocol(self._handle_datagram),
            sock=sock,
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

    def send_coverage_update(self, plan_id: int, cells: List[Tuple[int, int]]) -> None:
        """Send a coverage_update packet directly to the coordinator."""

        if not cells or self._listen_transport is None or not self._coord_port:
            return
        payload = {
            "msg_type": "coverage_update",
            "drone_id": self.drone_id,
            "plan_id": plan_id,
            "cells": [[int(ix), int(iy)] for ix, iy in cells],
            "timestamp": time.time(),
        }
        message = json.dumps(payload).encode("utf-8")
        self._listen_transport.sendto(message, (self._coord_host, self._coord_port))

    def get_swarm_snapshot(self) -> Dict[int, DroneState]:
        """Return a shallow copy of the swarm state dictionary."""

        return dict(self.swarm_state)

    def _handle_datagram(self, data: bytes, addr: Tuple[str, int]) -> None:
        """Parse incoming JSON packets and update swarm_state."""

        try:
            payload = json.loads(data.decode("utf-8"))
            msg_type = payload.get("msg_type") or payload.get("type") or "state"
            if msg_type == "state":
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
                    x_local=float(payload.get("x_local", 0.0)),
                    y_local=float(payload.get("y_local", 0.0)),
                )
                self.swarm_state[sender_id] = state
            elif msg_type == "region_assignment" and self._assignment_callback:
                self._assignment_callback(payload)
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
                            "msg_type": "state",
                            "drone_id": state.drone_id,
                            "lat": state.lat,
                            "lon": state.lon,
                            "alt_amsl": state.alt_amsl,
                            "heading_deg": state.heading_deg,
                            "timestamp": state.timestamp,
                            "x_local": state.x_local,
                            "y_local": state.y_local,
                        }
                    ).encode("utf-8")
                    for peer_id in range(1, self.swarm_size + 1):
                        if peer_id == self.drone_id:
                            continue
                        peer_port = self.base_port + peer_id
                        self._listen_transport.sendto(message, ("127.0.0.1", peer_port))
                    if self._coord_port:
                        self._listen_transport.sendto(message, (self._coord_host, self._coord_port))
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


class CoverageTracker:
    """Tracks visited cells and batches coverage_update messages."""

    def __init__(
        self,
        comm: SwarmComm,
        logger: DroneLoggerAdapter,
        batch_size: int = 5,
        report_interval: float = 5.0,
    ):
        self._comm = comm
        self._logger = logger
        self._batch_size = max(1, batch_size)
        self._report_interval = max(1.0, report_interval)
        self._covered: set[Tuple[int, int]] = set()
        self._pending: List[Tuple[int, int]] = []
        self._last_send = time.time()
        self._last_plan_id: Optional[int] = None
        self.total_batches = 0
        self.unique_cells = 0

    def mark_cell(self, plan_id: Optional[int], cell: Tuple[int, int]) -> None:
        if plan_id is None or cell in self._covered:
            return
        self._covered.add(cell)
        self.unique_cells += 1
        self._pending.append(cell)
        self._last_plan_id = plan_id
        if len(self._pending) >= self._batch_size:
            self.flush(plan_id)

    def maybe_flush(self, plan_id: Optional[int]) -> None:
        if not self._pending:
            return
        if time.time() - self._last_send >= self._report_interval:
            self.flush(plan_id)

    def flush(self, plan_id: Optional[int] = None) -> None:
        if not self._pending:
            return
        plan = plan_id if plan_id is not None else self._last_plan_id
        if plan is None:
            return
        cells = list(self._pending)
        self._comm.send_coverage_update(plan, cells)
        self._logger.info("Sent coverage_update for plan %s with %d cells.", plan, len(cells))
        self.total_batches += 1
        self._pending.clear()
        self._last_send = time.time()

    def finalize(self) -> None:
        self.flush()


async def connect_and_prepare(config: MissionConfig, logger: DroneLoggerAdapter) -> tuple[System, HomePosition]:
    """Connect to MAVSDK, ensure readiness, and fetch home position."""

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


async def start_local_state_updates(drone: System, comm: SwarmComm, config: MissionConfig) -> list[asyncio.Task]:
    """Spawn telemetry listeners that keep the latest local state updated for broadcasting."""

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
                x_local, y_local = latlon_to_local_xy(
                    position.latitude_deg,
                    position.longitude_deg,
                    config.target_lat,
                    config.target_lon,
                )
                state = DroneState(
                    drone_id=comm.drone_id,
                    lat=position.latitude_deg,
                    lon=position.longitude_deg,
                    alt_amsl=position.absolute_altitude_m,
                    heading_deg=heading_holder["value"],
                    timestamp=time.time(),
                    received_time=time.time(),
                    x_local=x_local,
                    y_local=y_local,
                )
                comm.update_local_state(state)
        except asyncio.CancelledError:
            pass

    tasks.append(asyncio.create_task(heading_worker()))
    tasks.append(asyncio.create_task(position_worker()))
    return tasks


async def run_mission(config: MissionConfig) -> None:
    """Execute the V-formation mission with Voronoi-based search."""

    logger = _create_logger(config.drone_id)
    log_configuration(config, logger)
    drone, home_pos = await connect_and_prepare(config, logger)

    assignment_state: Dict[str, Optional[RegionAssignment]] = {"current": None, "pending": None}
    assignment_event = asyncio.Event()

    def refresh_assignment_event() -> None:
        if assignment_state.get("current"):
            assignment_event.set()
        else:
            assignment_event.clear()

    def handle_assignment(payload: dict) -> None:
        try:
            target_id = int(payload.get("drone_id"))
        except (TypeError, ValueError):
            return
        if target_id != config.drone_id:
            return
        cells_raw = payload.get("cells") or []
        parsed_cells: List[Tuple[int, int]] = []
        for cell in cells_raw:
            try:
                ix, iy = int(cell[0]), int(cell[1])
                parsed_cells.append((ix, iy))
            except (TypeError, ValueError, IndexError):
                continue
        origin_xy = payload.get("origin_xy") or [0.0, 0.0]
        try:
            origin_x = float(origin_xy[0])
            origin_y = float(origin_xy[1])
        except (TypeError, ValueError, IndexError):
            origin_x = origin_y = 0.0
        try:
            cell_size = float(payload.get("cell_size", config.cell_size))
        except (TypeError, ValueError):
            cell_size = config.cell_size

        plan_id = payload.get("plan_id")
        try:
            plan_id_int = int(plan_id) if plan_id is not None else 0
        except (TypeError, ValueError):
            plan_id_int = 0
        active_ids = []
        for item in payload.get("active_ids") or []:
            try:
                active_ids.append(int(item))
            except (TypeError, ValueError):
                continue

        assignment = RegionAssignment(
            origin_xy=(origin_x, origin_y),
            cell_size=cell_size,
            cells=parsed_cells,
            received_time=time.time(),
            plan_id=plan_id_int,
            active_ids=active_ids,
        )
        if assignment_state["current"] is None:
            assignment_state["current"] = assignment
            refresh_assignment_event()
            logger.info(
                "Received region assignment plan %s with %d cells.",
                plan_id_int,
                len(parsed_cells),
            )
        else:
            assignment_state["pending"] = assignment
            logger.info(
                "Queued new assignment plan %s (%d cells). Will switch after current row.",
                plan_id_int,
                len(parsed_cells),
            )

    comm = SwarmComm(
        config.drone_id,
        config.swarm_size,
        config.swarm_broadcast_port_base,
        logger,
        assignment_callback=handle_assignment,
        coordinator_host=config.coordinator_host,
        coordinator_port=config.coordinator_port,
    )
    await comm.start()
    telem_tasks = await start_local_state_updates(drone, comm, config)

    search_alt_amsl = home_pos.altitude_m + config.target_alt_agl

    try:
        if config.takeoff_delay > 0:
            logger.info("Waiting %.1f seconds before takeoff (deconfliction delay).", config.takeoff_delay)
            await asyncio.sleep(config.takeoff_delay)

        if config.cruise_speed > 0:
            try:
                await drone.action.set_maximum_speed(config.cruise_speed)
                logger.info("Set cruise speed to %.1f m/s.", config.cruise_speed)
            except Exception as exc:
                logger.warning("Unable to set cruise speed: %s", exc)

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

        logger.info(
            "Nominal target lat=%.7f lon=%.7f; assigned V-waypoint lat=%.7f lon=%.7f; forward %.1f m, lateral %.1f m.",
            config.target_lat,
            config.target_lon,
            assigned_lat,
            assigned_lon,
            forward_offset,
            lateral_offset,
        )

        await drone.action.goto_location(assigned_lat, assigned_lon, search_alt_amsl, config.yaw_deg)
        await _wait_until_at_position(
            drone,
            target_lat=assigned_lat,
            target_lon=assigned_lon,
            target_alt_amsl=search_alt_amsl,
            horiz_threshold_m=5.0,
            vert_threshold_m=2.5,
            logger=logger,
        )
        logger.info("Reached assigned V-waypoint. Holding for %.1f seconds.", config.initial_loiter_seconds)
        await asyncio.sleep(config.initial_loiter_seconds)

        coverage_tracker = CoverageTracker(
            comm=comm,
            logger=logger,
            batch_size=int(config.coverage_batch_size),
            report_interval=float(config.coverage_report_interval),
        )

        centroid_latlon: Tuple[float, float] = (float("nan"), float("nan"))
        assignments_executed = await handle_coordinator_assignments(
            drone=drone,
            config=config,
            assignment_state=assignment_state,
            assignment_event=assignment_event,
            update_assignment_event=refresh_assignment_event,
            coverage_tracker=coverage_tracker,
            search_alt_amsl=search_alt_amsl,
            logger=logger,
        )

        if not assignments_executed:
            logger.warning("No coordinator assignment executed before search. Falling back to Voronoi partition.")
            await wait_for_voronoi_data(comm, config, logger)
            swarm_snapshot = comm.get_swarm_snapshot()
            voronoi_polygon = compute_voronoi_cell(config, swarm_snapshot, logger)
            centroid_latlon, centroid_local = determine_centroid(voronoi_polygon, config, logger)
            cell_area = float(voronoi_polygon.area)
            logger.info("Voronoi cell area ≈ %.1f m^2.", cell_area)
            send_voronoi_cell_to_collector(voronoi_polygon, centroid_local, cell_area, config, logger)

            await execute_search_mode(
                drone=drone,
                config=config,
                centroid_latlon=centroid_latlon,
                centroid_local=centroid_local,
                polygon=voronoi_polygon,
                search_alt_amsl=search_alt_amsl,
                logger=logger,
            )

        logger.info("Search pattern complete. Landing.")
        await drone.action.land()
        await _wait_until_not_in_air(drone, logger)

        logger.info("Disarming.")
        await drone.action.disarm()
        if assignments_executed:
            logger.info(
                "Mission completed with coordinator assignments. Coverage cells reported=%d in %d batches.",
                coverage_tracker.unique_cells,
                coverage_tracker.total_batches,
            )
        else:
            logger.info(
                "Mission completed via Voronoi fallback. centroid=(%.7f, %.7f).",
                centroid_latlon[0],
                centroid_latlon[1],
            )
    finally:
        for task in telem_tasks:
            task.cancel()
        if telem_tasks:
            await asyncio.gather(*telem_tasks, return_exceptions=True)
        await comm.stop()


def log_configuration(config: MissionConfig, logger: DroneLoggerAdapter) -> None:
    """Log startup configuration parameters."""

    logger.info(
        "Configuration: target=(%.7f, %.7f), target_alt=%.1f AGL, search_radius=%.1f, mode=%d, "
        "coverage_spacing=%.1f, spiral_step=%.1f, spiral_theta_step=%.2f",
        config.target_lat,
        config.target_lon,
        config.target_alt_agl,
        config.search_radius,
        config.search_mode,
        config.coverage_spacing,
        config.spiral_step,
        config.spiral_theta_step,
    )


async def wait_for_voronoi_data(comm: SwarmComm, config: MissionConfig, logger: DroneLoggerAdapter) -> None:
    """Wait for swarm_state to contain enough fresh entries before Voronoi."""

    deadline = time.time() + config.voronoi_wait_seconds
    while time.time() < deadline:
        snapshot = comm.get_swarm_snapshot()
        fresh = [
            (drone_id, state)
            for drone_id, state in snapshot.items()
            if time.time() - state.received_time <= config.swarm_age_threshold
        ]
        if len(fresh) >= max(2, min(config.swarm_size, 3)):
            return
        await asyncio.sleep(1.0)
    logger.warning(
        "Proceeding with Voronoi despite limited swarm data (available=%d).",
        len(comm.get_swarm_snapshot()),
    )


def compute_voronoi_cell(
    config: MissionConfig,
    swarm_snapshot: Dict[int, DroneState],
    logger: DroneLoggerAdapter,
) -> Polygon:
    """Compute the Voronoi cell for this drone within the search radius."""

    reference_lat = config.target_lat
    reference_lon = config.target_lon

    now = time.time()
    sites = []
    site_ids = []
    for drone_id, state in swarm_snapshot.items():
        age = now - state.received_time
        if age > config.swarm_age_threshold:
            continue
        x, y = latlon_to_local_xy(state.lat, state.lon, reference_lat, reference_lon)
        sites.append((x, y))
        site_ids.append(drone_id)
        logger.info(
            "Voronoi input: drone=%d lat=%.7f lon=%.7f -> local=(%.1f, %.1f) age=%.1fs",
            drone_id,
            state.lat,
            state.lon,
            x,
            y,
            age,
        )

    if not sites:
        logger.warning("No fresh swarm data available for Voronoi. Using circular search region.")
        return Point(0.0, 0.0).buffer(config.search_radius, resolution=128)

    if len(sites) == 1:
        logger.warning("Only one site available. Assigning entire circular search region.")
        return Point(0.0, 0.0).buffer(config.search_radius, resolution=128)

    sites_array = np.array(sites)
    vor = Voronoi(sites_array)

    regions, vertices = voronoi_finite_polygons_2d(vor, radius=config.search_radius * 4.0)
    bounding_poly = Point(0.0, 0.0).buffer(config.search_radius, resolution=128)
    polygons: Dict[int, Polygon] = {}

    for site_index, region in enumerate(regions):
        if not region:
            continue
        poly_coords = vertices[region]
        poly = Polygon(poly_coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        clipped = poly.intersection(bounding_poly)
        if clipped.is_empty:
            continue
        polygon_candidate = _largest_polygon(clipped)
        if polygon_candidate is not None:
            polygons[site_index] = polygon_candidate

    try:
        own_index = site_ids.index(config.drone_id)
    except ValueError:
        logger.warning("This drone ID not present in Voronoi sites. Using circular search region.")
        return bounding_poly

    polygon = polygons.get(own_index)
    if polygon is None or polygon.is_empty:
        logger.warning("Voronoi polygon empty. Using circular search region.")
        return bounding_poly

    logger.info(
        "Voronoi polygon vertices (local frame): %s",
        ["(%.1f, %.1f)" % (x, y) for x, y in polygon.exterior.coords],
    )
    return polygon


def voronoi_finite_polygons_2d(vor: Voronoi, radius: Optional[float] = None) -> tuple[list[list[int]], np.ndarray]:
    """
    Reconstruct infinite Voronoi regions to finite regions (SciPy cookbook recipe).
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D inputs.")

    new_regions: list[list[int]] = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    all_ridges: Dict[int, list[tuple[int, int, int]]] = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0 and v2 >= 0:
                continue

            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)


def determine_centroid(polygon: Polygon, config: MissionConfig, logger: DroneLoggerAdapter) -> tuple[Tuple[float, float], Tuple[float, float]]:
    """Compute centroid in local and lat/lon coordinates."""

    if polygon.is_empty:
        logger.warning("Voronoi polygon empty. Using mission target as centroid.")
        return (config.target_lat, config.target_lon), (0.0, 0.0)

    centroid_point = polygon.centroid
    centroid_local = (centroid_point.x, centroid_point.y)
    centroid_latlon = local_xy_to_latlon(centroid_local[0], centroid_local[1], config.target_lat, config.target_lon)

    logger.info(
        "Voronoi centroid local=(%.1f, %.1f) -> lat=%.7f lon=%.7f",
        centroid_local[0],
        centroid_local[1],
        centroid_latlon[0],
        centroid_latlon[1],
    )
    return centroid_latlon, centroid_local


async def execute_search_mode(
    drone: System,
    config: MissionConfig,
    centroid_latlon: Tuple[float, float],
    centroid_local: Tuple[float, float],
    polygon: Polygon,
    search_alt_amsl: float,
    logger: DroneLoggerAdapter,
) -> List[Tuple[float, float]]:
    """Run the requested search mode and return the flown waypoint list."""

    waypoints: List[Tuple[float, float]] = []

    if config.search_mode == 1:
        logger.info("Mode 1: flying to Voronoi centroid and loitering for %.1f seconds.", config.voronoi_loiter_seconds)
        await goto_waypoint_and_wait(drone, centroid_latlon, search_alt_amsl, logger)
        await asyncio.sleep(config.voronoi_loiter_seconds)
        return [centroid_latlon]

    if polygon.is_empty:
        logger.warning("Polygon empty for advanced mode. Falling back to centroid loiter.")
        await goto_waypoint_and_wait(drone, centroid_latlon, search_alt_amsl, logger)
        await asyncio.sleep(config.voronoi_loiter_seconds)
        return [centroid_latlon]

    if config.search_mode == 2:
        logger.info("Mode 2: lawn-mower coverage with spacing %.1f m.", config.coverage_spacing)
        path_xy = generate_lawnmower_path(polygon, config.coverage_spacing)
        if not path_xy:
            logger.warning("Lawn-mower path empty. Falling back to centroid loiter.")
            await goto_waypoint_and_wait(drone, centroid_latlon, search_alt_amsl, logger)
            await asyncio.sleep(config.voronoi_loiter_seconds)
            return [centroid_latlon]

        path_latlon = [
            local_xy_to_latlon(x, y, config.target_lat, config.target_lon) for x, y in path_xy
        ]
        logger.info("Generated %d lawn-mower waypoints.", len(path_latlon))

        await goto_waypoint_and_wait(drone, centroid_latlon, search_alt_amsl, logger)
        for latlon in path_latlon:
            await goto_waypoint_and_wait(drone, latlon, search_alt_amsl, logger)
            waypoints.append(latlon)
        await asyncio.sleep(config.loiter_after_pattern)
        return waypoints

    if config.search_mode == 3:
        logger.info(
            "Mode 3: spiral search with step %.1f m per revolution and theta step %.2f rad.",
            config.spiral_step,
            config.spiral_theta_step,
        )
        path_xy = generate_spiral_path(
            polygon=polygon,
            centroid=centroid_local,
            spiral_step=config.spiral_step,
            theta_step=config.spiral_theta_step,
            max_radius=config.search_radius,
        )
        if not path_xy:
            logger.warning("Spiral path empty. Falling back to centroid loiter.")
            await goto_waypoint_and_wait(drone, centroid_latlon, search_alt_amsl, logger)
            await asyncio.sleep(config.voronoi_loiter_seconds)
            return [centroid_latlon]

        path_latlon = [
            local_xy_to_latlon(x, y, config.target_lat, config.target_lon) for x, y in path_xy
        ]
        logger.info("Generated %d spiral waypoints.", len(path_latlon))

        await goto_waypoint_and_wait(drone, centroid_latlon, search_alt_amsl, logger)
        for latlon in path_latlon:
            await goto_waypoint_and_wait(drone, latlon, search_alt_amsl, logger)
            waypoints.append(latlon)
        await asyncio.sleep(config.loiter_after_pattern)
        return waypoints

    logger.warning("Unknown search mode %d. Defaulting to centroid loiter.", config.search_mode)
    await goto_waypoint_and_wait(drone, centroid_latlon, search_alt_amsl, logger)
    await asyncio.sleep(config.voronoi_loiter_seconds)
    return [centroid_latlon]


async def goto_waypoint_and_wait(
    drone: System,
    latlon: Tuple[float, float],
    altitude_amsl: float,
    logger: DroneLoggerAdapter,
) -> None:
    """Command the drone to fly to the waypoint and wait until arrival."""

    lat, lon = latlon
    logger.info("Navigating to waypoint lat=%.7f lon=%.7f alt=%.1f AMSL", lat, lon, altitude_amsl)
    await drone.action.goto_location(lat, lon, altitude_amsl, float("nan"))
    await _wait_until_at_position(
        drone,
        target_lat=lat,
        target_lon=lon,
        target_alt_amsl=altitude_amsl,
        horiz_threshold_m=5.0,
        vert_threshold_m=3.0,
        logger=logger,
    )


def send_voronoi_cell_to_collector(
    polygon: Polygon,
    centroid_local: Tuple[float, float],
    cell_area: float,
    config: MissionConfig,
    logger: DroneLoggerAdapter,
) -> None:
    """Send Voronoi polygon vertices to the visualization collector via UDP."""

    if polygon.is_empty:
        logger.warning("No Voronoi cell available – skipping collector send.")
        return

    vertices = list(polygon.exterior.coords)
    if not vertices:
        logger.warning("Voronoi polygon has no vertices – skipping collector send.")
        return

    payload = {
        "type": "voronoi_cell",
        "drone_id": config.drone_id,
        "target_lat": config.target_lat,
        "target_lon": config.target_lon,
        "search_radius": config.search_radius,
        "vertices_xy": [[float(x), float(y)] for x, y in vertices],
        "centroid_xy": [float(centroid_local[0]), float(centroid_local[1])],
        "mode": config.search_mode,
        "cell_area_m2": cell_area,
        "timestamp": time.time(),
    }

    message = json.dumps(payload).encode("utf-8")
    address = (config.vis_collector_host, config.vis_collector_port)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.sendto(message, address)
        logger.info(
            "Sending Voronoi cell to collector host=%s port=%d, vertices=%d",
            config.vis_collector_host,
            config.vis_collector_port,
            len(vertices),
        )
    except OSError as exc:
        logger.warning("Failed to send Voronoi cell to collector: %s", exc)


def generate_lawnmower_path(polygon: Polygon, spacing: float) -> List[Tuple[float, float]]:
    """Generate a simple axis-aligned lawn-mower path for the polygon."""

    minx, miny, maxx, maxy = polygon.bounds
    y = miny
    direction = 1
    path: List[Tuple[float, float]] = []

    while y <= maxy:
        line = LineString([(minx - spacing, y), (maxx + spacing, y)])
        segment = line.intersection(polygon)
        coords_sequences = extract_lines(segment)
        if coords_sequences:
            for coords in coords_sequences:
                seq = list(coords)
                if direction < 0:
                    seq.reverse()
                path.extend(seq)
            direction *= -1
        y += spacing

    return path


def generate_spiral_path(
    polygon: Polygon,
    centroid: Tuple[float, float],
    spiral_step: float,
    theta_step: float,
    max_radius: float,
) -> List[Tuple[float, float]]:
    """Generate spiral waypoints inside the polygon."""

    cx, cy = centroid
    theta = 0.0
    path: List[Tuple[float, float]] = []
    b = spiral_step / (2 * math.pi + 1e-6)

    while True:
        r = b * theta
        if r > max_radius:
            break
        x = cx + r * math.cos(theta)
        y = cy + r * math.sin(theta)
        point = Point(x, y)
        if polygon.contains(point):
            path.append((x, y))
        theta += theta_step

    return path


def convert_cells_to_local_points(
    cells: List[Tuple[int, int]],
    origin_xy: Tuple[float, float],
    cell_size: float,
) -> List[Tuple[int, int, float, float]]:
    """Convert grid indices to local XY centers."""

    ox, oy = origin_xy
    points: List[Tuple[int, int, float, float]] = []
    for ix, iy in cells:
        x = ox + (ix + 0.5) * cell_size
        y = oy + (iy + 0.5) * cell_size
        points.append((ix, iy, x, y))
    return points


def build_snake_path(points: List[Tuple[int, int, float, float]]) -> List[Tuple[int, int, float, float]]:
    """Sort cell centers in a snake (lawn-mower) order."""

    ordered: List[Tuple[int, int, float, float]] = []
    for row in build_snake_rows(points):
        ordered.extend(row)
    return ordered


def build_snake_rows(points: List[Tuple[int, int, float, float]]) -> List[List[Tuple[int, int, float, float]]]:
    """Return snake path grouped by rows (iy) for row-wise execution."""

    rows: Dict[int, List[Tuple[int, int, float, float]]] = {}
    for ix, iy, x, y in points:
        rows.setdefault(iy, []).append((ix, iy, x, y))

    ordered_rows: List[List[Tuple[int, int, float, float]]] = []
    for idx, (iy, row_cells) in enumerate(sorted(rows.items())):
        row_cells.sort(key=lambda item: item[0], reverse=(idx % 2 == 1))
        ordered_rows.append(row_cells)
    return ordered_rows


async def goto_local_xy(
    drone: System,
    config: MissionConfig,
    x_local: float,
    y_local: float,
    altitude_amsl: float,
    logger: DroneLoggerAdapter,
) -> Tuple[float, float]:
    """Fly to a local XY point converted to lat/lon."""

    lat, lon = local_xy_to_latlon(x_local, y_local, config.target_lat, config.target_lon)
    await drone.action.goto_location(lat, lon, altitude_amsl, config.yaw_deg)
    await _wait_until_at_position(
        drone,
        target_lat=lat,
        target_lon=lon,
        target_alt_amsl=altitude_amsl,
        horiz_threshold_m=5.0,
        vert_threshold_m=3.0,
        logger=logger,
    )
    return lat, lon


async def handle_coordinator_assignments(
    drone: System,
    config: MissionConfig,
    assignment_state: Dict[str, Optional[RegionAssignment]],
    assignment_event: asyncio.Event,
    update_assignment_event,
    coverage_tracker: CoverageTracker,
    search_alt_amsl: float,
    logger: DroneLoggerAdapter,
) -> bool:
    """Execute coordinator-provided assignments with dynamic reassignment support."""

    executed_any = False

    async def _wait_for_assignment(timeout: float) -> bool:
        if assignment_state.get("current"):
            return True
        try:
            await asyncio.wait_for(assignment_event.wait(), timeout=timeout)
            return assignment_state.get("current") is not None
        except asyncio.TimeoutError:
            return False

    if not await _wait_for_assignment(config.voronoi_wait_seconds):
        return False

    while await _wait_for_assignment(config.voronoi_wait_seconds):
        assignment = assignment_state.get("current")
        if assignment is None:
            continue
        executed_any = True
        result = await _fly_assignment_rows(
            drone=drone,
            config=config,
            assignment=assignment,
            search_alt_amsl=search_alt_amsl,
            logger=logger,
            coverage_tracker=coverage_tracker,
            assignment_state=assignment_state,
        )
        coverage_tracker.flush(assignment.plan_id)

        if result == "switch":
            assignment_state["current"] = assignment_state.get("pending")
            assignment_state["pending"] = None
            update_assignment_event()
            continue

        assignment_state["current"] = assignment_state.get("pending")
        assignment_state["pending"] = None
        update_assignment_event()
        if assignment_state["current"] is None:
            logger.info("Assignment complete – waiting for the next plan.")

    coverage_tracker.finalize()
    if coverage_tracker.unique_cells:
        logger.info(
            "Coverage summary: %d unique cells visited in %d batches.",
            coverage_tracker.unique_cells,
            coverage_tracker.total_batches,
        )
    return executed_any


async def _fly_assignment_rows(
    drone: System,
    config: MissionConfig,
    assignment: RegionAssignment,
    search_alt_amsl: float,
    logger: DroneLoggerAdapter,
    coverage_tracker: CoverageTracker,
    assignment_state: Dict[str, Optional[RegionAssignment]],
) -> str:
    """Fly the snake path row-by-row, allowing mid-row reassignment."""

    cell_points = convert_cells_to_local_points(assignment.cells, assignment.origin_xy, assignment.cell_size)
    plan_id = assignment.plan_id
    if not cell_points:
        logger.info("Assignment plan %s contains no cells. Holding for %.1f seconds.", plan_id, config.loiter_after_pattern)
        await asyncio.sleep(config.loiter_after_pattern)
        return "idle"

    rows = build_snake_rows(cell_points)
    total_rows = len(rows)
    for row_idx, row in enumerate(rows, start=1):
        for ix, iy, x_local, y_local in row:
            lat, lon = local_xy_to_latlon(x_local, y_local, config.target_lat, config.target_lon)
            logger.info(
                "Plan %s row %d/%d – visiting cell (%d, %d) at lat=%.7f lon=%.7f.",
                plan_id,
                row_idx,
                total_rows,
                ix,
                iy,
                lat,
                lon,
            )
            await drone.action.goto_location(lat, lon, search_alt_amsl, config.yaw_deg)
            await _wait_until_at_position(
                drone,
                target_lat=lat,
                target_lon=lon,
                target_alt_amsl=search_alt_amsl,
                horiz_threshold_m=5.0,
                vert_threshold_m=3.0,
                logger=logger,
            )
            coverage_tracker.mark_cell(plan_id, (ix, iy))
            coverage_tracker.maybe_flush(plan_id)

        logger.info("Completed row %d/%d for plan %s.", row_idx, total_rows, plan_id)
        if assignment_state.get("pending"):
            logger.info(
                "Pending assignment plan %s detected – switching after row %d.",
                assignment_state["pending"].plan_id,
                row_idx,
            )
            return "switch"

    logger.info("Assignment plan %s fully covered. Loitering %.1f seconds.", plan_id, config.loiter_after_pattern)
    await asyncio.sleep(config.loiter_after_pattern)
    return "done"


def extract_lines(geom) -> List[Iterable[Tuple[float, float]]]:
    """Extract coordinate sequences from a (multi)line intersection result."""

    if geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom.coords]
    try:
        return [g.coords for g in geom.geoms if isinstance(g, LineString)]
    except AttributeError:
        return []


def _largest_polygon(geometry) -> Optional[Polygon]:
    """Return the largest Polygon component within a geometry."""

    if geometry is None or geometry.is_empty:
        return None
    if isinstance(geometry, Polygon):
        return geometry
    if hasattr(geometry, "geoms"):
        polygons = [g for g in geometry.geoms if isinstance(g, Polygon) and not g.is_empty]
        if not polygons:
            return None
        return max(polygons, key=lambda g: g.area)
    return None


def latlon_to_local_xy(lat: float, lon: float, lat0: float, lon0: float) -> Tuple[float, float]:
    """Convert latitude/longitude to a local tangent plane (meters)."""

    meters_per_deg_lat = 111_320.0
    lat_rad = math.radians(lat0)
    meters_per_deg_lon = 111_320.0 * math.cos(lat_rad)
    x = (lon - lon0) * meters_per_deg_lon
    y = (lat - lat0) * meters_per_deg_lat
    return x, y


def local_xy_to_latlon(x: float, y: float, lat0: float, lon0: float) -> Tuple[float, float]:
    """Convert local tangent plane coordinates back to latitude/longitude."""

    meters_per_deg_lat = 111_320.0
    lat_rad = math.radians(lat0)
    meters_per_deg_lon = 111_320.0 * math.cos(lat_rad)
    lat = y / meters_per_deg_lat + lat0
    if abs(meters_per_deg_lon) < 1e-6:
        meters_per_deg_lon = 1e-6
    lon = x / meters_per_deg_lon + lon0
    return lat, lon


def local_xy_to_grid(x: float, y: float, cfg: GridAssignmentConfig) -> Tuple[int, int]:
    """Convert local XY (meters) to integer grid cell indices."""

    nx, ny = cfg.grid_size
    offset_x = -(nx * cfg.cell_size) / 2.0
    offset_y = -(ny * cfg.cell_size) / 2.0
    ix = int((x - offset_x) // cfg.cell_size)
    iy = int((y - offset_y) // cfg.cell_size)
    return ix, iy


@dataclass
class GridAssignmentConfig:
    cell_size: float
    grid_size: Tuple[int, int]


@dataclass
class RegionAssignment:
    origin_xy: Tuple[float, float]
    cell_size: float
    cells: List[Tuple[int, int]]
    received_time: float
    plan_id: int = 0
    active_ids: List[int] = field(default_factory=list)


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

    parser = argparse.ArgumentParser(description="Phase 5 Voronoi-enabled drone agent using MAVSDK.")
    parser.add_argument("--drone-id", type=int, required=True, help="Identifier of this drone (1-5).")
    parser.add_argument("--swarm-size", type=int, default=5, help="Total number of drones in the swarm.")
    parser.add_argument(
        "--mavsdk-address",
        default="127.0.0.1:50051",
        help="Address of the MAVSDK server in host:port format.",
    )
    parser.add_argument("--target-lat", type=float, required=True, help="Mission target latitude.")
    parser.add_argument("--target-lon", type=float, required=True, help="Mission target longitude.")
    parser.add_argument("--target-alt", type=float, required=True, help="Mission altitude AGL in meters.")
    parser.add_argument("--takeoff-alt", type=float, default=30.0, help="Takeoff altitude AGL in meters.")
    parser.add_argument("--yaw-deg", type=float, default=float("nan"), help="Yaw angle at waypoints (default: keep).")
    parser.add_argument(
        "--initial-loiter-seconds",
        type=float,
        default=5.0,
        help="Hold time at V-waypoint before Voronoi computation.",
    )
    parser.add_argument(
        "--loiter-after-pattern",
        type=float,
        default=15.0,
        help="Loiter time after completing coverage patterns.",
    )
    parser.add_argument(
        "--voronoi-loiter-seconds",
        type=float,
        default=60.0,
        help="Loiter duration at centroid for mode 1.",
    )
    parser.add_argument(
        "--swarm-broadcast-port-base",
        type=int,
        default=61000,
        help="Base UDP port for swarm mesh (default 61000).",
    )
    parser.add_argument(
        "--swarm-age-threshold",
        type=float,
        default=10.0,
        help="Max allowed age (seconds) for swarm states in Voronoi.",
    )
    parser.add_argument(
        "--voronoi-wait-seconds",
        type=float,
        default=10.0,
        help="Max seconds to wait for swarm data before Voronoi.",
    )
    parser.add_argument(
        "--search-radius",
        type=float,
        default=200.0,
        help="Bounding radius (meters) for Voronoi search region.",
    )
    parser.add_argument(
        "--search-mode",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Search mode: 1=centroid, 2=lawn-mower, 3=spiral.",
    )
    parser.add_argument(
        "--coverage-spacing",
        type=float,
        default=20.0,
        help="Spacing (meters) between lawn-mower stripes (mode 2).",
    )
    parser.add_argument(
        "--spiral-step",
        type=float,
        default=15.0,
        help="Radial step (meters per revolution) for spiral mode.",
    )
    parser.add_argument(
        "--spiral-theta-step",
        type=float,
        default=0.4,
        help="Angular increment (radians) between spiral waypoints.",
    )
    parser.add_argument(
        "--cruise-speed",
        type=float,
        default=15.0,
        help="Target cruise speed in m/s (set as MAV maximum speed).",
    )
    parser.add_argument(
        "--coverage-batch-size",
        type=int,
        default=5,
        help="Number of cells to include in each coverage_update batch.",
    )
    parser.add_argument(
        "--coverage-report-interval",
        type=float,
        default=5.0,
        help="Maximum seconds between automatic coverage_update flushes.",
    )
    parser.add_argument(
        "--vis-collector-host",
        type=str,
        default="127.0.0.1",
        help="Host/IP for Voronoi visualization collector.",
    )
    parser.add_argument(
        "--vis-collector-port",
        type=int,
        default=62000,
        help="UDP port for Voronoi visualization collector.",
    )
    parser.add_argument(
        "--grid-cell-size",
        type=float,
        default=20.0,
        help="Coverage grid cell size (meters) for assignments.",
    )
    parser.add_argument(
        "--coordinator-port",
        type=int,
        default=61000,
        help="UDP port where the coverage coordinator listens for state messages.",
    )
    parser.add_argument(
        "--coordinator-host",
        type=str,
        default="127.0.0.1",
        help="Host/IP of the coverage coordinator listener.",
    )
    parser.add_argument(
        "--takeoff-delay",
        type=float,
        default=None,
        help="Optional delay before takeoff (defaults to drone_id * 2 seconds).",
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
        initial_loiter_seconds=args.initial_loiter_seconds,
        loiter_after_pattern=args.loiter_after_pattern,
        voronoi_loiter_seconds=args.voronoi_loiter_seconds,
        swarm_broadcast_port_base=args.swarm_broadcast_port_base,
        swarm_age_threshold=args.swarm_age_threshold,
        voronoi_wait_seconds=args.voronoi_wait_seconds,
        search_radius=args.search_radius,
        search_mode=args.search_mode,
        coverage_spacing=args.coverage_spacing,
        spiral_step=args.spiral_step,
        spiral_theta_step=args.spiral_theta_step,
        coverage_batch_size=args.coverage_batch_size,
        coverage_report_interval=args.coverage_report_interval,
        cruise_speed=args.cruise_speed,
        connection_timeout=args.connection_timeout,
        readiness_timeout=args.readiness_timeout,
        takeoff_delay=takeoff_delay,
        vis_collector_host=args.vis_collector_host,
        vis_collector_port=args.vis_collector_port,
        cell_size=args.grid_cell_size,
        grid_origin=(0.0, 0.0),
        coordinator_port=args.coordinator_port,
        coordinator_host=args.coordinator_host,
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
# Drone 1 (leader, mode 1):
# python drone_agent_phase5_voronoi.py --drone-id 1 --mavsdk-address 127.0.0.1:50040 --target-lat <LAT> --target-lon <LON> --target-alt 30 --search-mode 1
#
# Drone 2 (mode 2 lawn-mower):
# python drone_agent_phase5_voronoi.py --drone-id 2 --mavsdk-address 127.0.0.1:50041 --target-lat <LAT> --target-lon <LON> --target-alt 30 --search-mode 2 --coverage-spacing 25
#
# Drone 3 (mode 3 spiral):
# python drone_agent_phase5_voronoi.py --drone-id 3 --mavsdk-address 127.0.0.1:50042 --target-lat <LAT> --target-lon <LON> --target-alt 30 --search-mode 3 --spiral-step 20
#
# Drone 4 (mode 1):
# python drone_agent_phase5_voronoi.py --drone-id 4 --mavsdk-address 127.0.0.1:50043 --target-lat <LAT> --target-lon <LON> --target-alt 30
#
# Drone 5 (mode 2):
# python drone_agent_phase5_voronoi.py --drone-id 5 --mavsdk-address 127.0.0.1:50044 --target-lat <LAT> --target-lon <LON> --target-alt 30 --search-mode 2 --coverage-spacing 20
