"""
Coverage coordinator for Phase 6 – grid-based Voronoi assignments.

Listens on the existing swarm UDP mesh, tracks active drones from "state"
messages, partitions an XY grid of the search region, and broadcasts
region_assignment payloads whenever the active set changes or on a periodic
timer. No coverage updates or MAVSDK control are implemented in this step.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import socket
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple


LOGGER = logging.getLogger("coverage_coordinator")


def latlon_to_local_xy(lat: float, lon: float, lat0: float, lon0: float) -> Tuple[float, float]:
    """Convert latitude/longitude to a local tangent plane (meters)."""

    meters_per_deg_lat = 111_320.0
    lat_rad = math.radians(lat0)
    meters_per_deg_lon = 111_320.0 * math.cos(lat_rad)
    x = (lon - lon0) * meters_per_deg_lon
    y = (lat - lat0) * meters_per_deg_lat
    return x, y


@dataclass
class DroneInfo:
    drone_id: int
    position_xy: Tuple[float, float]
    last_seen: float
    ready: bool = False


class CoverageCoordinator:
    def __init__(self, args: argparse.Namespace):
        self.target_lat = args.target_lat
        self.target_lon = args.target_lon
        self.search_radius = args.search_radius_m
        self.cell_size = args.cell_size_m
        self.failure_timeout = args.failure_timeout
        self.replan_interval = args.replan_interval
        self.base_port = args.mesh_base_port
        self.max_drones = args.max_drones
        self.listen_host = args.listen_host
        self.broadcast_host = args.broadcast_host
        self.listen_port = args.listen_port
        self.visualizer_host = args.visualizer_host
        self.visualizer_port = args.visualizer_port
        self.activation_radius = args.activation_radius_m

        self.grid_side = int(math.ceil((2 * self.search_radius) / self.cell_size))
        self.radius_sq = self.search_radius * self.search_radius
        offset = -(self.grid_side * self.cell_size) / 2.0
        self.grid_origin = (offset, offset)
        self.circle_cell_count = self._count_circle_cells()

        self.drone_infos: Dict[int, DroneInfo] = {}
        self.pending_replan = False
        self.last_replan_time = 0.0
        self.plan_counter = 0

    def _count_circle_cells(self) -> int:
        count = 0
        for ix in range(self.grid_side):
            for iy in range(self.grid_side):
                if self._cell_inside_circle(ix, iy):
                    count += 1
        return count

    def _cell_center_xy(self, ix: int, iy: int) -> Tuple[float, float]:
        x = self.grid_origin[0] + ix * self.cell_size + self.cell_size / 2.0
        y = self.grid_origin[1] + iy * self.cell_size + self.cell_size / 2.0
        return x, y

    def _cell_inside_circle(self, ix: int, iy: int) -> bool:
        x, y = self._cell_center_xy(ix, iy)
        return x * x + y * y <= self.radius_sq

    def handle_state_message(self, payload: dict) -> None:
        drone_id = payload.get("drone_id")
        lat = payload.get("lat")
        lon = payload.get("lon")
        x_local = payload.get("x_local")
        y_local = payload.get("y_local")
        if drone_id is None or lat is None or lon is None:
            return
        try:
            drone_id = int(drone_id)
            lat = float(lat)
            lon = float(lon)
        except (TypeError, ValueError):
            return
        if x_local is not None and y_local is not None:
            try:
                x = float(x_local)
                y = float(y_local)
            except (TypeError, ValueError):
                x, y = latlon_to_local_xy(lat, lon, self.target_lat, self.target_lon)
        else:
            x, y = latlon_to_local_xy(lat, lon, self.target_lat, self.target_lon)
        now = time.time()
        in_activation = (x * x + y * y) <= self.activation_radius * self.activation_radius
        info = self.drone_infos.get(drone_id)
        if info is None:
            LOGGER.info("Detected new drone %s.", drone_id)
            info = DroneInfo(drone_id=drone_id, position_xy=(x, y), last_seen=now, ready=in_activation)
            self.drone_infos[drone_id] = info
            if in_activation:
                LOGGER.info("  Drone %s entered search area – scheduling repartition.", drone_id)
                self.pending_replan = True
            return

        info.position_xy = (x, y)
        info.last_seen = now
        if not info.ready and in_activation:
            info.ready = True
            LOGGER.info("Drone %s reached search area – scheduling repartition.", drone_id)
            self.pending_replan = True

    def remove_inactive_drones(self) -> None:
        now = time.time()
        removed = []
        for drone_id, info in list(self.drone_infos.items()):
            if now - info.last_seen > self.failure_timeout:
                removed.append(drone_id)
                del self.drone_infos[drone_id]
        if removed:
            LOGGER.warning("Removed inactive drones: %s", removed)
            self.pending_replan = True

    def should_replan(self) -> bool:
        if not self.drone_infos:
            return False
        if self.pending_replan:
            return True
        if self.replan_interval > 0:
            return (time.time() - self.last_replan_time) >= self.replan_interval
        return False

    def compute_assignments(self) -> Dict[int, List[List[int]]]:
        active_ids = [did for did, info in self.drone_infos.items() if info.ready]
        if not active_ids:
            return {}
        assignments: Dict[int, List[List[int]]] = {drone_id: [] for drone_id in active_ids}
        positions = {drone_id: info.position_xy for drone_id, info in self.drone_infos.items() if info.ready}

        for ix in range(self.grid_side):
            for iy in range(self.grid_side):
                if not self._cell_inside_circle(ix, iy):
                    continue
                x, y = self._cell_center_xy(ix, iy)
                best_id = min(active_ids, key=lambda did: self._distance_sq((x, y), positions[did]))
                assignments[best_id].append([ix, iy])
        return assignments

    @staticmethod
    def _distance_sq(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx * dx + dy * dy

    async def broadcast_assignments(self, assignments: Dict[int, List[List[int]]]) -> None:
        if not assignments:
            return
        loop = asyncio.get_running_loop()
        active_ids = sorted(assignments.keys())
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setblocking(False)
        transport, _ = await loop.create_datagram_endpoint(
            asyncio.DatagramProtocol,
            sock=sock,
        )
        plan_id = self.plan_counter
        self.plan_counter += 1
        try:
            for drone_id, cells in assignments.items():
                payload = {
                    "msg_type": "region_assignment",
                    "drone_id": drone_id,
                    "origin_xy": list(self.grid_origin),
                    "cell_size": self.cell_size,
                    "cells": cells,
                    "search_radius": self.search_radius,
                    "active_ids": active_ids,
                    "plan_id": plan_id,
                }
                message = json.dumps(payload).encode("utf-8")
                dest = (self.broadcast_host, self.base_port + drone_id)
                transport.sendto(message, dest)
                if self.visualizer_port:
                    transport.sendto(message, (self.visualizer_host, self.visualizer_port))
        finally:
            transport.close()

    async def run_replan(self) -> None:
        assignments = self.compute_assignments()
        if not assignments:
            return
        self.last_replan_time = time.time()
        self.pending_replan = False

        total_cells = sum(len(cells) for cells in assignments.values())
        unassigned = max(self.circle_cell_count - total_cells, 0)
        unassigned_pct = (
            100.0 * unassigned / self.circle_cell_count if self.circle_cell_count > 0 else 0.0
        )

        LOGGER.info("Active drones: %d", len(assignments))
        for drone_id, cells in assignments.items():
            LOGGER.info("  Drone %d: %d cells assigned", drone_id, len(cells))
        LOGGER.info("Unassigned cells: %d (%.2f%%)", unassigned, unassigned_pct)

        await self.broadcast_assignments(assignments)


class MeshProtocol(asyncio.DatagramProtocol):
    def __init__(self, coordinator: CoverageCoordinator):
        self.coordinator = coordinator

    def datagram_received(self, data: bytes, addr):
        try:
            payload = json.loads(data.decode("utf-8"))
        except json.JSONDecodeError:
            return
        msg_type = payload.get("msg_type") or payload.get("type") or "state"
        if msg_type == "state":
            self.coordinator.handle_state_message(payload)


async def main() -> None:
    args = parse_args()
    coordinator = CoverageCoordinator(args)
    loop = asyncio.get_running_loop()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if hasattr(socket, "SO_REUSEPORT"):
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except OSError:
            pass
    sock.bind((args.listen_host, args.listen_port))
    transport, _ = await loop.create_datagram_endpoint(
        lambda: MeshProtocol(coordinator),
        sock=sock,
    )
    LOGGER.info("Coordinator listening on %s:%d", args.listen_host, args.listen_port)

    try:
        while True:
            coordinator.remove_inactive_drones()
            if coordinator.should_replan():
                await coordinator.run_replan()
            await asyncio.sleep(1.0)
    except KeyboardInterrupt:
        LOGGER.info("Coordinator stopped by user.")
    finally:
        transport.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 6 coverage coordinator.")
    parser.add_argument("--target-lat", type=float, required=True, help="Target latitude.")
    parser.add_argument("--target-lon", type=float, required=True, help="Target longitude.")
    parser.add_argument("--search-radius-m", type=float, default=400.0, help="Search radius in meters.")
    parser.add_argument("--cell-size-m", type=float, default=20.0, help="Grid cell size in meters.")
    parser.add_argument("--failure-timeout", type=float, default=60.0, help="Seconds before a drone is considered failed.")
    parser.add_argument("--replan-interval", type=float, default=0.0, help="Periodic replan interval (0 disables).")
    parser.add_argument("--mesh-base-port", type=int, default=61000, help="Base UDP port for swarm mesh.")
    parser.add_argument("--max-drones", type=int, default=5, help="Maximum number of drone IDs to monitor.")
    parser.add_argument("--listen-host", default="0.0.0.0", help="Host/IP to bind for listening sockets.")
    parser.add_argument("--listen-port", type=int, default=61000, help="UDP port where coordinator listens for state messages.")
    parser.add_argument("--broadcast-host", default="127.0.0.1", help="Host/IP used when sending assignments.")
    parser.add_argument("--visualizer-host", default="127.0.0.1", help="Host/IP for optional visualization listener.")
    parser.add_argument("--visualizer-port", type=int, default=62000, help="UDP port for visualization listener (0 to disable).")
    parser.add_argument(
        "--activation-radius-m",
        type=float,
        default=150.0,
        help="Distance from target before a drone participates in assignments.",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity.")
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    return args


if __name__ == "__main__":
    asyncio.run(main())
