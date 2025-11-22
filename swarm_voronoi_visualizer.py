"""
Swarm Voronoi visualizer.

Listens for Voronoi/partition broadcasts from the swarm (Phase 5+):
* Phase 5 agents send ``type = "voronoi_cell"`` with polygon vertices.
* Phase 6 coordinator sends ``msg_type = "region_assignment"`` plus grid cells.

Once every expected drone has reported in, all cells are plotted on a shared
local XY map and coverage diagnostics are printed. The tool can optionally keep
listening for future assignments/replans.
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
from dataclasses import dataclass
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union


LOGGER = logging.getLogger("swarm_voronoi_visualizer")


@dataclass
class VisualizerConfig:
    listen_host: str
    listen_port: int
    expected_drones: List[int]
    auto_plot: bool
    keep_listening: bool


def parse_expected(expected: str) -> List[int]:
    """Parse a comma-separated list of drone IDs into sorted integers."""

    if expected is None or expected.strip() == "":
        return []
    ids = []
    for token in expected.split(","):
        token = token.strip()
        if not token:
            continue
        ids.append(int(token))
    return sorted(set(ids))


def plot_regions(polygons: Dict[int, Polygon], diag: dict | None, block_display: bool = True) -> None:
    """Plot all Voronoi cells on a shared Matplotlib figure."""

    fig, ax = plt.subplots()
    search_radius = diag.get("search_radius") if diag else None
    coverage_pct = diag.get("coverage_pct") if diag else None
    gap_pct = diag.get("gap_pct") if diag else None
    overlap_pct = diag.get("overlap_pct") if diag else None

    for drone_id in sorted(polygons):
        poly = polygons[drone_id]
        if poly.is_empty:
            continue
        vertices = list(poly.exterior.coords)
        centroid = (poly.centroid.x, poly.centroid.y)
        area = poly.area

        if len(vertices) < 3:
            LOGGER.warning("Drone %d polygon has too few vertices (%d). Skipping plot.", drone_id, len(vertices))
            continue

        xs = [p[0] for p in vertices]
        ys = [p[1] for p in vertices]
        patch = ax.fill(xs, ys, alpha=0.3, label=f"Drone {drone_id}")
        edge_color = patch[0].get_facecolor()
        ax.plot(xs, ys, color=edge_color, linewidth=1.5)
        ax.plot(centroid[0], centroid[1], "ko", markersize=4)
        label = f"{drone_id}\n{area:.0f} m²"
        ax.text(centroid[0], centroid[1], label, fontsize=8, ha="center", va="center")

    if search_radius:
        theta = np.linspace(0, 2 * np.pi, 720)
        circle_x = search_radius * np.cos(theta)
        circle_y = search_radius * np.sin(theta)
        ax.plot(circle_x, circle_y, "k--", linewidth=1.0, label="Search boundary")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    title = "Swarm Voronoi Cells (Local XY, Circular Search Region)"
    if coverage_pct is not None:
        title += f"\nCoverage={coverage_pct:.2f}%, Gaps={gap_pct:.4f}%, Overlap={overlap_pct:.4f}%"
    ax.set_title(title)
    ax.grid(True)
    if polygons:
        ax.legend(loc="upper right")
    plt.show(block=block_display)
    if not block_display:
        plt.pause(0.1)


def log_diagnostics(polygons: Dict[int, Polygon], diag: dict | None, order: Iterable[int]) -> None:
    """Print coverage/overlap diagnostics and per-drone percentages."""

    if not diag:
        LOGGER.info("Swarm Voronoi diagnostics unavailable (missing search radius or polygons).")
        return

    circle_area = diag["circle_area"]
    LOGGER.info("=== Swarm Voronoi diagnostics (local XY) ===")
    LOGGER.info("Search radius         : %.1f m", diag["search_radius"])
    LOGGER.info("Circle area           : %.1f m² (100.0%%)", circle_area)
    LOGGER.info(
        "Covered area (union)  : %.1f m² (%.2f%%)",
        diag["covered_area"],
        diag["coverage_pct"],
    )
    LOGGER.info(
        "Uncovered gaps        : %.1f m² (%.4f%%)",
        diag["gap_area"],
        diag["gap_pct"],
    )
    LOGGER.info(
        "Overlap between cells : %.1f m² (%.4f%%)",
        diag["overlap_area"],
        diag["overlap_pct"],
    )
    LOGGER.info("")
    LOGGER.info("Per-drone cell areas (relative to full circle):")
    for drone_id in order:
        poly = polygons.get(drone_id)
        if poly is None or poly.is_empty:
            LOGGER.info("  Drone %d: no data received.", drone_id)
            continue
        area = poly.area
        pct = 100.0 * area / circle_area if circle_area > 0 else 0.0
        LOGGER.info("  Drone %d: %.1f m² (%.2f%%)", drone_id, area, pct)


def compute_diagnostics(polygons: Dict[int, Polygon], search_radius: float | None) -> dict | None:
    """Compute coverage/overlap diagnostics for already-built polygons."""

    if not search_radius or search_radius <= 0:
        LOGGER.warning("No valid search radius provided; skipping diagnostics.")
        return None

    circle_poly = Point(0.0, 0.0).buffer(search_radius, resolution=512)
    circle_area = circle_poly.area

    if polygons:
        union_poly = unary_union(list(polygons.values()))
    else:
        union_poly = Polygon()

    covered_poly = union_poly.intersection(circle_poly) if not union_poly.is_empty else Polygon()
    covered_area = covered_poly.area
    gap_area = max(circle_area - covered_area, 0.0)

    overlap_area = 0.0
    items = list(polygons.items())
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            try:
                inter = items[i][1].intersection(items[j][1]).intersection(circle_poly)
                overlap_area += inter.area
            except Exception as exc:
                LOGGER.debug(
                    "Failed to compute overlap between drones %s and %s: %s",
                    items[i][0],
                    items[j][0],
                    exc,
                )

    circle_inv = 1.0 / circle_area if circle_area > 0 else 0.0
    return {
        "search_radius": search_radius,
        "circle_area": circle_area,
        "covered_area": covered_area,
        "gap_area": gap_area,
        "overlap_area": overlap_area,
        "coverage_pct": covered_area * circle_inv * 100.0,
        "gap_pct": gap_area * circle_inv * 100.0,
        "overlap_pct": overlap_area * circle_inv * 100.0,
    }


def _polygon_from_vertices(drone_id: int, payload: dict) -> Polygon | None:
    """Convert vertex payload into a cleaned Shapely Polygon."""

    vertices = payload.get("vertices_xy") or []
    if len(vertices) < 3:
        LOGGER.warning("Drone %d polygon has too few vertices (%d).", drone_id, len(vertices))
        return None

    try:
        poly = Polygon(vertices)
    except Exception as exc:
        LOGGER.warning("Drone %d polygon could not be constructed: %s", drone_id, exc)
        return None

    if poly.is_empty:
        LOGGER.warning("Drone %d polygon is empty.", drone_id)
        return None

    if not poly.is_valid:
        LOGGER.warning("Drone %d polygon invalid; attempting to fix.", drone_id)
        poly = poly.buffer(0)
        if poly.is_empty:
            LOGGER.warning("Drone %d polygon remained empty after fix.", drone_id)
            return None

    if poly.geom_type == "MultiPolygon":
        try:
            poly = max(poly.geoms, key=lambda g: g.area)
        except ValueError:
            LOGGER.warning("Drone %d multipolygon has no components.", drone_id)
            return None

    return poly


def _polygon_from_assignment(payload: dict) -> Polygon | None:
    """Convert a region_assignment payload into a polygon via union of grid cells."""

    origin = payload.get("origin_xy")
    cell_size = payload.get("cell_size")
    cells = payload.get("cells") or []
    if origin is None or cell_size is None or not cells:
        return None
    try:
        origin_x = float(origin[0])
        origin_y = float(origin[1])
        cell_size = float(cell_size)
    except (TypeError, ValueError):
        return None

    boxes = []
    for cell in cells:
        if not isinstance(cell, (list, tuple)) or len(cell) != 2:
            continue
        try:
            ix = int(cell[0])
            iy = int(cell[1])
        except (TypeError, ValueError):
            continue
        x_min = origin_x + ix * cell_size
        y_min = origin_y + iy * cell_size
        boxes.append(box(x_min, y_min, x_min + cell_size, y_min + cell_size))

    if not boxes:
        return None
    try:
        return unary_union(boxes)
    except Exception as exc:
        LOGGER.warning("Failed to build polygon from assignment: %s", exc)
        return None


def run_visualizer(config: VisualizerConfig) -> None:
    """Receive Voronoi/assignment packets and plot using dynamic drone sets."""

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((config.listen_host, config.listen_port))
    LOGGER.info(
        "Listening for Voronoi cells on %s:%d (expected drones: %s)",
        config.listen_host,
        config.listen_port,
        ", ".join(map(str, config.expected_drones)) if config.expected_drones else "dynamic",
    )

    static_expected = set(config.expected_drones)
    current_expected = set(static_expected)
    current_plan_id: int | None = None
    polygons: Dict[int, Polygon] = {}
    latest_radius: float | None = None

    try:
        while True:
            data, addr = sock.recvfrom(65535)
            try:
                payload = json.loads(data.decode("utf-8"))
            except json.JSONDecodeError as exc:
                LOGGER.warning("Failed to decode JSON from %s: %s", addr, exc)
                continue

            msg_type = payload.get("msg_type") or payload.get("type") or ""
            drone_id = payload.get("drone_id")
            if drone_id is None:
                LOGGER.warning("Received payload without drone_id from %s", addr)
                continue

            polygon: Polygon | None = None
            plan_id = payload.get("plan_id")
            dynamic_expected_raw = payload.get("active_ids")
            dynamic_expected: set[int] | None = None
            if isinstance(dynamic_expected_raw, list):
                try:
                    dynamic_expected = {int(i) for i in dynamic_expected_raw}
                except (TypeError, ValueError):
                    dynamic_expected = None

            if plan_id is not None and plan_id != current_plan_id:
                LOGGER.info("Starting new assignment batch plan_id=%s", plan_id)
                polygons.clear()
                current_plan_id = plan_id
                current_expected = dynamic_expected or set(static_expected)
            elif dynamic_expected:
                current_expected = dynamic_expected
            elif not current_expected and static_expected:
                current_expected = set(static_expected)

            if msg_type == "voronoi_cell":
                polygon = _polygon_from_vertices(drone_id, payload)
                latest_radius = payload.get("search_radius", latest_radius)
                vertex_count = len(payload.get("vertices_xy") or [])
                LOGGER.info(
                    "Received Voronoi cell from drone_id=%s with %d vertices (mode=%s)",
                    drone_id,
                    vertex_count,
                    payload.get("mode"),
                )
            elif msg_type == "region_assignment":
                polygon = _polygon_from_assignment(payload)
                latest_radius = payload.get("search_radius", latest_radius)
                LOGGER.info(
                    "Received region assignment from drone_id=%s with %d cells.",
                    drone_id,
                    len(payload.get("cells") or []),
                )
            else:
                LOGGER.debug("Ignoring message with unexpected type '%s' from %s", msg_type, addr)
                continue

            if polygon is None or polygon.is_empty:
                LOGGER.warning("Polygon for drone %s is empty or invalid; skipping.", drone_id)
                continue

            polygons[drone_id] = polygon

            if current_expected:
                have_all = current_expected.issubset(polygons.keys())
            else:
                have_all = bool(polygons)
            LOGGER.debug(
                "Plan %s: collected cells from %s / expected %s",
                current_plan_id,
                sorted(polygons.keys()),
                sorted(current_expected) if current_expected else "any",
            )

            if config.auto_plot and have_all:
                order = sorted(current_expected) if current_expected else sorted(polygons.keys())
                subset = {did: polygons[did] for did in order}
                diag = compute_diagnostics(subset, latest_radius)
                log_diagnostics(subset, diag, order)
                plot_regions(subset, diag, block_display=not config.keep_listening)
                if not config.keep_listening:
                    LOGGER.info("All expected cells plotted. Exiting.")
                    break
                LOGGER.info("Keep-listening enabled – clearing cells for next batch.")
                polygons.clear()
                latest_radius = None
                if plan_id is not None:
                    current_plan_id = None
                current_expected = set(static_expected)
    finally:
        sock.close()


def parse_args() -> VisualizerConfig:
    parser = argparse.ArgumentParser(description="Swarm Voronoi visualizer.")
    parser.add_argument("--listen-host", default="0.0.0.0", help="Host/IP to bind UDP listener (default 0.0.0.0).")
    parser.add_argument("--listen-port", type=int, default=62000, help="UDP port to listen on (default 62000).")
    parser.add_argument(
        "--expected-drones",
        default="",
        help="Comma-separated list of drone IDs expected to report. Leave blank for dynamic detection.",
    )
    parser.add_argument(
        "--auto-plot",
        type=lambda x: str(x).lower() in {"1", "true", "yes"},
        default="True",
        help="Automatically plot once all drones have reported (default True).",
    )
    parser.add_argument(
        "--keep-listening",
        type=lambda x: str(x).lower() in {"1", "true", "yes"},
        default="False",
        help="After plotting, continue listening for new batches (default False).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s [%(levelname)s] %(message)s")

    expected_drones = parse_expected(args.expected_drones)
    config = VisualizerConfig(
        listen_host=args.listen_host,
        listen_port=args.listen_port,
        expected_drones=expected_drones,
        auto_plot=bool(args.auto_plot),
        keep_listening=bool(args.keep_listening),
    )
    return config


def main() -> None:
    config = parse_args()
    try:
        run_visualizer(config)
    except KeyboardInterrupt:
        LOGGER.info("Visualizer interrupted by user.")


if __name__ == "__main__":
    main()
