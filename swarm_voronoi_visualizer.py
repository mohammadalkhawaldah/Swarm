"""
Swarm Voronoi visualizer.

Listens for Voronoi cell broadcasts from Phase 5 drone agents and plots all
cells on a shared local XY map once every expected drone has reported in.
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
from shapely.geometry import Point, Polygon
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

    ids = []
    for token in expected.split(","):
        token = token.strip()
        if not token:
            continue
        ids.append(int(token))
    if not ids:
        raise ValueError("Expected at least one drone ID in --expected-drones.")
    return sorted(set(ids))


def plot_voronoi_cells(cells: Dict[int, dict], diag: dict | None) -> None:
    """Plot all Voronoi cells on a shared Matplotlib figure."""

    fig, ax = plt.subplots()
    search_radius = diag.get("search_radius") if diag else None
    coverage_pct = diag.get("coverage_pct") if diag else None
    gap_pct = diag.get("gap_pct") if diag else None
    overlap_pct = diag.get("overlap_pct") if diag else None

    for drone_id in sorted(cells):
        payload = cells[drone_id]
        vertices = payload.get("vertices_xy") or []
        centroid = payload.get("centroid_xy") or [0.0, 0.0]
        area = payload.get("cell_area_m2")

        if len(vertices) < 3:
            LOGGER.warning("Drone %d polygon has too few vertices (%d). Skipping plot.", drone_id, len(vertices))
            continue

        xs = [p[0] for p in vertices] + [vertices[0][0]]
        ys = [p[1] for p in vertices] + [vertices[0][1]]
        patch = ax.fill(xs, ys, alpha=0.3, label=f"Drone {drone_id}")
        edge_color = patch[0].get_facecolor()
        ax.plot(xs, ys, color=edge_color, linewidth=1.5)
        ax.plot(centroid[0], centroid[1], "ko", markersize=4)
        label = f"{drone_id}"
        if area is not None:
            label += f"\n{area:.0f} m²"
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
    if cells:
        ax.legend(loc="upper right")
    plt.show()


def log_diagnostics(cells: Dict[int, dict], diag: dict | None, order: Iterable[int]) -> None:
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
        payload = cells.get(drone_id)
        if payload is None:
            LOGGER.info("  Drone %d: no data received.", drone_id)
            continue
        area = payload.get("cell_area_m2", 0.0) or 0.0
        pct = 100.0 * area / circle_area if circle_area > 0 else 0.0
        mode = payload.get("mode")
        LOGGER.info("  Drone %d: %.1f m² (%.2f%%), mode=%s", drone_id, area, pct, mode)


def compute_diagnostics(cells: Dict[int, dict]) -> dict | None:
    """Build Shapely polygons and compute coverage/overlap diagnostics."""

    search_radius = None
    for payload in cells.values():
        radius = payload.get("search_radius")
        if radius is not None:
            search_radius = float(radius)
            break

    if not search_radius or search_radius <= 0:
        LOGGER.warning("No valid search radius found in payloads; skipping diagnostics.")
        return None

    circle_poly = Point(0.0, 0.0).buffer(search_radius, resolution=512)
    circle_area = circle_poly.area

    cell_polygons: Dict[int, Polygon] = {}
    for drone_id, payload in cells.items():
        poly = _payload_to_polygon(drone_id, payload)
        if poly is not None and not poly.is_empty:
            cell_polygons[drone_id] = poly

    if not cell_polygons:
        LOGGER.warning("No valid Voronoi polygons to analyze; diagnostics will be limited.")
        union_poly = Polygon()
    else:
        union_poly = unary_union(list(cell_polygons.values()))

    if union_poly.is_empty:
        covered_poly = Polygon()
    else:
        covered_poly = union_poly.intersection(circle_poly)

    covered_area = covered_poly.area
    gap_area = max(circle_area - covered_area, 0.0)

    overlap_area = 0.0
    items = list(cell_polygons.items())
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            try:
                inter = items[i][1].intersection(items[j][1]).intersection(circle_poly)
                overlap_area += inter.area
            except Exception as exc:
                LOGGER.debug("Failed to compute overlap between drones %s and %s: %s", items[i][0], items[j][0], exc)

    coverage_pct = 100.0 * covered_area / circle_area if circle_area > 0 else 0.0
    gap_pct = 100.0 * gap_area / circle_area if circle_area > 0 else 0.0
    overlap_pct = 100.0 * overlap_area / circle_area if circle_area > 0 else 0.0

    return {
        "search_radius": search_radius,
        "circle_poly": circle_poly,
        "circle_area": circle_area,
        "cell_polygons": cell_polygons,
        "covered_area": covered_area,
        "gap_area": gap_area,
        "overlap_area": overlap_area,
        "coverage_pct": coverage_pct,
        "gap_pct": gap_pct,
        "overlap_pct": overlap_pct,
    }


def _payload_to_polygon(drone_id: int, payload: dict) -> Polygon | None:
    """Convert payload vertices into a cleaned Shapely Polygon."""

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


def run_visualizer(config: VisualizerConfig) -> None:
    """Receive Voronoi cells via UDP and plot once all expected drones report in."""

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((config.listen_host, config.listen_port))
    LOGGER.info(
        "Listening for Voronoi cells on %s:%d (expected drones: %s)",
        config.listen_host,
        config.listen_port,
        ", ".join(map(str, config.expected_drones)),
    )

    expected_set = set(config.expected_drones)
    cells: Dict[int, dict] = {}

    try:
        while True:
            data, addr = sock.recvfrom(65535)
            try:
                payload = json.loads(data.decode("utf-8"))
            except json.JSONDecodeError as exc:
                LOGGER.warning("Failed to decode JSON from %s: %s", addr, exc)
                continue

            if payload.get("type") != "voronoi_cell":
                LOGGER.debug("Ignoring message with unexpected type from %s: %s", addr, payload.get("type"))
                continue

            drone_id = payload.get("drone_id")
            if drone_id is None:
                LOGGER.warning("Received Voronoi payload without drone_id from %s", addr)
                continue

            vertices = payload.get("vertices_xy") or []
            LOGGER.info(
                "Received Voronoi cell from drone_id=%s with %d vertices (mode=%s)",
                drone_id,
                len(vertices),
                payload.get("mode"),
            )
            cells[drone_id] = payload

            have_all = expected_set.issubset(cells.keys())
            LOGGER.debug("Currently have cells for drones: %s", sorted(cells.keys()))

            if config.auto_plot and have_all:
                subset = {drone_id: cells[drone_id] for drone_id in sorted(expected_set)}
                diag = compute_diagnostics(subset)
                log_diagnostics(subset, diag, sorted(expected_set))
                plot_voronoi_cells(subset, diag)
                if not config.keep_listening:
                    LOGGER.info("All expected cells plotted. Exiting.")
                    break
                LOGGER.info("Keep-listening enabled – clearing cells for next batch.")
                cells.clear()
    finally:
        sock.close()


def parse_args() -> VisualizerConfig:
    parser = argparse.ArgumentParser(description="Swarm Voronoi visualizer.")
    parser.add_argument("--listen-host", default="0.0.0.0", help="Host/IP to bind UDP listener (default 0.0.0.0).")
    parser.add_argument("--listen-port", type=int, default=62000, help="UDP port to listen on (default 62000).")
    parser.add_argument(
        "--expected-drones",
        default="1,2,3,4,5",
        help="Comma-separated list of drone IDs expected to report (default '1,2,3,4,5').",
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
