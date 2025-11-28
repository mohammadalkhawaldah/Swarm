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
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union


LOGGER = logging.getLogger("swarm_voronoi_visualizer")
DRONE_COLORS = ["#4c78a8", "#f58518", "#54a24b", "#b079a7", "#e45756", "#72b7b2", "#f28e2b", "#ff9da7"]


@dataclass
class VisualizerConfig:
    listen_host: str
    listen_port: int
    expected_drones: List[int]
    auto_plot: bool
    keep_listening: bool
    plot_interval: float


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


def _plot_polygon(ax, poly: Polygon, *, label: Optional[str] = None, color=None, alpha: float = 0.35):
    """Helper to draw polygons (including MultiPolygons) with consistent styling."""

    if poly.is_empty:
        return None

    polygons = [poly]
    if poly.geom_type == "MultiPolygon":
        polygons = [p for p in poly.geoms if not p.is_empty]
        if not polygons:
            return None

    face_color = color
    for idx, polygon in enumerate(polygons):
        vertices = list(polygon.exterior.coords)
        xs = [p[0] for p in vertices]
        ys = [p[1] for p in vertices]
        patch = ax.fill(
            xs,
            ys,
            alpha=alpha,
            color=face_color,
            label=label if idx == 0 else None,
        )
        face_color = face_color or patch[0].get_facecolor()
        ax.plot(xs, ys, color=face_color, linewidth=2.0)
    return face_color


def plot_regions(
    polygons: Dict[int, Polygon],
    diag: dict | None,
    block_display: bool = True,
    axes: Optional[Tuple["plt.Figure", "plt.Axes"]] = None,
    explored_overlay: Optional[Polygon] = None,
) -> Tuple["plt.Figure", "plt.Axes"]:
    """Plot all Voronoi cells on a shared Matplotlib figure.

    If ``axes`` is provided, the plot is redrawn within the same figure/axes
    (useful for live refreshing). Returns the figure/axes used.
    """

    if axes:
        fig, ax = axes
        ax.clear()
    else:
        fig, ax = plt.subplots()

    search_radius = diag.get("search_radius") if diag else None
    coverage_pct = diag.get("coverage_pct") if diag else None
    gap_pct = diag.get("gap_pct") if diag else None
    overlap_pct = diag.get("overlap_pct") if diag else None

    if explored_overlay and not explored_overlay.is_empty:
        _plot_polygon(ax, explored_overlay, label="Explored", color="#fdfdfd", alpha=1.0)

    for drone_id in sorted(polygons):
        poly = polygons[drone_id]
        if poly.is_empty:
            continue
        if explored_overlay and not explored_overlay.is_empty:
            try:
                poly = poly.difference(explored_overlay)
            except Exception:
                pass
            if poly.is_empty:
                continue
        color = DRONE_COLORS[(drone_id - 1) % len(DRONE_COLORS)]
        _plot_polygon(ax, poly, label=f"Drone {drone_id}", color=color)
        centroid = (poly.centroid.x, poly.centroid.y)
        area = poly.area
        ax.plot(centroid[0], centroid[1], "ko", markersize=4)
        ax.text(centroid[0], centroid[1], f"{drone_id}\n{area:.0f} m^2", fontsize=8, ha="center", va="center")

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
    if block_display:
        plt.show(block=True)
    else:
        fig.canvas.draw()
        fig.canvas.flush_events()

    return fig, ax


def plot_progress_map(
    explored_poly: Optional[Polygon],
    search_radius: Optional[float],
    summary: Optional[dict],
    axes: Tuple["plt.Figure", "plt.Axes"],
) -> None:
    """Draw the overall coverage progress (explored vs remaining)."""

    fig, ax = axes
    if not search_radius or search_radius <= 0:
        ax.set_title("Coverage progress (waiting for data)")
        fig.canvas.draw()
        fig.canvas.flush_events()
        return
    ax.clear()

    circle_poly = Point(0.0, 0.0).buffer(search_radius, resolution=512)
    _plot_polygon(ax, circle_poly, label="Search area", color="#ddeeff", alpha=0.3)

    if explored_poly and not explored_poly.is_empty:
        _plot_polygon(ax, explored_poly, label="Explored", color="#888888", alpha=0.5)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    explored_pct = summary["explored_pct"] if summary else 0.0
    remaining = summary["remaining"] if summary else 0
    total = summary["total"] if summary else 0
    explored_cells = total - remaining if total else 0
    ax.set_title(f"Coverage progress: explored={explored_pct:.2f}% ({explored_cells}/{total} cells)")
    ax.grid(True)
    fig.canvas.draw()
    fig.canvas.flush_events()


def log_diagnostics(polygons: Dict[int, Polygon], diag: dict | None, order: Iterable[int]) -> None:
    """Print coverage/overlap diagnostics and per-drone percentages."""

    if not diag:
        LOGGER.info("Swarm Voronoi diagnostics unavailable (missing search radius or polygons).")
        return

    circle_area = diag["circle_area"]
    LOGGER.info("=== Swarm Voronoi diagnostics (local XY) ===")
    LOGGER.info("Search radius         : %.1f m", diag["search_radius"])
    LOGGER.info("Circle area           : %.1f m^2 (100.0%%)", circle_area)

    coverage_info = diag.get("coverage_info")
    if coverage_info:
        explored = coverage_info.get("explored", 0)
        total = coverage_info.get("total", 0)
        explored_pct = coverage_info.get("explored_pct", 0.0)
        remaining = coverage_info.get("remaining", 0)
        remaining_pct = coverage_info.get("remaining_pct", 0.0)
        LOGGER.info(
            "Explored grid         : %.2f%% (%d / %d cells)",
            explored_pct,
            explored,
            total,
        )
        LOGGER.info(
            "Remaining cells       : %d (%.2f%%)",
            remaining,
            remaining_pct,
        )
    LOGGER.info(
        "Remaining plan area   : %.1f m^2 (%.2f%%)",
        diag["remaining_area"],
        diag["remaining_pct"],
    )
    LOGGER.info(
        "Overlap between cells : %.1f m^2 (%.4f%%)",
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
        LOGGER.info("  Drone %d: %.1f m^2 (%.2f%%)", drone_id, area, pct)


def compute_diagnostics(
    polygons: Dict[int, Polygon],
    search_radius: float | None,
    coverage_summary: dict | None = None,
    explored_poly: Polygon | None = None,
) -> dict | None:
    """Compute diagnostics for polygons plus optional coverage summary."""

    if not search_radius or search_radius <= 0:
        LOGGER.warning("No valid search radius provided; skipping diagnostics.")
        return None

    circle_poly = Point(0.0, 0.0).buffer(search_radius, resolution=512)
    circle_area = circle_poly.area

    union_poly = unary_union(list(polygons.values())) if polygons else Polygon()
    remaining_poly = union_poly.intersection(circle_poly) if not union_poly.is_empty else Polygon()
    remaining_area = remaining_poly.area
    remaining_pct = remaining_area / circle_area * 100.0 if circle_area > 0 else 0.0

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

    coverage_pct = coverage_summary["explored_pct"] if coverage_summary else (100.0 - remaining_pct)
    gap_pct = 100.0 - coverage_pct
    overlap_pct = overlap_area / circle_area * 100.0 if circle_area > 0 else 0.0

    return {
        "search_radius": search_radius,
        "circle_area": circle_area,
        "remaining_area": remaining_area,
        "remaining_pct": remaining_pct,
        "overlap_area": overlap_area,
        "overlap_pct": overlap_pct,
        "coverage_pct": coverage_pct,
        "gap_pct": gap_pct,
        "coverage_info": coverage_summary,
        "explored_poly": explored_poly,
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


def _polygon_from_cells(origin_xy, cell_size, cells) -> Optional[Polygon]:
    """Convert an origin + cell list into a polygon."""

    if origin_xy is None or cell_size is None or not cells:
        return None
    try:
        origin_x = float(origin_xy[0])
        origin_y = float(origin_xy[1])
        cell_size = float(cell_size)
    except (TypeError, ValueError, IndexError):
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
        LOGGER.warning("Failed to build polygon from cell list: %s", exc)
        return None


def _polygon_from_assignment(payload: dict) -> Polygon | None:
    """Convert a region_assignment payload into a polygon via union of grid cells."""

    origin = payload.get("origin_xy")
    cell_size = payload.get("cell_size")
    cells = payload.get("cells") or []
    return _polygon_from_cells(origin, cell_size, cells)


def run_visualizer(config: VisualizerConfig) -> None:
    """Receive Voronoi/assignment packets and plot using dynamic drone sets."""

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((config.listen_host, config.listen_port))
    sock.settimeout(1.0)
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
    latest_summary: Optional[dict] = None
    latest_explored_poly: Optional[Polygon] = None
    last_assignment_plot: Optional[float] = None
    last_progress_plot: Optional[float] = None

    plt.ion()
    assignment_axes = plt.subplots()
    try:
        assignment_axes[0].canvas.manager.set_window_title("Voronoi assignments")
    except Exception:
        pass
    progress_axes = plt.subplots()
    try:
        progress_axes[0].canvas.manager.set_window_title("Coverage progress")
    except Exception:
        pass
    plt.show(block=False)

    def emit_assignment_plot(reason: str) -> bool:
        nonlocal last_assignment_plot
        if not polygons:
            return False
        order = sorted(polygons.keys())
        display_subset: Dict[int, Polygon] = {
            did: poly for did, poly in polygons.items() if not poly.is_empty
        }
        if not display_subset:
            return False
        diag = compute_diagnostics(display_subset, latest_radius, latest_summary, latest_explored_poly)
        if not diag:
            return False
        LOGGER.info("Rendering plot (%s) for drones %s.", reason, order)
        log_diagnostics(display_subset, diag, order)
        plot_regions(
            display_subset,
            diag,
            block_display=False,
            axes=assignment_axes,
            explored_overlay=latest_explored_poly,
        )
        last_assignment_plot = time.time()
        return True

    try:
        while True:
            data = None
            addr = None
            try:
                data, addr = sock.recvfrom(65535)
            except socket.timeout:
                pass

            forced_reason = None
            if data:
                try:
                    payload = json.loads(data.decode("utf-8"))
                except json.JSONDecodeError as exc:
                    LOGGER.warning("Failed to decode JSON from %s: %s", addr, exc)
                    payload = None

                if payload:
                    msg_type = payload.get("msg_type") or payload.get("type") or ""
                    drone_id = payload.get("drone_id")
                    if drone_id is None and msg_type != "coverage_snapshot":
                        LOGGER.warning("Received payload without drone_id from %s", addr)
                        payload = None

                if payload:
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
                        latest_summary = None
                        latest_explored_poly = None
                        last_assignment_plot = None
                        last_progress_plot = None
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
                        summary_candidate = payload.get("coverage_summary")
                        if summary_candidate:
                            latest_summary = summary_candidate
                        explored_candidate = payload.get("explored_cells") or None
                        if explored_candidate:
                            latest_explored_poly = _polygon_from_cells(
                                payload.get("origin_xy"),
                                payload.get("cell_size"),
                                explored_candidate,
                            )
                        LOGGER.info(
                            "Received region assignment from drone_id=%s with %d cells.",
                            drone_id,
                            len(payload.get("cells") or []),
                        )
                        forced_reason = "assignment"
                    elif msg_type == "coverage_snapshot":
                        latest_summary = payload.get("coverage_summary")
                        latest_radius = payload.get("search_radius", latest_radius)
                        latest_explored_poly = _polygon_from_cells(
                            payload.get("origin_xy"),
                            payload.get("cell_size"),
                            payload.get("explored_cells") or [],
                        )
                        LOGGER.debug("Received coverage snapshot with %d cells.", len(payload.get("explored_cells") or []))
                        forced_reason = "coverage"
                        polygon = None
                    else:
                        LOGGER.debug("Ignoring message with unexpected type '%s' from %s", msg_type, addr)
                        polygon = None

                    if polygon is not None and not polygon.is_empty:
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
                    else:
                        polygon = None

            now = time.time()
            should_plot = False
            reason = ""
            have_all = current_expected.issubset(polygons.keys()) if current_expected else bool(polygons)
            if forced_reason == "assignment" and config.auto_plot and have_all:
                should_plot = True
                reason = "all-assignments"

            progress_due = False
            if latest_radius is not None:
                if forced_reason == "coverage":
                    progress_due = True
                elif config.plot_interval > 0 and (
                    last_progress_plot is None or (now - last_progress_plot) >= config.plot_interval
                ):
                    progress_due = True
            if progress_due:
                plot_progress_map(latest_explored_poly, latest_radius, latest_summary, progress_axes)
                last_progress_plot = now

            if should_plot and emit_assignment_plot(reason):
                if not config.keep_listening and reason == "all-assignments":
                    LOGGER.info("All expected cells plotted. Exiting.")
                    break
    finally:
        sock.close()
        if not config.keep_listening:
            plt.ioff()
            plt.show()


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
        "--plot-interval",
        type=float,
        default=5.0,
        help="Seconds between periodic map snapshots (0 to disable interval plotting).",
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
        plot_interval=max(0.0, float(args.plot_interval)),
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
