"""
eval/metrics/navigate_maze.py

Trajectory-based metric for navigate_maze task.

Success requires:
  1. Robot avoids EACH wall (lateral separation check when at wall's x range).
  2. Robot's final x position is past the last wall by pass_x_margin.

Also reports:
  - walls_cleared: how many walls the robot got past (by final x)
  - distance_traveled: total path length during the episode
  - distance_to_success: how far the robot was from the success x threshold
"""

import math
import torch
from ..base_metric import Metric, MetricResult


class NavigateMazeMetric(Metric):
    """
    Checks that the robot navigates around walls and reaches past them.

    Args:
        name:                Metric name.
        link_name:           Robot body link to track (default: pelvis).
        line_x_min/max:      X range for line compliance check.
        line_y_half_width:   Half-width of the line in Y. Set very large to disable.
        obstacle_positions:  List of (x, y) wall inner-edge positions in IsaacLab frame.
        avoidance_min_dist:  Min |py - obs_y| required when at the wall's x range.
        x_window:            Half-width of the x range around each wall to check.
        pass_x_margin:       How far past the LAST wall the robot must be at episode end.
    """

    higher_is_better = True

    def __init__(
        self,
        name: str,
        link_name: str = "pelvis",
        line_x_min: float = 0.0,
        line_x_max: float = 10.0,
        line_y_half_width: float = 100.0,
        obstacle_positions: list[tuple[float, float]] = ((1.5, 0.1), (3.0, -0.1)),
        avoidance_min_dist: float = 0.3,
        x_window: float = 0.5,
        pass_x_margin: float = 0.5,
    ):
        self.name = name
        self.link_name = link_name
        self.line_x_min = line_x_min
        self.line_x_max = line_x_max
        self.line_y_half_width = line_y_half_width
        self.obstacle_positions = list(obstacle_positions)
        self.avoidance_min_dist = avoidance_min_dist
        self.x_window = x_window
        self.pass_x_margin = pass_x_margin
        self._last_obs_x = max(ox for ox, _ in obstacle_positions)
        self._success_x = self._last_obs_x + pass_x_margin

        self._link_index = None
        self._always_on_line = True
        self._steps_on_line = 0
        self._steps_in_x_range = 0
        self._obs_min_lateral = [None] * len(self.obstacle_positions)
        self._final_px = 0.0
        self._prev_pos = None
        self._distance_traveled = 0.0

    def reset(self) -> None:
        self._link_index = None
        self._always_on_line = True
        self._steps_on_line = 0
        self._steps_in_x_range = 0
        self._obs_min_lateral = [None] * len(self.obstacle_positions)
        self._final_px = 0.0
        self._prev_pos = None
        self._distance_traveled = 0.0

    def _resolve_link_index(self, env) -> int:
        body_names = env.simulator._body_names
        if self.link_name not in body_names:
            raise ValueError(
                f"Link '{self.link_name}' not found. Available: {body_names}"
            )
        return body_names.index(self.link_name)

    def update(self, env, scene_lib) -> None:
        if self._link_index is None:
            self._link_index = self._resolve_link_index(env)

        pos = env.simulator._robot.data.body_pos_w[0, self._link_index]
        px = float(pos[0])
        py = float(pos[1])

        self._final_px = px

        # Distance traveled (2D path length)
        if self._prev_pos is not None:
            dx = px - self._prev_pos[0]
            dy = py - self._prev_pos[1]
            self._distance_traveled += math.sqrt(dx * dx + dy * dy)
        self._prev_pos = (px, py)

        # Line compliance (disabled when line_y_half_width is very large)
        if self.line_x_min <= px <= self.line_x_max:
            self._steps_in_x_range += 1
            if abs(py) <= self.line_y_half_width:
                self._steps_on_line += 1
            else:
                self._always_on_line = False

        # Per-wall avoidance
        for i, (obs_x, obs_y) in enumerate(self.obstacle_positions):
            if abs(px - obs_x) <= self.x_window:
                lateral = abs(py - obs_y)
                if self._obs_min_lateral[i] is None:
                    self._obs_min_lateral[i] = lateral
                else:
                    self._obs_min_lateral[i] = min(self._obs_min_lateral[i], lateral)

    def get_overlay(self) -> tuple[str, bool] | None:
        avoided = [
            (ml is not None and ml >= self.avoidance_min_dist)
            for ml in self._obs_min_lateral
        ]
        n = sum(avoided)
        passed = self._final_px > self._success_x
        success = all(avoided) and passed
        label = f"Avoided: {n}/{len(self.obstacle_positions)} | x={self._final_px:.1f}/{self._success_x:.1f}"
        return label, success

    def compute(self) -> MetricResult:
        obstacles_avoided = []
        for ml in self._obs_min_lateral:
            if ml is None:
                obstacles_avoided.append(False)
            else:
                obstacles_avoided.append(ml >= self.avoidance_min_dist)

        n_avoided = sum(obstacles_avoided)
        all_avoided = all(obstacles_avoided)
        passed_last = self._final_px > self._success_x
        success = all_avoided and passed_last

        # Walls cleared: count how many walls the robot got past (by final x)
        sorted_obs_x = sorted(ox for ox, _ in self.obstacle_positions)
        walls_cleared = sum(1 for ox in sorted_obs_x if self._final_px > ox + self.pass_x_margin)

        # Distance to success threshold
        distance_to_success = max(0.0, self._success_x - self._final_px)

        return MetricResult(
            value=float(n_avoided) / len(self.obstacle_positions),
            success=success,
            info={
                "obstacles_avoided": n_avoided,
                "total_obstacles": len(self.obstacle_positions),
                "walls_cleared": walls_cleared,
                "per_wall_clearance": [
                    round(ml, 3) if ml is not None else None
                    for ml in self._obs_min_lateral
                ],
                "avoidance_threshold": self.avoidance_min_dist,
                "final_x": round(self._final_px, 3),
                "success_x_threshold": self._success_x,
                "distance_to_success": round(distance_to_success, 3),
                "distance_traveled": round(self._distance_traveled, 3),
                "passed_last_wall": passed_last,
            },
        )
