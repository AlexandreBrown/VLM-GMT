"""
eval/metrics/walk_on_line.py

Trajectory-based metric for walk_on_green_line_avoid_obs.

Success requires BOTH:
  1. Robot pelvis stays within line Y bounds at all times (while in line X range).
  2. Robot avoids EACH obstacle: when pelvis is within ±x_window of an obstacle's
     x center, its lateral separation |py - obs_y| must stay >= avoidance_min_dist.

Forward progress alone does NOT count as avoidance because the GMT policy has
no collision response: the robot clips straight through fixed obstacles.
"""

import torch
from ..base_metric import Metric, MetricResult


class WalkOnLineMetric(Metric):
    """
    Checks that the robot walks along the green line and navigates around
    each obstacle with sufficient lateral clearance.

    Args:
        name:                Metric name.
        link_name:           Robot body link to track (default: pelvis).
        line_x_min/max:      X range of the line.
        line_y_half_width:   Half-width of the line in Y (±this).
        obstacle_positions:  List of (x, y) obstacle center positions in IsaacLab frame.
        avoidance_min_dist:  Min |py - obs_y| required when at the obstacle's x range.
        x_window:            Half-width of the x range around each obstacle to check avoidance.
    """

    higher_is_better = True

    def __init__(
        self,
        name: str,
        link_name: str = "pelvis",
        line_x_min: float = 0.25,
        line_x_max: float = 5.75,
        line_y_half_width: float = 0.5,
        obstacle_positions: list[tuple[float, float]] = ((1.5, 0.2), (3.0, -0.2), (4.5, 0.15)),
        avoidance_min_dist: float = 0.3,
        x_window: float = 0.5,
    ):
        self.name = name
        self.link_name = link_name
        self.line_x_min = line_x_min
        self.line_x_max = line_x_max
        self.line_y_half_width = line_y_half_width
        self.obstacle_positions = list(obstacle_positions)
        self.avoidance_min_dist = avoidance_min_dist
        self.x_window = x_window

        self._link_index = None
        self._always_on_line = True
        self._steps_on_line = 0
        self._steps_in_x_range = 0
        # Per-obstacle: None = robot hasn't entered x window yet, float = min lateral dist seen
        self._obs_min_lateral = [None] * len(self.obstacle_positions)

    def reset(self) -> None:
        self._link_index = None
        self._always_on_line = True
        self._steps_on_line = 0
        self._steps_in_x_range = 0
        self._obs_min_lateral = [None] * len(self.obstacle_positions)

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

        # Line compliance: check Y bounds while within line X range
        if self.line_x_min <= px <= self.line_x_max:
            self._steps_in_x_range += 1
            if abs(py) <= self.line_y_half_width:
                self._steps_on_line += 1
            else:
                self._always_on_line = False

        # Per-obstacle avoidance: track min lateral distance while in each obstacle's x window
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
        success = self._always_on_line and all(avoided)
        label = f"Obs avoided: {n}/{len(self.obstacle_positions)} | On line: {self._always_on_line}"
        return label, success

    def compute(self) -> MetricResult:
        obstacles_avoided = []
        for i, ml in enumerate(self._obs_min_lateral):
            if ml is None:
                # Never entered this obstacle's x window: not avoided
                obstacles_avoided.append(False)
            else:
                obstacles_avoided.append(ml >= self.avoidance_min_dist)

        n_avoided = sum(obstacles_avoided)
        all_avoided = all(obstacles_avoided)
        line_compliance = (
            self._steps_on_line / self._steps_in_x_range
            if self._steps_in_x_range > 0
            else 1.0
        )
        success = self._always_on_line and all_avoided

        return MetricResult(
            value=float(n_avoided) / len(self.obstacle_positions),
            success=success,
            info={
                "obstacles_avoided": n_avoided,
                "total_obstacles": len(self.obstacle_positions),
                "per_obstacle_min_lateral": [
                    round(ml, 3) if ml is not None else None
                    for ml in self._obs_min_lateral
                ],
                "avoidance_threshold": self.avoidance_min_dist,
                "always_on_line": self._always_on_line,
                "line_compliance": round(line_compliance, 4),
            },
        )
