"""
eval/metrics/walk_on_line.py

Trajectory-based metric for walk_on_green_line_avoid_obs.

Success requires both conditions checked at episode end:
  1. Robot pelvis stayed within line Y bounds at all times while within the line X range.
  2. Robot's FINAL position is past all obstacle X positions (plus margin).
     Checked at end only — briefly bouncing past an obstacle then retreating does NOT count.
"""

import torch
from ..base_metric import Metric, MetricResult


class WalkOnLineMetric(Metric):
    """
    Checks that the robot walks along the green line and clears all obstacles.

    Args:
        name:                  Metric name.
        link_name:             Robot body link to track (default: pelvis).
        line_x_min:            X coordinate where the line starts.
        line_x_max:            X coordinate where the line ends.
        line_y_half_width:     Half-width of the line in Y (robot must stay within ±this).
        obstacle_x_positions:  X positions of each obstacle center.
        obstacle_pass_margin:  How far past obstacle_x the robot must be AT EPISODE END to count.
    """

    higher_is_better = True

    def __init__(
        self,
        name: str,
        link_name: str = "pelvis",
        line_x_min: float = 0.25,
        line_x_max: float = 5.75,
        line_y_half_width: float = 0.5,
        obstacle_x_positions: tuple = (1.5, 3.0, 4.5),
        obstacle_pass_margin: float = 0.5,
    ):
        self.name = name
        self.link_name = link_name
        self.line_x_min = line_x_min
        self.line_x_max = line_x_max
        self.line_y_half_width = line_y_half_width
        self.obstacle_x_positions = list(obstacle_x_positions)
        self.obstacle_pass_margin = obstacle_pass_margin

        self._link_index = None
        self._always_on_line = True
        self._steps_on_line = 0
        self._steps_in_x_range = 0
        self._final_px = 0.0

    def reset(self) -> None:
        self._link_index = None
        self._always_on_line = True
        self._steps_on_line = 0
        self._steps_in_x_range = 0
        self._final_px = 0.0

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

        self._final_px = px  # always track final position

        # Check line compliance only while robot is within the line X range
        if self.line_x_min <= px <= self.line_x_max:
            self._steps_in_x_range += 1
            if abs(py) <= self.line_y_half_width:
                self._steps_on_line += 1
            else:
                self._always_on_line = False

    def get_overlay(self) -> tuple[str, bool] | None:
        obstacles_cleared = [
            self._final_px > obs_x + self.obstacle_pass_margin
            for obs_x in self.obstacle_x_positions
        ]
        n_cleared = sum(obstacles_cleared)
        success = self._always_on_line and n_cleared == len(self.obstacle_x_positions)
        label = f"Obs cleared: {n_cleared}/{len(self.obstacle_x_positions)} | On line: {self._always_on_line}"
        return label, success

    def compute(self) -> MetricResult:
        # Success requires robot's FINAL position to be past every obstacle
        obstacles_cleared = [
            self._final_px > obs_x + self.obstacle_pass_margin
            for obs_x in self.obstacle_x_positions
        ]
        n_cleared = sum(obstacles_cleared)
        all_cleared = n_cleared == len(self.obstacle_x_positions)
        line_compliance = (
            self._steps_on_line / self._steps_in_x_range
            if self._steps_in_x_range > 0
            else 1.0
        )
        success = self._always_on_line and all_cleared

        return MetricResult(
            value=float(n_cleared) / len(self.obstacle_x_positions),  # fraction cleared (0–1)
            success=success,
            info={
                "obstacles_cleared": n_cleared,
                "total_obstacles": len(self.obstacle_x_positions),
                "always_on_line": self._always_on_line,
                "line_compliance": round(line_compliance, 4),
                "final_x": round(self._final_px, 3),
                "steps_in_x_range": self._steps_in_x_range,
            },
        )
