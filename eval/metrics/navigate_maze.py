"""
eval/metrics/navigate_maze.py

Simple metric for navigate_maze: fraction of walls the robot passed.

A wall is "cleared" if the pelvis final x-position is past the wall's x
(plus a small margin). Wall world positions are read directly from the
simulator via scene object indices, so everything stays in world frame.
"""

import math
from ..base_metric import Metric, MetricResult


class NavigateMazeMetric(Metric):
    """
    Success = all walls cleared. Score = fraction of walls cleared.

    Args:
        name:             Metric name.
        link_name:        Robot body link to track (default: pelvis).
        obstacle_indices: Scene object indices of the walls to clear.
        pass_x_margin:    How far past a wall the pelvis must be (meters).
    """

    higher_is_better = True

    def __init__(
        self,
        name: str = "navigate_maze",
        link_name: str = "pelvis",
        obstacle_indices: tuple[int, ...] = (4, 5),
        pass_x_margin: float = 0.5,
    ):
        self.name = name
        self.link_name = link_name
        self.obstacle_indices = list(obstacle_indices)
        self.pass_x_margin = pass_x_margin

        self._link_index = None
        self._wall_xs = None   # world-frame wall x positions
        self._origin_x = None  # robot root world x at start
        self._final_px = 0.0
        self._prev_pos = None
        self._distance_traveled = 0.0

    def reset(self) -> None:
        self._link_index = None
        self._wall_xs = None
        self._origin_x = None
        self._final_px = 0.0
        self._prev_pos = None
        self._distance_traveled = 0.0

    def _resolve_link_index(self, env) -> int:
        body_names = list(env.simulator._robot.data.body_names)
        if self.link_name not in body_names:
            raise ValueError(
                f"Link '{self.link_name}' not found. Available: {body_names}"
            )
        return body_names.index(self.link_name)

    def update(self, env, scene_lib) -> None:
        if self._link_index is None:
            self._link_index = self._resolve_link_index(env)

        if self._wall_xs is None:
            self._wall_xs = [
                float(env.simulator._object[i].data.root_pos_w[0, 0])
                for i in self.obstacle_indices
            ]
            self._origin_x = float(env.simulator._robot.data.root_pos_w[0, 0])

        pos = env.simulator._robot.data.body_pos_w[0, self._link_index]
        px = float(pos[0])
        py = float(pos[1])

        self._final_px = px

        if self._prev_pos is not None:
            dx = px - self._prev_pos[0]
            dy = py - self._prev_pos[1]
            self._distance_traveled += math.sqrt(dx * dx + dy * dy)
        self._prev_pos = (px, py)

    def get_overlay(self) -> tuple[str, bool] | None:
        if self._wall_xs is None:
            return None
        cleared = sum(1 for wx in self._wall_xs if self._final_px > wx + self.pass_x_margin)
        total = len(self._wall_xs)
        return f"Walls cleared: {cleared}/{total}", cleared == total

    def compute(self) -> MetricResult:
        if self._wall_xs is None:
            return MetricResult(value=0.0, success=False)

        total = len(self._wall_xs)
        cleared = sum(1 for wx in self._wall_xs if self._final_px > wx + self.pass_x_margin)
        score = cleared / total

        final_local_x = self._final_px - self._origin_x
        return MetricResult(
            value=score,
            success=cleared == total,
            info={
                "walls_cleared": cleared,
                "total_walls": total,
                "wall_world_xs": [round(w, 3) for w in self._wall_xs],
                "final_px_world": round(self._final_px, 3),
                "final_px_local": round(final_local_x, 3),
                "pass_x_margin": self.pass_x_margin,
                "distance_traveled": round(self._distance_traveled, 3),
            },
        )
