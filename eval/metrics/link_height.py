"""
eval/metrics/link_height.py

Checks whether a robot link's Z position exceeds a height threshold.
Used for tasks like "raise hand up and keep it up".
"""

import torch
from ..base_metric import Metric, MetricResult


class LinkHeightMetric(Metric):
    """
    Success: link Z > height_threshold at episode end.

    Args:
        name:              Metric name.
        link_name:         Robot body link to track.
        height_threshold:  Z threshold (meters) for success.
        check_below:       If True, success = z < threshold. If False, success = z > threshold.
        use_mean:          If True, success based on mean Z over episode (sustained).
                           If False, success based on final Z only.
    """

    higher_is_better = True

    def __init__(
        self,
        name: str,
        link_name: str,
        height_threshold: float,
        check_below: bool = False,
        use_mean: bool = False,
        overlay_label: str | None = None,
    ):
        self.name = name
        self.link_name = link_name
        self.height_threshold = height_threshold
        self.check_below = check_below
        self.use_mean = use_mean
        self.overlay_label = overlay_label
        self.higher_is_better = not check_below

        self._link_index = None
        self._heights = []

    def reset(self) -> None:
        self._link_index = None
        self._heights = []

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

        z = float(env.simulator._robot.data.body_pos_w[0, self._link_index, 2])
        self._heights.append(z)

    def get_overlay(self) -> tuple[str, bool] | None:
        if not self._heights:
            return None
        z = self._heights[-1]
        label = self.overlay_label or self.name
        op = "<" if self.check_below else ">"
        ok = z < self.height_threshold if self.check_below else z > self.height_threshold
        return f"{label}: {z:.2f}m (need {op}{self.height_threshold:.2f})", ok

    def compute(self) -> MetricResult:
        if not self._heights:
            return MetricResult(value=0.0, success=False)

        final_z = self._heights[-1]
        mean_z = sum(self._heights) / len(self._heights)
        max_z = max(self._heights)
        eval_z = mean_z if self.use_mean else final_z

        success = eval_z < self.height_threshold if self.check_below else eval_z > self.height_threshold
        return MetricResult(
            value=eval_z,
            success=success,
            info={
                "final_z": round(final_z, 4),
                "mean_z": round(mean_z, 4),
                "max_z": round(max_z, 4),
                "threshold": self.height_threshold,
            },
        )
