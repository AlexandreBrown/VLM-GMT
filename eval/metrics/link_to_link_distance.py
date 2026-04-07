"""
eval/metrics/link_to_link_distance.py

Measures distance between two robot body links.
Used for tasks like "touch left leg with right hand".
"""

import torch
from ..base_metric import Metric, MetricResult


class LinkToLinkDistance(Metric):
    """
    Computes distance between two robot body links at episode end.

    Args:
        name:              Metric name string.
        link_a:            First robot body link name (e.g. "right_rubber_hand").
        link_b:            Second robot body link name (e.g. "left_knee").
        success_threshold: Distance in meters below which the episode is a success.
        use_min:           If True, success based on min distance during episode.
    """

    higher_is_better = False

    def __init__(
        self,
        name: str,
        link_a: str,
        link_b: str,
        success_threshold: float = 0.15,
        use_min: bool = False,
        overlay_label: str | None = None,
    ):
        self.name = name
        self.link_a = link_a
        self.link_b = link_b
        self.success_threshold = success_threshold
        self.use_min = use_min
        self.overlay_label = overlay_label

        self._index_a = None
        self._index_b = None
        self._distances = []

    def reset(self) -> None:
        self._index_a = None
        self._index_b = None
        self._distances = []

    def _resolve_link_index(self, env, link_name: str) -> int:
        body_names = list(env.simulator._robot.data.body_names)
        if link_name not in body_names:
            raise ValueError(
                f"Link '{link_name}' not found. Available: {body_names}"
            )
        return body_names.index(link_name)

    def update(self, env, scene_lib) -> None:
        if self._index_a is None:
            self._index_a = self._resolve_link_index(env, self.link_a)
            self._index_b = self._resolve_link_index(env, self.link_b)

        pos_a = env.simulator._robot.data.body_pos_w[:, self._index_a, :]
        pos_b = env.simulator._robot.data.body_pos_w[:, self._index_b, :]
        dist = torch.norm(pos_a - pos_b, dim=-1)
        self._distances.append(dist[0].item())

    def get_overlay(self) -> tuple[str, bool] | None:
        if not self._distances:
            return None
        d = self._distances[-1]
        label = self.overlay_label or self.name
        return f"{label}: {d:.3f}m", d < self.success_threshold

    def compute(self) -> MetricResult:
        if not self._distances:
            return MetricResult(value=float("inf"), success=False)

        final_dist = self._distances[-1]
        min_dist = min(self._distances)
        eval_dist = min_dist if self.use_min else final_dist

        return MetricResult(
            value=eval_dist,
            success=eval_dist < self.success_threshold,
            info={
                "final_dist": round(final_dist, 4),
                "min_dist": round(min_dist, 4),
                "threshold": self.success_threshold,
            },
        )
