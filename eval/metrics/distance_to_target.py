"""
eval/metrics/distance_to_target.py

Measures distance from a robot body link to a scene object at episode end.
Used for: reach_obj, walk_to_obj, point_at_obj.
"""

import torch
from ..base_metric import Metric, MetricResult


class DistanceToTarget(Metric):
    """
    Computes distance between a robot link and a scene object center.

    Records the minimum distance achieved during the episode (best attempt)
    and the final distance at episode end.

    Args:
        name:            Metric name string.
        link_name:       Robot rigid body link name (e.g. "right_wrist_yaw_link").
        object_index:    Index of the target object in the scene (default 0).
        success_threshold: Distance in meters below which the episode is a success.
        use_min:         If True, success is based on min distance during episode.
                         If False, success is based on final distance only.
        use_2d:          If True, measure XY distance only (ignore Z). Useful for
                         locomotion tasks where the robot root is always above the object.
        higher_is_better: Always False for distance metrics.
    """

    higher_is_better = False

    def __init__(
        self,
        name: str,
        link_name: str,
        object_index: int = 0,
        success_threshold: float = 0.1,
        use_min: bool = False,
        use_2d: bool = False,
        overlay_label: str | None = None,
        fixed_target_pos: list[float] | None = None,
    ):
        self.name = name
        self.link_name = link_name
        self.object_index = object_index
        self.success_threshold = success_threshold
        self.use_min = use_min
        self.use_2d = use_2d
        self.overlay_label = overlay_label
        self.fixed_target_pos = fixed_target_pos  # [x, y, z] in world frame; overrides object_index

        self._link_index = None      # resolved on first update
        self._distances = []         # distance at each step
        self._final_dist = None

    def reset(self) -> None:
        self._distances = []
        self._final_dist = None
        self._link_index = None      # re-resolve each episode (safe)

    def _resolve_link_index(self, env) -> int:
        """Find the rigid body index for self.link_name in simulator ordering."""
        # Use _robot.data.body_names (IsaacLab ordering) since we index into
        # _robot.data.body_pos_w directly. NOT _body_names (ProtoMotions ordering).
        body_names = list(env.simulator._robot.data.body_names)
        if self.link_name not in body_names:
            raise ValueError(
                f"Link '{self.link_name}' not found in robot bodies.\n"
                f"Available: {body_names}"
            )
        return body_names.index(self.link_name)

    def update(self, env, scene_lib) -> None:
        if self._link_index is None:
            self._link_index = self._resolve_link_index(env)

        # Robot link position: (num_envs, 3)
        link_pos = env.simulator._robot.data.body_pos_w[:, self._link_index, :]

        # Target position: fixed point or scene object
        if self.fixed_target_pos is not None:
            obj_pos = torch.tensor(self.fixed_target_pos, dtype=link_pos.dtype, device=link_pos.device).unsqueeze(0)
        else:
            obj_pos = env.simulator._object[self.object_index].data.root_pos_w

        if self.use_2d:
            dist = torch.norm(link_pos[:, :2] - obj_pos[:, :2], dim=-1)
        else:
            dist = torch.norm(link_pos - obj_pos, dim=-1)  # (num_envs,)
        self._distances.append(dist[0].item())

    def get_overlay(self) -> tuple[str, bool] | None:
        if self.use_min or not self._distances:
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
                "final_dist": final_dist,
                "min_dist": min_dist,
                "max_dist": max(self._distances),
                "mean_dist": sum(self._distances) / len(self._distances),
                "threshold": self.success_threshold,
            },
        )
