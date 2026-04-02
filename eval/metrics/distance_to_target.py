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
    ):
        self.name = name
        self.link_name = link_name
        self.object_index = object_index
        self.success_threshold = success_threshold
        self.use_min = use_min

        self._link_index = None      # resolved on first update
        self._distances = []         # distance at each step
        self._final_dist = None

    def reset(self) -> None:
        self._distances = []
        self._final_dist = None
        self._link_index = None      # re-resolve each episode (safe)

    def _resolve_link_index(self, env) -> int:
        """Find the rigid body index for self.link_name."""
        body_names = env.simulator._body_names  # list of str
        if self.link_name not in body_names:
            raise ValueError(
                f"Link '{self.link_name}' not found in robot bodies.\n"
                f"Available: {body_names}"
            )
        return body_names.index(self.link_name)

    def update(self, env, scene_lib) -> None:
        if self._link_index is None:
            self._link_index = self._resolve_link_index(env)

        # get_bodies_state() returns RobotState with rigid_body_pos: (num_envs, num_bodies, 3)
        bodies_state = env.simulator.get_bodies_state()
        link_pos = bodies_state.rigid_body_pos[:, self._link_index, :]  # (num_envs, 3)

        # Object world position: local translation + scene offset (env tiling)
        # _object_translations stores local coords; _scene_offsets stores per-env (x, y) offset
        obj_local = scene_lib._object_translations[self.object_index].clone()  # (3,)
        scene_offset = scene_lib._scene_offsets[0]  # (x, y) for env 0
        obj_pos = obj_local.to(link_pos.device)
        obj_pos[0] += scene_offset[0]
        obj_pos[1] += scene_offset[1]

        # Diagnostic — printed once per episode on first step
        if not self._distances:
            print(f"[DistanceToTarget] link_pos[0]     : {link_pos[0].tolist()}")
            print(f"[DistanceToTarget] obj_local       : {obj_local.tolist()}")
            print(f"[DistanceToTarget] scene_offset    : {scene_offset}")
            print(f"[DistanceToTarget] obj_pos (world) : {obj_pos.tolist()}")

        dist = torch.norm(link_pos - obj_pos, dim=-1)  # (num_envs,)
        self._distances.append(dist[0].item())         # env 0

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
                "mean_dist": sum(self._distances) / len(self._distances),
                "threshold": self.success_threshold,
            },
        )
