"""
eval/metrics/kneel_down.py

Composite metric for kneel_down_1_knee task.
Success requires ALL of:
  1. Left knee close to ground (z < knee_height_threshold)
  2. Left and right hands close to each other (dist < hands_close_threshold)
  3. Both hands close to left knee (dist < hands_to_knee_threshold)
  4. Pelvis low (z < pelvis_height_threshold)
"""

import torch
from ..base_metric import Metric, MetricResult


class KneelDownMetric(Metric):
    """
    Composite kneeling metric. All sub-conditions must hold at episode end.

    Args:
        knee_height_threshold:   Left knee z must be below this (meters).
        hands_close_threshold:   Max distance between left and right hands.
        hands_to_knee_threshold: Max distance from each hand to left knee.
        pelvis_height_threshold: Pelvis z must be below this (meters).
    """

    name = "kneel_down"
    higher_is_better = False

    def __init__(
        self,
        knee_height_threshold: float = 0.35,
        hands_close_threshold: float = 0.20,
        hands_to_knee_threshold: float = 0.25,
        pelvis_height_threshold: float = 0.55,
    ):
        self.knee_height_threshold = knee_height_threshold
        self.hands_close_threshold = hands_close_threshold
        self.hands_to_knee_threshold = hands_to_knee_threshold
        self.pelvis_height_threshold = pelvis_height_threshold

        self._indices = None  # resolved on first update
        self._snapshots = []  # store per-step values

    def reset(self) -> None:
        self._indices = None
        self._snapshots = []

    def _resolve_indices(self, env) -> dict[str, int]:
        body_names = list(env.simulator._robot.data.body_names)
        needed = {
            "left_knee": "left_knee_link",
            "right_knee": "right_knee_link",
            "right_hand": "right_rubber_hand",
            "left_hand": "left_rubber_hand",
            "pelvis": "pelvis",
        }
        indices = {}
        for key, link_name in needed.items():
            if link_name not in body_names:
                raise ValueError(
                    f"Link '{link_name}' not found. Available: {body_names}"
                )
            indices[key] = body_names.index(link_name)
        return indices

    def update(self, env, scene_lib) -> None:
        if self._indices is None:
            self._indices = self._resolve_indices(env)

        pos = env.simulator._robot.data.body_pos_w[0]  # (num_bodies, 3)
        self._snapshots.append({
            "left_knee_z": float(pos[self._indices["left_knee"], 2]),
            "right_knee_z": float(pos[self._indices["right_knee"], 2]),
            "left_hand_z": float(pos[self._indices["left_hand"], 2]),
            "right_hand_z": float(pos[self._indices["right_hand"], 2]),
            "pelvis_z": float(pos[self._indices["pelvis"], 2]),
            "hands_dist": float(torch.norm(
                pos[self._indices["left_hand"]] - pos[self._indices["right_hand"]]
            )),
            "left_hand_to_knee": float(torch.norm(
                pos[self._indices["left_hand"]] - pos[self._indices["left_knee"]]
            )),
            "right_hand_to_knee": float(torch.norm(
                pos[self._indices["right_hand"]] - pos[self._indices["left_knee"]]
            )),
        })

    def get_overlay(self) -> tuple[str, bool] | None:
        if not self._snapshots:
            return None
        s = self._snapshots[-1]
        checks = self._check(s)
        n_pass = sum(checks.values())
        return f"Kneel: {n_pass}/{len(checks)} conditions met", n_pass == len(checks)

    def _check(self, s: dict) -> dict[str, bool]:
        return {
            "knee_low": s["left_knee_z"] < self.knee_height_threshold,
            "hands_close": s["hands_dist"] < self.hands_close_threshold,
            "lh_near_knee": s["left_hand_to_knee"] < self.hands_to_knee_threshold,
            "rh_near_knee": s["right_hand_to_knee"] < self.hands_to_knee_threshold,
            "pelvis_low": s["pelvis_z"] < self.pelvis_height_threshold,
        }

    def compute(self) -> MetricResult:
        if not self._snapshots:
            return MetricResult(value=0.0, success=False)

        final = self._snapshots[-1]
        checks = self._check(final)
        n_pass = sum(checks.values())
        score = n_pass / len(checks)

        print(f"[KneelDownMetric] Final state:")
        print(f"  left_knee_z:  {final['left_knee_z']:.3f}m  (need <{self.knee_height_threshold})")
        print(f"  right_knee_z: {final['right_knee_z']:.3f}m")
        print(f"  left_hand_z:  {final['left_hand_z']:.3f}m")
        print(f"  right_hand_z: {final['right_hand_z']:.3f}m")
        print(f"  pelvis_z:     {final['pelvis_z']:.3f}m  (need <{self.pelvis_height_threshold})")
        print(f"  hands_dist:   {final['hands_dist']:.3f}m  (need <{self.hands_close_threshold})")
        print(f"  lh->knee:     {final['left_hand_to_knee']:.3f}m  (need <{self.hands_to_knee_threshold})")
        print(f"  rh->knee:     {final['right_hand_to_knee']:.3f}m  (need <{self.hands_to_knee_threshold})")
        check_str = " | ".join(f"{k}={'Y' if v else 'N'}" for k, v in checks.items())
        print(f"  Checks: {check_str}")
        print(f"  Score: {score:.2f} ({n_pass}/{len(checks)})")

        return MetricResult(
            value=score,
            success=score > 0,
            info={
                **{k: round(v, 4) if isinstance(v, float) else v for k, v in final.items()},
                **{f"check_{k}": v for k, v in checks.items()},
                "conditions_met": f"{n_pass}/{len(checks)}",
            },
        )
