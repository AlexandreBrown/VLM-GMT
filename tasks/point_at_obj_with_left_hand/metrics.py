"""Metrics for point_at_obj_with_left_hand task.

Success: dist(left_hand, object) < 0.15m at episode end.
Object is scene index 1 (index 0 = pedestal).
"""

from eval.metrics import DistanceToTarget


def get_metrics():
    return [
        DistanceToTarget(
            name="dist_left_hand_to_obj",
            link_name="left_rubber_hand",
            object_index=1,
            success_threshold=0.15,
            use_min=False,
            overlay_label="Dist To Object",
        ),
        DistanceToTarget(
            name="dist_left_hand_to_obj_min",
            link_name="left_rubber_hand",
            object_index=1,
            success_threshold=0.15,
            use_min=True,
        ),
    ]
