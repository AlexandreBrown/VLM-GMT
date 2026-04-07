"""Metrics for walk_to_obj task.

Success: 2D (XY) dist(pelvis, box) < 0.5m at episode end.
Uses XY-only distance because the robot pelvis is always ~0.8m above the ground box.
"""

from eval.metrics import DistanceToTarget


def get_metrics():
    return [
        DistanceToTarget(
            name="dist_pelvis_to_box_2d",
            link_name="pelvis",
            object_index=0,
            success_threshold=0.5,
            use_min=False,
            use_2d=True,
            overlay_label="XY Dist To Box",
        ),
        DistanceToTarget(
            name="dist_pelvis_to_box_2d_min",
            link_name="pelvis",
            object_index=0,
            success_threshold=0.5,
            use_min=True,
            use_2d=True,
        ),
    ]
