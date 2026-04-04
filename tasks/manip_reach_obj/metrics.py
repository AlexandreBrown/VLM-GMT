"""Metrics for manip_reach_obj task.

Success: dist(right_hand, cube) < 0.15m during the episode.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from eval.metrics import DistanceToTarget


def get_metrics():
    return [
        DistanceToTarget(
            name="dist_right_hand_to_cube",
            link_name="right_rubber_hand",
            object_index=1,  # index 0 = table, index 1 = cube
            success_threshold=0.15,
            use_min=False,
            overlay_label="Distance To Cube",
        ),
        DistanceToTarget(
            name="dist_right_hand_to_cube_min",
            link_name="right_rubber_hand",
            object_index=1,
            success_threshold=0.15,
            use_min=True,
        ),
    ]
