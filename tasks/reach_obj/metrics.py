"""
tasks/reach_obj/metrics.py — Metric definitions for reach_obj task.

To add metrics for a new task, copy this file to tasks/<task_name>/metrics.py
and define get_metrics() returning a list of Metric instances.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from eval.metrics import DistanceToTarget


def get_metrics():
    """
    Returns the list of metrics for reach_obj.

    Success criterion: right wrist within 0.1m of cube at episode end.
    Also tracks minimum distance achieved during episode.
    """
    return [
        DistanceToTarget(
            name="dist_right_wrist_to_cube",
            link_name="right_wrist_yaw_link",
            object_index=0,
            success_threshold=0.1,
            use_min=False,   # final distance (strictest)
        ),
        DistanceToTarget(
            name="dist_right_wrist_to_cube_min",
            link_name="right_wrist_yaw_link",
            object_index=0,
            success_threshold=0.1,
            use_min=True,    # best distance during episode (more lenient)
        ),
    ]
