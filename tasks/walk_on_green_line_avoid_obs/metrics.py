"""Metrics for walk_on_green_line_avoid_obs task.

Success: robot stays on the green line (±0.5m Y) at all times AND avoids each
obstacle with sufficient lateral clearance (no clipping through).

Obstacle avoidance is checked by measuring |pelvis_y - obstacle_y| when the
robot's pelvis_x is within ±0.5m of the obstacle's x center. If the lateral
distance drops below 0.3m at any point in that window, the robot clipped
through the obstacle instead of navigating around it.

Obstacle positions match create_scene.py defaults.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from eval.metrics import WalkOnLineMetric


def get_metrics():
    return [
        WalkOnLineMetric(
            name="walk_on_line",
            link_name="pelvis",
            line_x_min=0.25,
            line_x_max=5.75,
            line_y_half_width=0.5,
            obstacle_positions=[
                (1.5, 0.20),   # obs1: slightly left
                (3.0, -0.20),  # obs2: slightly right
                (4.5, 0.15),   # obs3: slightly left
            ],
            avoidance_min_dist=0.3,
            x_window=0.5,
        ),
    ]
