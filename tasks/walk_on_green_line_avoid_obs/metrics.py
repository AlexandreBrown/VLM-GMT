"""Metrics for walk_on_green_line_avoid_obs task.

Success: robot stays within the green line Y bounds at all times AND passes
all 3 obstacle X positions during the episode.

Line: center at (3.0, 0.0), 5.5m long (x: 0.25–5.75), 1.0m wide (y: ±0.5).
Obstacles at x = 1.5, 3.0, 4.5 (default positions from create_scene.py).
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
            obstacle_x_positions=(1.5, 3.0, 4.5),
            obstacle_pass_margin=0.3,
        ),
    ]
