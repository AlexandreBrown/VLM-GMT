"""Metrics for navigate_maze task.

Success: robot avoids both internal walls AND final pelvis x past wall 2 + 0.5m.

Obstacle positions are near the wall inner edges (y~0) with a small offset
to indicate which side the wall is on:
  Wall 1: (1.5, +0.1) — wall blocks y>0, edge at y=0
  Wall 2: (3.0, -0.1) — wall blocks y<0, edge at y=0

avoidance_min_dist = 0.3: a straight walk at y=0 gives dist=0.1 < 0.3 -> fails.
A correct avoidance at y=-0.3 gives dist=0.4 >= 0.3 -> passes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from eval.metrics import NavigateMazeMetric


def get_metrics():
    return [
        NavigateMazeMetric(
            name="navigate_maze",
            link_name="pelvis",
            line_x_min=-1.0,
            line_x_max=6.0,
            line_y_half_width=100.0,  # no lateral constraint, just avoidance
            obstacle_positions=[
                (1.5,  0.1),  # wall 1 inner edge (blocks y>0)
                (3.0, -0.1),  # wall 2 inner edge (blocks y<0)
            ],
            avoidance_min_dist=0.3,
            x_window=0.5,
            pass_x_margin=0.5,
        ),
    ]
