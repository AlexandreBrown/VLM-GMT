"""Metrics for navigate_maze task.

Simple metric: score = walls_cleared / total_walls.
A wall is cleared when the pelvis final world x is past the wall's world x
by pass_x_margin.

Scene object indices (see tasks/navigate_maze/create_scene.py):
  0: north boundary, 1: south, 2: west, 3: east
  4: wall 1, 5: wall 2
"""

from eval.metrics import NavigateMazeMetric


def get_metrics():
    return [
        NavigateMazeMetric(
            name="navigate_maze",
            link_name="pelvis",
            obstacle_indices=(4, 5),
            pass_x_margin=0.5,
        ),
    ]
