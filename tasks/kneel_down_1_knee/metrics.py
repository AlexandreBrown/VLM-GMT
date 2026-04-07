"""Metrics for kneel_down_1_knee task.

Success is partial: score = fraction of conditions met (0.0 to 1.0).
Conditions checked at episode end:
  - Left knee low (z < 0.35m)
  - Both hands close to each other (dist < 0.20m)
  - Both hands close to left knee (dist < 0.25m)
  - Pelvis low (z < 0.55m)
"""

from eval.metrics import KneelDownMetric


def get_metrics():
    return [
        KneelDownMetric(
            knee_height_threshold=0.35,
            hands_close_threshold=0.20,
            hands_to_knee_threshold=0.25,
            pelvis_height_threshold=0.55,
        ),
    ]
