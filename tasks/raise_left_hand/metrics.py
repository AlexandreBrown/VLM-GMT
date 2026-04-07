"""Metrics for raise_left_hand task.

Success: left hand Z > 1.3m at episode end (above head height ~1.2m).
"""

from eval.metrics import LinkHeightMetric


def get_metrics():
    return [
        LinkHeightMetric(
            name="left_hand_height",
            link_name="left_rubber_hand",
            height_threshold=1.3,
            use_mean=False,
            overlay_label="Left Hand Z",
        ),
    ]
