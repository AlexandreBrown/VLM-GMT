"""Metrics for raise_right_hand task.

Success: right hand Z > 1.3m at episode end (above head height ~1.2m).
"""

from eval.metrics import LinkHeightMetric


def get_metrics():
    return [
        LinkHeightMetric(
            name="right_hand_height",
            link_name="right_rubber_hand",
            height_threshold=1.3,
            use_mean=False,
            overlay_label="Right Hand Z",
        ),
    ]
