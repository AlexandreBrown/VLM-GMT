"""Metrics for touch_left_leg_with_right_hand task.

Success: dist(right_rubber_hand, left_knee_link) < 0.20m at episode end.
"""

from eval.metrics import LinkToLinkDistance


def get_metrics():
    return [
        LinkToLinkDistance(
            name="dist_right_hand_to_left_knee",
            link_a="right_rubber_hand",
            link_b="left_knee_link",
            success_threshold=0.2,
            use_min=False,
            overlay_label="R.Hand→L.Knee",
        ),
    ]
