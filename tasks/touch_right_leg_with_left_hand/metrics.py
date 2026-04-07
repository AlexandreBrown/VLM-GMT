"""Metrics for touch_right_leg_with_left_hand task.

Success: dist(left_rubber_hand, right_knee) < 0.15m at episode end.
"""

from eval.metrics import LinkToLinkDistance


def get_metrics():
    return [
        LinkToLinkDistance(
            name="dist_left_hand_to_right_knee",
            link_a="left_rubber_hand",
            link_b="right_knee_link",
            success_threshold=0.15,
            use_min=False,
            overlay_label="L.Hand→R.Knee",
        ),
    ]
