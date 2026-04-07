"""Metrics for kneel_down_1_knee task.

Success: pelvis Z < 0.5m at episode end (kneeling lowers pelvis from ~0.8m).
"""

from eval.metrics import LinkHeightMetric


def get_metrics():
    return [
        LinkHeightMetric(
            name="pelvis_height",
            link_name="pelvis",
            height_threshold=0.5,
            check_below=True,
            use_mean=False,
            overlay_label="Pelvis Z (need <0.5)",
        ),
    ]
