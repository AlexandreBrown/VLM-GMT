"""
eval/base_metric.py — Abstract metric interface.

Each task defines metrics by subclassing Metric and registering them
in tasks/<task_name>/metrics.py. The eval loop calls update() every
step and compute() at episode end.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricResult:
    value: float              # continuous value (e.g. mean distance)
    success: bool             # did this episode succeed?
    info: dict = field(default_factory=dict)  # optional extra data


class Metric(ABC):
    """
    Abstract base class for task metrics.

    Lifecycle per episode:
        reset() → update() × N steps → compute() → reset() → ...
    """

    name: str                 # e.g. "dist_right_wrist_to_cube"
    higher_is_better: bool    # for logging/display

    @abstractmethod
    def reset(self) -> None:
        """Called at the start of each episode."""
        ...

    @abstractmethod
    def update(self, env, scene_lib) -> None:
        """
        Called every step with the current env state.

        Args:
            env:       ProtoMotions BaseEnv instance.
            scene_lib: ProtoMotions SceneLib instance.
        """
        ...

    @abstractmethod
    def compute(self) -> MetricResult:
        """
        Called at episode end. Returns the metric result.
        Should NOT reset internal state (reset() does that).
        """
        ...
