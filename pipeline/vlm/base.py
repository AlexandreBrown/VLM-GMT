from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class VLMBase(ABC):

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def query_constraints(
        self,
        image_rgb: np.ndarray,
        task_description: str,
    ) -> list[dict[str, Any]]:
        """Return a list of kinematic constraints predicted from the egocentric image.

        Each constraint is a dict with keys:
            frame_id:  int     — target frame index (0-based, 30 fps)
            type:      str     — "right-hand" | "left-hand" | "right-foot" | "left-foot"
            position:  list    — [x, y, z] in IsaacLab world frame (meters)
                                  +X forward, +Y left, +Z up

        The VLM is given the coordinate system, robot dimensions, and motion
        duration in its prompt so it can reason about metric positions.
        """
        ...
