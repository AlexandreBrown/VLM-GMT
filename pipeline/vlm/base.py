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
        task_description: str | None = None,
    ) -> list[dict[str, Any]]:
        """Predict kinematic constraints from an egocentric image.

        System prompt is loaded from prompts/system.txt.
        Task prompt is loaded from tasks/<task>/vlm_prompt.txt,
        or overridden by task_description if provided.

        Each returned constraint is a dict:
            frame_id:  int   — target frame (0-based, 30 fps)
            type:      str   — "right-hand" | "left-hand" | "right-foot" | "left-foot"
            position:  list  — [x, y, z] in world frame (meters), +X fwd, +Y left, +Z up
        """
        ...
