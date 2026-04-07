from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class VLMBase(ABC):

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def query_constraints(
        self,
        image_rgb: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """Predict kinematic constraints from an optional egocentric image.

        If image_rgb is None, the VLM reasons from text only (body knowledge).
        System prompt: prompts/system.txt
        Task prompt: tasks/<task>/vlm_prompt.txt

        Each returned constraint is a dict:
            frame_id:  int
            type:      str
            position:  list[float]  — [x, y, z] in world frame (meters)
        """
        ...
