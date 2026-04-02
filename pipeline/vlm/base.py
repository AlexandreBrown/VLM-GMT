from abc import ABC, abstractmethod
import numpy as np


class VLMBase(ABC):

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def query_object_pixels(
        self,
        image_rgb: np.ndarray,
        object_description: str,
    ) -> tuple[float, float]:
        """Return estimated pixel coordinates (u, v) of the object center."""
        ...
