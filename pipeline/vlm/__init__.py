from .base import VLMBase
from .qwen import QwenVLM

REGISTRY: dict[str, type[VLMBase]] = {
    "qwen2.5-vl-7b": QwenVLM,
}


def load_vlm(name: str = "qwen2.5-vl-7b", **kwargs) -> VLMBase:
    if name not in REGISTRY:
        raise ValueError(f"Unknown VLM '{name}'. Available: {list(REGISTRY)}")
    vlm = REGISTRY[name](**kwargs)
    vlm.load()
    return vlm
