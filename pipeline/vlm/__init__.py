from .base import VLMBase
from .qwen import QwenVLM
from .qwen35 import Qwen35VLM

REGISTRY: dict[str, type[VLMBase]] = {
    "qwen2.5-vl-7b":    QwenVLM,
    "qwen2.5-vl-32b":   QwenVLM,
    "qwen2.5-vl-72b":   QwenVLM,
    "qwen3.5-27b":      Qwen35VLM,
    "qwen3.5-35b-a3b":  Qwen35VLM,
}

HF_MODEL_IDS = {
    "qwen2.5-vl-7b":    "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2.5-vl-32b":   "Qwen/Qwen2.5-VL-32B-Instruct",
    "qwen2.5-vl-72b":   "Qwen/Qwen2.5-VL-72B-Instruct",
    "qwen3.5-27b":      "Qwen/Qwen3.5-27B-FP8",    # requires flash-linear-attention + causal-conv1d
    "qwen3.5-35b-a3b":  "Qwen/Qwen3.5-35B-A3B-FP8",  # requires flash-linear-attention + causal-conv1d
}


def load_vlm(name: str = "qwen2.5-vl-32b", *, vlm_gmt_root: str, **kwargs) -> VLMBase:
    """Load and initialize a VLM by short name.

    Args:
        name: Short VLM name (e.g. 'qwen2.5-vl-32b').
        vlm_gmt_root: Path to VLM-GMT root directory (required, no default).
        **kwargs: Forwarded to the VLM constructor.
    """
    if name not in REGISTRY:
        raise ValueError(f"Unknown VLM '{name}'. Available: {list(REGISTRY)}")
    hf_model_id = HF_MODEL_IDS[name]
    vlm = REGISTRY[name](model_name=hf_model_id, vlm_gmt_root=vlm_gmt_root, **kwargs)
    vlm.load()
    return vlm
