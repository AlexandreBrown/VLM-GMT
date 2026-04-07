"""Qwen2.5-VL adapter using AutoModelForImageTextToText.

Supports Qwen2.5-VL-7B, 32B, 72B. Default: 32B with 4-bit (~16GB, fits in 48GB L40S).

pip install transformers>=4.50.0 accelerate bitsandbytes
"""

import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .base import VLMBase


def _load_prompt(path: str | Path) -> str:
    return Path(path).read_text().strip()


class QwenVLM(VLMBase):

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-32B-Instruct",
        device: str = "cuda",
        dtype: str = "bfloat16",
        load_in_4bit: bool = True,
        num_frames: int = 90,
        task: str = "reach_obj",
        task_description: str | None = None,
        vlm_gmt_root: str | Path = None,
    ):
        if vlm_gmt_root is None:
            raise ValueError("vlm_gmt_root must be provided to QwenVLM")
        self.vlm_gmt_root = Path(vlm_gmt_root)
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit
        self.num_frames = num_frames
        self.task = task
        self.task_description = task_description
        self._model = None
        self._processor = None

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

        quant_str = "4-bit" if self.load_in_4bit else self.dtype
        print(f"[QwenVLM] Loading {self.model_name} ({quant_str})...")
        self._processor = AutoProcessor.from_pretrained(self.model_name)

        load_kwargs = dict(device_map="auto")
        if self.load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        else:
            load_kwargs["torch_dtype"] = getattr(torch, self.dtype)

        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_name, **load_kwargs
        )
        self._model.eval()
        print("[QwenVLM] Ready.")

    def _get_system_prompt(self) -> str:
        text = _load_prompt(self.vlm_gmt_root / "prompts" / "system.txt")
        return text.format(num_frames=self.num_frames, max_frame=self.num_frames - 1)

    def _get_task_prompt(self) -> str:
        if self.task_description:
            p = Path(self.task_description)
            if p.is_file():
                return _load_prompt(p)
            return self.task_description
        path = self.vlm_gmt_root / "tasks" / self.task / "vlm_prompt.txt"
        if not path.exists():
            raise FileNotFoundError(f"No VLM prompt for task '{self.task}' at {path}")
        return _load_prompt(path)

    def query_constraints(
        self,
        image_rgb: np.ndarray | None = None,
        task_description: str | None = None,
    ) -> list[dict[str, Any]]:
        import torch

        assert self._model is not None, "Call load() first."

        system = self._get_system_prompt()
        task_text = task_description or self._get_task_prompt()

        if image_rgb is not None:
            pil_img = Image.fromarray(image_rgb)
            user_text = (
                f"Task: {task_text}\n"
                "Now output the JSON array containing the kinematic constraints you believe would help achieve the task.\n"
                "You MUST output an array with a least one constraint, even if you think there are no constraints needed.\n"
            )
            user_content = [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": user_text},
            ]
        else:
            user_text = (
                f"Task: {task_text}"
            )
            user_content = [{"type": "text", "text": user_text}]

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": user_content},
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=4096)

        raw = self._processor.decode(
            output_ids[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        ).strip()

        print(f"[QwenVLM] Raw response:\n{raw}")
        constraints = self._parse(raw)
        if not constraints:
            raise ValueError(f"[QwenVLM] Model returned 0 constraints. Raw: {raw!r}")
        return constraints

    @staticmethod
    def _parse(raw: str) -> list[dict[str, Any]]:
        import json

        clean = re.sub(r"```[a-z]*", "", raw).replace("```", "").strip()
        # Strip // comments (common LLM habit with JSON)
        clean = re.sub(r"//[^\n]*", "", clean)

        # Try full parse first
        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            # Extract first balanced [...] block
            data = None
            start = clean.find("[")
            if start != -1:
                depth = 0
                for i, ch in enumerate(clean[start:], start):
                    if ch == "[":
                        depth += 1
                    elif ch == "]":
                        depth -= 1
                        if depth == 0:
                            try:
                                data = json.loads(clean[start : i + 1])
                            except json.JSONDecodeError:
                                pass
                            break
            if data is None:
                raise ValueError(f"[QwenVLM] Could not parse response: {raw!r}")

        if isinstance(data, dict):
            data = [data]

        return [
            {
                "frame_id": int(item["frame_id"]),
                "type": str(item["type"]),
                "position": [float(v) for v in item["position"]],
            }
            for item in data
        ]
