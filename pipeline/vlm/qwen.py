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


class QwenVLM(VLMBase):

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-32B-Instruct",
        device: str = "cuda",
        dtype: str = "bfloat16",
        load_in_4bit: bool = True,
        num_frames: int = 90,
        pitch_deg: float = 50.0,
        task: str = "reach_obj",
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
        self.pitch_deg = pitch_deg
        self.task = task
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

    def _load_system_prompt(self) -> str:
        path = self.vlm_gmt_root / "prompts" / "system.txt"
        if not path.exists():
            raise FileNotFoundError(f"System prompt not found at {path}")
        text = path.read_text().strip()
        return text.format(num_frames=self.num_frames, max_frame=self.num_frames - 1, pitch_deg=self.pitch_deg)

    def _load_task_prompt(self) -> str:
        path = self.vlm_gmt_root / "tasks" / self.task / "vlm_prompt.txt"
        if not path.exists():
            raise FileNotFoundError(f"VLM prompt not found at {path}")
        return path.read_text().strip()

    def query_constraints(
        self,
        image_rgb: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        import torch

        assert self._model is not None, "Call load() first."

        system = self._load_system_prompt()
        task_prompt = self._load_task_prompt()

        if image_rgb is not None:
            pil_img = Image.fromarray(image_rgb)
            user_content = [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": task_prompt},
            ]
        else:
            user_content = [{"type": "text", "text": task_prompt}]

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
        print(f"[QwenVLM] Model returned 0 constraints. Raw: {raw!r}")
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

        parsed = []
        for item in data:
            ctype = str(item["type"])
            entry = {"frame_id": int(item["frame_id"]), "type": ctype}
            if ctype == "fullbody":
                entry["positions"] = {
                    str(k): [float(v) for v in pos]
                    for k, pos in item["positions"].items()
                }
            else:
                entry["position"] = [float(v) for v in item["position"]]
            parsed.append(entry)
        return parsed
