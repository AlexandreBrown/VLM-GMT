"""Qwen2.5-VL-7B-Instruct adapter (~14GB bf16).

pip install transformers>=4.50.0 qwen-vl-utils accelerate
"""

import re
import base64
import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .base import VLMBase

VLM_GMT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_prompt(path: str | Path) -> str:
    return Path(path).read_text().strip()


def get_system_prompt(num_frames: int) -> str:
    text = load_prompt(VLM_GMT_ROOT / "prompts" / "system.txt")
    return text.format(num_frames=num_frames, max_frame=num_frames - 1)


def get_task_prompt(task: str, override: str | None = None) -> str:
    if override:
        p = Path(override)
        if p.is_file():
            return load_prompt(p)
        return override
    path = VLM_GMT_ROOT / "tasks" / task / "vlm_prompt.txt"
    if not path.exists():
        raise FileNotFoundError(f"No VLM prompt for task '{task}' at {path}")
    return load_prompt(path)


class QwenVLM(VLMBase):

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-72B-Instruct",
        device: str = "cuda",
        dtype: str = "bfloat16",
        load_in_4bit: bool = True,
        num_frames: int = 90,
        task: str = "reach_obj",
        task_description: str | None = None,
    ):
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
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        torch_dtype = getattr(torch, self.dtype)
        quant_str = "4-bit" if self.load_in_4bit else self.dtype
        print(f"[QwenVLM] Loading {self.model_name} ({quant_str})...")
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        load_kwargs = dict(torch_dtype=torch_dtype, device_map="auto")
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            load_kwargs.pop("torch_dtype")  # incompatible with bnb 4-bit
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            **load_kwargs,
        )
        self._model.eval()
        print("[QwenVLM] Ready.")

    def query_constraints(
        self,
        image_rgb: np.ndarray,
        task_description: str | None = None,
    ) -> list[dict[str, Any]]:
        import json
        import torch

        assert self._model is not None, "Call load() first."

        pil_img = Image.fromarray(image_rgb)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        system = get_system_prompt(self.num_frames)
        task_text = task_description or get_task_prompt(self.task, self.task_description)
        user = (
            f"Task: {task_text}\n"
            "Look at the image carefully. Think step by step:\n"
            "1. What objects are visible and where are they relative to the robot?\n"
            "2. What body parts need to be constrained and at which frames to achieve the task?\n"
            "3. What are the estimated 3D world positions of those targets?\n"
            "Now output the JSON array of constraints. You MUST output at least 1 constraint."
        )

        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"data:image/jpeg;base64,{b64}"},
                    {"type": "text", "text": user},
                ],
            },
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text], images=[pil_img], return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=512)

        input_len = inputs["input_ids"].shape[1]
        raw = self._processor.batch_decode(
            output_ids[:, input_len:], skip_special_tokens=True
        )[0].strip()

        print(f"[QwenVLM] Raw response:\n{raw}")
        constraints = self._parse(raw)
        if not constraints:
            raise ValueError(f"[QwenVLM] Model returned 0 constraints. Raw response: {raw!r}")
        return constraints

    @staticmethod
    def _parse(raw: str) -> list[dict[str, Any]]:
        import json

        clean = re.sub(r"```[a-z]*", "", raw).replace("```", "").strip()
        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            m = re.search(r"\[.*\]", clean, re.DOTALL)
            if m:
                data = json.loads(m.group(0))
            else:
                raise ValueError(f"[QwenVLM] Could not parse response: {raw!r}")

        if isinstance(data, dict):
            data = [data]

        constraints = []
        for item in data:
            constraints.append({
                "frame_id": int(item["frame_id"]),
                "type": str(item["type"]),
                "position": [float(v) for v in item["position"]],
            })
        return constraints
