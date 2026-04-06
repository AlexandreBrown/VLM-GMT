"""Qwen3.5 VLM adapter (e.g. Qwen3.5-35B-A3B-FP8).

Qwen3.5 is a native multimodal model (early fusion). Key differences from Qwen2.5-VL:
- Uses AutoModelForCausalLM (new hybrid MoE + Gated DeltaNet architecture)
- Thinking mode on by default: output starts with <think>...</think> before the JSON
- FP8 variant loads directly without bitsandbytes (weights already quantized)

pip install transformers>=4.50.0 accelerate
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


class Qwen35VLM(VLMBase):

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-27B-FP8",
        device: str = "cuda",
        num_frames: int = 90,
        task: str = "reach_obj",
        task_description: str | None = None,
        load_in_4bit: bool = False,  # not needed for FP8 variant
    ):
        self.model_name = model_name
        self.device = device
        self.num_frames = num_frames
        self.task = task
        self.task_description = task_description
        self.load_in_4bit = load_in_4bit
        self._model = None
        self._processor = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        print(f"[Qwen35VLM] Loading {self.model_name} ...")
        self._processor = AutoProcessor.from_pretrained(self.model_name)

        # FP8 weights are already quantized — torch_dtype="auto" picks them up natively.
        # AutoModelForImageTextToText is the correct class for VL generation models.
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_name, torch_dtype="auto", device_map="auto"
        )
        self._model.eval()
        print("[Qwen35VLM] Ready.")

    def query_constraints(
        self,
        image_rgb: np.ndarray | None = None,
        task_description: str | None = None,
    ) -> list[dict[str, Any]]:
        import torch

        assert self._model is not None, "Call load() first."

        system = get_system_prompt(self.num_frames)
        task_text = task_description or get_task_prompt(self.task, self.task_description)

        if image_rgb is not None:
            pil_img = Image.fromarray(image_rgb)
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            user = (
                f"Task: {task_text}\n"
                "Look at the image carefully. Think step by step:\n"
                "1. What objects are visible and where are they relative to the robot?\n"
                "2. What body parts need to be constrained and at which frames to achieve the task?\n"
                "3. What are the estimated 3D world positions of those targets?\n"
                "Now output the JSON array of constraints."
            )
            user_content = [
                {"type": "image", "image": f"data:image/jpeg;base64,{b64}"},
                {"type": "text", "text": user},
            ]
        else:
            pil_img = None
            user = (
                f"Task: {task_text}\n"
                "Think step by step:\n"
                "1. What body parts need to be constrained and at which frames to achieve the task?\n"
                "2. What are the estimated 3D world positions of those targets (relative to robot start)?\n"
                "Now output the JSON array of constraints."
            )
            user_content = [{"type": "text", "text": user}]

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        proc_kwargs = {"text": [text], "return_tensors": "pt"}
        if pil_img is not None:
            proc_kwargs["images"] = [pil_img]
        inputs = self._processor(**proc_kwargs).to(self.device)

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=4096)

        input_len = inputs["input_ids"].shape[1]
        raw = self._processor.batch_decode(
            output_ids[:, input_len:], skip_special_tokens=True
        )[0].strip()

        print(f"[Qwen35VLM] Raw response:\n{raw}")
        constraints = self._parse(raw)
        if not constraints:
            raise ValueError(f"[Qwen35VLM] Model returned 0 constraints. Raw: {raw!r}")
        return constraints

    @staticmethod
    def _parse(raw: str) -> list[dict[str, Any]]:
        import json

        # Strip thinking block (<think>...</think>) if present
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        clean = re.sub(r"```[a-z]*", "", raw).replace("```", "").strip()
        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            m = re.search(r"\[.*\]", clean, re.DOTALL)
            if m:
                data = json.loads(m.group(0))
            else:
                raise ValueError(f"[Qwen35VLM] Could not parse response: {raw!r}")

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
