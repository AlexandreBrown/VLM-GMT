"""
Qwen2.5-VL-7B-Instruct adapter.
~14GB bf16, fits on a single RTX 5080 / A100.

pip install transformers>=4.50.0 qwen-vl-utils accelerate
"""

import re
import base64
import io
from typing import Any

import numpy as np
from PIL import Image

from .base import VLMBase


SYSTEM_PROMPT = """\
You are an embodied AI controlling a humanoid robot. You see through the robot's \
head-mounted camera (egocentric view).

Coordinate system (IsaacLab / MuJoCo convention):
  +X = forward (the direction the robot faces)
  +Y = left
  +Z = up
  Origin is on the ground beneath the robot.

Robot reference dimensions:
  Pelvis height: ~0.8 m
  Head / camera height: ~1.2 m
  Arm reach from shoulder: ~0.6 m

Motion is generated at 30 fps. Total frames: {num_frames}.

Your task: look at the image and output kinematic constraints that will guide \
the robot's motion to accomplish the described task. Output ONLY a JSON array, \
no explanation, no markdown fences. Each element must have exactly these keys:
  "frame_id": int — target frame index (0 to {max_frame})
  "type": str — one of "right-hand", "left-hand", "right-foot", "left-foot"
  "position": [x, y, z] — target position in world coordinates (meters)

Example output for reaching a cup 0.5 m forward on a 0.75 m table:
[{{"frame_id": 60, "type": "right-hand", "position": [0.5, 0.0, 0.75]}}]
"""

USER_PROMPT = "Task: {task_description}\nOutput the JSON array of constraints."


class QwenVLM(VLMBase):

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        dtype: str = "bfloat16",
        num_frames: int = 90,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.num_frames = num_frames
        self._model = None
        self._processor = None

    def load(self) -> None:
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        torch_dtype = getattr(torch, self.dtype)
        print(f"[QwenVLM] Loading {self.model_name} ({self.dtype})...")
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            device_map=self.device,
        )
        self._model.eval()
        print("[QwenVLM] Ready.")

    def query_constraints(
        self,
        image_rgb: np.ndarray,
        task_description: str,
    ) -> list[dict[str, Any]]:
        import json
        import torch

        assert self._model is not None, "Call load() first."

        pil_img = Image.fromarray(image_rgb)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        system = SYSTEM_PROMPT.format(
            num_frames=self.num_frames,
            max_frame=self.num_frames - 1,
        )
        user = USER_PROMPT.format(task_description=task_description)

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
            output_ids = self._model.generate(**inputs, max_new_tokens=256)

        input_len = inputs["input_ids"].shape[1]
        raw = self._processor.batch_decode(
            output_ids[:, input_len:], skip_special_tokens=True
        )[0].strip()

        return self._parse(raw)

    @staticmethod
    def _parse(raw: str) -> list[dict[str, Any]]:
        import json

        # Strip markdown fences if present
        clean = re.sub(r"```[a-z]*", "", raw).replace("```", "").strip()
        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            # Try to find a JSON array in the response
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
