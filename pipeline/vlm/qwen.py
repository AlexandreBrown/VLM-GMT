"""
Qwen2.5-VL-7B-Instruct adapter.
~14GB bf16, fits on a single RTX 5080 / A100.

pip install transformers>=4.50.0 qwen-vl-utils accelerate
"""

import re
import base64
import io
import numpy as np
from PIL import Image

from .base import VLMBase


PROMPT_TEMPLATE = (
    "You are analyzing a robotics simulation image. "
    "Identify the {object_description} in the image. "
    "Respond with ONLY a JSON object in this exact format: "
    '{{"u": <pixel_col_float>, "v": <pixel_row_float>}} '
    "where u is the horizontal pixel coordinate and v is the vertical pixel coordinate "
    "of the object's center. No explanation, no markdown, just the JSON."
)


class QwenVLM(VLMBase):

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        dtype: str = "bfloat16",
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
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

    def query_object_pixels(
        self,
        image_rgb: np.ndarray,
        object_description: str,
    ) -> tuple[float, float]:
        import json
        import torch

        assert self._model is not None, "Call load() first."

        pil_img = Image.fromarray(image_rgb)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        prompt = PROMPT_TEMPLATE.format(object_description=object_description)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"data:image/jpeg;base64,{b64}"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text], images=[pil_img], return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=64)

        input_len = inputs["input_ids"].shape[1]
        raw = self._processor.batch_decode(
            output_ids[:, input_len:], skip_special_tokens=True
        )[0].strip()

        return self._parse(raw)

    @staticmethod
    def _parse(raw: str) -> tuple[float, float]:
        import json
        try:
            clean = re.sub(r"```[a-z]*", "", raw).replace("```", "").strip()
            data = json.loads(clean)
            return float(data["u"]), float(data["v"])
        except Exception:
            m = re.search(r'"?u"?\s*:\s*([\d.]+).*?"?v"?\s*:\s*([\d.]+)', raw)
            if m:
                return float(m.group(1)), float(m.group(2))
            raise ValueError(f"[QwenVLM] Could not parse response: {raw!r}")
