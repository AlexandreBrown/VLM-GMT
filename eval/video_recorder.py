"""Side-by-side video recorder: third-person + egocentric views with metric overlays."""

import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


class VideoRecorder:
    """Records side-by-side frames across all episodes into a single video."""

    def __init__(self, output_dir: str, fps: int = 30, separator_frames: int = 15):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.separator_frames = separator_frames
        self._frames = []
        self._episode = 0
        self._font = None

    def _get_font(self):
        if self._font is None:
            try:
                self._font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20
                )
            except (OSError, IOError):
                self._font = ImageFont.load_default()
        return self._font

    def new_episode(self):
        self._episode += 1

    def capture_frame(
        self,
        third_person_rgb: np.ndarray,
        ego_rgb: np.ndarray | None,
        metrics: list,
        episode_num: int = 0,
    ):
        """Compose one frame: third-person (left) + ego (right) + metric overlays."""
        tp_img = Image.fromarray(third_person_rgb)

        if ego_rgb is not None:
            ego_img = Image.fromarray(ego_rgb)
            ego_img = ego_img.resize(tp_img.size)
            combined = Image.new("RGB", (tp_img.width * 2, tp_img.height))
            combined.paste(tp_img, (0, 0))
            combined.paste(ego_img, (tp_img.width, 0))
        else:
            combined = tp_img

        draw = ImageDraw.Draw(combined)
        font = self._get_font()

        # Episode label
        draw.text(
            (10, 10), f"Episode {episode_num + 1}", fill=(255, 255, 255), font=font
        )
        y_offset = 38

        for m in metrics:
            overlay = m.get_overlay()
            if overlay is None:
                continue
            text, is_success = overlay
            color = (0, 255, 0) if is_success else (255, 80, 80)
            draw.text((10, y_offset), text, fill=color, font=font)
            y_offset += 28

        self._frames.append(np.array(combined))

    def save(self, filename: str) -> str | None:
        """Write all accumulated frames to a single mp4."""
        if not self._frames:
            return None

        import cv2

        path = self.output_dir / filename
        h, w = self._frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"VP80"), self.fps, (w, h)
        )
        for frame in self._frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        print(
            f"[video] Saved {len(self._frames)} frames ({self._episode} episodes) to {path}"
        )
        return str(path)
