"""Create an empty scene for kneel_down_1_knee (no objects needed).

Usage:
    python tasks/kneel_down_1_knee/create_scene.py \
        --output outputs/kneel_down_1_knee_scene.pt
"""

import argparse
from protomotions.components.scene_lib import Scene, SceneLib


def create_scene(output: str = "outputs/kneel_down_1_knee_scene.pt") -> None:
    scene = Scene(objects=[])
    SceneLib.save_scenes_to_file([scene], output)
    print(f"[create_scene] Saved empty scene to '{output}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/kneel_down_1_knee_scene.pt")
    args = parser.parse_args()
    create_scene(output=args.output)
