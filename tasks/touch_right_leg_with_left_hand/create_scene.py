"""Create an empty scene for touch_right_leg_with_left_hand (no objects needed).

Usage:
    python tasks/touch_right_leg_with_left_hand/create_scene.py \
        --output outputs/touch_right_leg_with_left_hand_scene.pt
"""

import argparse
from protomotions.components.scene_lib import Scene, SceneLib


def create_scene(output: str = "outputs/touch_right_leg_with_left_hand_scene.pt") -> None:
    scene = Scene(objects=[])
    SceneLib.save_scenes_to_file([scene], output)
    print(f"[create_scene] Saved empty scene to '{output}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/touch_right_leg_with_left_hand_scene.pt")
    args = parser.parse_args()
    create_scene(output=args.output)
