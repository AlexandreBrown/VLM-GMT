"""Create a point_at_obj_with_left_hand scene.

A blue object on a thin pedestal placed to the RIGHT (-Y) and slightly
forward of the robot at chest height. The robot must reach across with
its LEFT hand to point at it.

Scene objects (in order):
  0 — pedestal (gray, thin)
  1 — target object (blue cube)

All positions in IsaacLab frame: x=forward, y=left, z=up.
Robot starts at origin facing +x.

Usage:
    python tasks/point_at_obj_with_left_hand/create_scene.py \
        --output outputs/point_at_obj_with_left_hand_scene.pt
"""

import argparse
from protomotions.components.scene_lib import (
    BoxSceneObject,
    ObjectOptions,
    Scene,
    SceneLib,
)

# Object at ~0.6m forward, 0.2m to the RIGHT, chest height (~0.9m)
# Placed right so naive "raise left hand" misses — robot must reach across
OBJ_POS = (0.6, -0.2, 0.9)
OBJ_SIZE = 0.08

PEDESTAL_WIDTH = 0.12
PEDESTAL_DEPTH = 0.12


def create_scene(
    obj_pos: tuple = OBJ_POS,
    obj_size: float = OBJ_SIZE,
    output: str = "outputs/point_at_obj_with_left_hand_scene.pt",
) -> None:
    pedestal_height = obj_pos[2] - obj_size / 2
    pedestal = BoxSceneObject(
        width=PEDESTAL_WIDTH,
        depth=PEDESTAL_DEPTH,
        height=pedestal_height,
        translation=(obj_pos[0], obj_pos[1], pedestal_height / 2),
        rotation=(0.0, 0.0, 0.0, 1.0),
        options=ObjectOptions(fix_base_link=True, color=(0.5, 0.5, 0.5)),
    )
    obj = BoxSceneObject(
        width=obj_size,
        depth=obj_size,
        height=obj_size,
        translation=tuple(obj_pos),
        rotation=(0.0, 0.0, 0.0, 1.0),
        options=ObjectOptions(fix_base_link=True, color=(0.1, 0.3, 0.9)),
    )

    scene = Scene(objects=[pedestal, obj])
    SceneLib.save_scenes_to_file([scene], output)
    print(f"[create_scene] Saved '{output}'")
    print(f"  object: pos={obj_pos}, size={obj_size}m, color=blue")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--obj-pos", nargs=3, type=float, default=list(OBJ_POS), metavar=("X", "Y", "Z")
    )
    parser.add_argument(
        "--output", default="outputs/point_at_obj_with_left_hand_scene.pt"
    )
    args = parser.parse_args()
    create_scene(obj_pos=tuple(args.obj_pos), output=args.output)
