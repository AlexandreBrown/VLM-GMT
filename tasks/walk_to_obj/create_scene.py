"""Create a walk_to_obj scene: a colored box on the ground ~2m in front of the robot.

The robot starts at the origin facing +x. The box is placed at x=2.0m so it
is reachable in ~2-3s at normal walking speed (~0.7-1.0 m/s).

Usage:
    python tasks/walk_to_obj/create_scene.py \
        --box-pos 2.0 0.0 0.25 \
        --output outputs/walk_to_obj_scene.pt
"""

import argparse
from protomotions.components.scene_lib import (
    BoxSceneObject,
    ObjectOptions,
    Scene,
    SceneLib,
)


def create_walk_scene(
    box_pos: tuple = (1.5, -0.8, 0.25),
    box_width: float = 0.4,
    box_depth: float = 0.4,
    box_height: float = 0.5,
    box_color: tuple = (0.2, 0.6, 0.2),
    output: str = "outputs/walk_to_obj_scene.pt",
) -> None:
    box = BoxSceneObject(
        width=box_width,
        depth=box_depth,
        height=box_height,
        translation=tuple(box_pos),
        rotation=(0.0, 0.0, 0.0, 1.0),
        options=ObjectOptions(fix_base_link=True, color=box_color),
    )

    scene = Scene(objects=[box])
    SceneLib.save_scenes_to_file([scene], output)
    print(f"[create_scene] Saved '{output}'")
    print(f"  box: pos={box_pos}, size={box_width}x{box_depth}x{box_height}m, color={box_color}")


def parse_color(s: str) -> tuple:
    return tuple(float(x) for x in s.split(","))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--box-pos", nargs=3, type=float, default=[1.5, -0.8, 0.25],
                        metavar=("X", "Y", "Z"))
    parser.add_argument("--box-width", type=float, default=0.4)
    parser.add_argument("--box-depth", type=float, default=0.4)
    parser.add_argument("--box-height", type=float, default=0.5)
    parser.add_argument("--box-color", type=parse_color, default=(0.2, 0.6, 0.2),
                        help="RGB floats, e.g. '0.2,0.6,0.2'")
    parser.add_argument("--output", default="outputs/walk_to_obj_scene.pt")
    args = parser.parse_args()
    create_walk_scene(
        tuple(args.box_pos), args.box_width, args.box_depth, args.box_height,
        args.box_color, args.output,
    )
