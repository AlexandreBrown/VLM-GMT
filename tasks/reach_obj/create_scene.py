"""
tasks/reach_obj/create_scene.py — Create a static reach_obj scene for ProtoMotions.

Usage (run from ProtoMotions root):
    python /path/to/VLM-GMT/tasks/reach_obj/create_scene.py \
        --cube-pos 0.6 0.0 0.4 \
        --output reach_obj_scene.pt

Pass the output .pt to inference_agent.py / env_kinematic_playback.py via --scenes-file.
"""

import argparse
import torch
from protomotions.components.scene_lib import (
    BoxSceneObject,
    ObjectOptions,
    Scene,
    SceneLib,
)


def create_reach_scene(
    cube_pos: tuple = (0.6, 0.0, 0.4),
    cube_size: float = 0.08,
    output: str = "reach_obj_scene.pt",
) -> None:
    """
    Args:
        cube_pos:  (x, y, z) world frame center. x=forward, y=left, z=up.
        cube_size: Side length in meters.
        output:    Output .pt path.
    """
    cube = BoxSceneObject(
        width=cube_size,
        depth=cube_size,
        height=cube_size,
        translation=tuple(cube_pos),
        rotation=(0.0, 0.0, 0.0, 1.0),  # identity quaternion (xyzw)
        options=ObjectOptions(fix_base_link=True),
    )
    scene = Scene(objects=[cube])
    SceneLib.save_scenes_to_file([scene], output)
    print(f"[create_scene] Saved '{output}' | cube_pos={cube_pos} size={cube_size}m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cube-pos", nargs=3, type=float, default=[0.6, 0.0, 0.4],
        metavar=("X", "Y", "Z"),
        help="Cube center in world frame (default: 0.6 0.0 0.4)",
    )
    parser.add_argument("--cube-size", type=float, default=0.08)
    parser.add_argument("--output", default="reach_obj_scene.pt")
    args = parser.parse_args()
    create_reach_scene(tuple(args.cube_pos), args.cube_size, args.output)
