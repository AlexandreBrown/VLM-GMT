"""Create a reach_obj scene with a table and a cube on top.

Usage:
    python tasks/manip_reach_obj/create_scene.py \
        --cube-pos 0.6 0.0 0.4 \
        --output outputs/manip_reach_obj_scene.pt
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
    table_width: float = 0.4,
    table_depth: float = 0.4,
    output: str = "outputs/manip_reach_obj_scene.pt",
) -> None:
    """
    Args:
        cube_pos:    (x, y, z) cube center in world frame. x=forward, y=left, z=up.
        cube_size:   Cube side length in meters.
        table_width: Table surface width (X) in meters.
        table_depth: Table surface depth (Y) in meters.
        output:      Output .pt path.
    """
    # Table height so cube sits on top: table_top = cube_z - cube_size/2
    table_top = cube_pos[2] - cube_size / 2
    table_center_z = table_top / 2

    table = BoxSceneObject(
        width=table_width,
        depth=table_depth,
        height=table_top,
        translation=(cube_pos[0], cube_pos[1], table_center_z),
        rotation=(0.0, 0.0, 0.0, 1.0),
        options=ObjectOptions(fix_base_link=True),
    )

    cube = BoxSceneObject(
        width=cube_size,
        depth=cube_size,
        height=cube_size,
        translation=tuple(cube_pos),
        rotation=(0.0, 0.0, 0.0, 1.0),
        options=ObjectOptions(fix_base_link=True),
    )

    scene = Scene(objects=[table, cube])
    SceneLib.save_scenes_to_file([scene], output)
    print(f"[create_scene] Saved '{output}'")
    print(f"  cube: pos={cube_pos}, size={cube_size}m")
    print(f"  table: {table_width}x{table_depth}m, height={table_top:.3f}m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cube-pos", nargs=3, type=float, default=[0.6, 0.0, 0.4],
                        metavar=("X", "Y", "Z"))
    parser.add_argument("--cube-size", type=float, default=0.08)
    parser.add_argument("--table-width", type=float, default=0.4)
    parser.add_argument("--table-depth", type=float, default=0.4)
    parser.add_argument("--output", default="outputs/manip_reach_obj_scene.pt")
    args = parser.parse_args()
    create_reach_scene(tuple(args.cube_pos), args.cube_size,
                       args.table_width, args.table_depth, args.output)
