"""Create a manip_reach_obj scene: table with a cube on top.

Usage:
    python tasks/manip_reach_obj/create_scene.py \
        --cube-pos 0.6 0.0 0.4 \
        --output outputs/manip_reach_obj_scene.pt
"""

import argparse
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
    cube_color: tuple = (0.8, 0.1, 0.1),
    table_color: tuple = (0.6, 0.5, 0.3),
    output: str = "outputs/manip_reach_obj_scene.pt",
) -> None:
    table_top = cube_pos[2] - cube_size / 2
    table_center_z = table_top / 2

    table = BoxSceneObject(
        width=table_width,
        depth=table_depth,
        height=table_top,
        translation=(cube_pos[0], cube_pos[1], table_center_z),
        rotation=(0.0, 0.0, 0.0, 1.0),
        options=ObjectOptions(fix_base_link=True, color=table_color),
    )

    cube = BoxSceneObject(
        width=cube_size,
        depth=cube_size,
        height=cube_size,
        translation=tuple(cube_pos),
        rotation=(0.0, 0.0, 0.0, 1.0),
        options=ObjectOptions(fix_base_link=True, color=cube_color),
    )

    scene = Scene(objects=[table, cube])
    SceneLib.save_scenes_to_file([scene], output)
    print(f"[create_scene] Saved '{output}'")
    print(f"  table: {table_width}x{table_depth}m, height={table_top:.3f}m, color={table_color}")
    print(f"  cube: pos={cube_pos}, size={cube_size}m, color={cube_color}")


def parse_color(s: str) -> tuple:
    """Parse 'r,g,b' string to (float, float, float)."""
    return tuple(float(x) for x in s.split(","))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cube-pos", nargs=3, type=float, default=[0.6, 0.0, 0.4],
                        metavar=("X", "Y", "Z"))
    parser.add_argument("--cube-size", type=float, default=0.08)
    parser.add_argument("--table-width", type=float, default=0.4)
    parser.add_argument("--table-depth", type=float, default=0.4)
    parser.add_argument("--cube-color", type=parse_color, default=(0.8, 0.1, 0.1),
                        help="RGB floats, e.g. '0.8,0.1,0.1'")
    parser.add_argument("--table-color", type=parse_color, default=(0.6, 0.5, 0.3),
                        help="RGB floats, e.g. '0.6,0.5,0.3'")
    parser.add_argument("--output", default="outputs/manip_reach_obj_scene.pt")
    args = parser.parse_args()
    create_reach_scene(
        tuple(args.cube_pos), args.cube_size,
        args.table_width, args.table_depth,
        args.cube_color, args.table_color, args.output,
    )
