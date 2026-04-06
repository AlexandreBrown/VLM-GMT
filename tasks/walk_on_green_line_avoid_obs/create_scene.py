"""Create a walk_on_green_line_avoid_obs scene.

A 1.0m-wide green line runs 5.5m forward from the robot. Three colored
obstacle boxes sit on the line at staggered lateral offsets, blocking
naive straight-line walking. The robot must navigate around each one
while staying on the line.

Scene objects (in order):
  0 — green line (flat box, ground level)
  1 — obstacle 1 (red,    small,  right-of-center)
  2 — obstacle 2 (orange, medium, left-of-center)
  3 — obstacle 3 (red,    small,  right-of-center)

All positions in IsaacLab frame: x=forward, y=left, z=up.
Robot starts at origin facing +x.

Usage:
    python tasks/walk_on_green_line_avoid_obs/create_scene.py \
        --output outputs/walk_on_green_line_avoid_obs_scene.pt
"""

import argparse
from protomotions.components.scene_lib import (
    BoxSceneObject,
    ObjectOptions,
    Scene,
    SceneLib,
)

# ── Defaults ──────────────────────────────────────────────────────────────────
# Line: 1.0m wide, 5.5m long, very thin (0.02m), center at (3.0, 0.0, 0.01)
LINE_POS   = (3.0, 0.0, 0.01)
LINE_WIDTH = 1.0    # y-axis
LINE_DEPTH = 5.5    # x-axis
LINE_HEIGHT = 0.02

# Obstacle positions and sizes (x_forward, y_left, z_center)
# Obs 1: slightly left (+y), small
OBS1_POS  = (1.5,  0.20, 0.30)
OBS1_SIZE = (0.30, 0.30, 0.60)  # (width=y, depth=x, height=z)

# Obs 2: slightly right (-y), medium
OBS2_POS  = (3.0, -0.20, 0.35)
OBS2_SIZE = (0.35, 0.35, 0.70)

# Obs 3: slightly left (+y), small
OBS3_POS  = (4.5,  0.15, 0.25)
OBS3_SIZE = (0.25, 0.30, 0.50)

LINE_END_X = LINE_POS[0] + LINE_DEPTH / 2  # 5.75m — used as metric target


def create_scene(
    line_pos: tuple = LINE_POS,
    line_width: float = LINE_WIDTH,
    line_depth: float = LINE_DEPTH,
    line_height: float = LINE_HEIGHT,
    obs1_pos: tuple = OBS1_POS,
    obs1_size: tuple = OBS1_SIZE,
    obs2_pos: tuple = OBS2_POS,
    obs2_size: tuple = OBS2_SIZE,
    obs3_pos: tuple = OBS3_POS,
    obs3_size: tuple = OBS3_SIZE,
    output: str = "outputs/walk_on_green_line_avoid_obs_scene.pt",
) -> None:
    line = BoxSceneObject(
        width=line_width,
        depth=line_depth,
        height=line_height,
        translation=tuple(line_pos),
        rotation=(0.0, 0.0, 0.0, 1.0),
        options=ObjectOptions(fix_base_link=True, color=(0.2, 0.7, 0.2)),
    )
    obs1 = BoxSceneObject(
        width=obs1_size[0],
        depth=obs1_size[1],
        height=obs1_size[2],
        translation=tuple(obs1_pos),
        rotation=(0.0, 0.0, 0.0, 1.0),
        options=ObjectOptions(fix_base_link=True, color=(0.8, 0.1, 0.1)),
    )
    obs2 = BoxSceneObject(
        width=obs2_size[0],
        depth=obs2_size[1],
        height=obs2_size[2],
        translation=tuple(obs2_pos),
        rotation=(0.0, 0.0, 0.0, 1.0),
        options=ObjectOptions(fix_base_link=True, color=(0.9, 0.5, 0.1)),
    )
    obs3 = BoxSceneObject(
        width=obs3_size[0],
        depth=obs3_size[1],
        height=obs3_size[2],
        translation=tuple(obs3_pos),
        rotation=(0.0, 0.0, 0.0, 1.0),
        options=ObjectOptions(fix_base_link=True, color=(0.8, 0.1, 0.1)),
    )

    scene = Scene(objects=[line, obs1, obs2, obs3])
    SceneLib.save_scenes_to_file([scene], output)

    end_x = line_pos[0] + line_depth / 2
    print(f"[create_scene] Saved '{output}'")
    print(f"  line:  center={line_pos}, {line_width}m wide x {line_depth}m long")
    print(f"  obs1:  pos={obs1_pos}, size={obs1_size}")
    print(f"  obs2:  pos={obs2_pos}, size={obs2_size}")
    print(f"  obs3:  pos={obs3_pos}, size={obs3_size}")
    print(f"  line end (metric target): x={end_x:.2f}, y=0.0")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/walk_on_green_line_avoid_obs_scene.pt")
    # Obstacle position overrides (optional)
    parser.add_argument("--obs1-pos", nargs=3, type=float, default=list(OBS1_POS), metavar=("X", "Y", "Z"))
    parser.add_argument("--obs2-pos", nargs=3, type=float, default=list(OBS2_POS), metavar=("X", "Y", "Z"))
    parser.add_argument("--obs3-pos", nargs=3, type=float, default=list(OBS3_POS), metavar=("X", "Y", "Z"))
    args = parser.parse_args()
    create_scene(
        obs1_pos=tuple(args.obs1_pos),
        obs2_pos=tuple(args.obs2_pos),
        obs3_pos=tuple(args.obs3_pos),
        output=args.output,
    )
