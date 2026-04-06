"""Create a navigate_maze scene.

Enclosed corridor with 2 staggering internal walls:

    |--------------------------|
    |          |||             |
    | x        |||     |||    |
    |                  |||    |
    |--------------------------|

  x = robot start (origin)
  ||| = internal walls (red blocks top half, orange blocks bottom half)
  ---- = boundary walls (gray)

The robot must go RIGHT to pass wall 1, then LEFT to pass wall 2.

Scene objects (in order):
  0 — north boundary (gray)
  1 — south boundary (gray)
  2 — west boundary  (gray, behind robot)
  3 — east boundary  (gray, end)
  4 — wall 1 (red,    blocks top/left half at x=2.0)
  5 — wall 2 (orange, blocks bottom/right half at x=4.0)

All positions in IsaacLab frame: x=forward, y=left, z=up.
Robot starts at origin facing +x.

Usage:
    python tasks/navigate_maze/create_scene.py \
        --output outputs/navigate_maze_scene.pt
"""

import argparse
from protomotions.components.scene_lib import (
    BoxSceneObject,
    ObjectOptions,
    Scene,
    SceneLib,
)

# ── Defaults ──────────────────────────────────────────────────────────────────
WALL_THIN = 0.1
WALL_HEIGHT = 1.0

# Corridor bounds
CORRIDOR_X_MIN = -1.0
CORRIDOR_X_MAX = 5.0
CORRIDOR_Y_MIN = -1.5
CORRIDOR_Y_MAX = 1.5
CORRIDOR_LENGTH = CORRIDOR_X_MAX - CORRIDOR_X_MIN  # 6.0m
CORRIDOR_WIDTH = CORRIDOR_Y_MAX - CORRIDOR_Y_MIN    # 3.0m
CORRIDOR_X_CENTER = (CORRIDOR_X_MIN + CORRIDOR_X_MAX) / 2  # 2.0
CORRIDOR_Y_CENTER = 0.0

# Internal wall 1 at x=1.5: blocks top half (y from 0 to +1.5)
# Robot must pass through bottom gap (y from -1.5 to 0)
WALL1_X = 1.5
WALL1_POS = (WALL1_X, CORRIDOR_Y_MAX / 2, WALL_HEIGHT / 2)  # center at y=+0.75
WALL1_SIZE = (WALL_THIN, CORRIDOR_WIDTH / 2, WALL_HEIGHT)    # spans half the corridor

# Internal wall 2 at x=3.0: blocks bottom half (y from -1.5 to 0)
# Robot must pass through top gap (y from 0 to +1.5)
WALL2_X = 3.0
WALL2_POS = (WALL2_X, CORRIDOR_Y_MIN / 2, WALL_HEIGHT / 2)  # center at y=-0.75
WALL2_SIZE = (WALL_THIN, CORRIDOR_WIDTH / 2, WALL_HEIGHT)


def _make_wall(width, depth, height, translation, color):
    return BoxSceneObject(
        width=width,
        depth=depth,
        height=height,
        translation=translation,
        rotation=(0.0, 0.0, 0.0, 1.0),
        options=ObjectOptions(fix_base_link=True, color=color),
    )


GRAY = (0.5, 0.5, 0.5)
RED = (0.8, 0.1, 0.1)
ORANGE = (0.9, 0.5, 0.1)


def create_scene(
    wall1_pos: tuple = WALL1_POS,
    wall1_size: tuple = WALL1_SIZE,
    wall2_pos: tuple = WALL2_POS,
    wall2_size: tuple = WALL2_SIZE,
    output: str = "outputs/navigate_maze_scene.pt",
) -> None:
    h = WALL_HEIGHT / 2

    # Boundary walls
    north = _make_wall(CORRIDOR_LENGTH, WALL_THIN, WALL_HEIGHT,
                       (CORRIDOR_X_CENTER, CORRIDOR_Y_MAX, h), GRAY)
    south = _make_wall(CORRIDOR_LENGTH, WALL_THIN, WALL_HEIGHT,
                       (CORRIDOR_X_CENTER, CORRIDOR_Y_MIN, h), GRAY)
    west = _make_wall(WALL_THIN, CORRIDOR_WIDTH, WALL_HEIGHT,
                      (CORRIDOR_X_MIN, CORRIDOR_Y_CENTER, h), GRAY)
    east = _make_wall(WALL_THIN, CORRIDOR_WIDTH, WALL_HEIGHT,
                      (CORRIDOR_X_MAX, CORRIDOR_Y_CENTER, h), GRAY)

    # Internal walls
    wall1 = _make_wall(wall1_size[0], wall1_size[1], wall1_size[2],
                       tuple(wall1_pos), RED)
    wall2 = _make_wall(wall2_size[0], wall2_size[1], wall2_size[2],
                       tuple(wall2_pos), ORANGE)

    scene = Scene(objects=[north, south, west, east, wall1, wall2])
    SceneLib.save_scenes_to_file([scene], output)

    print(f"[create_scene] Saved '{output}'")
    print(f"  corridor: x=[{CORRIDOR_X_MIN}, {CORRIDOR_X_MAX}], y=[{CORRIDOR_Y_MIN}, {CORRIDOR_Y_MAX}]")
    print(f"  wall1 (red):    pos={wall1_pos} (blocks top half, pass BOTTOM/RIGHT)")
    print(f"  wall2 (orange): pos={wall2_pos} (blocks bottom half, pass TOP/LEFT)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/navigate_maze_scene.pt")
    args = parser.parse_args()
    create_scene(output=args.output)
