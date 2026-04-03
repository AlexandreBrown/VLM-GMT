"""
pipeline/generate_constraints.py — Build Kimodo constraint objects in memory.

Returns a list of constraint objects ready to pass to model(constraint_lst=...).
No JSON serialization needed — Kimodo's in-memory path uses global_joints_positions
directly, so no IK is required.

Supported constraint types: right-hand, left-hand, right-foot, left-foot, root2d.
Multiple constraints of different types / frame indices are supported.

Coordinate conventions
----------------------
- IsaacLab: x forward, y left, z up
- Kimodo:   x forward, y up,   z lateral
Use `isaaclab_to_kimodo()` to convert.

Adding a new task
-----------------
Add a new function `constraints_<task>_<condition>(skeleton, ..., device)` that
returns a list of constraint objects, then register it in `build_constraints()`.
"""

import numpy as np

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

KIMODO_MODEL   = "kimodo-g1-rp"
G1_ROOT_HEIGHT = 0.793  # pelvis height in Kimodo y-up standing pose (meters)

LIMB_EFFECTOR_JOINT = {
    "right-hand": "right_hand_roll_skel",
    "left-hand":  "left_hand_roll_skel",
    "right-foot": "right_toe_base",
    "left-foot":  "left_toe_base",
}

# --------------------------------------------------------------------------- #
# Coordinate helpers
# --------------------------------------------------------------------------- #

def isaaclab_to_kimodo(pos) -> list:
    """IsaacLab (x forward, y left, z up) → Kimodo (x forward, y up, z lateral)."""
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    return [x, z, y]


def pixels_to_world(u, v, fx, fy, cx, cy, cam_T_world, assumed_world_z):
    """Back-project pixel (u,v) to world 3D via ray–plane intersection at z=assumed_world_z."""
    ray_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    R, t = cam_T_world[:3, :3], cam_T_world[:3, 3]
    ray_world = R @ ray_cam
    if abs(ray_world[2]) < 1e-6:
        raise ValueError("Ray is nearly horizontal — cannot intersect the z-plane.")
    lam = (assumed_world_z - t[2]) / ray_world[2]
    pos = t + lam * ray_world
    pos[2] = assumed_world_z
    return pos.astype(np.float32)

# --------------------------------------------------------------------------- #
# Low-level constraint builders
# --------------------------------------------------------------------------- #

def make_limb_constraint(skeleton, constraint_type: str, target_kimodo: list,
                          frame_index: int, device: str = "cpu"):
    """
    Build a single hand or foot EndEffectorConstraintSet.

    Runs FK in rest pose to get a valid skeleton state, then overrides the
    target EE joint's global position with `target_kimodo`. Kimodo's diffusion
    uses global_joints_positions directly (in-memory path) — no IK needed.

    Args:
        constraint_type: 'right-hand' | 'left-hand' | 'right-foot' | 'left-foot'
        target_kimodo:   [x, y, z] in Kimodo y-up coordinates
        frame_index:     frame at which to apply the constraint (30fps)
    """
    import torch
    from kimodo.constraints import (
        RightHandConstraintSet, LeftHandConstraintSet,
        RightFootConstraintSet, LeftFootConstraintSet,
    )
    from kimodo.geometry import axis_angle_to_matrix

    cls_map = {
        "right-hand": RightHandConstraintSet,
        "left-hand":  LeftHandConstraintSet,
        "right-foot": RightFootConstraintSet,
        "left-foot":  LeftFootConstraintSet,
    }
    if constraint_type not in cls_map:
        raise ValueError(f"Unknown type '{constraint_type}'. Valid: {list(cls_map.keys())}")

    n_joints  = skeleton.nbjoints
    root_pos  = torch.tensor([[0.0, G1_ROOT_HEIGHT, 0.0]], dtype=torch.float32, device=device)
    rest_mats = axis_angle_to_matrix(torch.zeros(1, n_joints, 3, device=device))

    global_rots, global_pos, _ = skeleton.fk(rest_mats, root_pos)

    # Override target EE joint's global position
    effector_idx          = skeleton.bone_index[LIMB_EFFECTOR_JOINT[constraint_type]]
    global_pos            = global_pos.clone()
    global_pos[0, effector_idx] = torch.tensor(target_kimodo, dtype=torch.float32, device=device)

    smooth_root_2d = root_pos[:, [0, 2]]
    frame_indices  = torch.tensor([frame_index], device=device)

    print(f"  [{constraint_type}] frame={frame_index}  target={[round(v,3) for v in target_kimodo]}")

    return cls_map[constraint_type](
        skeleton=skeleton,
        frame_indices=frame_indices,
        global_joints_positions=global_pos,
        global_joints_rots=global_rots,
        smooth_root_2d=smooth_root_2d,
    )


def make_root2d_constraint(skeleton, x: float, z: float, frame_index: int, device: str = "cpu"):
    """Root2D constraint: fix robot root (x, z) position at a given frame."""
    import torch
    from kimodo.constraints import Root2DConstraintSet

    print(f"  [root2d] frame={frame_index}  x={x:.3f}  z={z:.3f}")
    return Root2DConstraintSet(
        skeleton=skeleton,
        frame_indices=torch.tensor([frame_index], device=device),
        smooth_root_2d=torch.tensor([[x, z]], device=device),
    )

# --------------------------------------------------------------------------- #
# Task / condition recipes
# --------------------------------------------------------------------------- #

def constraints_reach_obj_gt(skeleton, cube_world_pos, frame_index: int, device: str) -> list:
    """
    GT upper bound for reach_obj: right hand constrained to cube world position.

    cube_world_pos: [x, y, z] in IsaacLab coordinates
    """
    target = isaaclab_to_kimodo(cube_world_pos)
    print(f"[GT] cube (IsaacLab)={list(np.round(cube_world_pos, 3))}  →  Kimodo={[round(v,3) for v in target]}")
    return [make_limb_constraint(skeleton, "right-hand", target, frame_index, device)]


def constraints_reach_obj_vlm(skeleton, image_rgb, fx, fy, cx, cy, cam_T_world,
                               assumed_world_z, object_description, vlm_name,
                               frame_index: int, device: str) -> list:
    """
    VLM condition for reach_obj: VLM estimates pixel → world → right hand constraint.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pipeline.vlm import load_vlm

    vlm = load_vlm(vlm_name)
    u, v = vlm.query_object_pixels(image_rgb, object_description)
    print(f"[VLM] pixel: u={u:.1f}  v={v:.1f}")

    target_world = pixels_to_world(u, v, fx, fy, cx, cy, cam_T_world, assumed_world_z)
    print(f"[VLM] world (IsaacLab): {target_world.tolist()}")
    target = isaaclab_to_kimodo(target_world)

    return [make_limb_constraint(skeleton, "right-hand", target, frame_index, device)]

# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def build_constraints(task: str, condition: str, skeleton, device: str, **kwargs) -> list:
    """
    Build and return a list of Kimodo constraint objects for a given task/condition.

    kwargs are task-specific (see individual recipe functions above).
    Returns [] for baseline (no constraints).

    Example — reach_obj gt:
        constraints = build_constraints(
            "reach_obj", "gt", skeleton, device,
            cube_world_pos=np.array([0.6, 0.0, 0.4]),
            frame_index=45,
        )

    Example — multi-constraint (future tasks):
        constraints = [
            make_limb_constraint(skeleton, "right-hand", target_rh, frame_5, device),
            make_limb_constraint(skeleton, "left-foot",  target_lf, frame_18, device),
            make_root2d_constraint(skeleton, x=0.3, z=0.0, frame_index=30, device=device),
        ]
    """
    if condition == "baseline":
        return []

    if task == "reach_obj":
        frame_index = kwargs.get("frame_index", 45)
        if condition == "gt":
            return constraints_reach_obj_gt(
                skeleton, np.array(kwargs["cube_world_pos"], dtype=np.float32),
                frame_index, device,
            )
        elif condition == "vlm":
            return constraints_reach_obj_vlm(
                skeleton,
                kwargs["image_rgb"],
                kwargs["fx"], kwargs["fy"], kwargs["cx"], kwargs["cy"],
                kwargs["cam_T_world"],
                kwargs.get("assumed_world_z", 0.4),
                kwargs.get("object_description", "red cube"),
                kwargs.get("vlm_name", "qwen2.5-vl-7b"),
                frame_index, device,
            )

    raise ValueError(f"Unknown task/condition: {task}/{condition}")
