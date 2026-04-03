"""
pipeline/generate_constraints.py — Generate Kimodo constraint JSON files.

Writes a constraints.json compatible with kimodo's load_constraints_lst().
Supported types: root2d, right-hand, left-hand, right-foot, left-foot.

How it works
------------
Kimodo's hand/foot constraint classes (e.g. RightHandConstraintSet) accept
global joint positions and rotations directly — no IK required. We run FK in
the robot's rest pose (all joints zero) to get a valid global skeleton state,
then override the target end-effector's position with our desired world target.
Kimodo's diffusion model handles finding a motion that reaches that position.

Usage
-----
    # GT for reach_obj (run from VLM-GMT root, in kimodo env)
    python pipeline/generate_constraints.py \\
        --task reach_obj --condition gt \\
        --cube-world-pos 0.6 0.0 0.4 \\
        --output outputs/reach_obj/gt/constraints.json

    # VLM for reach_obj
    python pipeline/generate_constraints.py \\
        --task reach_obj --condition vlm \\
        --image outputs/reach_obj/sim_frame.png \\
        --camera-intrinsics 500 500 320 240 \\
        --camera-extrinsic-npy outputs/reach_obj/camera_T_world.npy \\
        --output outputs/reach_obj/vlm/constraints.json
"""

import argparse
import numpy as np
from pathlib import Path

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

KIMODO_MODEL   = "kimodo-g1-rp"
G1_ROOT_HEIGHT = 0.793  # pelvis height in Kimodo y-up standing pose (meters)

# End-effector joint names in the G1 skeleton (last joint of each limb chain)
LIMB_EFFECTOR_JOINT = {
    "right-hand": "right_hand_roll_skel",
    "left-hand":  "left_hand_roll_skel",
    "right-foot": "right_toe_base",
    "left-foot":  "left_toe_base",
}

# --------------------------------------------------------------------------- #
# Coordinate helpers
# --------------------------------------------------------------------------- #

def isaaclab_to_kimodo(pos: np.ndarray) -> list:
    """IsaacLab (x forward, y left, z up) → Kimodo (x forward, y up, z lateral)."""
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    return [x, z, y]


def pixels_to_world(
    u: float, v: float,
    fx: float, fy: float, cx: float, cy: float,
    cam_T_world: np.ndarray,
    assumed_world_z: float,
) -> np.ndarray:
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
# Constraint builder
# --------------------------------------------------------------------------- #

def build_limb_constraint(skeleton, constraint_type: str, target_kimodo: list,
                           frame_index: int, device: str = "cpu"):
    """
    Build a hand or foot EndEffectorConstraintSet.

    Runs FK in rest pose, overrides the target EE joint's global position with
    `target_kimodo`, then passes the full skeleton state to the constraint class.
    No IK needed — Kimodo's diffusion handles finding a motion that reaches the target.

    constraint_type: 'right-hand' | 'left-hand' | 'right-foot' | 'left-foot'
    target_kimodo:   [x, y, z] in Kimodo y-up coordinates
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
        raise ValueError(f"Unknown constraint type '{constraint_type}'. "
                         f"Valid: {list(cls_map.keys())}")

    n_joints = skeleton.nbjoints
    root_pos = torch.tensor([[0.0, G1_ROOT_HEIGHT, 0.0]], dtype=torch.float32, device=device)

    # Rest pose: all joints at zero rotation
    rest_aa   = torch.zeros(1, n_joints, 3, device=device)
    rest_mats = axis_angle_to_matrix(rest_aa)

    # FK → global positions and rotations for the full skeleton in rest pose
    global_rots, global_pos, _ = skeleton.fk(rest_mats, root_pos)

    # Override target joint position with our desired world target
    effector_joint = LIMB_EFFECTOR_JOINT[constraint_type]
    effector_idx   = skeleton.bone_index[effector_joint]
    target_tensor  = torch.tensor(target_kimodo, dtype=torch.float32, device=device)
    global_pos[0, effector_idx] = target_tensor

    print(f"[{constraint_type}] target (Kimodo)={target_kimodo}  "
          f"rest-pose effector={global_pos[0, effector_idx].tolist()}")

    # smooth_root_2d: robot stays at origin (x=0, z=0)
    smooth_root_2d = root_pos[:, [0, 2]]
    frame_indices  = torch.tensor([frame_index], device=device)

    return cls_map[constraint_type](
        skeleton=skeleton,
        frame_indices=frame_indices,
        global_joints_positions=global_pos,
        global_joints_rots=global_rots,
        smooth_root_2d=smooth_root_2d,
    )


def build_root2d_constraint(skeleton, x: float, z: float, frame_index: int, device: str = "cpu"):
    """Build a Root2DConstraintSet (root x,z waypoint in Kimodo coords)."""
    import torch
    from kimodo.constraints import Root2DConstraintSet

    return Root2DConstraintSet(
        skeleton=skeleton,
        frame_indices=torch.tensor([frame_index], device=device),
        smooth_root_2d=torch.tensor([[x, z]], device=device),
    )

# --------------------------------------------------------------------------- #
# Task-specific recipes
# --------------------------------------------------------------------------- #

def constraints_reach_obj_gt(skeleton, cube_world_pos: np.ndarray, frame_index: int,
                              device: str) -> list:
    """GT upper bound: right hand constrained to cube position."""
    target = isaaclab_to_kimodo(cube_world_pos)
    print(f"[GT] cube (IsaacLab)={cube_world_pos.tolist()}  →  Kimodo={target}")
    return [build_limb_constraint(skeleton, "right-hand", target, frame_index, device)]


def constraints_reach_obj_vlm(skeleton, image_rgb, fx, fy, cx, cy, cam_T_world,
                               assumed_world_z, object_description, vlm_name,
                               frame_index: int, device: str) -> list:
    """VLM condition: pixel → world pos → right hand constraint."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pipeline.vlm import load_vlm

    vlm = load_vlm(vlm_name)
    u, v = vlm.query_object_pixels(image_rgb, object_description)
    print(f"[VLM] pixel: u={u:.1f}  v={v:.1f}")

    target_world = pixels_to_world(u, v, fx, fy, cx, cy, cam_T_world, assumed_world_z)
    print(f"[VLM] world (IsaacLab): {target_world.tolist()}")
    target = isaaclab_to_kimodo(target_world)
    print(f"[VLM] Kimodo: {target}")

    return [build_limb_constraint(skeleton, "right-hand", target, frame_index, device)]

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Generate Kimodo constraint JSON for a given task and condition."
    )
    parser.add_argument("--task", default="reach_obj",
                        help="Task name (default: reach_obj)")
    parser.add_argument("--condition", choices=["gt", "vlm"], required=True)
    parser.add_argument("--output", required=True,
                        help="Output path for constraints.json")
    parser.add_argument("--frame-index", type=int, default=45,
                        help="Keyframe index at 30fps (default 45 = 1.5s)")
    parser.add_argument("--kimodo-model", default=KIMODO_MODEL)

    # GT
    parser.add_argument("--cube-world-pos", nargs=3, type=float, metavar=("X", "Y", "Z"),
                        help="Cube position in IsaacLab coords (x y z)")

    # VLM
    parser.add_argument("--image")
    parser.add_argument("--camera-intrinsics", nargs=4, type=float,
                        metavar=("FX", "FY", "CX", "CY"))
    parser.add_argument("--camera-extrinsic-npy")
    parser.add_argument("--assumed-world-z", type=float, default=0.4)
    parser.add_argument("--object-description", default="red cube")
    parser.add_argument("--vlm-name", default="qwen2.5-vl-7b")

    args = parser.parse_args()

    import torch
    from kimodo import load_model
    from kimodo.constraints import save_constraints_lst

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[generate_constraints] device={device}")
    print(f"[generate_constraints] Loading model ({args.kimodo_model})...")
    model = load_model(args.kimodo_model, device=device)
    skeleton = model.skeleton

    if args.task == "reach_obj":
        if args.condition == "gt":
            if args.cube_world_pos is None:
                parser.error("--cube-world-pos required for reach_obj gt")
            constraints = constraints_reach_obj_gt(
                skeleton, np.array(args.cube_world_pos, dtype=np.float32),
                args.frame_index, device,
            )
        else:  # vlm
            if not all([args.image, args.camera_intrinsics, args.camera_extrinsic_npy]):
                parser.error("--image, --camera-intrinsics, --camera-extrinsic-npy required for vlm")
            from PIL import Image as PILImage
            image_rgb = np.array(PILImage.open(args.image).convert("RGB"))
            fx, fy, cx, cy = args.camera_intrinsics
            cam_T_world = np.load(args.camera_extrinsic_npy).astype(np.float32)
            constraints = constraints_reach_obj_vlm(
                skeleton, image_rgb, fx, fy, cx, cy, cam_T_world,
                args.assumed_world_z, args.object_description, args.vlm_name,
                args.frame_index, device,
            )
    else:
        raise ValueError(f"Unknown task '{args.task}'. Add a recipe above.")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_constraints_lst(str(out), constraints)
    print(f"[generate_constraints] Saved {len(constraints)} constraint(s) → {out}")


if __name__ == "__main__":
    main()
