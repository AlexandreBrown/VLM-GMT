"""
pipeline/generate_constraints.py — Generate Kimodo constraint JSON files.

Writes a constraints.json compatible with kimodo's load_constraints_lst().
Supported types: root2d, right-hand, left-hand, right-foot, left-foot.

How it works
------------
For hand/foot constraints, Kimodo stores local joint rotations in the JSON.
When loaded, FK is run to recover global joint positions, and the constrained
joints are used as position targets during diffusion. This means we need IK to
find the joint angles that place the end-effector at the desired world position.

We use a simple gradient-descent IK on Kimodo's own FK function (no external
IK library required). Only the relevant arm/leg joints are optimized.

Usage
-----
    # GT for reach_obj (run from VLM-GMT root)
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

KIMODO_MODEL = "kimodo-g1-rp"
G1_ROOT_HEIGHT = 0.793  # pelvis height in Kimodo y-up coords (standing pose)

# Joint index ranges in the G1 34-joint skeleton
RIGHT_ARM_INDICES = list(range(26, 34))  # shoulder_pitch … hand_roll
LEFT_ARM_INDICES  = list(range(18, 26))
RIGHT_LEG_INDICES = list(range(8, 15))
LEFT_LEG_INDICES  = list(range(1, 8))

# IK end-effector joint (last joint in each limb chain)
LIMB_EFFECTOR = {
    "right-hand": ("right_hand_roll_skel", RIGHT_ARM_INDICES),
    "left-hand":  ("left_hand_roll_skel",  LEFT_ARM_INDICES),
    "right-foot": ("right_toe_base",        RIGHT_LEG_INDICES),
    "left-foot":  ("left_toe_base",         LEFT_LEG_INDICES),
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
# IK
# --------------------------------------------------------------------------- #

def solve_ik(
    skeleton,
    target_pos_kimodo: list,
    effector_joint_name: str,
    limb_joint_indices: list,
    device: str = "cpu",
    lr: float = 0.05,
    steps: int = 400,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """
    Gradient-descent IK using Kimodo's FK.

    Optimises only the joints in `limb_joint_indices`; all other joints stay
    at zero (rest pose). Returns (local_aa [1, N_joints, 3], root_pos [1, 3]).
    """
    import torch
    from kimodo.geometry import axis_angle_to_matrix

    n_joints = skeleton.nbjoints
    root_pos = torch.tensor([[0.0, G1_ROOT_HEIGHT, 0.0]], dtype=torch.float32, device=device)
    target    = torch.tensor(target_pos_kimodo, dtype=torch.float32, device=device)
    joint_idx = skeleton.bone_index[effector_joint_name]
    limb_idx  = torch.tensor(limb_joint_indices, device=device)

    arm_aa = torch.zeros(len(limb_joint_indices), 3, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([arm_aa], lr=lr)

    for step in range(steps):
        optimizer.zero_grad()

        full_aa = torch.zeros(1, n_joints, 3, device=device)
        full_aa[0, limb_idx] = arm_aa

        rot_mats = axis_angle_to_matrix(full_aa)
        _, global_pos, _ = skeleton.fk(rot_mats, root_pos)

        hand_pos = global_pos[0, joint_idx]
        loss = ((hand_pos - target) ** 2).sum()
        loss.backward()
        optimizer.step()

        if (step + 1) % 100 == 0:
            dist = loss.item() ** 0.5
            print(f"  IK [{step+1}/{steps}] dist={dist:.4f}m  pos={hand_pos.detach().cpu().numpy().round(3).tolist()}")

    # Final full-body axis-angle (only arm joints non-zero)
    full_aa_final = torch.zeros(1, n_joints, 3, device=device)
    full_aa_final[0, limb_idx] = arm_aa.detach()

    # Recompute global state with final angles
    rot_mats = axis_angle_to_matrix(full_aa_final)
    _, global_pos_final, _ = skeleton.fk(rot_mats, root_pos)
    dist_final = ((global_pos_final[0, joint_idx] - target) ** 2).sum().item() ** 0.5
    print(f"  IK done: dist={dist_final:.4f}m  hand={global_pos_final[0, joint_idx].detach().cpu().numpy().round(3).tolist()}")

    return full_aa_final, root_pos

# --------------------------------------------------------------------------- #
# Constraint builders
# --------------------------------------------------------------------------- #

def build_limb_constraint(skeleton, constraint_type: str, target_kimodo: list,
                           frame_index: int, device: str = "cpu"):
    """
    Build a hand or foot EndEffectorConstraintSet via IK.

    constraint_type: one of 'right-hand', 'left-hand', 'right-foot', 'left-foot'
    target_kimodo:   [x, y, z] in Kimodo y-up coordinates
    """
    import torch
    from kimodo.geometry import axis_angle_to_matrix
    from kimodo.constraints import (
        RightHandConstraintSet, LeftHandConstraintSet,
        RightFootConstraintSet, LeftFootConstraintSet,
    )

    cls_map = {
        "right-hand": RightHandConstraintSet,
        "left-hand":  LeftHandConstraintSet,
        "right-foot": RightFootConstraintSet,
        "left-foot":  LeftFootConstraintSet,
    }
    if constraint_type not in cls_map:
        raise ValueError(f"Unknown constraint type '{constraint_type}'. "
                         f"Choose from {list(cls_map.keys())}")

    effector_joint, limb_indices = LIMB_EFFECTOR[constraint_type]
    print(f"[IK] {constraint_type} → target={target_kimodo}")

    local_aa, root_pos = solve_ik(skeleton, target_kimodo, effector_joint, limb_indices, device)

    rot_mats = axis_angle_to_matrix(local_aa)
    global_joints_rots, global_joints_positions, _ = skeleton.fk(rot_mats, root_pos)

    # smooth_root_2d = root x,z (robot stays at origin for reach_obj)
    smooth_root_2d = root_pos[:, [0, 2]]  # shape [1, 2]
    frame_indices  = torch.tensor([frame_index], device=device)

    return cls_map[constraint_type](
        skeleton=skeleton,
        frame_indices=frame_indices,
        global_joints_positions=global_joints_positions,
        global_joints_rots=global_joints_rots,
        smooth_root_2d=smooth_root_2d,
    )


def build_root2d_constraint(skeleton, x: float, z: float, frame_index: int, device: str = "cpu"):
    """Build a Root2DConstraintSet for a given root x,z position (Kimodo coords)."""
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
    """VLM condition: VLM pixel → world pos → right hand constraint."""
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
    parser.add_argument("--condition", choices=["gt", "vlm"], required=True,
                        help="Condition: gt (ground truth) or vlm (vision model)")
    parser.add_argument("--output", required=True,
                        help="Output path for constraints.json")
    parser.add_argument("--frame-index", type=int, default=45,
                        help="Keyframe index at 30fps (default 45 = 1.5s)")
    parser.add_argument("--kimodo-model", default=KIMODO_MODEL)

    # GT
    parser.add_argument("--cube-world-pos", nargs=3, type=float, metavar=("X", "Y", "Z"),
                        help="Cube position in IsaacLab coords (x y z)")

    # VLM
    parser.add_argument("--image", help="Path to RGB image for VLM")
    parser.add_argument("--camera-intrinsics", nargs=4, type=float,
                        metavar=("FX", "FY", "CX", "CY"))
    parser.add_argument("--camera-extrinsic-npy",
                        help="Path to 4x4 camera-to-world transform .npy")
    parser.add_argument("--assumed-world-z", type=float, default=0.4,
                        help="Assumed z height (IsaacLab) for ray-plane intersection")
    parser.add_argument("--object-description", default="red cube")
    parser.add_argument("--vlm-name", default="qwen2.5-vl-7b")

    args = parser.parse_args()

    import torch
    from kimodo import load_model
    from kimodo.constraints import save_constraints_lst

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[generate_constraints] device={device}")
    print(f"[generate_constraints] Loading model skeleton ({args.kimodo_model})...")
    model = load_model(args.kimodo_model, device=device)
    skeleton = model.skeleton

    # ── Build constraints ──────────────────────────────────────────────────
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
        raise ValueError(f"Unknown task '{args.task}'. Add a recipe above for new tasks.")

    # ── Save ───────────────────────────────────────────────────────────────
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_constraints_lst(str(out), constraints)
    print(f"[generate_constraints] Saved {len(constraints)} constraint(s) → {out}")


if __name__ == "__main__":
    main()
