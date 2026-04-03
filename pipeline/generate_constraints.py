"""
pipeline/generate_constraints.py — Generate Kimodo kinematic constraints.

Three modes:
    gt    Ground-truth target position from sim state (upper bound)
    vlm   VLM estimates target position from a camera image
    none  No constraint (text-only baseline, produces no output file)

Requires kimodo installed (run in kimodo env on cluster).

Usage:
    # GT
    python VLM-GMT/pipeline/generate_constraints.py --mode gt \
        --cube-world-pos 0.6 0.0 0.4 \
        --output VLM-GMT/outputs/reach_obj/gt/constraints.json

    # VLM
    python VLM-GMT/pipeline/generate_constraints.py --mode vlm \
        --image sim_frame.png \
        --camera-intrinsics 500 500 320 240 \
        --camera-extrinsic-npy camera_T_world.npy \
        --output VLM-GMT/outputs/reach_obj/vlm/constraints.json

    # Baseline (no file generated)
    python VLM-GMT/pipeline/generate_constraints.py --mode none
"""

import argparse
import numpy as onp
from pathlib import Path


G1_ROOT_HEIGHT = 0.793   # default G1 pelvis height in standing pose (meters)
CONSTRAINT_FRAME = 45    # 30fps → 1.5s into a 3s motion


# ---------------------------------------------------------------------------
# Kimodo constraint builders
# Uses Kimodo Python API — no IK needed.
# EndEffectorConstraintSet takes global joint positions; Kimodo resolves
# joint angles internally during generation.
# ---------------------------------------------------------------------------

def _load_skeleton(model_name: str = "kimodo-g1-rp"):
    from kimodo import load_model
    return load_model(model_name, device="cpu").skeleton


def build_root2d(target_world_pos: onp.ndarray, frame_index: int) -> object:
    """Root2D waypoint: walk root toward target XZ position by frame_index."""
    import torch
    from kimodo.constraints import Root2DConstraintSet

    smooth_root_2d = torch.tensor([
        [0.0, 0.0],
        [float(target_world_pos[0]), float(target_world_pos[2])],
    ])
    return Root2DConstraintSet(
        skeleton=_load_skeleton(),
        frame_indices=[0, frame_index],
        smooth_root_2d=smooth_root_2d,
    )


def build_right_hand(
    target_world_pos: onp.ndarray,
    frame_index: int,
    root_world_pos: onp.ndarray,
) -> object:
    """
    Right-hand end-effector constraint at target_world_pos.
    Sets the RightHand joint to the target; Kimodo handles the rest.
    """
    import torch
    from kimodo.constraints import RightHandConstraintSet

    skeleton = _load_skeleton()
    joint_names = list(skeleton.bone_order_names)
    n_joints = len(joint_names)

    global_positions = torch.zeros(1, n_joints, 3)
    global_rots = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(1, n_joints, 3, 3).clone()

    rh_idx = joint_names.index("right_hand_roll_skel")
    global_positions[0, rh_idx] = torch.tensor(target_world_pos, dtype=torch.float32)

    smooth_root_2d = torch.tensor([[float(root_world_pos[0]), float(root_world_pos[2])]])

    return RightHandConstraintSet(
        skeleton=skeleton,
        frame_indices=[frame_index],
        global_joints_positions=global_positions,
        global_joints_rots=global_rots,
        smooth_root_2d=smooth_root_2d,
    )


# ---------------------------------------------------------------------------
# Camera unprojection (pixels → world 3D)
# ---------------------------------------------------------------------------

def pixels_to_world(
    u: float, v: float,
    fx: float, fy: float, cx: float, cy: float,
    cam_T_world: onp.ndarray,
    assumed_world_z: float,
) -> onp.ndarray:
    """Back-project pixel (u, v) to world 3D via ray-plane intersection at z=assumed_world_z."""
    ray_cam = onp.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    R, t = cam_T_world[:3, :3], cam_T_world[:3, 3]
    ray_world = R @ ray_cam
    if abs(ray_world[2]) < 1e-6:
        raise ValueError("Ray nearly horizontal — can't intersect z-plane.")
    lam = (assumed_world_z - t[2]) / ray_world[2]
    pos = t + lam * ray_world
    pos[2] = assumed_world_z
    return pos.astype(onp.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["gt", "vlm", "none"], required=True)
    parser.add_argument("--output", default="constraints.json")
    parser.add_argument("--frame-index", type=int, default=CONSTRAINT_FRAME)
    parser.add_argument(
        "--root-world-pos", nargs=3, type=float,
        default=[0.0, 0.0, G1_ROOT_HEIGHT], metavar=("X", "Y", "Z"),
    )
    # GT
    parser.add_argument("--cube-world-pos", nargs=3, type=float, metavar=("X", "Y", "Z"))
    # VLM
    parser.add_argument("--image")
    parser.add_argument("--camera-intrinsics", nargs=4, type=float, metavar=("FX", "FY", "CX", "CY"))
    parser.add_argument("--camera-extrinsic-npy")
    parser.add_argument("--assumed-world-z", type=float, default=0.4)
    parser.add_argument("--object-description", default="red cube")
    parser.add_argument("--vlm-name", default="qwen2.5-vl-7b")

    args = parser.parse_args()

    if args.mode == "none":
        print("[generate_constraints] mode=none: no file generated.")
        return

    root_world_pos = onp.array(args.root_world_pos, dtype=onp.float32)

    if args.mode == "gt":
        if args.cube_world_pos is None:
            parser.error("--cube-world-pos required for --mode gt")
        target_pos = onp.array(args.cube_world_pos, dtype=onp.float32)
        print(f"[generate_constraints] GT target: {target_pos}")

    elif args.mode == "vlm":
        if not all([args.image, args.camera_intrinsics, args.camera_extrinsic_npy]):
            parser.error("--image, --camera-intrinsics, --camera-extrinsic-npy required for --mode vlm")
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from pipeline.vlm import load_vlm
        from PIL import Image as PILImage

        image_rgb = onp.array(PILImage.open(args.image).convert("RGB"))
        fx, fy, cx, cy = args.camera_intrinsics
        cam_T_world = onp.load(args.camera_extrinsic_npy).astype(onp.float32)

        print(f"[generate_constraints] Querying VLM ({args.vlm_name})...")
        vlm = load_vlm(args.vlm_name)
        u, v = vlm.query_object_pixels(image_rgb, args.object_description)
        print(f"[generate_constraints] VLM pixel estimate: u={u:.1f} v={v:.1f}")

        target_pos = pixels_to_world(u, v, fx, fy, cx, cy, cam_T_world, args.assumed_world_z)
        print(f"[generate_constraints] Unprojected world pos: {target_pos}")

    from kimodo.constraints import save_constraints_lst

    constraints = [
        build_root2d(target_pos, args.frame_index),
        build_right_hand(target_pos, args.frame_index, root_world_pos),
    ]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_constraints_lst(str(out), constraints)
    print(f"[generate_constraints] Saved to '{out}'")


if __name__ == "__main__":
    main()
