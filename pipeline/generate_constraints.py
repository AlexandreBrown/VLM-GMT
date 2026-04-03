"""
pipeline/generate_constraints.py — Generate Kimodo EndEffector constraints.

Uses EndEffectorConstraintSet with explicit joint_names for all tasks.
Kimodo only constrains the specified joints, leaving everything else free.
Max 20 keyframes per constraint type (Kimodo limit).

Available G1 joint names: see G1_JOINT_NAMES below.

Three modes:
    gt    Ground-truth positions from sim state (upper bound)
    vlm   VLM estimates positions from a camera image
    none  No constraint (text-only baseline, no output file)

Requires kimodo installed (run in kimodo env).

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

    # Baseline
    python VLM-GMT/pipeline/generate_constraints.py --mode none
"""

import argparse
import numpy as np
from pathlib import Path


# G1 skeleton joint names (34 joints, from skeleton.bone_order_names)
G1_JOINT_NAMES = [
    "pelvis_skel",
    "left_hip_pitch_skel", "left_hip_roll_skel", "left_hip_yaw_skel",
    "left_knee_skel", "left_ankle_pitch_skel", "left_ankle_roll_skel", "left_toe_base",
    "right_hip_pitch_skel", "right_hip_roll_skel", "right_hip_yaw_skel",
    "right_knee_skel", "right_ankle_pitch_skel", "right_ankle_roll_skel", "right_toe_base",
    "waist_yaw_skel", "waist_roll_skel", "waist_pitch_skel",
    "left_shoulder_pitch_skel", "left_shoulder_roll_skel", "left_shoulder_yaw_skel",
    "left_elbow_skel", "left_wrist_roll_skel", "left_wrist_pitch_skel",
    "left_wrist_yaw_skel", "left_hand_roll_skel",
    "right_shoulder_pitch_skel", "right_shoulder_roll_skel", "right_shoulder_yaw_skel",
    "right_elbow_skel", "right_wrist_roll_skel", "right_wrist_pitch_skel",
    "right_wrist_yaw_skel", "right_hand_roll_skel",
]

G1_ROOT_HEIGHT = 0.793   # default G1 pelvis height in standing pose (meters)
MAX_CONSTRAINT_FRAMES = 20  # Kimodo hard limit per constraint type


def _load_skeleton(model_name: str = "kimodo-g1-rp"):
    from kimodo import load_model
    return load_model(model_name, device="cpu").skeleton


def build_end_effector_constraints(
    keyframes: list[dict],
    root_height: float = G1_ROOT_HEIGHT,
) -> list:
    """
    Build Kimodo EndEffectorConstraintSet from keyframes.

    Kimodo only constrains the joints listed in each keyframe's "joints" dict.
    Everything else is left free (no zero-position artifacts).

    Args:
        keyframes: list of dicts, each:
            {
                "frame_index": int,
                "joints": {joint_name: [x, y, z], ...}  # world positions
            }
            All keyframes must specify the same set of joint names.
            If empty, returns [].
            If > 20, first 20 are used.

    Returns:
        List containing a single EndEffectorConstraintSet, or [] if no keyframes.
    """
    import torch
    from kimodo.constraints import EndEffectorConstraintSet

    if not keyframes:
        return []

    if len(keyframes) > MAX_CONSTRAINT_FRAMES:
        print(f"[build_end_effector_constraints] {len(keyframes)} keyframes exceeds limit {MAX_CONSTRAINT_FRAMES}, using first {MAX_CONSTRAINT_FRAMES}.")
        keyframes = keyframes[:MAX_CONSTRAINT_FRAMES]

    skeleton = _load_skeleton()
    n_joints = len(G1_JOINT_NAMES)
    T = len(keyframes)

    # Collect EE names (must be from EE_JOINT_NAMES)
    constrained_ee_names = list(keyframes[0].get("joints", {}).keys())
    for ee_name in constrained_ee_names:
        if ee_name not in EE_JOINT_NAMES:
            raise ValueError(f"Unknown EE name: '{ee_name}'. Valid: {EE_JOINT_NAMES}")

    global_positions = torch.zeros(T, n_joints, 3)
    global_rots = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(T, n_joints, 3, 3).clone()
    frame_indices = []

    # Map from high-level EE name to skeleton joint index for position setting
    # Kimodo uses the last joint in each chain as the position target
    ee_to_skel_joint = {
        "RightHand": "right_hand_roll_skel",
        "LeftHand": "left_hand_roll_skel",
        "RightFoot": "right_toe_base",
        "LeftFoot": "left_toe_base",
        "Hips": "pelvis_skel",
    }

    for t, kf in enumerate(keyframes):
        frame_indices.append(kf["frame_index"])

        # Set pelvis to standing height
        pelvis_idx = G1_JOINT_NAMES.index("pelvis_skel")
        global_positions[t, pelvis_idx] = torch.tensor([0.0, root_height, 0.0])

        for ee_name, world_pos in kf.get("joints", {}).items():
            skel_joint = ee_to_skel_joint[ee_name]
            idx = G1_JOINT_NAMES.index(skel_joint)
            global_positions[t, idx] = torch.tensor(world_pos, dtype=torch.float32)

    smooth_root_2d = global_positions[:, G1_JOINT_NAMES.index("pelvis_skel"), [0, 2]]  # (T, 2)

    return [EndEffectorConstraintSet(
        skeleton=skeleton,
        frame_indices=frame_indices,
        global_joints_positions=global_positions,
        global_joints_rots=global_rots,
        smooth_root_2d=smooth_root_2d,
        joint_names=constrained_ee_names,
    )]


# ---------------------------------------------------------------------------
# GT keyframe builder for reach_obj
# ---------------------------------------------------------------------------

# Valid end-effector names for EndEffectorConstraintSet
EE_JOINT_NAMES = ["LeftFoot", "RightFoot", "LeftHand", "RightHand", "Hips"]


def isaaclab_to_kimodo(pos: np.ndarray) -> list:
    """Convert IsaacLab (x, y, z-up) to Kimodo (x, y-up, z) convention."""
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    return [x, z, y]  # Kimodo: x=forward, y=height, z=lateral


def gt_reach_keyframes(target_world_pos: np.ndarray, frame_index: int) -> list[dict]:
    """GT keyframes for reach_obj: right hand at target position.

    target_world_pos: IsaacLab convention (x, y, z-up).
    Converted to Kimodo y-up convention internally.
    """
    return [{
        "frame_index": frame_index,
        "joints": {
            "RightHand": isaaclab_to_kimodo(target_world_pos),
        }
    }]


# ---------------------------------------------------------------------------
# Camera unprojection
# ---------------------------------------------------------------------------

def pixels_to_world(
    u: float, v: float,
    fx: float, fy: float, cx: float, cy: float,
    cam_T_world: np.ndarray,
    assumed_world_z: float,
) -> np.ndarray:
    """Back-project pixel (u, v) to world 3D via ray-plane intersection at z=assumed_world_z."""
    ray_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    R, t = cam_T_world[:3, :3], cam_T_world[:3, 3]
    ray_world = R @ ray_cam
    if abs(ray_world[2]) < 1e-6:
        raise ValueError("Ray nearly horizontal — can't intersect z-plane.")
    lam = (assumed_world_z - t[2]) / ray_world[2]
    pos = t + lam * ray_world
    pos[2] = assumed_world_z
    return pos.astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["gt", "vlm", "none"], required=True)
    parser.add_argument("--output", default="constraints.json")
    parser.add_argument("--frame-index", type=int, default=45,
                        help="Keyframe index (30fps, default 45 = 1.5s)")
    parser.add_argument("--root-height", type=float, default=G1_ROOT_HEIGHT)

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

    if args.mode == "gt":
        if args.cube_world_pos is None:
            parser.error("--cube-world-pos required for --mode gt")
        target_pos = np.array(args.cube_world_pos, dtype=np.float32)
        print(f"[generate_constraints] GT target: {target_pos}")
        keyframes = gt_reach_keyframes(target_pos, args.frame_index)

    elif args.mode == "vlm":
        if not all([args.image, args.camera_intrinsics, args.camera_extrinsic_npy]):
            parser.error("--image, --camera-intrinsics, --camera-extrinsic-npy required for --mode vlm")
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from pipeline.vlm import load_vlm
        from PIL import Image as PILImage

        image_rgb = np.array(PILImage.open(args.image).convert("RGB"))
        fx, fy, cx, cy = args.camera_intrinsics
        cam_T_world = np.load(args.camera_extrinsic_npy).astype(np.float32)

        print(f"[generate_constraints] Querying VLM ({args.vlm_name})...")
        vlm = load_vlm(args.vlm_name)
        u, v = vlm.query_object_pixels(image_rgb, args.object_description)
        print(f"[generate_constraints] VLM pixel estimate: u={u:.1f} v={v:.1f}")

        target_pos = pixels_to_world(u, v, fx, fy, cx, cy, cam_T_world, args.assumed_world_z)
        print(f"[generate_constraints] Unprojected world pos: {target_pos}")
        keyframes = gt_reach_keyframes(target_pos, args.frame_index)

    from kimodo.constraints import save_constraints_lst

    constraints = build_end_effector_constraints(keyframes, root_height=args.root_height)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_constraints_lst(str(out), constraints)
    print(f"[generate_constraints] Saved {len(keyframes)} keyframe(s) to '{out}'")


if __name__ == "__main__":
    main()
