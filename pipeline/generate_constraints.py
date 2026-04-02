"""
pipeline/generate_constraints.py — Generate Kimodo kinematic constraints.

Supports three modes:
    --mode gt     Ground-truth cube position → IK → constraint (upper bound)
    --mode vlm    VLM image estimate → IK → constraint
    --mode none   No constraint (text-only baseline)

Usage (run from ProtoMotions root, with VLM-GMT on PYTHONPATH):
    # GT
    python VLM-GMT/pipeline/generate_constraints.py --mode gt \
        --cube-world-pos 0.6 0.0 0.4 \
        --output constraints.json

    # VLM
    python VLM-GMT/pipeline/generate_constraints.py --mode vlm \
        --image sim_frame.png \
        --camera-intrinsics 500 500 320 240 \
        --camera-extrinsic-npy camera_T_world.npy \
        --output constraints.json

    # Baseline
    python VLM-GMT/pipeline/generate_constraints.py --mode none

Requires pyroki (separate env or same env):
    git clone https://github.com/chungmin99/pyroki && pip install -e pyroki
"""

import argparse
import json
import numpy as onp
from pathlib import Path


# ---------------------------------------------------------------------------
# G1 config
# ---------------------------------------------------------------------------
G1_URDF_DEFAULT = "protomotions/data/assets/urdf/for_retargeting/g1_29dof.urdf"
G1_RIGHT_WRIST_LINK = "right_wrist_yaw_link"
G1_ROOT_HEIGHT = 0.793       # default standing pelvis height (meters)
CONSTRAINT_FRAME = 45        # keyframe index (30fps → 1.5s into a 3s motion)


# ---------------------------------------------------------------------------
# IK via PyRoki
# ---------------------------------------------------------------------------

def solve_ik_right_wrist(
    target_world_pos: onp.ndarray,
    urdf_path: str,
    root_world_pos: onp.ndarray,
) -> onp.ndarray:
    """
    Single-frame IK: find G1 joint angles so right wrist reaches target_world_pos.

    Returns:
        joint_angles: (n_actuated_dofs,) float32 array.
    """
    import jax.numpy as jnp
    import jaxlie
    import jaxls
    import pyroki as pk
    from yourdfpy import URDF

    urdf_obj = URDF.load(urdf_path)
    robot = pk.Robot.from_urdf(urdf_obj)

    link_names = list(robot.links.names)
    ee_idx = link_names.index(G1_RIGHT_WRIST_LINK)

    # Target relative to robot root
    target_root_rel = jnp.array(target_world_pos - root_world_pos, dtype=jnp.float32)

    var_joints = robot.joint_var_cls(0)

    @jaxls.Cost.factory
    def ee_position_cost(var_values, var_joints):
        cfg = var_values[var_joints]
        fk = jaxlie.SE3(robot.forward_kinematics(cfg=cfg))
        ee_pos = fk[ee_idx].translation()
        return ee_pos - target_root_rel

    solution = (
        jaxls.LeastSquaresProblem(
            costs=[
                ee_position_cost(var_joints),
                pk.costs.limit_constraint(robot, var_joints),
            ],
            variables=[var_joints],
        )
        .analyze()
        .solve()
    )

    return onp.array(solution[var_joints])


def build_right_hand_constraint(
    target_world_pos: onp.ndarray,
    frame_index: int,
    urdf_path: str,
    root_world_pos: onp.ndarray,
) -> dict:
    """Run IK and format as a Kimodo right-hand constraint dict."""
    from yourdfpy import URDF

    joint_angles = solve_ik_right_wrist(target_world_pos, urdf_path, root_world_pos)

    # Get actuated joint axes from URDF
    urdf_obj = URDF.load(urdf_path)
    actuated_joints = [
        j for j in urdf_obj.robot.joints
        if j.type in ("revolute", "continuous", "prismatic")
    ]
    axes = onp.array(
        [j.axis if j.axis is not None else [0.0, 0.0, 1.0] for j in actuated_joints],
        dtype=onp.float32,
    )  # (J, 3)

    n_joints = len(actuated_joints)
    angles = onp.zeros(n_joints, dtype=onp.float32)
    n = min(len(joint_angles), n_joints)
    angles[:n] = joint_angles[:n]

    # axis-angle per joint: axis * angle → shape (J, 3)
    local_joints_rot = (axes * angles[:, None]).astype(onp.float32)

    return {
        "type": "right-hand",
        "frame_indices": [frame_index],
        "local_joints_rot": local_joints_rot[None].tolist(),  # (1, J, 3)
        "root_positions": [root_world_pos.tolist()],           # (1, 3)
    }


def build_root2d_constraint(
    target_world_pos: onp.ndarray,
    frame_index: int,
) -> dict:
    """Walk root toward target XZ position by frame_index."""
    return {
        "type": "root2d",
        "frame_indices": [0, frame_index],
        "smooth_root_2d": [[0.0, 0.0], [float(target_world_pos[0]), float(target_world_pos[2])]],
    }


# ---------------------------------------------------------------------------
# Camera unprojection
# ---------------------------------------------------------------------------

def pixels_to_world(
    u: float, v: float,
    fx: float, fy: float, cx: float, cy: float,
    cam_T_world: onp.ndarray,
    assumed_world_z: float,
) -> onp.ndarray:
    """Back-project pixel (u, v) to 3D world via ray-plane intersection at z=assumed_world_z."""
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
    parser.add_argument("--urdf", default=G1_URDF_DEFAULT)
    parser.add_argument(
        "--root-world-pos", nargs=3, type=float, default=[0.0, 0.0, G1_ROOT_HEIGHT],
        metavar=("X", "Y", "Z"),
        help="Robot root (pelvis) position in world frame (default: standing at origin)",
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
        print("[generate_constraints] mode=none: no constraint file generated.")
        return

    root_world_pos = onp.array(args.root_world_pos, dtype=onp.float32)

    # --- Determine target position ---
    if args.mode == "gt":
        if args.cube_world_pos is None:
            parser.error("--cube-world-pos required for --mode gt")
        target_pos = onp.array(args.cube_world_pos, dtype=onp.float32)
        print(f"[generate_constraints] GT target: {target_pos}")

    elif args.mode == "vlm":
        if not all([args.image, args.camera_intrinsics, args.camera_extrinsic_npy]):
            parser.error("--image, --camera-intrinsics, --camera-extrinsic-npy required for vlm.")
        from PIL import Image as PILImage
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from pipeline.vlm import load_vlm

        image_rgb = onp.array(PILImage.open(args.image).convert("RGB"))
        fx, fy, cx, cy = args.camera_intrinsics
        cam_T_world = onp.load(args.camera_extrinsic_npy).astype(onp.float32)

        print(f"[generate_constraints] Querying VLM ({args.vlm_name})...")
        vlm = load_vlm(args.vlm_name)
        u, v = vlm.query_object_pixels(image_rgb, args.object_description)
        print(f"[generate_constraints] VLM pixel estimate: u={u:.1f} v={v:.1f}")

        target_pos = pixels_to_world(u, v, fx, fy, cx, cy, cam_T_world, args.assumed_world_z)
        print(f"[generate_constraints] Unprojected world pos: {target_pos}")

    # --- Build constraints ---
    print("[generate_constraints] Running IK (PyRoki)...")
    constraints = [
        build_root2d_constraint(target_pos, args.frame_index),
        build_right_hand_constraint(target_pos, args.frame_index, args.urdf, root_world_pos),
    ]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(constraints, f, indent=2)
    print(f"[generate_constraints] Saved to '{out}'")


if __name__ == "__main__":
    main()
