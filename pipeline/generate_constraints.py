"""
pipeline/generate_constraints.py — Build Kimodo constraint objects in memory.

Returns a list of constraint objects ready to pass to model(constraint_lst=...).
No JSON serialization needed — Kimodo's in-memory path uses global_joints_positions
directly, so no IK is required.

Supported constraint types: right-hand, left-hand, right-foot, left-foot, root2d.
Multiple constraints of different types / frame indices are supported.

Coordinate conventions
----------------------
- IsaacLab/MuJoCo/ProtoMotions: x=forward, y=left,    z=up   (z-up)
- Kimodo:                        x=left,    y=up,      z=forward (y-up)
Mapping (from MujocoQposConverter): kimodo = [mujoco_y, mujoco_z, mujoco_x]
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

# End-effector joint per limb — used to identify the tip of each constrained chain.
# For hands the chain is [wrist_yaw, hand_roll]; we target the last joint (hand_roll)
# so the gripper/palm reaches the object, not the forearm.
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
    """
    Convert from IsaacLab/MuJoCo/ProtoMotions world frame to Kimodo frame.

    IsaacLab: x=forward, y=left,  z=up   (z-up, x-forward)
    Kimodo:   x=left,   y=up,     z=forward  (y-up, z-forward)

    From MujocoQposConverter.mujoco_to_kimodo_matrix = [[0,1,0],[0,0,1],[1,0,0]]:
      kimodo_x = mujoco_y  (left)
      kimodo_y = mujoco_z  (up)
      kimodo_z = mujoco_x  (forward)
    """
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    return [y, z, x]  # kimodo: [left, up, forward]


# --------------------------------------------------------------------------- #
# Low-level constraint builders
# --------------------------------------------------------------------------- #

def make_limb_constraint(skeleton, constraint_type: str, target_kimodo: list,
                          frame_index, device: str = "cpu"):
    """
    Build a hand or foot EndEffectorConstraintSet (single frame or multi-frame).

    Runs FK in rest pose to get a valid skeleton state, then overrides ALL
    position-constrained joints so the end effector lands on `target_kimodo`
    while maintaining consistent bone-length offsets for parent joints in the
    chain.  (The constraint class constrains every joint returned by
    ``skeleton.expand_joint_names``; if we only override the tip joint, the
    parent stays at its rest-pose position and the diffusion model compromises
    between the two, pulling the hand above the target.)

    Kimodo's diffusion uses global_joints_positions directly — no IK needed.

    Args:
        constraint_type: 'right-hand' | 'left-hand' | 'right-foot' | 'left-foot'
        target_kimodo:   [x, y, z] in Kimodo y-up coordinates
        frame_index:     int or list[int] — frame(s) at which to apply constraint (30fps)
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

    # Normalise frame_index to a list
    if isinstance(frame_index, int):
        frame_indices_list = [frame_index]
    else:
        frame_indices_list = list(frame_index)
    T = len(frame_indices_list)

    n_joints   = skeleton.nbjoints
    root_pos_1 = torch.tensor([[0.0, G1_ROOT_HEIGHT, 0.0]], dtype=torch.float32, device=device)
    rest_mats  = axis_angle_to_matrix(torch.zeros(1, n_joints, 3, device=device))

    # FK in rest pose → [1, n_joints, 3] / [1, n_joints, 3, 3]
    global_rots_1, global_pos_1, _ = skeleton.fk(rest_mats, root_pos_1)

    # Expand to T frames (same rest pose repeated)
    global_pos  = global_pos_1.expand(T, -1, -1).clone()        # [T, n_joints, 3]
    global_rots = global_rots_1.expand(T, -1, -1, -1).clone()   # [T, n_joints, 3, 3]

    # --- Override ALL position-constrained joints, not just the tip ----------
    # The constraint class (via expand_joint_names) will constrain every joint
    # in the limb chain.  We place the end-effector (last joint) at the target
    # and shift the other joints by their rest-pose offset so bone lengths stay
    # consistent and the diffusion model doesn't get conflicting targets.
    constraint_cls = cls_map[constraint_type]
    _, pos_joint_names = skeleton.expand_joint_names(constraint_cls.joint_names)

    ee_name = LIMB_EFFECTOR_JOINT[constraint_type]
    ee_idx  = skeleton.bone_index[ee_name]
    ee_rest = global_pos_1[0, ee_idx]                 # [3]

    target_t = torch.tensor(target_kimodo, dtype=torch.float32, device=device)

    print(f"  [{constraint_type}] frames={frame_indices_list}")
    print(f"    end-effector joint: {ee_name}")
    print(f"    rest-pose EE pos:   {[round(v,3) for v in ee_rest.tolist()]}")
    print(f"    target (Kimodo):    {[round(v,3) for v in target_kimodo]}")

    for jname in pos_joint_names:
        jidx   = skeleton.bone_index[jname]
        offset = global_pos_1[0, jidx] - ee_rest      # rest-pose offset from EE
        global_pos[:, jidx] = target_t + offset
        print(f"    {jname} (idx={jidx}): offset={[round(v,4) for v in offset.tolist()]}")

    # Match from_dict() device convention:
    #   frame_indices  → CPU, everything else → skeleton device (CUDA)
    frame_indices  = torch.tensor(frame_indices_list)          # CPU
    smooth_root_2d = root_pos_1[:, [0, 2]].expand(T, -1).clone()  # [T, 2], CUDA

    return constraint_cls(
        skeleton=skeleton,
        frame_indices=frame_indices,           # CPU
        global_joints_positions=global_pos,   # CUDA
        global_joints_rots=global_rots,        # CUDA
        smooth_root_2d=smooth_root_2d,         # CUDA
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

def constraints_reach_obj_gt(skeleton, cube_world_pos, frame_index, device: str) -> list:
    """
    GT upper bound for reach_obj: right hand constrained to cube world position.

    cube_world_pos: [x, y, z] in IsaacLab coordinates
    frame_index: int or list[int] — which frames to constrain
    """
    target = isaaclab_to_kimodo(cube_world_pos)
    print(f"[GT] cube (IsaacLab)={list(np.round(cube_world_pos, 3))}  →  Kimodo={[round(v,3) for v in target]}")
    return [make_limb_constraint(skeleton, "right-hand", target, frame_index, device)]


def constraints_vlm(skeleton, task, image_rgb, task_description, vlm_name,
                     num_frames, device: str) -> list:
    """VLM predicts 3D constraint positions from an egocentric image.

    Loads system prompt from prompts/system.txt and task prompt from
    tasks/<task>/vlm_prompt.txt (or uses task_description override).
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pipeline.vlm import load_vlm

    vlm = load_vlm(vlm_name, num_frames=num_frames, task=task,
                    task_description=task_description)
    raw_constraints = vlm.query_constraints(image_rgb)
    print(f"[VLM] predicted {len(raw_constraints)} constraint(s):")

    constraint_objects = []
    for c in raw_constraints:
        ctype = c["type"]
        pos_isaaclab = np.array(c["position"], dtype=np.float32)
        frame_id = c["frame_id"]
        target = isaaclab_to_kimodo(pos_isaaclab)
        print(f"  {ctype} frame={frame_id}  IsaacLab={pos_isaaclab.tolist()}  Kimodo={[round(v,3) for v in target]}")
        constraint_objects.append(
            make_limb_constraint(skeleton, ctype, target, frame_id, device)
        )
    return constraint_objects

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

    # VLM condition is task-agnostic (prompt loaded from tasks/<task>/vlm_prompt.txt)
    if condition == "vlm":
        return constraints_vlm(
            skeleton, task,
            kwargs["image_rgb"],
            kwargs.get("task_description"),
            kwargs.get("vlm_name", "qwen2.5-vl-7b"),
            kwargs["num_frames"],
            device,
        )

    # GT conditions are task-specific
    if task == "manip_reach_obj":
        frame_index = kwargs.get("frame_index", 45)
        if condition == "gt":
            return constraints_reach_obj_gt(
                skeleton, np.array(kwargs["cube_world_pos"], dtype=np.float32),
                frame_index, device,
            )

    raise ValueError(f"Unknown task/condition: {task}/{condition}")
