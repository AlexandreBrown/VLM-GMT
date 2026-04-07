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
# Generic constraint loader (delegates to kimodo API)
# --------------------------------------------------------------------------- #

def load_constraints_from_json(skeleton, json_path: str, device: str) -> list:
    """Load constraints from a JSON file exported by the Kimodo UI.

    Supports any constraint type (fullbody, root2d, end-effector, etc.)
    via kimodo.constraints.load_constraints_lst.
    """
    from pathlib import Path
    from kimodo.constraints import load_constraints_lst

    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Constraint file not found: {path}")

    return load_constraints_lst(str(path), skeleton, device=device)


# --------------------------------------------------------------------------- #
# Task / condition recipes
# --------------------------------------------------------------------------- #

def constraints_walk_to_obj_gt(skeleton, box_world_pos, frame_index, device: str) -> list:
    """
    GT upper bound for walk_to_obj: root2d constrained to box XY position.

    box_world_pos: [x, y, z] in IsaacLab coordinates (only x, y used)
    frame_index: int or list[int] — constraint applied at the last frame only
    """
    x_isaaclab = float(box_world_pos[0])  # forward
    y_isaaclab = float(box_world_pos[1])  # left
    # Kimodo: x=left, z=forward  →  kimodo_x = isaaclab_y, kimodo_z = isaaclab_x
    kimodo_x = y_isaaclab
    kimodo_z = x_isaaclab
    if isinstance(frame_index, int):
        frame_index = [frame_index]
    
    constraints = []
    for fi in frame_index:
        print(f"[GT walk_to_obj] box XY (IsaacLab): ({x_isaaclab:.3f}, {y_isaaclab:.3f})")
        print(f"  → root2d (Kimodo): x={kimodo_x:.3f}, z={kimodo_z:.3f}  frame={fi}")
        constraints.append(make_root2d_constraint(skeleton, x=kimodo_x, z=kimodo_z,
                                   frame_index=fi, device=device))
    return constraints


def constraints_navigate_maze_gt(
    skeleton, obs_world_positions: list, line_end_x: float, num_frames: int, device: str
) -> list:
    """
    GT upper bound: three root2d waypoints per wall + final waypoint.

    For each wall at (bx, by):
      A — x = bx - 0.4,  y = avoidance   (approach: move to gap side)
      B — x = bx + 0.5,  y = avoidance   (clear: still on gap side, past wall)
      C — x = bx + 1.0,  y = avoidance   (hold: extra room before turning for next wall)
    Final constraint at (line_end_x, y=0): robot reaches end.

    The extra "hold" waypoint gives the GMT tracker margin to catch up
    before the robot needs to turn for the next wall.

    avoidance_y = -sign(by) * 0.6: well into the gap (gap center at ±0.75).
    Frame indices scale linearly with x / (line_end_x + 1.0).
    """
    import math

    constraints = []
    sorted_obs = sorted(obs_world_positions, key=lambda p: float(p[0]))
    total_x = line_end_x + 1.0

    def frame_for_x(x):
        return max(1, min(int(x / total_x * num_frames), num_frames - 2))

    for obs_pos in sorted_obs:
        bx = float(obs_pos[0])
        by = float(obs_pos[1])

        sign = math.copysign(1.0, by) if abs(by) > 0.01 else 1.0
        avoidance_y = -sign * 0.6

        for x_offset, label in [(-0.4, "approach"), (0.5, "clear"), (1.0, "hold")]:
            x = bx + x_offset
            f = frame_for_x(x)
            print(f"[GT navigate_maze] wall ({bx:.2f}, {by:.2f})"
                  f" {label}: kimodo_z={x:.2f}, kimodo_x={avoidance_y:.2f}, frame={f}")
            constraints.append(make_root2d_constraint(
                skeleton, x=avoidance_y, z=x, frame_index=f, device=device
            ))

    # Final: center of corridor end
    print(f"[GT navigate_maze] final: kimodo_z={line_end_x:.2f}, kimodo_x=0.0, frame={num_frames-1}")
    constraints.append(make_root2d_constraint(
        skeleton, x=0.0, z=float(line_end_x), frame_index=num_frames - 1, device=device
    ))
    return constraints


def constraints_reach_obj_gt(skeleton, cube_world_pos, frame_index, device: str) -> list:
    """
    GT upper bound for reach_obj: right hand constrained to cube world position.

    cube_world_pos: [x, y, z] in IsaacLab coordinates
    frame_index: int or list[int] — which frames to constrain
    """
    target = isaaclab_to_kimodo(cube_world_pos)
    print(f"[GT] cube (IsaacLab)={list(np.round(cube_world_pos, 3))}  →  Kimodo={[round(v,3) for v in target]}")
    return [make_limb_constraint(skeleton, "right-hand", target, frame_index, device)]


def query_vlm_raw(task: str, image_rgb, vlm_name: str,
                   load_in_4bit: bool, num_frames: int, pitch_deg: float,
                   output_dir: str, vlm_gmt_root: str) -> list:
    """Run the VLM and return raw constraint dicts. No skeleton required.

    Call this BEFORE loading Kimodo to avoid concurrent GPU memory usage.
    Returns a list of dicts: [{"type": ..., "position": [...], "frame_id": ...}, ...]
    """
    import json
    from pathlib import Path
    from pipeline.vlm import load_vlm

    vlm = load_vlm(vlm_name, vlm_gmt_root=vlm_gmt_root, num_frames=num_frames,
                    pitch_deg=pitch_deg, task=task, load_in_4bit=load_in_4bit)
    raw_constraints = vlm.query_constraints(image_rgb)
    print(f"[VLM] predicted {len(raw_constraints)} constraint(s):")

    if output_dir:
        log_path = Path(output_dir) / "vlm_constraints.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(raw_constraints, f, indent=2)
        print(f"[VLM] Saved constraints to {log_path}")

    return raw_constraints


def constraints_vlm(skeleton, task, image_rgb, vlm_name,
                     num_frames, output_dir, device: str, load_in_4bit: bool = True,
                     vlm_gmt_root: str = None) -> list:
    """VLM predicts 3D constraint positions from an egocentric image.

    Loads system prompt from prompts/system.txt and task prompt from
    tasks/<task>/vlm_prompt.txt.
    Saves raw VLM predictions to <output_dir>/vlm_constraints.json.
    """
    import json
    from pathlib import Path
    from pipeline.vlm import load_vlm

    if vlm_gmt_root is None:
        raise ValueError("vlm_gmt_root must be provided")
    vlm = load_vlm(vlm_name, vlm_gmt_root=vlm_gmt_root, num_frames=num_frames,
                    task=task, load_in_4bit=load_in_4bit)
    raw_constraints = vlm.query_constraints(image_rgb)
    print(f"[VLM] predicted {len(raw_constraints)} constraint(s):")

    # Save raw VLM output for debugging
    if output_dir:
        log_path = Path(output_dir) / "vlm_constraints.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(raw_constraints, f, indent=2)
        print(f"[VLM] Saved constraints to {log_path}")

    return _build_vlm_constraints_from_raw(skeleton, raw_constraints, device)


def _build_vlm_constraints_from_raw(skeleton, raw_constraints: list, device: str) -> list:
    """Convert raw VLM dicts to Kimodo constraint objects using the skeleton."""
    import torch
    from kimodo.constraints import FullBodyConstraintSet
    from kimodo.geometry import axis_angle_to_matrix

    constraint_objects = []
    for c in raw_constraints:
        ctype = c["type"]
        frame_id = c["frame_id"]

        if ctype == "fullbody":
            # VLM predicts joint positions in IsaacLab frame. Convert to Kimodo
            # and build a FullBodyConstraintSet via FK from rest pose with
            # overridden joint positions.
            positions = c["positions"]  # dict: joint_name → [x, y, z]
            print(f"  fullbody frame={frame_id}  joints={len(positions)}")
            # Build global_joints_positions from VLM predictions
            n_joints = skeleton.nbjoints
            root_pos = torch.tensor([[0.0, G1_ROOT_HEIGHT, 0.0]], dtype=torch.float32, device=device)
            rest_mats = axis_angle_to_matrix(torch.zeros(1, n_joints, 3, device=device))
            global_rots, global_pos, _ = skeleton.fk(rest_mats, root_pos)

            # Map short VLM names to skeleton bone names
            vlm_to_skel = {name.replace("_skel", ""): name for name in skeleton.bone_index}
            for jname, pos_isaac in positions.items():
                skel_name = vlm_to_skel.get(jname) or vlm_to_skel.get(jname + "_skel")
                if skel_name and skel_name in skeleton.bone_index:
                    jidx = skeleton.bone_index[skel_name]
                    pos_kimodo = isaaclab_to_kimodo(pos_isaac)
                    global_pos[0, jidx] = torch.tensor(pos_kimodo, dtype=torch.float32, device=device)

            # If VLM provided a pelvis position, use it for root
            if "pelvis" in positions:
                pelvis_kimodo = isaaclab_to_kimodo(positions["pelvis"])
                root_pos = torch.tensor([pelvis_kimodo], dtype=torch.float32, device=device)

            constraint_objects.append(FullBodyConstraintSet(
                skeleton=skeleton,
                frame_indices=torch.tensor([frame_id]),
                global_joints_positions=global_pos,
                global_joints_rots=global_rots,
                smooth_root_2d=root_pos[:, [0, 2]],
            ))
        elif ctype == "root2d":
            pos_isaaclab = np.array(c["position"], dtype=np.float32)
            print(f"  {ctype} frame={frame_id}  IsaacLab={pos_isaaclab.tolist()}")
            constraint_objects.append(
                make_root2d_constraint(skeleton, x=float(pos_isaaclab[1]),
                                       z=float(pos_isaaclab[0]),
                                       frame_index=frame_id, device=device)
            )
        else:
            pos_isaaclab = np.array(c["position"], dtype=np.float32)
            print(f"  {ctype} frame={frame_id}  IsaacLab={pos_isaaclab.tolist()}")
            target = isaaclab_to_kimodo(pos_isaaclab)
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

    # VLM condition: accept pre-queried raw dicts (from query_vlm_raw) or run VLM inline
    if condition == "vlm":
        if "raw_vlm_constraints" in kwargs:
            return _build_vlm_constraints_from_raw(skeleton, kwargs["raw_vlm_constraints"], device)
        return constraints_vlm(
            skeleton, task,
            kwargs["image_rgb"],
            kwargs.get("vlm_name", "qwen2.5-vl-32b"),
            kwargs["num_frames"],
            kwargs.get("output_dir"),
            device,
            load_in_4bit=kwargs.get("load_in_4bit", True),
            vlm_gmt_root=kwargs.get("vlm_gmt_root"),
        )

    # GT conditions are task-specific
    frame_index = kwargs.get("frame_index", 45)

    if task in ("manip_reach_obj", "point_at_obj_with_right_hand", "raise_right_hand", "touch_left_leg_with_right_hand") and condition == "gt":
        return constraints_reach_obj_gt(
            skeleton, np.array(kwargs["cube_world_pos"], dtype=np.float32),
            frame_index, device,
        )

    if task in ("point_at_obj_with_left_hand", "raise_left_hand", "touch_right_leg_with_left_hand") and condition == "gt":
        target = isaaclab_to_kimodo(np.array(kwargs["cube_world_pos"], dtype=np.float32))
        return [make_limb_constraint(skeleton, "left-hand", target, frame_index, device)]

    if task == "walk_to_obj" and condition == "gt":
        return constraints_walk_to_obj_gt(
            skeleton, np.array(kwargs["box_world_pos"], dtype=np.float32),
            frame_index, device,
        )

    if task == "navigate_maze" and condition == "gt":
        return constraints_navigate_maze_gt(
            skeleton,
            [np.array(p, dtype=np.float32) for p in kwargs["obs_world_positions"]],
            kwargs.get("line_end_x", 5.75),
            kwargs["num_frames"],
            device,
        )

    if task == "kneel_down_1_knee" and condition == "gt":
        return load_constraints_from_json(
            skeleton, kwargs["constraint_json"], device,
        )

    raise ValueError(f"Unknown task/condition: {task}/{condition}")
