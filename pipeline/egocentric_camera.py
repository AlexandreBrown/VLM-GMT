"""Egocentric head camera for the G1 robot in IsaacLab.

Injects a Camera sensor onto the G1 head body without modifying ProtoMotions.
Works headed and headless.

Usage:
    # Before building components:
    patch_scene_with_egocentric_camera(pitch_deg=60)

    # After env.reset():
    ego_cam = get_egocentric_camera(env.simulator)
    orient_robot_toward_objects(env.simulator)

    # Capture at any step:
    frame = capture_egocentric_frame(ego_cam)
    save_egocentric_frame(frame, "outputs/reach_obj")
"""

import math

import numpy as np

# Camera defaults
EGO_CAM_WIDTH = 640
EGO_CAM_HEIGHT = 480
EGO_CAM_FOCAL_LENGTH = 24.0  # mm
EGO_CAM_HORIZONTAL_APERTURE = 20.955  # mm
EGO_CAM_CLIPPING = (0.05, 50.0)  # near / far in meters
EGO_CAM_DEFAULT_PITCH_DEG = 50.0  # degrees downward
EGO_CAM_OFFSET_FORWARD = 0.15  # meters forward from head center

# Key for the sensor in InteractiveScene
_SCENE_KEY = "egocentric_camera"


def _pitch_to_quat(pitch_deg: float) -> tuple:
    """Downward pitch (degrees) to quaternion (w, x, y, z).

    In "world" convention, rotation around +Y pitches the camera down.
    """
    half = math.radians(pitch_deg) / 2
    return (math.cos(half), 0.0, math.sin(half), 0.0)


def patch_scene_with_egocentric_camera(
    width: int = EGO_CAM_WIDTH,
    height: int = EGO_CAM_HEIGHT,
    pitch_deg: float = EGO_CAM_DEFAULT_PITCH_DEG,
    offset_forward: float = EGO_CAM_OFFSET_FORWARD,
):
    """Add a head-mounted camera to the ProtoMotions scene config.

    Call after AppLauncher init, before build_all_components().

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        pitch_deg: Downward tilt (0 = horizontal, 90 = straight down).
    """
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg
    from protomotions.simulator.isaaclab.utils.scene import SceneCfg

    cam_cfg = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/pelvis/head/EgoCamera",
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=EGO_CAM_FOCAL_LENGTH,
            focus_distance=400.0,
            horizontal_aperture=EGO_CAM_HORIZONTAL_APERTURE,
            clipping_range=EGO_CAM_CLIPPING,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(offset_forward, 0.0, 0.0),
            rot=_pitch_to_quat(pitch_deg),
            convention="world",
        ),
        update_latest_camera_pose=True,
    )

    _original_init = SceneCfg.__init__

    def _patched_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)
        setattr(self, _SCENE_KEY, cam_cfg)

    SceneCfg.__init__ = _patched_init
    print(f"[ego_camera] Patched SceneCfg: {width}x{height}, pitch={pitch_deg}°")


def get_egocentric_camera(simulator):
    """Get the Camera sensor from the simulator scene. Call after env.reset()."""
    return simulator._scene[_SCENE_KEY]


def capture_egocentric_frame(camera, env_idx: int = 0, dt: float = 0.0) -> dict:
    """Capture one RGB frame. Returns {"image_rgb": (H, W, 3) uint8 ndarray}."""
    camera.update(dt)
    rgb = camera.data.output["rgb"][env_idx].cpu().numpy()
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    return {"image_rgb": rgb.astype(np.uint8)}


def orient_robot_with_yaw(simulator, yaw_deg: float = 0.0, env_idx: int = 0):
    """Set the robot root yaw to a fixed angle. Call after env.reset().

    yaw_deg=0 matches the default eval orientation (robot faces +X).
    Adjust to match whatever the eval starting pose is for the task.
    """
    import torch

    yaw = math.radians(yaw_deg)
    robot = simulator._robot
    root_state = robot.data.default_root_state[env_idx].clone().unsqueeze(0)
    root_state[0, :3] = robot.data.root_pos_w[env_idx]
    root_state[0, 3] = math.cos(yaw / 2)  # w
    root_state[0, 4] = 0.0  # x
    root_state[0, 5] = 0.0  # y
    root_state[0, 6] = math.sin(yaw / 2)  # z
    root_state[0, 7:] = 0.0

    env_ids = torch.tensor([env_idx], device=robot.device)
    robot.write_root_state_to_sim(root_state.to(robot.device), env_ids)
    simulator._sim.step(render=True)
    print(f"[ego_camera] Robot yaw set to {yaw_deg:.1f}°")


def orient_robot_toward_objects(simulator, env_idx: int = 0):
    """Yaw the robot root to face the first scene object. Call after env.reset()."""
    import torch

    robot = simulator._robot
    objects = getattr(simulator, "_object", [])
    if not objects:
        return

    robot_pos = robot.data.root_pos_w[env_idx].cpu()
    obj_pos = objects[0].data.root_pos_w[env_idx].cpu()

    dx = float(obj_pos[0] - robot_pos[0])
    dy = float(obj_pos[1] - robot_pos[1])
    yaw = math.atan2(dy, dx)

    root_state = robot.data.default_root_state[env_idx].clone().unsqueeze(0)
    root_state[0, :3] = robot.data.root_pos_w[env_idx]
    root_state[0, 3] = math.cos(yaw / 2)  # w
    root_state[0, 4] = 0.0  # x
    root_state[0, 5] = 0.0  # y
    root_state[0, 6] = math.sin(yaw / 2)  # z
    root_state[0, 7:] = 0.0

    env_ids = torch.tensor([env_idx], device=robot.device)
    robot.write_root_state_to_sim(root_state.to(robot.device), env_ids)
    simulator._sim.step(render=True)

    print(f"[ego_camera] Oriented robot toward object: yaw={math.degrees(yaw):.1f}°")


def save_egocentric_frame(frame: dict, output_dir: str, prefix: str = "ego") -> str:
    """Save RGB image to <output_dir>/<prefix>.png. Returns the path."""
    from pathlib import Path
    from PIL import Image

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    img_path = out / f"{prefix}.png"
    Image.fromarray(frame["image_rgb"]).save(str(img_path))
    print(f"[ego_camera] Saved: {img_path}")
    return str(img_path)
