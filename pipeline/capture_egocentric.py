"""Capture an egocentric RGB image from the G1 head camera.

Sets up a minimal ProtoMotions environment, orients the robot toward
scene objects, and saves one egocentric frame. Works headed or headless.

Usage (from ProtoMotions root):
    VLMGMT=~/Documents/vlm_project/VLM-GMT
    python $VLMGMT/pipeline/capture_egocentric.py \
        --experiment-path examples/experiments/mimic/mlp.py \
        --motion-file data/motion_for_trackers/g1_bones_seed_mini.pt \
        --robot-name g1 --simulator isaaclab --num-envs 1 \
        --scenes-file $VLMGMT/outputs/reach_obj_scene.pt \
        --output-dir $VLMGMT/outputs/reach_obj
"""

import argparse
import sys
import os

# VLM-GMT root on sys.path so `pipeline.*` imports work
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

parser = argparse.ArgumentParser(description="Capture egocentric frame from G1 head camera")
parser.add_argument("--robot-name", type=str, required=True)
parser.add_argument("--simulator", type=str, required=True)
parser.add_argument("--num-envs", type=int, required=True)
parser.add_argument("--motion-file", type=str, required=True)
parser.add_argument("--experiment-path", type=str, required=True)
parser.add_argument("--experiment-name", type=str, default="ego_capture")
parser.add_argument("--scenes-file", type=str, default=None)
parser.add_argument("--output-dir", type=str, required=True)
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--pitch-deg", type=float, default=60.0,
                    help="Camera downward tilt in degrees (0=horizontal, 90=straight down)")
parser.add_argument("--robot-yaw-deg", type=float, default=0.0,
                    help="Fixed robot yaw in degrees (0 = faces +X, matches eval default). "
                         "Tweak to match the eval starting orientation for the task.")
parser.add_argument("--warmup-steps", type=int, default=10)
parser.add_argument("--prefix", type=str, default="ego")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

# IsaacLab must be imported before torch
from protomotions.utils.simulator_imports import import_simulator_before_torch
AppLauncher = import_simulator_before_torch(args.simulator)

from pathlib import Path
import importlib.util
import torch


def load_experiment_module(path: str):
    """Load an experiment .py file as a module."""
    spec = importlib.util.spec_from_file_location("experiment_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    device = torch.device("cuda:0")

    experiment_module = load_experiment_module(args.experiment_path)

    # Launch simulator
    extra_simulator_params = {}
    if args.simulator == "isaaclab":
        app_launcher = AppLauncher({
            "headless": args.headless,
            "device": str(device),
            "enable_cameras": True,
        })
        extra_simulator_params["simulation_app"] = app_launcher.app

    # Patch scene config to include head camera (must happen before build)
    from pipeline.egocentric_camera import (
        patch_scene_with_egocentric_camera,
        get_egocentric_camera,
        capture_egocentric_frame,
        save_egocentric_frame,
        orient_robot_with_yaw,
    )
    patch_scene_with_egocentric_camera(
        width=args.width, height=args.height, pitch_deg=args.pitch_deg
    )

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Build configs
    from protomotions.utils.config_builder import build_standard_configs
    configs = build_standard_configs(
        args=args,
        terrain_config_fn=getattr(experiment_module, "terrain_config"),
        scene_lib_config_fn=getattr(experiment_module, "scene_lib_config"),
        motion_lib_config_fn=getattr(experiment_module, "motion_lib_config"),
        env_config_fn=getattr(experiment_module, "env_config"),
        configure_robot_and_simulator_fn=getattr(
            experiment_module, "configure_robot_and_simulator", None
        ),
        agent_config_fn=None,
    )

    robot_config = configs["robot"]
    simulator_config = configs["simulator"]
    env_config = configs["env"]

    # Minimal env: no control, obs, reward, or termination components
    env_config.show_terrain_markers = False
    env_config.control_components = {}
    env_config.termination_components = {}
    env_config.observation_components = {}
    env_config.reward_components = {}

    # Build components and environment
    from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator
    from protomotions.utils.component_builder import build_all_components
    from protomotions.envs.base_env.env import BaseEnv

    terrain_config, simulator_config = convert_friction_for_simulator(
        configs["terrain"], simulator_config
    )
    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=configs["scene_lib"],
        motion_lib_config=configs["motion_lib"],
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=device,
        **extra_simulator_params,
    )
    env = BaseEnv(
        config=env_config,
        robot_config=robot_config,
        device=device,
        terrain=components["terrain"],
        scene_lib=components["scene_lib"],
        motion_lib=components["motion_lib"],
        simulator=components["simulator"],
    )

    try:
        print("[capture] Resetting environment ...")
        env.reset()

        orient_robot_with_yaw(env.simulator, yaw_deg=args.robot_yaw_deg, env_idx=0)

        # Render-only warmup (no physics) so robot stays standing
        for _ in range(args.warmup_steps):
            env.simulator._sim.render()

        # Capture
        ego_cam = get_egocentric_camera(env.simulator)
        frame = capture_egocentric_frame(ego_cam, env_idx=0)
        img_path = save_egocentric_frame(frame, args.output_dir, prefix=args.prefix)

        print(f"[capture] Image shape: {frame['image_rgb'].shape}")
        print(f"[capture] Saved to: {img_path}")
    finally:
        env.close()
        print("[capture] Done.")


if __name__ == "__main__":
    main()
