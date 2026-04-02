"""
eval/run_eval.py — Run GMT inference and compute task-specific metrics.

Modeled on ProtoMotions/protomotions/inference_agent.py. Reuses the same
stack (env, agent, simulator) but adds a per-step metric hook and saves
a JSON results file.

Usage (run from ProtoMotions root):
    python VLM-GMT/eval/run_eval.py \
        --checkpoint data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt \
        --motion-file VLM-GMT/outputs/reach_obj/baseline/motion.pt \
        --scenes-file VLM-GMT/outputs/reach_obj_scene.pt \
        --task reach_obj \
        --condition baseline \
        --num-episodes 20 \
        --simulator isaaclab \
        --output-dir VLM-GMT/outputs/reach_obj/results

Results are saved to:
    <output-dir>/<condition>_results.json
"""

# ── argparse before any torch import (IsaacLab requirement) ──────────────────
import argparse


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--motion-file", required=True)
    parser.add_argument("--scenes-file", required=True)
    parser.add_argument("--task", required=True, help="Task name, e.g. 'reach_obj'")
    parser.add_argument("--condition", required=True, help="Condition label for output, e.g. 'baseline'")
    parser.add_argument("--simulator", required=True)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--output-dir", default="VLM-GMT/outputs/results")
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument(
        "--protomotions-root", required=True,
        help="Path to ProtoMotions root directory (scripts use relative asset paths)."
    )
    return parser


parser = create_parser()
args, _ = parser.parse_known_args()

from protomotions.utils.simulator_imports import import_simulator_before_torch
AppLauncher = import_simulator_before_torch(args.simulator)

# ── safe to import torch now ──────────────────────────────────────────────────
import json
import importlib
import logging
import sys
import os
from pathlib import Path
from dataclasses import asdict

import torch
from lightning.fabric import Fabric

from protomotions.utils.hydra_replacement import get_class
from protomotions.utils.fabric_config import FabricConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
log = logging.getLogger(__name__)


def load_task_metrics(task_name: str):
    """Dynamically load get_metrics() from tasks/<task_name>/metrics.py."""
    vlm_gmt_root = Path(__file__).resolve().parent.parent
    metrics_path = vlm_gmt_root / "tasks" / task_name / "metrics.py"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"No metrics file found for task '{task_name}' at {metrics_path}.\n"
            f"Create tasks/{task_name}/metrics.py with a get_metrics() function."
        )
    spec = importlib.util.spec_from_file_location(f"{task_name}_metrics", metrics_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_metrics()


def run_episode(env, agent, metrics, max_steps: int = 300):
    """
    Run a single episode, calling metric.update() every step.
    Returns list of MetricResult (one per metric).
    """
    for m in metrics:
        m.reset()

    scene_lib = env.scene_lib
    env_ids = torch.arange(env.num_envs, device=env.device)

    obs, _ = env.reset(env_ids)
    obs = agent.add_agent_info_to_obs(obs)
    obs_td = agent.obs_dict_to_tensordict(obs)

    for _ in range(max_steps):
        with torch.no_grad():
            model_outs = agent.model(obs_td)
            actions = model_outs.get("mean_action", model_outs.get("action"))

        obs, rewards, dones, terminated, extras = env.step(actions)
        obs = agent.add_agent_info_to_obs(obs)
        obs_td = agent.obs_dict_to_tensordict(obs)

        for m in metrics:
            m.update(env, scene_lib)

        if dones[0].item():
            break

    return [m.compute() for m in metrics]


def main():
    global args
    args = parser.parse_args()

    # ProtoMotions uses relative asset paths — must run from its root
    import os
    os.chdir(Path(args.protomotions_root).resolve())

    checkpoint = Path(args.checkpoint)
    resolved_configs_path = checkpoint.parent / "resolved_configs_inference.pt"
    assert resolved_configs_path.exists(), f"Missing {resolved_configs_path}"

    resolved_configs = torch.load(resolved_configs_path, map_location="cpu", weights_only=False)
    robot_config     = resolved_configs["robot"]
    simulator_config = resolved_configs["simulator"]
    terrain_config   = resolved_configs.get("terrain")
    scene_lib_config = resolved_configs["scene_lib"]
    motion_lib_config = resolved_configs["motion_lib"]
    env_config       = resolved_configs["env"]
    agent_config     = resolved_configs["agent"]

    # Switch simulator if needed
    current_simulator = simulator_config._target_.split(".")[-3]
    if args.simulator != current_simulator:
        from protomotions.simulator.factory import update_simulator_config_for_test
        simulator_config = update_simulator_config_for_test(
            current_simulator_config=simulator_config,
            new_simulator=args.simulator,
            robot_config=robot_config,
        )

    from protomotions.utils.inference_utils import apply_backward_compatibility_fixes
    apply_backward_compatibility_fixes(robot_config, simulator_config, env_config)

    # Apply CLI overrides
    simulator_config.num_envs = args.num_envs
    motion_lib_config.motion_file = args.motion_file
    scene_lib_config.scene_file = args.scenes_file
    simulator_config.headless = args.headless

    accelerator = "cpu" if args.simulator == "mujoco" else "gpu"
    fabric_config = FabricConfig(accelerator=accelerator, devices=1, num_nodes=1, loggers=[], callbacks=[])
    fabric: Fabric = Fabric(**asdict(fabric_config))
    fabric.launch()

    simulator_extra_params = {}
    if args.simulator == "isaaclab":
        app_launcher = AppLauncher({"headless": args.headless, "device": str(fabric.device)})
        simulator_extra_params["simulation_app"] = app_launcher.app

    from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator
    terrain_config, simulator_config = convert_friction_for_simulator(terrain_config, simulator_config)

    from protomotions.utils.component_builder import build_all_components
    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=fabric.device,
        save_dir=None,
        **simulator_extra_params,
    )

    from protomotions.envs.base_env.env import BaseEnv
    EnvClass = get_class(env_config._target_)
    env: BaseEnv = EnvClass(
        config=env_config,
        robot_config=robot_config,
        device=fabric.device,
        terrain=components["terrain"],
        scene_lib=components["scene_lib"],
        motion_lib=components["motion_lib"],
        simulator=components["simulator"],
    )

    from protomotions.agents.base_agent.agent import BaseAgent
    AgentClass = get_class(agent_config._target_)
    agent: BaseAgent = AgentClass(config=agent_config, env=env, fabric=fabric, root_dir=checkpoint.parent)
    agent.setup()
    agent.load(str(checkpoint), load_env=False)

    # Load task metrics
    metrics = load_task_metrics(args.task)
    log.info(f"Loaded {len(metrics)} metrics for task '{args.task}': {[m.name for m in metrics]}")

    # Run episodes
    all_results = {m.name: [] for m in metrics}

    log.info(f"Running {args.num_episodes} episodes...")
    for ep in range(args.num_episodes):
        episode_results = run_episode(env, agent, metrics)
        for i, result in enumerate(episode_results):
            all_results[metrics[i].name].append({
                "value": result.value,
                "success": result.success,
                **result.info,
            })
        log.info(
            f"Episode {ep+1}/{args.num_episodes} — "
            + " | ".join(
                f"{metrics[i].name}={result.value:.3f} ({'✓' if result.success else '✗'})"
                for i, result in enumerate(episode_results)
            )
        )

    # Aggregate
    summary = {}
    for name, results in all_results.items():
        values = [r["value"] for r in results]
        successes = [r["success"] for r in results]
        summary[name] = {
            "success_rate": sum(successes) / len(successes),
            "mean_value": sum(values) / len(values),
            "min_value": min(values),
            "max_value": max(values),
            "n_episodes": len(results),
        }

    output = {
        "task": args.task,
        "condition": args.condition,
        "checkpoint": str(checkpoint),
        "motion_file": args.motion_file,
        "scenes_file": args.scenes_file,
        "num_episodes": args.num_episodes,
        "summary": summary,
        "episodes": all_results,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.condition}_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print(f"RESULTS — task={args.task} condition={args.condition}")
    print("=" * 60)
    for name, s in summary.items():
        print(f"  {name}:")
        print(f"    success_rate : {s['success_rate']:.1%} ({int(s['success_rate']*args.num_episodes)}/{args.num_episodes})")
        print(f"    mean_value   : {s['mean_value']:.4f}m")
        print(f"    min_value    : {s['min_value']:.4f}m")
    print("=" * 60)
    print(f"\nSaved to: {out_path}")

    if hasattr(env.simulator, "shutdown"):
        env.simulator.shutdown()


if __name__ == "__main__":
    main()
