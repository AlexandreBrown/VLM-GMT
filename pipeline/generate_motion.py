"""
pipeline/generate_motion.py — Generate a Kimodo motion and package for ProtoMotions.

Builds constraints in memory (no JSON, no IK) and passes them directly to
model(constraint_lst=...). Then converts output CSV → motion.pt.

Usage (run from VLM-GMT root, in kimodo env)
----
    # Baseline (no constraints)
    python pipeline/generate_motion.py \\
        --condition baseline \\
        --output-dir outputs/reach_obj/baseline \\
        --protomotions-root /path/to/ProtoMotions

    # GT
    python pipeline/generate_motion.py \\
        --condition gt \\
        --cube-world-pos 0.6 0.0 0.4 \\
        --output-dir outputs/reach_obj/gt \\
        --protomotions-root /path/to/ProtoMotions

    # VLM
    python pipeline/generate_motion.py \\
        --condition vlm \\
        --image outputs/reach_obj/ego.png \\
        --task-description "Reach the red cube with your right hand." \\
        --output-dir outputs/reach_obj/vlm \\
        --protomotions-root /path/to/ProtoMotions
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

KIMODO_MODEL = "kimodo-g1-rp"
DEFAULT_DURATION = 3.0
DENOISING_STEPS = 100
SEED = 42


def convert_and_package(
    csv_path: Path, output_dir: Path, protomotions_root: Path
) -> Path:
    """CSV → per-motion .motion files → MotionLib .pt batch."""
    proto_dir = output_dir / "proto"
    pt_path = output_dir / "motion.pt"
    proto_dir.mkdir(parents=True, exist_ok=True)

    print("[generate_motion] Converting CSV → .motion ...")
    subprocess.run(
        [
            sys.executable,
            str(protomotions_root / "data/scripts/convert_g1_csv_to_proto.py"),
            "--input-dir",
            str(csv_path.parent.resolve()),
            "--output-dir",
            str(proto_dir),
            "--input-fps",
            "30",
            "--output-fps",
            "30",
            "--pos-units",
            "m",
            "--rot-format",
            "quat_wxyz",
            "--joint-units",
            "rad",
            "--no-has-header",
            "--no-has-frame-column",
            "--force-remake",
        ],
        check=True,
        cwd=str(protomotions_root),
    )

    print("[generate_motion] Packaging .motion → .pt ...")
    subprocess.run(
        [
            sys.executable,
            str(protomotions_root / "protomotions/components/motion_lib.py"),
            "--motion-path",
            str(proto_dir),
            "--output-file",
            str(pt_path),
        ],
        check=True,
        cwd=str(protomotions_root),
    )

    print(f"[generate_motion] MotionLib written: {pt_path}")
    return pt_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Kimodo motion (optionally constrained) and package as motion.pt."
    )
    parser.add_argument("--task", default="manip_reach_obj")
    parser.add_argument("--condition", choices=["baseline", "gt", "vlm"], required=True)
    parser.add_argument("--prompt", default=None,
                        help="Kimodo text prompt. If not set, loads from tasks/<task>/kimodo_prompt.txt")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    parser.add_argument("--kimodo-model", default=KIMODO_MODEL)
    parser.add_argument("--diffusion-steps", type=int, default=DENOISING_STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--frame-index",
        type=int,
        default=None,
        help="Single keyframe index for constraint (30fps). "
             "If not set, --constraint-last-n-frames is used instead.",
    )
    parser.add_argument(
        "--constraint-last-n-frames",
        type=int,
        default=20,
        help="Constrain the last N frames of the motion (default 20). "
             "Ignored if --frame-index is set.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--protomotions-root", required=True)

    # GT: manip_reach_obj
    parser.add_argument(
        "--cube-world-pos", nargs=3, type=float, metavar=("X", "Y", "Z")
    )
    # GT: walk_to_obj
    parser.add_argument(
        "--box-world-pos", nargs=3, type=float, metavar=("X", "Y", "Z")
    )

    # VLM
    parser.add_argument("--image", help="Egocentric RGB image for VLM condition")
    parser.add_argument(
        "--task-description",
        default="Reach the red cube with your right hand.",
        help="Natural-language task description passed to the VLM",
    )
    parser.add_argument("--vlm-name", default="qwen3.5-27b")
    parser.add_argument("--vlm-no-4bit", action="store_true", default=False,
                        help="Disable 4-bit quantization (use bfloat16, requires more VRAM)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    protomotions_root = Path(args.protomotions_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add VLM-GMT root to sys.path for pipeline imports
    vlm_gmt_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(vlm_gmt_root))

    # Load kimodo text prompt from file if not provided
    if args.prompt is None:
        prompt_path = vlm_gmt_root / "tasks" / args.task / "kimodo_prompt.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(f"No kimodo prompt for task '{args.task}' at {prompt_path}")
        args.prompt = prompt_path.read_text().strip()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[generate_motion] device={device}  condition={args.condition}")

    # ── VLM: query BEFORE loading Kimodo to avoid concurrent GPU memory usage ──
    # VLM only needs the image; skeleton is not required at this stage.
    raw_vlm_constraints = None
    if args.condition == "vlm":
        if args.image is None:
            parser.error("--image required for condition=vlm")
        from PIL import Image as PILImage
        from pipeline.generate_constraints import query_vlm_raw
        import gc

        image_rgb = np.array(PILImage.open(args.image).convert("RGB"))
        num_frames_approx = int(args.duration * 30)  # 30fps estimate before model loads
        raw_vlm_constraints = query_vlm_raw(
            task=args.task,
            image_rgb=image_rgb,
            task_description=args.task_description,
            vlm_name=args.vlm_name,
            load_in_4bit=not args.vlm_no_4bit,
            num_frames=num_frames_approx,
            output_dir=str(output_dir),
        )
        # Free VLM memory before loading Kimodo
        gc.collect()
        import torch as _torch
        _torch.cuda.empty_cache()
        print("[generate_motion] VLM unloaded. Loading Kimodo ...")

    # ── Load Kimodo ────────────────────────────────────────────────────────
    from kimodo import load_model
    from kimodo.tools import seed_everything
    from kimodo.exports.mujoco import MujocoQposConverter

    print(f"[generate_motion] Loading model '{args.kimodo_model}' ...")
    model = load_model(args.kimodo_model, device=device)

    # ── Build constraints (skeleton now available) ─────────────────────────
    from pipeline.generate_constraints import build_constraints

    num_frames = int(args.duration * model.fps)
    constraint_kwargs = {"num_frames": num_frames}

    if args.condition == "gt":
        if args.frame_index is not None:
            frame_index = args.frame_index
        else:
            n = min(args.constraint_last_n_frames, num_frames)
            frame_index = list(range(num_frames - n, num_frames))
        print(f"[generate_motion] GT constraint frames: {frame_index}")
        constraint_kwargs["frame_index"] = frame_index
        if args.task == "manip_reach_obj":
            if args.cube_world_pos is None:
                parser.error("--cube-world-pos required for condition=gt with task=manip_reach_obj")
            constraint_kwargs["cube_world_pos"] = args.cube_world_pos
        elif args.task == "walk_to_obj":
            if args.box_world_pos is None:
                parser.error("--box-world-pos required for condition=gt with task=walk_to_obj")
            constraint_kwargs["box_world_pos"] = args.box_world_pos

    elif args.condition == "vlm":
        constraint_kwargs["raw_vlm_constraints"] = raw_vlm_constraints

    print(f"[generate_motion] Building constraints ...")
    constraint_lst = build_constraints(
        args.task, args.condition, model.skeleton, device, **constraint_kwargs
    )
    if constraint_lst:
        print(f"[generate_motion] {len(constraint_lst)} constraint set(s) ready.")
    else:
        print("[generate_motion] No constraints (baseline).")

    # ── Generate motion ────────────────────────────────────────────────────
    if args.seed is not None:
        seed_everything(args.seed)

    print(
        f"[generate_motion] Generating: '{args.prompt}'  ({num_frames} frames @ {model.fps}fps)"
    )

    output = model(
        [args.prompt],
        [num_frames],
        constraint_lst=constraint_lst,
        num_denoising_steps=args.diffusion_steps,
        num_samples=1,
        multi_prompt=True,
        post_processing=False,  # not supported for G1
        return_numpy=True,
    )

    # ── Save CSV ───────────────────────────────────────────────────────────
    csv_path = output_dir / f"{args.task}.csv"
    converter = MujocoQposConverter(model.skeleton)
    qpos = converter.dict_to_qpos(output, device)
    converter.save_csv(qpos, str(csv_path))
    print(f"[generate_motion] CSV saved: {csv_path}")

    # ── Convert + package ──────────────────────────────────────────────────
    pt_path = convert_and_package(csv_path, output_dir, protomotions_root)

    print(f"\n{'='*60}")
    print(f"Done!  motion.pt → {pt_path}")
    print(f"\nKinematic preview (from ProtoMotions root):")
    print(f"  python examples/env_kinematic_playback.py \\")
    print(f"    --experiment-path examples/experiments/mimic/mlp.py \\")
    print(
        f"    --motion-file {pt_path} --robot-name g1 --simulator isaaclab --num-envs 1"
    )
    print(f"\nGMT inference (from ProtoMotions root):")
    print(f"  python protomotions/inference_agent.py \\")
    print(
        f"    --checkpoint data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt \\"
    )
    print(f"    --motion-file {pt_path} --simulator isaaclab --num-envs 1")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
