"""
pipeline/generate_motion.py — Generate a Kimodo motion and package for ProtoMotions.

Closely mirrors kimodo/scripts/generate.py, adding CSV → motion.pt packaging.

Usage (run from VLM-GMT root or anywhere; pass --protomotions-root explicitly)
------
    # Baseline (no constraints)
    python pipeline/generate_motion.py \\
        --output-dir outputs/reach_obj/baseline \\
        --protomotions-root /path/to/ProtoMotions

    # GT (with pre-generated constraints)
    python pipeline/generate_motion.py \\
        --constraints outputs/reach_obj/gt/constraints.json \\
        --output-dir outputs/reach_obj/gt \\
        --protomotions-root /path/to/ProtoMotions

    # Custom prompt / duration
    python pipeline/generate_motion.py \\
        --prompt "A person squats down slowly." \\
        --duration 4.0 \\
        --output-dir outputs/squat/baseline \\
        --protomotions-root /path/to/ProtoMotions
"""

import argparse
import subprocess
import sys
from pathlib import Path

import torch

KIMODO_MODEL   = "kimodo-g1-rp"
DEFAULT_PROMPT = "A person reaches forward with their right hand to grab an object in front of them."
DEFAULT_DURATION  = 3.0
DENOISING_STEPS   = 100
SEED              = 42


def convert_and_package(csv_path: Path, output_dir: Path, protomotions_root: Path) -> Path:
    """CSV → .motion (per-motion ProtoMotions format) → .pt (MotionLib batch)."""
    proto_dir = output_dir / "proto"
    pt_path   = output_dir / "motion.pt"
    proto_dir.mkdir(parents=True, exist_ok=True)

    print("[generate_motion] Converting CSV → .motion ...")
    subprocess.run([
        sys.executable,
        str(protomotions_root / "data/scripts/convert_g1_csv_to_proto.py"),
        "--input-dir",       str(csv_path.parent.resolve()),
        "--output-dir",      str(proto_dir),
        "--input-fps",       "30",
        "--output-fps",      "30",
        "--pos-units",       "m",
        "--rot-format",      "quat_wxyz",
        "--joint-units",     "rad",
        "--no-has-header",
        "--no-has-frame-column",
        "--force-remake",
    ], check=True, cwd=str(protomotions_root))

    print("[generate_motion] Packaging .motion → .pt ...")
    subprocess.run([
        sys.executable,
        str(protomotions_root / "protomotions/components/motion_lib.py"),
        "--motion-path", str(proto_dir),
        "--output-file", str(pt_path),
    ], check=True, cwd=str(protomotions_root))

    print(f"[generate_motion] MotionLib written: {pt_path}")
    return pt_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Kimodo motion (optionally constrained) and package as motion.pt."
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT,
                        help="Text prompt for Kimodo")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION,
                        help="Motion duration in seconds (default: 3.0)")
    parser.add_argument("--model", default=KIMODO_MODEL,
                        help="Kimodo model name")
    parser.add_argument("--constraints", default=None,
                        help="Path to constraints.json (omit for baseline)")
    parser.add_argument("--diffusion-steps", type=int, default=DENOISING_STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write reach.csv, proto/, and motion.pt")
    parser.add_argument("--protomotions-root", required=True,
                        help="Path to ProtoMotions root directory")
    args = parser.parse_args()

    output_dir       = Path(args.output_dir).resolve()
    protomotions_root = Path(args.protomotions_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[generate_motion] device={device}")

    # ── Load model ─────────────────────────────────────────────────────────
    from kimodo import load_model
    from kimodo.constraints import load_constraints_lst
    from kimodo.tools import seed_everything
    from kimodo.exports.mujoco import MujocoQposConverter

    print(f"[generate_motion] Loading model '{args.model}' ...")
    model = load_model(args.model, device=device)

    # ── Load constraints ───────────────────────────────────────────────────
    if args.constraints:
        print(f"[generate_motion] Loading constraints: {args.constraints}")
        constraint_lst = load_constraints_lst(args.constraints, model.skeleton, device=device)
        print(f"[generate_motion] {len(constraint_lst)} constraint set(s):")
        for c in constraint_lst:
            print(f"    {c.__class__.__name__}")
    else:
        print("[generate_motion] No constraints — baseline run.")
        constraint_lst = []

    # ── Generate motion ────────────────────────────────────────────────────
    if args.seed is not None:
        seed_everything(args.seed)

    num_frames = int(args.duration * model.fps)
    print(f"[generate_motion] '{args.prompt}'  ({num_frames} frames @ {model.fps} fps)")

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
    csv_path = output_dir / "reach.csv"
    converter = MujocoQposConverter(model.skeleton)
    qpos = converter.dict_to_qpos(output, device)
    converter.save_csv(qpos, str(output_dir / "reach"))
    print(f"[generate_motion] CSV saved: {csv_path}")

    # ── Convert + package ──────────────────────────────────────────────────
    pt_path = convert_and_package(csv_path, output_dir, protomotions_root)

    # ── Usage hints ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Done!  motion.pt → {pt_path}")
    print(f"\nKinematic preview (run from ProtoMotions root):")
    print(f"  python examples/env_kinematic_playback.py \\")
    print(f"    --experiment-path examples/experiments/mimic/mlp.py \\")
    print(f"    --motion-file {pt_path} \\")
    print(f"    --robot-name g1 --simulator isaaclab --num-envs 1")
    print(f"\nGMT inference (run from ProtoMotions root):")
    print(f"  python protomotions/inference_agent.py \\")
    print(f"    --checkpoint data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt \\")
    print(f"    --motion-file {pt_path} \\")
    print(f"    --simulator isaaclab --num-envs 1")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
