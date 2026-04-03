"""
pipeline/generate_motion.py — Generate a motion with Kimodo and package it for ProtoMotions.

Pipeline:
    1. [Optional] Generate kinematic constraints (gt / vlm / none)
    2. Generate G1 CSV via Kimodo Python API
    3. Convert CSV → .motion (ProtoMotions convert_g1_csv_to_proto.py)
    4. Package .motion → .pt MotionLib (ProtoMotions motion_lib.py)

Usage (run from ProtoMotions root):
    # Baseline
    python VLM-GMT/pipeline/generate_motion.py \
        --condition baseline \
        --prompt "A person reaches forward with their right hand." \
        --output-dir VLM-GMT/outputs/reach_obj/baseline

    # GT
    python VLM-GMT/pipeline/generate_motion.py \
        --condition gt \
        --prompt "A person reaches forward with their right hand." \
        --cube-world-pos 0.6 0.0 0.4 \
        --output-dir VLM-GMT/outputs/reach_obj/gt

    # VLM
    python VLM-GMT/pipeline/generate_motion.py \
        --condition vlm \
        --prompt "A person reaches forward with their right hand." \
        --image VLM-GMT/outputs/reach_obj/sim_frame.png \
        --camera-intrinsics 500 500 320 240 \
        --camera-extrinsic-npy VLM-GMT/outputs/reach_obj/camera_T_world.npy \
        --output-dir VLM-GMT/outputs/reach_obj/vlm
"""

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_PROMPT = "A person reaches forward with their right hand to grab an object in front of them."
MOTION_DURATION = 3.0
NUM_FRAMES = int(MOTION_DURATION * 30)  # 90 frames @ 30fps
DENOISING_STEPS = 100
SEED = 42
KIMODO_MODEL = "kimodo-g1-rp"


def generate_csv(
    prompt: str,
    output_csv: Path,
    constraint_lst: list = None,
    seed: int = SEED,
    model_name: str = KIMODO_MODEL,
):
    """
    Generate G1 CSV via Kimodo Python API.
    Passes constraint objects directly to avoid JSON serialization loss.
    """
    import torch
    from kimodo import load_model
    from kimodo.tools import seed_everything
    from kimodo.exports.mujoco import MujocoQposConverter

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if seed is not None:
        seed_everything(seed)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[generate_motion] Loading Kimodo model '{model_name}' on {device}...")
    model = load_model(model_name, device=device)

    if constraint_lst:
        print(f"[generate_motion] Using {len(constraint_lst)} constraint set(s) in-memory.")
    else:
        print("[generate_motion] No constraints (text only).")
        constraint_lst = []

    print(f"[generate_motion] Generating: '{prompt}'")
    output = model(
        [prompt],
        [NUM_FRAMES],
        num_denoising_steps=DENOISING_STEPS,
        constraint_lst=constraint_lst,
        multi_prompt=True,
        post_processing=False,  # does not work for G1
        return_numpy=True,
    )

    converter = MujocoQposConverter(model.skeleton)
    qpos = converter.dict_to_qpos(output, device)
    csv_stem = str(output_csv.parent / output_csv.stem)
    converter.save_csv(qpos, csv_stem + ".csv")
    print(f"[generate_motion] CSV saved: {output_csv}")


def convert_and_package(csv_path: Path, output_dir: Path, protomotions_root: Path) -> Path:
    """Convert CSV → .motion → .pt."""
    proto_dir = output_dir.resolve() / "proto"
    pt_path = output_dir.resolve() / "motion.pt"
    proto_dir.mkdir(parents=True, exist_ok=True)

    print("[generate_motion] Converting CSV → .motion...")
    subprocess.run([
        sys.executable, str(protomotions_root / "data/scripts/convert_g1_csv_to_proto.py"),
        "--input-dir", str(csv_path.parent.resolve()),
        "--output-dir", str(proto_dir),
        "--input-fps", "30", "--output-fps", "30",
        "--pos-units", "m",
        "--rot-format", "quat_wxyz",
        "--joint-units", "rad",
        "--no-has-header",
        "--no-has-frame-column",
        "--force-remake",
    ], check=True, cwd=str(protomotions_root))

    print("[generate_motion] Packaging .motion → .pt...")
    subprocess.run([
        sys.executable, str(protomotions_root / "protomotions/components/motion_lib.py"),
        "--motion-path", str(proto_dir),
        "--output-file", str(pt_path),
    ], check=True, cwd=str(protomotions_root))

    print(f"[generate_motion] MotionLib: {pt_path}")
    return pt_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["baseline", "gt", "vlm"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--kimodo-model", default=KIMODO_MODEL)
    parser.add_argument(
        "--protomotions-root", required=True,
        help="Path to ProtoMotions root directory.",
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Skip Kimodo generation and use this existing CSV directly (for testing).",
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

    parser.add_argument("--frame-index", type=int, default=45)

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protomotions_root = Path(args.protomotions_root).resolve()

    csv_path = output_dir / "reach.csv"
    vlm_gmt_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(vlm_gmt_root))

    # --- Build constraints in-memory ---
    constraint_lst = []
    if args.condition != "baseline":
        from pipeline.generate_constraints import build_end_effector_constraints, gt_reach_keyframes, pixels_to_world
        import numpy as np

        if args.condition == "gt":
            if args.cube_world_pos is None:
                parser.error("--cube-world-pos required for --condition gt")
            target_pos = np.array(args.cube_world_pos, dtype=np.float32)
            print(f"[generate_motion] GT target (IsaacLab): {target_pos}")
            keyframes = gt_reach_keyframes(target_pos, args.frame_index)

        elif args.condition == "vlm":
            if not all([args.image, args.camera_intrinsics, args.camera_extrinsic_npy]):
                parser.error("--image, --camera-intrinsics, --camera-extrinsic-npy required for vlm.")
            from pipeline.vlm import load_vlm
            from PIL import Image as PILImage

            image_rgb = np.array(PILImage.open(args.image).convert("RGB"))
            fx, fy, cx, cy = args.camera_intrinsics
            cam_T_world = np.load(args.camera_extrinsic_npy).astype(np.float32)

            print(f"[generate_motion] Querying VLM ({args.vlm_name})...")
            vlm = load_vlm(args.vlm_name)
            u, v = vlm.query_object_pixels(image_rgb, args.object_description)
            print(f"[generate_motion] VLM pixel: u={u:.1f} v={v:.1f}")

            target_pos = pixels_to_world(u, v, fx, fy, cx, cy, cam_T_world, args.assumed_world_z)
            print(f"[generate_motion] VLM world pos (IsaacLab): {target_pos}")
            keyframes = gt_reach_keyframes(target_pos, args.frame_index)

        constraint_lst = build_end_effector_constraints(keyframes)
        print(f"[generate_motion] Built {len(constraint_lst)} constraint set(s) in-memory.")

    # --- Generate motion ---
    if args.csv_path is not None:
        import shutil
        csv_path = (output_dir / "reach.csv").resolve()
        shutil.copy(Path(args.csv_path).resolve(), csv_path)
        print(f"[generate_motion] Skipping Kimodo, using existing CSV: {args.csv_path}")
    else:
        generate_csv(
            prompt=args.prompt,
            output_csv=csv_path,
            constraint_lst=constraint_lst if constraint_lst else None,
            seed=args.seed,
            model_name=args.kimodo_model,
        )

    # --- Convert + package ---
    pt_path = convert_and_package(csv_path, output_dir, protomotions_root)

    scenes_file = str(vlm_gmt_root / "outputs" / "reach_obj_scene.pt")

    print(f"\n[generate_motion] Done — condition={args.condition}")
    print(f"  MotionLib: {pt_path}")
    print(f"\nKinematic playback:")
    print(f"  python examples/env_kinematic_playback.py \\")
    print(f"    --experiment-path examples/experiments/mimic/mlp.py \\")
    print(f"    --motion-file {pt_path} \\")
    print(f"    --robot-name g1 --simulator isaaclab --num-envs 1 \\")
    print(f"    --scenes-file {scenes_file}")
    print(f"\nGMT inference:")
    print(f"  python protomotions/inference_agent.py \\")
    print(f"    --checkpoint data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt \\")
    print(f"    --motion-file {pt_path} \\")
    print(f"    --simulator isaaclab --num-envs 1 \\")
    print(f"    --scenes-file {scenes_file}")


if __name__ == "__main__":
    main()
