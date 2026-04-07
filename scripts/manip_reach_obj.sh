#!/bin/bash
# Task: manip_reach_obj
# Robot reaches a red cube on a table with its right hand.
# Success: dist(right_wrist, cube) < 0.15m at episode end.
#
# Run order:
#   1. Create scene        (local)
#   2. Generate baseline   (cluster)
#   3. Capture ego image   (local)
#   4. Generate gt + vlm   (cluster)
#   5. Kinematic playback  (local, optional sanity check)
#   6. Eval                (local)
#
# Edit variables below before running.

# ── Variables ────────────────────────────────────────────────────────────────
VLMGMT=~/Documents/vlm_project/VLM-GMT
PROTOMOTIONS=~/Documents/vlm_project/ProtoMotions
CKPT=$PROTOMOTIONS/data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt

CUBE_POS="0.6 0.0 0.4"

# ── 1. Create scene (local) ──────────────────────────────────────────────────
python $VLMGMT/tasks/manip_reach_obj/create_scene.py \
    --cube-pos $CUBE_POS \
    --output $VLMGMT/outputs/manip_reach_obj_scene.pt

# ── 2. Generate baseline motion — CLUSTER ────────────────────────────────────
# VLMGMT=~/kimodo_test/VLM-GMT
# PROTOMOTIONS=~/kimodo_test/ProtoMotions
# export HF_HOME=$SCRATCH/huggingface_cache
python $VLMGMT/pipeline/generate_motion.py \
    --task manip_reach_obj --condition baseline \
    --output-dir $VLMGMT/outputs/manip_reach_obj/baseline \
    --protomotions-root $PROTOMOTIONS --vlm-gmt-root $VLMGMT

# ── 3. Capture ego image (local, from $PROTOMOTIONS) ─────────────────────────
cd $PROTOMOTIONS
python $VLMGMT/pipeline/capture_egocentric.py \
    --experiment-path examples/experiments/mimic/mlp.py \
    --motion-file $VLMGMT/outputs/manip_reach_obj/baseline/motion.pt \
    --robot-name g1 --simulator isaaclab --num-envs 1 \
    --scenes-file $VLMGMT/outputs/manip_reach_obj_scene.pt \
    --output-dir $VLMGMT/outputs/manip_reach_obj \
    --pitch-deg 50 --robot-yaw-deg 0 --prefix ego

# ── 4. Generate gt + vlm motions — CLUSTER ───────────────────────────────────
# GT
python $VLMGMT/pipeline/generate_motion.py \
    --task manip_reach_obj --condition gt \
    --cube-world-pos $CUBE_POS \
    --output-dir $VLMGMT/outputs/manip_reach_obj/gt \
    --protomotions-root $PROTOMOTIONS --vlm-gmt-root $VLMGMT

# VLM 32B (scp ego.png to cluster first)
python $VLMGMT/pipeline/generate_motion.py \
    --task manip_reach_obj --condition vlm \
    --image $VLMGMT/outputs/manip_reach_obj/ego.png \
    --vlm-name qwen2.5-vl-32b --pitch-deg 50 \
    --output-dir $VLMGMT/outputs/manip_reach_obj/vlm_32b \
    --protomotions-root $PROTOMOTIONS --vlm-gmt-root $VLMGMT

# VLM 7B
python $VLMGMT/pipeline/generate_motion.py \
    --task manip_reach_obj --condition vlm \
    --image $VLMGMT/outputs/manip_reach_obj/ego.png \
    --vlm-name qwen2.5-vl-7b --pitch-deg 50 \
    --output-dir $VLMGMT/outputs/manip_reach_obj/vlm_7b \
    --protomotions-root $PROTOMOTIONS --vlm-gmt-root $VLMGMT

# ── 5. Kinematic playback (local, from $PROTOMOTIONS) ────────────────────────
cd $PROTOMOTIONS
python examples/env_kinematic_playback.py \
    --experiment-path examples/experiments/mimic/mlp.py \
    --motion-file $VLMGMT/outputs/manip_reach_obj/baseline/motion.pt \
    --robot-name g1 --simulator isaaclab --num-envs 1 \
    --scenes-file $VLMGMT/outputs/manip_reach_obj_scene.pt

# ── 6. Eval (local, from $PROTOMOTIONS) ──────────────────────────────────────
cd $PROTOMOTIONS
for COND in baseline gt vlm_7b vlm_32b; do
    python $VLMGMT/eval/run_eval.py \
        --checkpoint $CKPT \
        --motion-file $VLMGMT/outputs/manip_reach_obj/${COND}/motion.pt \
        --scenes-file $VLMGMT/outputs/manip_reach_obj_scene.pt \
        --task manip_reach_obj --condition ${COND} \
        --num-episodes 50 --simulator isaaclab \
        --output-dir $VLMGMT/outputs/manip_reach_obj/results \
        --protomotions-root $PROTOMOTIONS --vlm-gmt-root $VLMGMT
done
