#!/bin/bash
# Task: walk_to_obj
# Robot walks toward a green box offset laterally so baseline (straight walk) misses it.
# Success: dist_2d(pelvis, box) < 0.5m at episode end.
#
# Local commands: create scene, capture ego, kinematic playback, eval.
# Cluster commands: generate_motion (baseline / gt / vlm).
#
# Edit variables below before running.

# ── Variables ────────────────────────────────────────────────────────────────
VLMGMT=~/Documents/vlm_project/VLM-GMT
PROTOMOTIONS=~/Documents/vlm_project/ProtoMotions
CKPT=$PROTOMOTIONS/data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt

BOX_POS="3.0 -1.1 0.25"

# ── 1. Create scene (local) ──────────────────────────────────────────────────
python $VLMGMT/tasks/walk_to_obj/create_scene.py \
    --box-pos $BOX_POS \
    --output $VLMGMT/outputs/walk_to_obj_scene.pt

# ── 2. Capture ego image (local, from $PROTOMOTIONS) ─────────────────────────
# Requires a baseline motion.pt to exist (generate it first on cluster).
cd $PROTOMOTIONS
python $VLMGMT/pipeline/capture_egocentric.py \
    --experiment-path examples/experiments/mimic/mlp.py \
    --motion-file $VLMGMT/outputs/walk_to_obj/baseline/motion.pt \
    --robot-name g1 --simulator isaaclab --num-envs 1 \
    --scenes-file $VLMGMT/outputs/walk_to_obj_scene.pt \
    --output-dir $VLMGMT/outputs/walk_to_obj \
    --pitch-deg 30 --robot-yaw-deg 0 --prefix ego_frame

# ── 3. Generate motion — CLUSTER ─────────────────────────────────────────────
# VLMGMT=~/kimodo_test/VLM-GMT
# PROTOMOTIONS=~/kimodo_test/ProtoMotions
# export HF_HOME=$SCRATCH/huggingface_cache

# Baseline
python $VLMGMT/pipeline/generate_motion.py \
    --task walk_to_obj --condition baseline \
    --output-dir $VLMGMT/outputs/walk_to_obj/baseline \
    --protomotions-root $PROTOMOTIONS

# GT
python $VLMGMT/pipeline/generate_motion.py \
    --task walk_to_obj --condition gt \
    --box-world-pos $BOX_POS \
    --output-dir $VLMGMT/outputs/walk_to_obj/gt \
    --protomotions-root $PROTOMOTIONS

# VLM (scp ego_frame.png to cluster first)
python $VLMGMT/pipeline/generate_motion.py \
    --task walk_to_obj --condition vlm \
    --image $VLMGMT/outputs/walk_to_obj/ego_frame.png \
    --vlm-name qwen2.5-vl-32b \
    --output-dir $VLMGMT/outputs/walk_to_obj/vlm \
    --protomotions-root $PROTOMOTIONS

# ── 4. Kinematic playback (local, from $PROTOMOTIONS) ────────────────────────
cd $PROTOMOTIONS
python examples/env_kinematic_playback.py \
    --experiment-path examples/experiments/mimic/mlp.py \
    --motion-file $VLMGMT/outputs/walk_to_obj/baseline/motion.pt \
    --robot-name g1 --simulator isaaclab --num-envs 1 \
    --scenes-file $VLMGMT/outputs/walk_to_obj_scene.pt

# ── 5. Eval (local, from $PROTOMOTIONS) ──────────────────────────────────────
cd $PROTOMOTIONS
for COND in baseline gt vlm; do
    python $VLMGMT/eval/run_eval.py \
        --checkpoint $CKPT \
        --motion-file $VLMGMT/outputs/walk_to_obj/${COND}/motion.pt \
        --scenes-file $VLMGMT/outputs/walk_to_obj_scene.pt \
        --task walk_to_obj --condition ${COND} \
        --num-episodes 50 --simulator isaaclab \
        --output-dir $VLMGMT/outputs/walk_to_obj/results \
        --protomotions-root $PROTOMOTIONS
done
