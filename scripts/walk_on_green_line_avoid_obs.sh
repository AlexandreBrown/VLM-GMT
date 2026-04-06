#!/bin/bash
# Task: walk_on_green_line_avoid_obs
# Robot walks along a 1m-wide green line and navigates around 3 colored obstacles on it.
# Success: pelvis stays within line Y bounds (±0.5m) AND passes all 3 obstacles during episode.
#
# Local commands: create scene, capture ego, kinematic playback, eval.
# Cluster commands: generate_motion (baseline / gt / vlm).
#
# Edit variables below before running.

# ── Variables ────────────────────────────────────────────────────────────────
VLMGMT=~/Documents/vlm_project/VLM-GMT
PROTOMOTIONS=~/Documents/vlm_project/ProtoMotions
CKPT=$PROTOMOTIONS/data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt

# Obstacle positions (default, matching create_scene.py defaults)
OBS1="1.5 0.20 0.30"
OBS2="3.0 -0.20 0.35"
OBS3="4.5 0.15 0.25"

# ── 1. Create scene (local) ──────────────────────────────────────────────────
python $VLMGMT/tasks/walk_on_green_line_avoid_obs/create_scene.py \
    --output $VLMGMT/outputs/walk_on_green_line_avoid_obs_scene.pt

# ── 2. Capture ego image (local, from $PROTOMOTIONS) ─────────────────────────
# Requires a baseline motion.pt to exist (generate it first on cluster).
cd $PROTOMOTIONS
python $VLMGMT/pipeline/capture_egocentric.py \
    --experiment-path examples/experiments/mimic/mlp.py \
    --motion-file $VLMGMT/outputs/walk_on_green_line_avoid_obs/baseline/motion.pt \
    --robot-name g1 --simulator isaaclab --num-envs 1 \
    --scenes-file $VLMGMT/outputs/walk_on_green_line_avoid_obs_scene.pt \
    --output-dir $VLMGMT/outputs/walk_on_green_line_avoid_obs \
    --pitch-deg 30 --robot-yaw-deg 0 --prefix ego_frame

# ── 3. Generate motion — CLUSTER ─────────────────────────────────────────────
# VLMGMT=~/kimodo_test/VLM-GMT
# PROTOMOTIONS=~/kimodo_test/ProtoMotions
# export HF_HOME=$SCRATCH/huggingface_cache

# Baseline
python $VLMGMT/pipeline/generate_motion.py \
    --task walk_on_green_line_avoid_obs --condition baseline \
    --output-dir $VLMGMT/outputs/walk_on_green_line_avoid_obs/baseline \
    --protomotions-root $PROTOMOTIONS

# GT (obstacle positions default to create_scene.py values; pass explicitly if changed)
python $VLMGMT/pipeline/generate_motion.py \
    --task walk_on_green_line_avoid_obs --condition gt \
    --obs1-world-pos $OBS1 \
    --obs2-world-pos $OBS2 \
    --obs3-world-pos $OBS3 \
    --output-dir $VLMGMT/outputs/walk_on_green_line_avoid_obs/gt \
    --protomotions-root $PROTOMOTIONS

# VLM (scp ego_frame.png to cluster first)
python $VLMGMT/pipeline/generate_motion.py \
    --task walk_on_green_line_avoid_obs --condition vlm \
    --image $VLMGMT/outputs/walk_on_green_line_avoid_obs/ego_frame.png \
    --vlm-name qwen2.5-vl-32b \
    --output-dir $VLMGMT/outputs/walk_on_green_line_avoid_obs/vlm \
    --protomotions-root $PROTOMOTIONS

# ── 4. Kinematic playback (local, from $PROTOMOTIONS) ────────────────────────
cd $PROTOMOTIONS
python examples/env_kinematic_playback.py \
    --experiment-path examples/experiments/mimic/mlp.py \
    --motion-file $VLMGMT/outputs/walk_on_green_line_avoid_obs/baseline/motion.pt \
    --robot-name g1 --simulator isaaclab --num-envs 1 \
    --scenes-file $VLMGMT/outputs/walk_on_green_line_avoid_obs_scene.pt

# ── 5. Eval (local, from $PROTOMOTIONS) ──────────────────────────────────────
cd $PROTOMOTIONS
for COND in baseline gt vlm; do
    python $VLMGMT/eval/run_eval.py \
        --checkpoint $CKPT \
        --motion-file $VLMGMT/outputs/walk_on_green_line_avoid_obs/${COND}/motion.pt \
        --scenes-file $VLMGMT/outputs/walk_on_green_line_avoid_obs_scene.pt \
        --task walk_on_green_line_avoid_obs --condition ${COND} \
        --num-episodes 50 --simulator isaaclab \
        --output-dir $VLMGMT/outputs/walk_on_green_line_avoid_obs/results \
        --protomotions-root $PROTOMOTIONS
done
