#!/bin/bash
# Task: navigate_maze
# Robot navigates around 2 staggering walls (maze-like) on default terrain.
# Wall 1 blocks left side, wall 2 blocks right side.
# Success: avoids both walls laterally AND final pelvis x past wall 2 + 0.5m.
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

# Wall inner-edge positions (matching create_scene.py / metrics.py)
OBS1="2.0 0.1 0.5"
OBS2="4.0 -0.1 0.5"

# ── 1. Create scene (local) ──────────────────────────────────────────────────
python $VLMGMT/tasks/navigate_maze/create_scene.py \
    --output $VLMGMT/outputs/navigate_maze_scene.pt

# ── 2. Generate baseline motion — CLUSTER ────────────────────────────────────
# VLMGMT=~/kimodo_test/VLM-GMT
# PROTOMOTIONS=~/kimodo_test/ProtoMotions
# export HF_HOME=$SCRATCH/huggingface_cache
python $VLMGMT/pipeline/generate_motion.py \
    --task navigate_maze --condition baseline \
    --duration 10 \
    --output-dir $VLMGMT/outputs/navigate_maze/baseline \
    --protomotions-root $PROTOMOTIONS

# ── 3. Capture ego image (local, from $PROTOMOTIONS) ─────────────────────────
cd $PROTOMOTIONS
python $VLMGMT/pipeline/capture_egocentric.py \
    --experiment-path examples/experiments/mimic/mlp.py \
    --motion-file $VLMGMT/outputs/navigate_maze/baseline/motion.pt \
    --robot-name g1 --simulator isaaclab --num-envs 1 \
    --scenes-file $VLMGMT/outputs/navigate_maze_scene.pt \
    --output-dir $VLMGMT/outputs/navigate_maze \
    --pitch-deg 30 --robot-yaw-deg 0 --prefix ego

# ── 4. Generate gt + vlm motions — CLUSTER ───────────────────────────────────
# GT (2 root2d constraints per wall: before + past, plus final waypoint)
python $VLMGMT/pipeline/generate_motion.py \
    --task navigate_maze --condition gt \
    --duration 10 \
    --obs1-world-pos $OBS1 \
    --obs2-world-pos $OBS2 \
    --output-dir $VLMGMT/outputs/navigate_maze/gt \
    --protomotions-root $PROTOMOTIONS

# VLM (scp ego.png to cluster first)
python $VLMGMT/pipeline/generate_motion.py \
    --task navigate_maze --condition vlm \
    --duration 10 \
    --image $VLMGMT/outputs/navigate_maze/ego.png \
    --vlm-name qwen2.5-vl-32b \
    --output-dir $VLMGMT/outputs/navigate_maze/vlm \
    --protomotions-root $PROTOMOTIONS

# ── 5. Kinematic playback (local, from $PROTOMOTIONS) ────────────────────────
cd $PROTOMOTIONS
python examples/env_kinematic_playback.py \
    --experiment-path examples/experiments/mimic/mlp.py \
    --motion-file $VLMGMT/outputs/navigate_maze/baseline/motion.pt \
    --robot-name g1 --simulator isaaclab --num-envs 1 \
    --scenes-file $VLMGMT/outputs/navigate_maze_scene.pt

# ── 6. Eval (local, from $PROTOMOTIONS) ──────────────────────────────────────
cd $PROTOMOTIONS
for COND in baseline gt vlm; do
    python $VLMGMT/eval/run_eval.py \
        --checkpoint $CKPT \
        --motion-file $VLMGMT/outputs/navigate_maze/${COND}/motion.pt \
        --scenes-file $VLMGMT/outputs/navigate_maze_scene.pt \
        --task navigate_maze --condition ${COND} \
        --num-episodes 50 --simulator isaaclab \
        --output-dir $VLMGMT/outputs/navigate_maze/results \
        --protomotions-root $PROTOMOTIONS
done
