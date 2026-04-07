#!/bin/bash
# Task: point_at_obj_with_right_hand
# Robot points at a blue object to its right with its right hand.
# Success: dist(right_hand, obj) < 0.15m at episode end.
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

OBJ_POS="0.5 -0.3 0.9"

# ── 1. Create scene (local) ──────────────────────────────────────────────────
python $VLMGMT/tasks/point_at_obj_with_right_hand/create_scene.py \
    --obj-pos $OBJ_POS \
    --output $VLMGMT/outputs/point_at_obj_with_right_hand_scene.pt

# ── 2. Generate baseline motion — CLUSTER ────────────────────────────────────
# VLMGMT=~/kimodo_test/VLM-GMT
# PROTOMOTIONS=~/kimodo_test/ProtoMotions
# export HF_HOME=$SCRATCH/huggingface_cache
python $VLMGMT/pipeline/generate_motion.py \
    --task point_at_obj_with_right_hand --condition baseline \
    --output-dir $VLMGMT/outputs/point_at_obj_with_right_hand/baseline \
    --protomotions-root $PROTOMOTIONS --vlm-gmt-root $VLMGMT

# ── 3. Capture ego image (local, from $PROTOMOTIONS) ─────────────────────────
cd $PROTOMOTIONS
python $VLMGMT/pipeline/capture_egocentric.py \
    --experiment-path examples/experiments/mimic/mlp.py \
    --motion-file $VLMGMT/outputs/point_at_obj_with_right_hand/baseline/motion.pt \
    --robot-name g1 --simulator isaaclab --num-envs 1 \
    --scenes-file $VLMGMT/outputs/point_at_obj_with_right_hand_scene.pt \
    --output-dir $VLMGMT/outputs/point_at_obj_with_right_hand \
    --pitch-deg 50 --robot-yaw-deg 0 --prefix ego

# ── 4. Generate gt + vlm motions — CLUSTER ───────────────────────────────────
# GT
python $VLMGMT/pipeline/generate_motion.py \
    --task point_at_obj_with_right_hand --condition gt \
    --cube-world-pos $OBJ_POS \
    --output-dir $VLMGMT/outputs/point_at_obj_with_right_hand/gt \
    --protomotions-root $PROTOMOTIONS --vlm-gmt-root $VLMGMT

# VLM (scp ego.png to cluster first)
python $VLMGMT/pipeline/generate_motion.py \
    --task point_at_obj_with_right_hand --condition vlm \
    --image $VLMGMT/outputs/point_at_obj_with_right_hand/ego.png \
    --vlm-name qwen2.5-vl-32b \
    --output-dir $VLMGMT/outputs/point_at_obj_with_right_hand/vlm \
    --protomotions-root $PROTOMOTIONS --vlm-gmt-root $VLMGMT

# ── 5. Kinematic playback (local, from $PROTOMOTIONS) ────────────────────────
cd $PROTOMOTIONS
python examples/env_kinematic_playback.py \
    --experiment-path examples/experiments/mimic/mlp.py \
    --motion-file $VLMGMT/outputs/point_at_obj_with_right_hand/baseline/motion.pt \
    --robot-name g1 --simulator isaaclab --num-envs 1 \
    --scenes-file $VLMGMT/outputs/point_at_obj_with_right_hand_scene.pt

# ── 6. Eval (local, from $PROTOMOTIONS) ──────────────────────────────────────
cd $PROTOMOTIONS
for COND in baseline gt vlm; do
    python $VLMGMT/eval/run_eval.py \
        --checkpoint $CKPT \
        --motion-file $VLMGMT/outputs/point_at_obj_with_right_hand/${COND}/motion.pt \
        --scenes-file $VLMGMT/outputs/point_at_obj_with_right_hand_scene.pt \
        --task point_at_obj_with_right_hand --condition ${COND} \
        --num-episodes 50 --simulator isaaclab \
        --output-dir $VLMGMT/outputs/point_at_obj_with_right_hand/results \
        --protomotions-root $PROTOMOTIONS --vlm-gmt-root $VLMGMT
done
