#!/bin/bash
# Task: touch_right_leg_with_left_hand (text-only)
# Robot bends down and touches its right knee with its left hand.
# Success: dist(left_rubber_hand, right_knee) < 0.15m at episode end.
#
# Text-only task: no ego capture, VLM queried without image.
#
# Run order:
#   1. Create empty scene  (local)
#   2. Generate baseline   (cluster)
#   3. Generate gt + vlm   (cluster)
#   4. Eval                (local)
#
# Edit variables below before running.

# ── Variables ────────────────────────────────────────────────────────────────
VLMGMT=~/Documents/vlm_project/VLM-GMT
PROTOMOTIONS=~/Documents/vlm_project/ProtoMotions
CKPT=$PROTOMOTIONS/data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt

# GT left-hand target: right knee position (slightly right, knee height)
HAND_POS="0.0 -0.1 0.45"

# ── 1. Create empty scene (local) ────────────────────────────────────────────
python $VLMGMT/tasks/touch_right_leg_with_left_hand/create_scene.py \
    --output $VLMGMT/outputs/touch_right_leg_with_left_hand_scene.pt

# ── 2. Generate baseline motion — CLUSTER ────────────────────────────────────
# VLMGMT=~/kimodo_test/VLM-GMT
# PROTOMOTIONS=~/kimodo_test/ProtoMotions
# export HF_HOME=$SCRATCH/huggingface_cache
python $VLMGMT/pipeline/generate_motion.py \
    --task touch_right_leg_with_left_hand --condition baseline \
    --output-dir $VLMGMT/outputs/touch_right_leg_with_left_hand/baseline \
    --protomotions-root $PROTOMOTIONS --vlm-gmt-root $VLMGMT

# ── 3. Generate gt + vlm motions — CLUSTER ───────────────────────────────────
# GT
python $VLMGMT/pipeline/generate_motion.py \
    --task touch_right_leg_with_left_hand --condition gt \
    --cube-world-pos $HAND_POS \
    --output-dir $VLMGMT/outputs/touch_right_leg_with_left_hand/gt \
    --protomotions-root $PROTOMOTIONS --vlm-gmt-root $VLMGMT

# VLM 32B (text-only, no --image)
python $VLMGMT/pipeline/generate_motion.py \
    --task touch_right_leg_with_left_hand --condition vlm \
    --vlm-name qwen2.5-vl-32b \
    --output-dir $VLMGMT/outputs/touch_right_leg_with_left_hand/vlm_32b \
    --protomotions-root $PROTOMOTIONS --vlm-gmt-root $VLMGMT

# VLM 7B (text-only, no --image)
python $VLMGMT/pipeline/generate_motion.py \
    --task touch_right_leg_with_left_hand --condition vlm \
    --vlm-name qwen2.5-vl-7b \
    --output-dir $VLMGMT/outputs/touch_right_leg_with_left_hand/vlm_7b \
    --protomotions-root $PROTOMOTIONS --vlm-gmt-root $VLMGMT

# ── 4. Eval (local, from $PROTOMOTIONS) ──────────────────────────────────────
cd $PROTOMOTIONS
for COND in baseline gt vlm_7b vlm_32b; do
    python $VLMGMT/eval/run_eval.py \
        --checkpoint $CKPT \
        --motion-file $VLMGMT/outputs/touch_right_leg_with_left_hand/${COND}/motion.pt \
        --scenes-file $VLMGMT/outputs/touch_right_leg_with_left_hand_scene.pt \
        --task touch_right_leg_with_left_hand --condition ${COND} \
        --num-episodes 50 --simulator isaaclab \
        --output-dir $VLMGMT/outputs/touch_right_leg_with_left_hand/results \
        --protomotions-root $PROTOMOTIONS --vlm-gmt-root $VLMGMT
done
