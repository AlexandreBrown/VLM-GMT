#!/bin/bash
# Task: raise_left_hand (text-only, no scene objects)
# Robot raises its left hand above its head and keeps it there.
# Success: left hand Z > 1.3m at episode end.
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

# GT hand target: above head, slightly left
HAND_POS="0.1 0.2 1.5"

# ── 1. Create empty scene (local) ────────────────────────────────────────────
python $VLMGMT/tasks/raise_left_hand/create_scene.py \
    --output $VLMGMT/outputs/raise_left_hand_scene.pt

# ── 2. Generate baseline motion — CLUSTER ────────────────────────────────────
# VLMGMT=~/kimodo_test/VLM-GMT
# PROTOMOTIONS=~/kimodo_test/ProtoMotions
# export HF_HOME=$SCRATCH/huggingface_cache
python $VLMGMT/pipeline/generate_motion.py \
    --task raise_left_hand --condition baseline \
    --output-dir $VLMGMT/outputs/raise_left_hand/baseline \
    --protomotions-root $PROTOMOTIONS --vlm-gmt-root $VLMGMT

# ── 3. Generate gt + vlm motions — CLUSTER ───────────────────────────────────
# GT
python $VLMGMT/pipeline/generate_motion.py \
    --task raise_left_hand --condition gt \
    --cube-world-pos $HAND_POS \
    --output-dir $VLMGMT/outputs/raise_left_hand/gt \
    --protomotions-root $PROTOMOTIONS --vlm-gmt-root $VLMGMT

# VLM 32B (text-only, no --image)
python $VLMGMT/pipeline/generate_motion.py \
    --task raise_left_hand --condition vlm \
    --vlm-name qwen2.5-vl-32b \
    --output-dir $VLMGMT/outputs/raise_left_hand/vlm_32b \
    --protomotions-root $PROTOMOTIONS --vlm-gmt-root $VLMGMT

# VLM 7B (text-only, no --image)
python $VLMGMT/pipeline/generate_motion.py \
    --task raise_left_hand --condition vlm \
    --vlm-name qwen2.5-vl-7b \
    --output-dir $VLMGMT/outputs/raise_left_hand/vlm_7b \
    --protomotions-root $PROTOMOTIONS --vlm-gmt-root $VLMGMT

# ── 4. Eval (local, from $PROTOMOTIONS) ──────────────────────────────────────
cd $PROTOMOTIONS
for COND in baseline gt vlm_7b vlm_32b; do
    python $VLMGMT/eval/run_eval.py \
        --checkpoint $CKPT \
        --motion-file $VLMGMT/outputs/raise_left_hand/${COND}/motion.pt \
        --scenes-file $VLMGMT/outputs/raise_left_hand_scene.pt \
        --task raise_left_hand --condition ${COND} \
        --num-episodes 50 --simulator isaaclab \
        --output-dir $VLMGMT/outputs/raise_left_hand/results \
        --protomotions-root $PROTOMOTIONS --vlm-gmt-root $VLMGMT
done
