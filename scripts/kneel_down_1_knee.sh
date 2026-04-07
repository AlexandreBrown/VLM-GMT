#!/bin/bash
# Task: kneel_down_1_knee (text-only, fullbody constraint)
# Robot kneels down on one knee with hands on the other knee.
# Success: pelvis Z < 0.5m at episode end.
#
# Text-only task: no ego capture, VLM queried without image.
# GT uses fullbody constraint exported from Kimodo UI.
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

# GT fullbody constraint JSON (exported from Kimodo UI)
FULLBODY_JSON=$VLMGMT/constraints/output_kneel_down_constraints.json

# ── 1. Create empty scene (local) ────────────────────────────────────────────
python $VLMGMT/tasks/kneel_down_1_knee/create_scene.py \
    --output $VLMGMT/outputs/kneel_down_1_knee_scene.pt

# ── 2. Generate baseline motion — CLUSTER ────────────────────────────────────
# VLMGMT=~/kimodo_test/VLM-GMT
# PROTOMOTIONS=~/kimodo_test/ProtoMotions
# export HF_HOME=$SCRATCH/huggingface_cache
python $VLMGMT/pipeline/generate_motion.py \
    --task kneel_down_1_knee --condition baseline \
    --output-dir $VLMGMT/outputs/kneel_down_1_knee/baseline \
    --protomotions-root $PROTOMOTIONS --vlm-gmt-root $VLMGMT

# ── 3. Generate gt + vlm motions — CLUSTER ───────────────────────────────────
# GT (fullbody constraint from Kimodo UI export)
python $VLMGMT/pipeline/generate_motion.py \
    --task kneel_down_1_knee --condition gt \
    --constraint-json $FULLBODY_JSON \
    --output-dir $VLMGMT/outputs/kneel_down_1_knee/gt \
    --protomotions-root $PROTOMOTIONS --vlm-gmt-root $VLMGMT

# VLM (text-only, no --image)
python $VLMGMT/pipeline/generate_motion.py \
    --task kneel_down_1_knee --condition vlm \
    --vlm-name qwen2.5-vl-32b \
    --output-dir $VLMGMT/outputs/kneel_down_1_knee/vlm \
    --protomotions-root $PROTOMOTIONS --vlm-gmt-root $VLMGMT

# ── 4. Eval (local, from $PROTOMOTIONS) ──────────────────────────────────────
cd $PROTOMOTIONS
for COND in baseline gt vlm; do
    python $VLMGMT/eval/run_eval.py \
        --checkpoint $CKPT \
        --motion-file $VLMGMT/outputs/kneel_down_1_knee/${COND}/motion.pt \
        --scenes-file $VLMGMT/outputs/kneel_down_1_knee_scene.pt \
        --task kneel_down_1_knee --condition ${COND} \
        --num-episodes 50 --simulator isaaclab \
        --output-dir $VLMGMT/outputs/kneel_down_1_knee/results \
        --protomotions-root $PROTOMOTIONS --vlm-gmt-root $VLMGMT
done
