#!/bin/bash
# Capture per-task initial conditions (link/object positions at reset).
# Runs run_eval.py with --capture-initial-only for every task and writes
# outputs/initial_conditions.json.
#
# Run this from VLM-GMT root. Must be executed on a box with ProtoMotions
# + IsaacLab (i.e. your local machine). Safe to re-run.

set -e

VLMGMT=~/Documents/vlm_project/VLM-GMT
PROTOMOTIONS=~/Documents/vlm_project/ProtoMotions
CKPT=$PROTOMOTIONS/data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt

OUT=$VLMGMT/outputs/initial_conditions.json
rm -f $OUT

# (task_name, condition_dir) pairs. Condition only picks which motion.pt
# to load; initial state is condition-independent so we use "gt" everywhere
# (any condition with an existing motion.pt works).
TASKS=(
  "manip_reach_obj"
  "walk_to_obj"
  "navigate_maze"
  "point_at_obj_with_right_hand"
  "point_at_obj_with_left_hand"
  "raise_right_hand"
  "raise_left_hand"
  "kneel_down_1_knee"
  "touch_left_leg_with_right_hand"
  "touch_right_leg_with_left_hand"
)

cd $PROTOMOTIONS

for T in "${TASKS[@]}"; do
  echo ""
  echo "=== Capturing initial conditions for $T ==="
  python $VLMGMT/eval/run_eval.py \
      --checkpoint $CKPT \
      --motion-file $VLMGMT/outputs/$T/gt/motion.pt \
      --scenes-file $VLMGMT/outputs/${T}_scene.pt \
      --task $T \
      --condition gt \
      --simulator isaaclab \
      --num-envs 1 \
      --num-episodes 1 \
      --output-dir /tmp/vlmgmt_init_capture \
      --headless \
      --no-video \
      --capture-initial-only \
      --initial-output $OUT \
      --protomotions-root $PROTOMOTIONS \
      --vlm-gmt-root $VLMGMT
done

echo ""
echo "Done. Wrote $OUT"
