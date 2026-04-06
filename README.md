# VLM-GMT

Evaluating VLM-augmented kinematic constraints for General Motion Tracking in humanoid loco-manipulation.

## Overview

Can off-the-shelf Vision-Language Models generate useful kinematic constraints from egocentric observations? This project tests that by combining [Kimodo](https://research.nvidia.com/labs/sil/projects/kimodo/) (text+constraint → motion) with a pretrained GMT policy on the Unitree G1 via [ProtoMotions](https://github.com/NVlabs/ProtoMotions).

### Three Conditions (per task)

| Condition | Input to Kimodo | Purpose |
|---|---|---|
| **Baseline** | Text only | Lower bound |
| **VLM** | Text + VLM-predicted kinematic constraints | Main experiment |
| **GT** | Text + ground-truth kinematic constraints | Upper bound |

## Setup

### Dependencies

```bash
git clone https://github.com/NVlabs/ProtoMotions.git
uv pip install -e ProtoMotions
uv pip install dm_control easydict matplotlib

git clone https://github.com/AlexandreBrown/VLM-GMT.git
```

Commands that need IsaacLab (capture, playback, eval) must be run from `$PROTOMOTIONS` because ProtoMotions uses relative asset paths.

### IsaacLab

Required locally for kinematic playback, GMT inference, eval, and egocentric capture. Follow the [IsaacLab installation guide](https://isaac-sim.github.io/IsaacLab/).

### Kimodo (motion generation, cluster only)

Kimodo requires >=17 GB VRAM. Follow the Kimodo docs to install.

### VLM deps (VLM condition only)

```bash
uv pip install transformers>=4.50.0 accelerate qwen-vl-utils Pillow
```

## Repository Structure

```
VLM-GMT/
├── prompts/
│   └── system.txt                       # Shared VLM system prompt
├── tasks/
│   ├── manip_reach_obj/
│   │   ├── create_scene.py              # Red cube on a table
│   │   ├── metrics.py                   # dist(right_wrist, cube) < 0.15m
│   │   ├── kimodo_prompt.txt
│   │   └── vlm_prompt.txt
│   ├── walk_to_obj/
│   │   ├── create_scene.py              # Green box on the ground
│   │   ├── metrics.py                   # dist_2d(pelvis, box) < 0.5m
│   │   ├── kimodo_prompt.txt
│   │   └── vlm_prompt.txt
│   └── walk_on_green_line_avoid_obs/
│       ├── create_scene.py              # Green line + 3 obstacle boxes
│       ├── metrics.py                   # dist_2d(pelvis, line_end) < 0.5m
│       ├── kimodo_prompt.txt
│       └── vlm_prompt.txt
├── pipeline/
│   ├── generate_motion.py               # Constraints → Kimodo → motion.pt  (cluster)
│   ├── generate_constraints.py          # GT / VLM → Kimodo constraint objects
│   ├── capture_egocentric.py            # Capture G1 head camera image  (local)
│   ├── egocentric_camera.py             # Camera sensor utilities
│   └── vlm/
│       ├── __init__.py                  # Registry + load_vlm()
│       ├── base.py                      # Abstract VLMBase
│       └── qwen.py                      # Qwen2.5-VL (7B / 32B / 72B)
├── eval/
│   ├── run_eval.py                      # GMT inference + metrics  (local)
│   ├── video_recorder.py                # Optional video capture
│   └── metrics/
│       ├── base.py
│       └── distance_to_target.py        # Supports 2D, 3D, fixed target pos
├── outputs/                             # Generated data (gitignored)
└── README.md
```

## Common variables

```bash
VLMGMT=~/Documents/vlm_project/VLM-GMT          # local
PROTOMOTIONS=~/Documents/vlm_project/ProtoMotions  # local
CKPT=$PROTOMOTIONS/data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt

# Cluster
VLMGMT=~/kimodo_test/VLM-GMT
PROTOMOTIONS=~/kimodo_test/ProtoMotions
export HF_HOME=$SCRATCH/huggingface_cache
```

`generate_motion.py` runs on the **cluster** (needs Kimodo).
`capture_egocentric.py`, kinematic playback, and eval run **locally** (need IsaacLab).
`capture_egocentric.py` must be run from `$PROTOMOTIONS` (relative experiment paths).

---

## Task: manip_reach_obj

Robot reaches a red cube on a table with its right hand.
**Success:** `dist(right_wrist, cube) < 0.15m` at episode end.

### Create scene (local)

```bash
python $VLMGMT/tasks/manip_reach_obj/create_scene.py \
    --cube-pos 0.6 0.0 0.4 \
    --output $VLMGMT/outputs/manip_reach_obj_scene.pt
```

### Capture ego image (local, from $PROTOMOTIONS)

```bash
cd $PROTOMOTIONS
python $VLMGMT/pipeline/capture_egocentric.py \
    --experiment-path examples/experiments/mimic/mlp.py \
    --motion-file $VLMGMT/outputs/manip_reach_obj/baseline/motion.pt \
    --robot-name g1 --simulator isaaclab --num-envs 1 \
    --scenes-file $VLMGMT/outputs/manip_reach_obj_scene.pt \
    --output-dir $VLMGMT/outputs/manip_reach_obj \
    --pitch-deg 50 --robot-yaw-deg 0 --prefix ego_frame
```

### Generate motion (cluster)

```bash
# Baseline
python $VLMGMT/pipeline/generate_motion.py \
    --task manip_reach_obj --condition baseline \
    --output-dir $VLMGMT/outputs/manip_reach_obj/baseline \
    --protomotions-root $PROTOMOTIONS

# GT
python $VLMGMT/pipeline/generate_motion.py \
    --task manip_reach_obj --condition gt \
    --cube-world-pos 0.6 0.0 0.4 \
    --output-dir $VLMGMT/outputs/manip_reach_obj/gt \
    --protomotions-root $PROTOMOTIONS

# VLM (scp ego_frame.png to cluster first)
python $VLMGMT/pipeline/generate_motion.py \
    --task manip_reach_obj --condition vlm \
    --image $VLMGMT/outputs/manip_reach_obj/ego_frame.png \
    --vlm-name qwen2.5-vl-32b \
    --output-dir $VLMGMT/outputs/manip_reach_obj/vlm \
    --protomotions-root $PROTOMOTIONS
```

### Kinematic playback (local, from $PROTOMOTIONS)

```bash
cd $PROTOMOTIONS
python examples/env_kinematic_playback.py \
    --experiment-path examples/experiments/mimic/mlp.py \
    --motion-file $VLMGMT/outputs/manip_reach_obj/baseline/motion.pt \
    --robot-name g1 --simulator isaaclab --num-envs 1 \
    --scenes-file $VLMGMT/outputs/manip_reach_obj_scene.pt
```

### Eval (local, from $PROTOMOTIONS)

```bash
cd $PROTOMOTIONS
python $VLMGMT/eval/run_eval.py \
    --checkpoint $CKPT \
    --motion-file $VLMGMT/outputs/manip_reach_obj/baseline/motion.pt \
    --scenes-file $VLMGMT/outputs/manip_reach_obj_scene.pt \
    --task manip_reach_obj --condition baseline \
    --num-episodes 10 --simulator isaaclab \
    --output-dir $VLMGMT/outputs/manip_reach_obj/results \
    --protomotions-root $PROTOMOTIONS
```

Replace `baseline` with `gt` or `vlm` for other conditions.

---

## Task: walk_to_obj

Robot walks toward a green box offset laterally so baseline (straight walk) misses it.
**Success:** `dist_2d(pelvis, box) < 0.5m` at episode end.

### Create scene (local)

```bash
python $VLMGMT/tasks/walk_to_obj/create_scene.py \
    --box-pos 3.0 -1.1 0.25 \
    --output $VLMGMT/outputs/walk_to_obj_scene.pt
```

### Capture ego image (local, from $PROTOMOTIONS)

```bash
cd $PROTOMOTIONS
python $VLMGMT/pipeline/capture_egocentric.py \
    --experiment-path examples/experiments/mimic/mlp.py \
    --motion-file $VLMGMT/outputs/walk_to_obj/baseline/motion.pt \
    --robot-name g1 --simulator isaaclab --num-envs 1 \
    --scenes-file $VLMGMT/outputs/walk_to_obj_scene.pt \
    --output-dir $VLMGMT/outputs/walk_to_obj \
    --pitch-deg 30 --robot-yaw-deg 0 --prefix ego_frame
```

### Generate motion (cluster)

```bash
# Baseline
python $VLMGMT/pipeline/generate_motion.py \
    --task walk_to_obj --condition baseline \
    --output-dir $VLMGMT/outputs/walk_to_obj/baseline \
    --protomotions-root $PROTOMOTIONS

# GT
python $VLMGMT/pipeline/generate_motion.py \
    --task walk_to_obj --condition gt \
    --box-world-pos 3.0 -1.1 0.25 \
    --output-dir $VLMGMT/outputs/walk_to_obj/gt \
    --protomotions-root $PROTOMOTIONS

# VLM (scp ego_frame.png to cluster first)
python $VLMGMT/pipeline/generate_motion.py \
    --task walk_to_obj --condition vlm \
    --image $VLMGMT/outputs/walk_to_obj/ego_frame.png \
    --vlm-name qwen2.5-vl-32b \
    --output-dir $VLMGMT/outputs/walk_to_obj/vlm \
    --protomotions-root $PROTOMOTIONS
```

### Kinematic playback (local, from $PROTOMOTIONS)

```bash
cd $PROTOMOTIONS
python examples/env_kinematic_playback.py \
    --experiment-path examples/experiments/mimic/mlp.py \
    --motion-file $VLMGMT/outputs/walk_to_obj/baseline/motion.pt \
    --robot-name g1 --simulator isaaclab --num-envs 1 \
    --scenes-file $VLMGMT/outputs/walk_to_obj_scene.pt
```

### Eval (local, from $PROTOMOTIONS)

```bash
cd $PROTOMOTIONS
python $VLMGMT/eval/run_eval.py \
    --checkpoint $CKPT \
    --motion-file $VLMGMT/outputs/walk_to_obj/baseline/motion.pt \
    --scenes-file $VLMGMT/outputs/walk_to_obj_scene.pt \
    --task walk_to_obj --condition baseline \
    --num-episodes 10 --simulator isaaclab \
    --output-dir $VLMGMT/outputs/walk_to_obj/results \
    --protomotions-root $PROTOMOTIONS
```

Replace `baseline` with `gt` or `vlm` for other conditions.

---

## Task: walk_on_green_line_avoid_obs

Robot walks along a 1m-wide green line and navigates around 3 colored obstacles on it.
Baseline (straight walk) collides with all obstacles.
**Success:** pelvis stays within line Y bounds (±0.5m) at all times AND passes all 3 obstacle X positions during the episode.

### Create scene (local)

```bash
python $VLMGMT/tasks/walk_on_green_line_avoid_obs/create_scene.py \
    --output $VLMGMT/outputs/walk_on_green_line_avoid_obs_scene.pt
```

### Capture ego image (local, from $PROTOMOTIONS)

```bash
cd $PROTOMOTIONS
python $VLMGMT/pipeline/capture_egocentric.py \
    --experiment-path examples/experiments/mimic/mlp.py \
    --motion-file $VLMGMT/outputs/walk_on_green_line_avoid_obs/baseline/motion.pt \
    --robot-name g1 --simulator isaaclab --num-envs 1 \
    --scenes-file $VLMGMT/outputs/walk_on_green_line_avoid_obs_scene.pt \
    --output-dir $VLMGMT/outputs/walk_on_green_line_avoid_obs \
    --pitch-deg 30 --robot-yaw-deg 0 --prefix ego_frame
```

### Generate motion (cluster)

```bash
# Baseline
python $VLMGMT/pipeline/generate_motion.py \
    --task walk_on_green_line_avoid_obs --condition baseline \
    --output-dir $VLMGMT/outputs/walk_on_green_line_avoid_obs/baseline \
    --protomotions-root $PROTOMOTIONS

# GT (uses default obstacle positions matching create_scene.py defaults)
python $VLMGMT/pipeline/generate_motion.py \
    --task walk_on_green_line_avoid_obs --condition gt \
    --output-dir $VLMGMT/outputs/walk_on_green_line_avoid_obs/gt \
    --protomotions-root $PROTOMOTIONS

# VLM (scp ego_frame.png to cluster first)
python $VLMGMT/pipeline/generate_motion.py \
    --task walk_on_green_line_avoid_obs --condition vlm \
    --image $VLMGMT/outputs/walk_on_green_line_avoid_obs/ego_frame.png \
    --vlm-name qwen2.5-vl-32b \
    --output-dir $VLMGMT/outputs/walk_on_green_line_avoid_obs/vlm \
    --protomotions-root $PROTOMOTIONS
```

If you change obstacle positions in `create_scene.py`, pass them explicitly to GT:
```bash
python $VLMGMT/pipeline/generate_motion.py \
    --task walk_on_green_line_avoid_obs --condition gt \
    --obs1-world-pos 1.5 0.20 0.30 \
    --obs2-world-pos 3.0 -0.20 0.35 \
    --obs3-world-pos 4.5 0.15 0.25 \
    --output-dir $VLMGMT/outputs/walk_on_green_line_avoid_obs/gt \
    --protomotions-root $PROTOMOTIONS
```

### Kinematic playback (local, from $PROTOMOTIONS)

```bash
cd $PROTOMOTIONS
python examples/env_kinematic_playback.py \
    --experiment-path examples/experiments/mimic/mlp.py \
    --motion-file $VLMGMT/outputs/walk_on_green_line_avoid_obs/baseline/motion.pt \
    --robot-name g1 --simulator isaaclab --num-envs 1 \
    --scenes-file $VLMGMT/outputs/walk_on_green_line_avoid_obs_scene.pt
```

### Eval (local, from $PROTOMOTIONS)

```bash
cd $PROTOMOTIONS
python $VLMGMT/eval/run_eval.py \
    --checkpoint $CKPT \
    --motion-file $VLMGMT/outputs/walk_on_green_line_avoid_obs/baseline/motion.pt \
    --scenes-file $VLMGMT/outputs/walk_on_green_line_avoid_obs_scene.pt \
    --task walk_on_green_line_avoid_obs --condition baseline \
    --num-episodes 10 --simulator isaaclab \
    --output-dir $VLMGMT/outputs/walk_on_green_line_avoid_obs/results \
    --protomotions-root $PROTOMOTIONS
```

Replace `baseline` with `gt` or `vlm` for other conditions.

---

## Adding a New Task

1. Create `tasks/<task_name>/create_scene.py`
2. Create `tasks/<task_name>/metrics.py` with `get_metrics() → list[Metric]`
3. Create `tasks/<task_name>/kimodo_prompt.txt` and `vlm_prompt.txt`
4. Add GT constraint logic to `pipeline/generate_constraints.py`
5. Add GT arg handling to `pipeline/generate_motion.py`

## Adding a New VLM

1. Subclass `pipeline/vlm/base.py:VLMBase`
2. Implement `load()` and `query_constraints(image_rgb, task_description) → list[dict]`
   - `image_rgb` may be `None` for text-only tasks
3. Register in `pipeline/vlm/__init__.py:REGISTRY` and `HF_MODEL_IDS`
