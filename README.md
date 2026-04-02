# VLM-GMT

Evaluating and learning VLM-augmented kinematic constraints for General Motion Tracking in humanoid loco-manipulation.

## Overview

This project investigates whether off-the-shelf Vision-Language Models (VLMs) can generate kinematic constraints for [Kimodo](https://research.nvidia.com/labs/sil/projects/kimodo/) to enable vision-aware motion generation, tracked by a pretrained General Motion Tracking (GMT) policy on the Unitree G1 robot via [ProtoMotions](https://github.com/NVlabs/ProtoMotions).

### Hypothesis

Motion tracking policies are inherently limited to following a reference trajectory. Even with a VLM on top, this limits their ability to handle multi-modal commands. This project quantifies that limitation and investigates whether it justifies a new learning-based approach.

### Three Conditions (per task)

| Condition | Input to Kimodo | Purpose |
|---|---|---|
| **Baseline** | Text only | Lower bound |
| **VLM** | Text + VLM-estimated kinematic constraint | Main experiment |
| **GT** | Text + ground-truth kinematic constraint | Upper bound |

## Setup

### Local (simulation + GMT inference)

Requires IsaacLab + ProtoMotions. Follow the [ProtoMotions installation guide](https://nvlabs.github.io/ProtoMotions/getting_started/installation.html).

```bash
git clone https://github.com/AlexandreBrown/VLM-GMT.git

# Add ProtoMotions to PYTHONPATH so VLM-GMT scripts can import it
export PYTHONPATH=$PYTHONPATH:/path/to/ProtoMotions
```

All VLM-GMT commands below assume you run from the `VLM-GMT/` directory with `PYTHONPATH` set.

### Cluster (motion generation with Kimodo)

IsaacLab is **not** needed on the cluster. Only Kimodo + ProtoMotions conversion scripts are required.

```bash
# 1. Install Kimodo (follow Kimodo docs)

# 2. Clone ProtoMotions (no LFS needed, conversion scripts are pure Python)
git clone https://github.com/NVlabs/ProtoMotions.git

# 3. Install ProtoMotions and missing deps into your Kimodo env
uv pip install -e ProtoMotions
uv pip install dm_control easydict

# 4. Clone this repo
git clone https://github.com/AlexandreBrown/VLM-GMT.git
```

### Constraint generation (PyRoki IK, optional — gt/vlm conditions only)

PyRoki requires a separate env due to JAX conflicts:

```bash
conda create -n pyroki python=3.10
conda activate pyroki
git clone https://github.com/chungmin99/pyroki && pip install -e pyroki
```

### VLM deps (vlm condition only)

```bash
pip install transformers>=4.50.0 accelerate qwen-vl-utils Pillow
```

## Repository Structure

```
VLM-GMT/
├── tasks/
│   └── reach_obj/
│       └── create_scene.py       # Create ProtoMotions scene .pt
├── pipeline/
│   ├── generate_motion.py        # End-to-end: constraints → Kimodo → .pt
│   ├── generate_constraints.py   # GT / VLM / none → Kimodo constraint JSON
│   └── vlm/
│       ├── __init__.py           # Registry + load_vlm()
│       ├── base.py               # Abstract VLMBase interface
│       └── qwen.py               # Qwen2.5-VL-7B-Instruct
├── outputs/                      # Generated scenes, motions, logs (gitignored)
├── requirements.txt
└── README.md
```

## Tasks

### reach_obj

Robot reaches a static object with its right hand. Vision-dominant task.

**Success metric:** `dist(right_wrist_pos, cube_pos) < 0.1m` at episode end.

**Create scene** (run from ProtoMotions root):
```bash
python VLM-GMT/tasks/reach_obj/create_scene.py \
    --cube-pos 0.6 0.0 0.4 \
    --output VLM-GMT/outputs/reach_obj_scene.pt
```

**Generate motion** (run from ProtoMotions root):
```bash
# Baseline
python VLM-GMT/pipeline/generate_motion.py \
    --condition baseline \
    --prompt "A person reaches forward with their right hand to grab an object." \
    --output-dir VLM-GMT/outputs/reach_obj/baseline

# GT (upper bound)
python VLM-GMT/pipeline/generate_motion.py \
    --condition gt \
    --cube-world-pos 0.6 0.0 0.4 \
    --output-dir VLM-GMT/outputs/reach_obj/gt

# VLM
python VLM-GMT/pipeline/generate_motion.py \
    --condition vlm \
    --image VLM-GMT/outputs/reach_obj/sim_frame.png \
    --camera-intrinsics 500 500 320 240 \
    --camera-extrinsic-npy VLM-GMT/outputs/reach_obj/camera_T_world.npy \
    --output-dir VLM-GMT/outputs/reach_obj/vlm
```

**Kinematic playback** (verify motion + scene alignment):
```bash
python examples/env_kinematic_playback.py \
    --experiment-path examples/experiments/mimic/mlp.py \
    --motion-file VLM-GMT/outputs/reach_obj/baseline/motion.pt \
    --robot-name g1 --simulator isaaclab --num-envs 1 \
    --scenes-file VLM-GMT/outputs/reach_obj_scene.pt
```

**GMT inference (visualize):**
```bash
# From VLM-GMT/ with PYTHONPATH set
export PYTHONPATH=$PYTHONPATH:/path/to/ProtoMotions

python -m protomotions.inference_agent \
    --checkpoint /path/to/ProtoMotions/data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt \
    --motion-file outputs/reach_obj/baseline/motion.pt \
    --simulator isaaclab --num-envs 1 \
    --scenes-file outputs/reach_obj_scene.pt
```

**Eval (metrics + results JSON):**
```bash
# From VLM-GMT/
python eval/run_eval.py \
    --checkpoint /path/to/ProtoMotions/data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt \
    --motion-file outputs/reach_obj/baseline/motion.pt \
    --scenes-file outputs/reach_obj_scene.pt \
    --task reach_obj \
    --condition baseline \
    --num-episodes 20 \
    --simulator isaaclab \
    --output-dir outputs/reach_obj/results
```

## Adding a New Task

1. Create `tasks/<task_name>/create_scene.py`
2. Use `pipeline/generate_motion.py` with a task-specific prompt
3. Add a task-specific success metric to `eval/` (coming soon)

## Adding a New VLM

1. Subclass `pipeline/vlm/base.py:VLMBase`
2. Implement `load()` and `query_object_pixels(image_rgb, object_description) → (u, v)`
3. Register in `pipeline/vlm/__init__.py:REGISTRY`
