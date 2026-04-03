# VLM-GMT

Evaluating VLM-augmented kinematic constraints for General Motion Tracking in humanoid loco-manipulation.

## Overview

Can off-the-shelf Vision-Language Models generate useful kinematic constraints from egocentric observations? This project tests that by combining [Kimodo](https://research.nvidia.com/labs/sil/projects/kimodo/) (text+constraint → motion) with a pretrained GMT policy on the Unitree G1 via [ProtoMotions](https://github.com/NVlabs/ProtoMotions).

The VLM sees an egocentric image, receives the robot's coordinate system and dimensions in its prompt, and directly outputs 3D constraint positions — no camera calibration or depth estimation needed.

### Three Conditions (per task)

| Condition | Input to Kimodo | Purpose |
|---|---|---|
| **Baseline** | Text only | Lower bound |
| **VLM** | Text + VLM-predicted kinematic constraints | Main experiment |
| **GT** | Text + ground-truth kinematic constraints | Upper bound |

## Setup

### Local (simulation + inference + eval)

Requires IsaacLab + ProtoMotions. Follow the [ProtoMotions installation guide](https://nvlabs.github.io/ProtoMotions/getting_started/installation.html).

```bash
git clone https://github.com/AlexandreBrown/VLM-GMT.git
export PROTOMOTIONS_ROOT=/path/to/ProtoMotions
```

Commands that need IsaacLab (capture, playback, inference, eval) must be run from `$PROTOMOTIONS_ROOT` because ProtoMotions uses relative asset paths.

### Motion generation (Kimodo)

Kimodo requires **>=17 GB VRAM**. If running the VLM condition on the same GPU, the VLM adds ~14 GB (total ~31 GB; an A100 40 GB+ is recommended).

IsaacLab is **not** needed for motion generation. Only Kimodo + ProtoMotions conversion scripts.

```bash
# Install Kimodo (follow Kimodo docs)
# Clone ProtoMotions (no LFS needed)
git clone https://github.com/NVlabs/ProtoMotions.git
uv pip install -e ProtoMotions
uv pip install dm_control easydict

# Clone this repo
git clone https://github.com/AlexandreBrown/VLM-GMT.git
```

### VLM deps (VLM condition only)

```bash
uv pip install transformers>=4.50.0 accelerate qwen-vl-utils Pillow
```

## Repository Structure

```
VLM-GMT/
├── prompts/
│   └── system.txt                  # Shared VLM system prompt (coord system, robot dims)
├── tasks/
│   └── manip_reach_obj/
│       ├── create_scene.py         # Create scene .pt
│       ├── vlm_prompt.txt          # Task-specific VLM prompt
│       └── metrics.py              # Eval metrics
├── pipeline/
│   ├── generate_motion.py          # Constraints → Kimodo → motion.pt
│   ├── generate_constraints.py     # GT / VLM → Kimodo constraints
│   ├── capture_egocentric.py       # Capture head camera image (requires IsaacLab)
│   ├── egocentric_camera.py        # Camera sensor utilities
│   └── vlm/
│       ├── __init__.py             # Registry + load_vlm()
│       ├── base.py                 # Abstract VLMBase
│       └── qwen.py                 # Qwen2.5-VL-7B-Instruct
├── eval/
│   ├── run_eval.py                 # Run GMT + compute metrics
│   ├── base_metric.py              # Metric interface
│   └── metrics/
│       └── distance_to_target.py   # Distance-based metric
├── outputs/                        # Generated data (gitignored)
└── README.md
```

## Pipeline (manip_reach_obj example)

### 1. Create scene

```bash
cd VLM-GMT
python tasks/manip_reach_obj/create_scene.py \
    --cube-pos 0.6 0.0 0.4 \
    --output outputs/manip_reach_obj_scene.pt
```

### 2. Capture egocentric image (requires IsaacLab)

```bash
cd $PROTOMOTIONS_ROOT
python ~/path/to/VLM-GMT/pipeline/capture_egocentric.py \
    --experiment-path examples/experiments/mimic/mlp.py \
    --motion-file data/motion_for_trackers/g1_bones_seed_mini.pt \
    --robot-name g1 --simulator isaaclab --num-envs 1 \
    --scenes-file ~/path/to/VLM-GMT/outputs/manip_reach_obj_scene.pt \
    --output-dir ~/path/to/VLM-GMT/outputs/manip_reach_obj \
    --pitch-deg 50
```

### 3. Generate motion (requires Kimodo, >=17 GB VRAM)

```bash
cd VLM-GMT

# Baseline (text only, no constraints)
python pipeline/generate_motion.py \
    --condition baseline \
    --output-dir outputs/manip_reach_obj/baseline \
    --protomotions-root /path/to/ProtoMotions

# GT (ground-truth constraint at cube position)
python pipeline/generate_motion.py \
    --condition gt \
    --cube-world-pos 0.6 0.0 0.4 \
    --output-dir outputs/manip_reach_obj/gt \
    --protomotions-root /path/to/ProtoMotions

# VLM (constraint predicted from egocentric image, >=31 GB VRAM with Qwen2.5-VL-7B)
python pipeline/generate_motion.py \
    --condition vlm \
    --image outputs/manip_reach_obj/ego.png \
    --task manip_reach_obj \
    --output-dir outputs/manip_reach_obj/vlm \
    --protomotions-root /path/to/ProtoMotions
```

### 4. Kinematic playback (requires IsaacLab)

```bash
cd $PROTOMOTIONS_ROOT
python examples/env_kinematic_playback.py \
    --experiment-path examples/experiments/mimic/mlp.py \
    --motion-file ~/path/to/VLM-GMT/outputs/manip_reach_obj/gt/motion.pt \
    --robot-name g1 --simulator isaaclab --num-envs 1 \
    --scenes-file ~/path/to/VLM-GMT/outputs/manip_reach_obj_scene.pt
```

### 5. GMT inference (requires IsaacLab)

```bash
cd $PROTOMOTIONS_ROOT
python protomotions/inference_agent.py \
    --checkpoint data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt \
    --motion-file ~/path/to/VLM-GMT/outputs/manip_reach_obj/gt/motion.pt \
    --simulator isaaclab --num-envs 1 \
    --scenes-file ~/path/to/VLM-GMT/outputs/manip_reach_obj_scene.pt
```

### 6. Eval (requires IsaacLab)

```bash
cd $PROTOMOTIONS_ROOT
python ~/path/to/VLM-GMT/eval/run_eval.py \
    --checkpoint data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt \
    --motion-file ~/path/to/VLM-GMT/outputs/manip_reach_obj/gt/motion.pt \
    --scenes-file ~/path/to/VLM-GMT/outputs/manip_reach_obj_scene.pt \
    --task manip_reach_obj --condition gt \
    --num-episodes 20 --simulator isaaclab \
    --protomotions-root $PROTOMOTIONS_ROOT \
    --output-dir ~/path/to/VLM-GMT/outputs/manip_reach_obj/results
```

## VLM Prompt System

Prompts are split into two files:

- **`prompts/system.txt`** — shared across all tasks. Contains coordinate system, robot dimensions, output format, and an example. Has `{num_frames}` and `{max_frame}` placeholders filled at runtime.
- **`tasks/<task>/vlm_prompt.txt`** — task-specific instruction (e.g., "reach the red cube with your right hand").

The VLM receives the system prompt as the system message and the task prompt + egocentric image as the user message. It outputs a JSON array of kinematic constraints with 3D world-frame positions directly — no camera calibration needed.

Override the task prompt at runtime with `--task-description "custom text"` or `--task-description path/to/prompt.txt`.

## Adding a New Task

1. Create `tasks/<task_name>/create_scene.py` (if the task has scene objects)
2. Create `tasks/<task_name>/vlm_prompt.txt` with the task-specific VLM prompt
3. Create `tasks/<task_name>/metrics.py` with a `get_metrics()` function returning a list of `Metric` instances
4. Add a GT constraint recipe in `generate_constraints.py` (if applicable)

## Adding a New VLM

1. Subclass `pipeline/vlm/base.py:VLMBase`
2. Implement `load()` and `query_constraints(image_rgb, task_description) -> list[dict]`
3. Register in `pipeline/vlm/__init__.py:REGISTRY`
