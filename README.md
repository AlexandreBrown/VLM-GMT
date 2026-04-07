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

**IsaacLab** is required locally (capture, playback, eval). Follow the [IsaacLab installation guide](https://isaac-sim.github.io/IsaacLab/).

**Kimodo** is required on the cluster for motion generation (>=17 GB VRAM). Follow the Kimodo docs.

**VLM deps** (VLM condition only):
```bash
uv pip install transformers>=4.50.0 accelerate qwen-vl-utils Pillow
```

### Environment variables

```bash
# Local
export VLMGMT=~/Documents/vlm_project/VLM-GMT
export PROTOMOTIONS=~/Documents/vlm_project/ProtoMotions
export CKPT=$PROTOMOTIONS/data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt

# Cluster
export VLMGMT=~/kimodo_test/VLM-GMT
export PROTOMOTIONS=~/kimodo_test/ProtoMotions
export HF_HOME=$SCRATCH/huggingface_cache
```

Commands that need IsaacLab must be run from `$PROTOMOTIONS` (relative asset paths).

## Repository Structure

```
VLM-GMT/
├── prompts/
│   └── system.txt                       # Shared VLM system prompt
├── tasks/
│   ├── manip_reach_obj/                 # Red cube on a table
│   ├── walk_to_obj/                     # Green box on the ground
│   ├── navigate_maze/                   # 2 staggering walls (maze-like)
│   ├── point_at_obj_with_right_hand/    # Blue object on pedestal (left), point with right hand
│   ├── point_at_obj_with_left_hand/     # Blue object on pedestal (right), point with left hand
│   └── raise_right_hand/               # Text-only: raise right hand above head
│       Each task contains: create_scene.py, metrics.py, kimodo_prompt.txt, vlm_prompt.txt
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
│       ├── distance_to_target.py        # Supports 2D, 3D, fixed target pos
│       └── navigate_maze.py              # Trajectory-based line compliance metric
├── scripts/                             # Per-task command scripts
│   ├── manip_reach_obj.sh
│   ├── walk_to_obj.sh
│   ├── navigate_maze.sh
│   ├── point_at_obj_with_right_hand.sh
│   ├── point_at_obj_with_left_hand.sh
│   └── raise_right_hand.sh
└── outputs/                             # Generated data (gitignored)
```

## Tasks

Each task has a script in `scripts/` with all commands (create scene, capture, generate motion, playback, eval). Edit the variables at the top of each script to match your paths.

| Task | Script | Success metric |
|---|---|---|
| **manip_reach_obj** | `scripts/manip_reach_obj.sh` | `dist(right_wrist, cube) < 0.15m` at episode end |
| **walk_to_obj** | `scripts/walk_to_obj.sh` | `dist_2d(pelvis, box) < 0.5m` at episode end |
| **navigate_maze** | `scripts/navigate_maze.sh` | Avoids both walls laterally AND final x past wall 2 + 0.5m |
| **point_at_obj_with_right_hand** | `scripts/point_at_obj_with_right_hand.sh` | `dist(right_hand, obj) < 0.15m` at episode end |
| **point_at_obj_with_left_hand** | `scripts/point_at_obj_with_left_hand.sh` | `dist(left_hand, obj) < 0.15m` at episode end |
| **raise_right_hand** | `scripts/raise_right_hand.sh` | `right_hand_z > 1.3m` at episode end (text-only) |

### Pipeline order per task

1. **Create scene** (local)
2. **Generate baseline motion** (cluster)
3. **Capture ego image** (local, from `$PROTOMOTIONS`) — requires baseline motion.pt
4. **Generate GT and VLM motions** (cluster) — VLM requires ego.png transferred from local
5. **Kinematic playback** (local) — optional sanity check
6. **Eval** (local, from `$PROTOMOTIONS`)

## Adding a New Task

1. Create `tasks/<task_name>/create_scene.py`
2. Create `tasks/<task_name>/metrics.py` with `get_metrics() → list[Metric]`
3. Create `tasks/<task_name>/kimodo_prompt.txt` and `vlm_prompt.txt`
4. Add GT constraint logic to `pipeline/generate_constraints.py`
5. Add GT arg handling to `pipeline/generate_motion.py`
6. Create `scripts/<task_name>.sh`

## Adding a New VLM

1. Subclass `pipeline/vlm/base.py:VLMBase`
2. Implement `load()` and `query_constraints(image_rgb, task_description) → list[dict]`
   - `image_rgb` may be `None` for text-only tasks (VLM reasons from body knowledge)
3. Register in `pipeline/vlm/__init__.py:REGISTRY` and `HF_MODEL_IDS`
