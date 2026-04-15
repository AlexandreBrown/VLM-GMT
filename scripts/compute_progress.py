"""Compute per-episode progress ∈ [0, 1] for every task and plot results.

Progress is computed from each task's existing eval json (outputs/<task>/results/<cond>_results.json)
plus the per-task initial conditions in outputs/initial_conditions.json (produced by
scripts/capture_initial_conditions.sh).

Progress definition (one score per episode per task):

  Distance-based tasks (reach_obj, walk_to_obj, point_*, touch_*):
      initial = task_initial_dist (from initial_conditions.json)
      value   = final_dist of the episode
      progress = clip((initial - value) / (initial - threshold), 0, 1)

  Raise hand tasks (raise_left_hand, raise_right_hand):
      initial = task_initial_z
      value   = final_z of the episode
      target  = threshold (1.3 m)
      progress = clip((value - initial) / (target - initial), 0, 1)

  Kneel down:
      progress = episode "value" (already the partial score in [0, 1])

  Navigate maze:
      cleared = walls_cleared
      total   = total_walls
      if cleared == total:
          progress = 1
      else:
          prev_milestone = origin_x if cleared == 0 else wall_world_xs[cleared-1]
          next_wall      = wall_world_xs[cleared]
          frac = clip((final_x - prev_milestone) / (next_wall - prev_milestone), 0, 1)
          progress = (cleared + frac) / total
      origin_x is read from initial_conditions.json (falls back to final_px_world - final_px_local).

Outputs (all under <out-dir>, default figs/):
  progress_by_category.png     Aggregate bar plot, vision vs text
  progress_vision.png          Per-task bar plot, vision tasks
  progress_text.png            Per-task bar plot, text-only tasks
  progress_data.json           Raw per-task and aggregate values
  progress_per_task_means.csv  One row per (task, method), columns: task,method,mean_progress
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = ["baseline", "vlm_7b", "vlm_32b", "gt"]
COND_LABELS = {
    "baseline": "Baseline",
    "vlm_7b":   "Qwen 2.5 VL 7B",
    "vlm_32b":  "Qwen 2.5 VL 32B",
    "gt":       "GT (oracle)",
}
COND_COLORS = {
    "baseline": "#7f7f7f",
    "vlm_7b":   "#ff7f0e",
    "vlm_32b":  "#1f77b4",
    "gt":       "#2ca02c",
}

TASKS = [
    # (task_dir, short_name, display_name, category, metric_name, kind)
    ("manip_reach_obj",                "reach_obj",        "Reach Object",              "vision", "dist_right_hand_to_cube",      "dist"),
    ("walk_to_obj",                    "walk_to_obj",      "Walk to Object",            "vision", "dist_pelvis_to_box_2d",        "dist"),
    ("navigate_maze",                  "navigate_maze",    "Navigate Maze",             "vision", "navigate_maze",                "navigate_maze"),
    ("point_at_obj_with_right_hand",   "point_right",      "Point at Object (Right)",   "vision", "dist_right_hand_to_obj",       "dist"),
    ("point_at_obj_with_left_hand",    "point_left",       "Point at Object (Left)",    "vision", "dist_left_hand_to_obj",        "dist"),
    ("raise_right_hand",               "raise_right_hand", "Raise Right Hand",          "text",   "right_hand_height",            "raise_hand"),
    ("raise_left_hand",                "raise_left_hand",  "Raise Left Hand",           "text",   "left_hand_height",             "raise_hand"),
    ("kneel_down_1_knee",              "kneel_down",       "Kneel Down",                "text",   "kneel_down",                   "kneel"),
    ("touch_left_leg_with_right_hand", "touch_left_knee",  "Touch Left Knee (R.Hand)",  "text",   "dist_right_hand_to_left_knee", "dist"),
    ("touch_right_leg_with_left_hand", "touch_right_knee", "Touch Right Knee (L.Hand)", "text",   "dist_left_hand_to_right_knee", "dist"),
]


def clip01(x):
    return max(0.0, min(1.0, x))


def progress_dist(initial, final, threshold):
    if initial <= threshold:
        return 1.0 if final <= threshold else 0.0
    if final >= initial:
        return 0.0
    return clip01((initial - final) / (initial - threshold))


def progress_raise(initial_z, final_z, target_z):
    if initial_z >= target_z:
        return 1.0 if final_z >= target_z else 0.0
    if final_z <= initial_z:
        return 0.0
    if final_z >= target_z:
        return 1.0
    return clip01((final_z - initial_z) / (target_z - initial_z))


def progress_navigate(ep, origin_x, wall_xs):
    cleared = ep["walls_cleared"]
    total = ep["total_walls"]
    if cleared >= total:
        return 1.0
    final_x = ep["final_px_world"]
    next_wall = wall_xs[cleared]
    prev_milestone = origin_x if cleared == 0 else wall_xs[cleared - 1]
    denom = next_wall - prev_milestone
    frac = 0.0 if denom <= 0 else clip01((final_x - prev_milestone) / denom)
    return (cleared + frac) / total


def episode_progress(ep, kind, task_init, metric_name):
    if kind == "dist":
        initial = task_init[metric_name]["initial_dist"]
        return progress_dist(initial, ep["final_dist"], ep["threshold"])
    if kind == "raise_hand":
        initial_z = task_init[metric_name]["initial_z"]
        target_z = ep["threshold"]
        return progress_raise(initial_z, ep["final_z"], target_z)
    if kind == "kneel":
        return float(ep["value"])
    if kind == "navigate_maze":
        init = task_init[metric_name]
        origin_x = init.get("origin_x_world")
        wall_xs = init["wall_world_xs"]
        if origin_x is None:
            origin_x = ep["final_px_world"] - ep["final_px_local"]
        return progress_navigate(ep, origin_x, wall_xs)
    raise ValueError(kind)


def load_episodes(root, task_dir, cond, metric_name):
    p = root / "outputs" / task_dir / "results" / f"{cond}_results.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")
    d = json.loads(p.read_text())
    eps = d["episodes"].get(metric_name)
    if eps is None:
        raise KeyError(f"Metric {metric_name} missing in {p}")
    return eps


def _grouped_bar(ax, tasks, cat_filter, task_means):
    filtered = [t for t in tasks if t[3] == cat_filter]
    shorts = [t[1] for t in filtered]
    displays = [t[2] for t in filtered]
    xticks = np.arange(len(filtered))
    n = len(CONDITIONS)
    w = 0.8 / n
    for i, c in enumerate(CONDITIONS):
        vals = [task_means[s][c] for s in shorts]
        off = xticks - 0.4 + w / 2 + i * w
        bars = ax.bar(off, vals, w,
                      label=COND_LABELS[c], color=COND_COLORS[c],
                      alpha=0.55 if c == "gt" else 1.0,
                      edgecolor="black", linewidth=0.4)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xticks)
    ax.set_xticklabels(displays, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Mean progress score  ↑", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm-gmt-root", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--initial-conditions", type=Path, default=None,
                    help="Path to initial_conditions.json (default: outputs/initial_conditions.json).")
    args = ap.parse_args()

    root = args.vlm_gmt_root
    out_dir = args.out_dir or (root / "figs")
    out_dir.mkdir(parents=True, exist_ok=True)

    init_path = args.initial_conditions or (root / "outputs" / "initial_conditions.json")
    if not init_path.exists():
        raise FileNotFoundError(
            f"Missing {init_path}. Run scripts/capture_initial_conditions.sh first."
        )
    initial = json.loads(init_path.read_text())

    task_means = {}               # {short_name: {cond: mean_progress}}
    cat_pool = {(cat, c): [] for cat in ("vision", "text") for c in CONDITIONS}

    for task_dir, short, _display, cat, metric, kind in TASKS:
        if task_dir not in initial:
            raise KeyError(
                f"Task '{task_dir}' missing from {init_path}. "
                f"Re-run scripts/capture_initial_conditions.sh."
            )
        task_init = initial[task_dir]
        task_means[short] = {}
        for c in CONDITIONS:
            eps = load_episodes(root, task_dir, c, metric)
            scores = [episode_progress(e, kind, task_init, metric) for e in eps]
            task_means[short][c] = float(np.mean(scores))
            cat_pool[(cat, c)].extend(scores)

    cat_means = {(cat, c): float(np.mean(cat_pool[(cat, c)])) for (cat, c) in cat_pool}

    # Print summary
    print("Mean progress per task:")
    for short in task_means:
        s = task_means[short]
        print(f"  {short:<20}  baseline={s['baseline']:.3f}  7B={s['vlm_7b']:.3f}  "
              f"32B={s['vlm_32b']:.3f}  GT={s['gt']:.3f}")
    print("\nMean progress per category:")
    for cat in ("vision", "text"):
        print(f"  {cat}:")
        for c in CONDITIONS:
            print(f"    {COND_LABELS[c]:<20}  {cat_means[(cat, c)]:.3f}")

    # Save raw numbers
    (out_dir / "progress_data.json").write_text(json.dumps({
        "per_task": task_means,
        "per_category": {f"{cat}/{c}": v for (cat, c), v in cat_means.items()},
        "initial_conditions_path": str(init_path),
    }, indent=2))

    # CSV per-task means
    with open(out_dir / "progress_per_task_means.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "category", "baseline", "vlm_7b", "vlm_32b", "gt"])
        for task_dir, short, _display, cat, _, _ in TASKS:
            m = task_means[short]
            w.writerow([short, cat,
                        f"{m['baseline']:.4f}",
                        f"{m['vlm_7b']:.4f}",
                        f"{m['vlm_32b']:.4f}",
                        f"{m['gt']:.4f}"])

    # ---- Plot 1: category aggregate ----
    fig, ax = plt.subplots(figsize=(8, 5))
    xticks = np.arange(2)
    n = len(CONDITIONS)
    w = 0.8 / n
    for i, c in enumerate(CONDITIONS):
        vals = [cat_means[("vision", c)], cat_means[("text", c)]]
        off = xticks - 0.4 + w / 2 + i * w
        bars = ax.bar(off, vals, w,
                      label=COND_LABELS[c], color=COND_COLORS[c],
                      alpha=0.55 if c == "gt" else 1.0,
                      edgecolor="black", linewidth=0.4)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xticks)
    ax.set_xticklabels(["Vision-dependent tasks", "Text-only tasks"], fontsize=11)
    ax.set_ylabel("Mean progress score  ↑", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("Mean progress across tasks, by category", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "progress_by_category.png", dpi=200)
    plt.close(fig)

    # ---- Plot 2: per-task (vision) ----
    fig, ax = plt.subplots(figsize=(9, 5))
    _grouped_bar(ax, TASKS, "vision", task_means)
    ax.set_title("Mean progress per vision-dependent task", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "progress_vision.png", dpi=200)
    plt.close(fig)

    # ---- Plot 3: per-task (text) ----
    fig, ax = plt.subplots(figsize=(9, 5))
    _grouped_bar(ax, TASKS, "text", task_means)
    ax.set_title("Mean progress per text-only task", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "progress_text.png", dpi=200)
    plt.close(fig)

    print(f"\nWrote plots + csv to {out_dir}")


if __name__ == "__main__":
    main()
