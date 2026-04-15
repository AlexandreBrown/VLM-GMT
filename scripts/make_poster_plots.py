"""Poster plots: IQM-aggregated success by task category.

For each task we compute a per-episode score in [0, 1]:
  - distance-based tasks: 1.0 if min_dist < threshold else 0.0 (best-in-episode success)
  - kneel_down / navigate_maze: per-episode partial score (already in [0,1])
  - raise_*_hand: binary success from the height metric

We then concatenate all episode scores within a category (vision / text),
per method, and take the interquartile mean (IQM, middle 50%).

Output: figs/iqm_by_category.png and figs/iqm_by_category_data.json.

Usage:
    python scripts/make_poster_plots.py --vlm-gmt-root /path/to/VLM-GMT
"""

import argparse
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

# (task_dir, category, metric_name, kind)
#   kind: "dist_min" -> episode success = min_dist < threshold
#         "partial"  -> use episode value directly (already in [0,1])
#         "binary"   -> use episode success flag
TASKS = [
    ("manip_reach_obj",                "vision", "dist_right_hand_to_cube",      "dist_min"),
    ("walk_to_obj",                    "vision", "dist_pelvis_to_box_2d",        "dist_min"),
    ("navigate_maze",                  "vision", "navigate_maze",                "partial"),
    ("point_at_obj_with_right_hand",   "vision", "dist_right_hand_to_obj",       "dist_min"),
    ("point_at_obj_with_left_hand",    "vision", "dist_left_hand_to_obj",        "dist_min"),
    ("raise_right_hand",               "text",   "right_hand_height",            "binary"),
    ("raise_left_hand",                "text",   "left_hand_height",             "binary"),
    ("kneel_down_1_knee",              "text",   "kneel_down",                   "partial"),
    ("touch_left_leg_with_right_hand", "text",   "dist_right_hand_to_left_knee", "dist_min"),
    ("touch_right_leg_with_left_hand", "text",   "dist_left_hand_to_right_knee", "dist_min"),
]


def load_episodes(root: Path, task_dir: str, cond: str, metric_name: str):
    p = root / "outputs" / task_dir / "results" / f"{cond}_results.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing results file: {p}")
    d = json.loads(p.read_text())
    eps = d["episodes"].get(metric_name)
    if eps is None:
        raise KeyError(f"Metric {metric_name} missing in {p}")
    return eps


def episode_scores(episodes, kind):
    """Return a list of per-episode scores in [0, 1]."""
    out = []
    for e in episodes:
        if kind == "dist_min":
            thr = e["threshold"]
            out.append(1.0 if e["min_dist"] < thr else 0.0)
        elif kind == "partial":
            out.append(float(e["value"]))
        elif kind == "binary":
            out.append(1.0 if e["success"] else 0.0)
        else:
            raise ValueError(kind)
    return out


def iqm(values):
    """Interquartile mean: mean of values between 25th and 75th percentile."""
    if not values:
        return float("nan")
    a = np.sort(np.asarray(values, dtype=float))
    n = len(a)
    lo = int(np.floor(n * 0.25))
    hi = int(np.ceil(n * 0.75))
    if hi <= lo:
        return float(a.mean())
    return float(a[lo:hi].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm-gmt-root", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    root = args.vlm_gmt_root
    out_dir = args.out_dir or (root / "figs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate episode scores by (category, condition)
    bucket = {
        ("vision", c): [] for c in CONDITIONS
    }
    bucket.update({
        ("text", c): [] for c in CONDITIONS
    })

    for task_dir, cat, metric, kind in TASKS:
        for c in CONDITIONS:
            eps = load_episodes(root, task_dir, c, metric)
            scores = episode_scores(eps, kind)
            bucket[(cat, c)].extend(scores)

    # Compute IQM per (category, condition)
    iqm_table = {
        (cat, c): iqm(bucket[(cat, c)]) for (cat, c) in bucket
    }

    # Print summary
    print("IQM success score per category (middle 50% of per-episode scores):")
    for cat in ("vision", "text"):
        print(f"  {cat}:")
        for c in CONDITIONS:
            print(f"    {COND_LABELS[c]:<20} {iqm_table[(cat, c)]:.3f}  (n_episodes={len(bucket[(cat, c)])})")

    # Save raw numbers
    (out_dir / "iqm_by_category_data.json").write_text(json.dumps(
        {f"{cat}/{c}": iqm_table[(cat, c)] for (cat, c) in iqm_table},
        indent=2,
    ))

    # Grouped bar plot: x ticks = {vision, text}, 4 bars per tick
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ["vision", "text"]
    xticks = np.arange(len(categories))
    n_conds = len(CONDITIONS)
    width = 0.8 / n_conds
    for i, c in enumerate(CONDITIONS):
        vals = [iqm_table[(cat, c)] for cat in categories]
        offsets = xticks - 0.4 + width / 2 + i * width
        bars = ax.bar(
            offsets, vals, width,
            label=COND_LABELS[c], color=COND_COLORS[c],
            alpha=0.55 if c == "gt" else 1.0,
            edgecolor="black", linewidth=0.4,
        )
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9,
            )
    ax.set_xticks(xticks)
    ax.set_xticklabels(["Vision-dependent tasks", "Text-only tasks"], fontsize=11)
    ax.set_ylabel("IQM success score  ↑", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("IQM success across tasks, by category and method", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "iqm_by_category.png", dpi=200)
    plt.close(fig)

    print(f"\nWrote {out_dir/'iqm_by_category.png'}")


if __name__ == "__main__":
    main()
