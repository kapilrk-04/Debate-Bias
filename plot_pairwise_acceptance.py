#!/usr/bin/env python3
"""
Plot pairwise judge acceptance rates as annotated heatmaps.

- Excludes the 'overall' column
- Per-column maximum shown in bold
- One plot each for diff (4 judges) and same (3 judges)

Usage:
  python plot_pairwise_acceptance.py [--diff pairwise_acceptance_diff.csv]
                                     [--same pairwise_acceptance_same.csv]
                                     [--out-dir pairwise_acceptance_plots]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Display names
# ---------------------------------------------------------------------------

JUDGE_DISPLAY = {
    "gemma":   "Gemma-3-12B",
    "gpt":     "GPT-4o-nano",
    "krutrim": "Krutrim-2",
    "human":   "Human",
}

COL_DISPLAY = {
    "cat_bias":                      "Bias\n(category)",
    "cat_stereotype":                "Stereotype\n(category)",
    "task_bias_judgement_negative":  "Bias Judg.\nNeg",
    "task_bias_judgement_positive":  "Bias Judg.\nPos",
    "task_bias_plausible_negative":  "Bias Plaus.\nNeg",
    "task_bias_plausible_positive":  "Bias Plaus.\nPos",
    "task_stereotype_judgement":     "Stereo. Judg.",
    "task_stereotype_plausible":     "Stereo. Plaus.",
}

# Preferred column order (category first, then tasks)
COL_ORDER = [
    "cat_bias",
    "cat_stereotype",
    "task_bias_judgement_negative",
    "task_bias_judgement_positive",
    "task_bias_plausible_negative",
    "task_bias_plausible_positive",
    "task_stereotype_judgement",
    "task_stereotype_plausible",
]

# Thin vertical divider between category cols and task cols
CATEGORY_DIVIDER_AFTER = 1   # divider after the 2nd column (index 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pair_label(row: pd.Series) -> str:
    a = JUDGE_DISPLAY.get(row["judge_a"], row["judge_a"])
    b = JUDGE_DISPLAY.get(row["judge_b"], row["judge_b"])
    return f"{a} × {b}"


def plot_acceptance(df: pd.DataFrame, title: str, out_path: str):
    """
    Annotated heatmap: rows = judge pairs, columns = tasks/categories.
    Bold = column-wise maximum.
    """
    # ── build row/col labels ──────────────────────────────────────────────
    row_labels = [_pair_label(row) for _, row in df.iterrows()]

    # keep only columns present in data and in preferred order
    cols = [c for c in COL_ORDER if c in df.columns]
    col_labels = [COL_DISPLAY.get(c, c) for c in cols]

    mat = df[cols].values.astype(float)
    n_rows, n_cols = mat.shape

    # column-wise max mask
    col_max = np.nanmax(mat, axis=0)
    is_max = np.isclose(mat, col_max[np.newaxis, :])

    # ── figure ────────────────────────────────────────────────────────────
    fig_w = max(10, 1.35 * n_cols)
    fig_h = max(3.5, 0.65 * n_rows)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = plt.get_cmap("YlGn").copy()
    cmap.set_bad("lightgrey")
    im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap,
                   vmin=0.4, vmax=1.0, aspect="auto")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Acceptance Rate", fontsize=10)

    # axes ticks
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=9, ha="center")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9)

    # move x-axis ticks to top
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    # cell annotations
    for i in range(n_rows):
        for j in range(n_cols):
            val = mat[i, j]
            if np.isnan(val):
                continue
            bold = is_max[i, j]
            txt = f"{val:.3f}"
            ax.text(j, i, txt,
                    ha="center", va="center",
                    fontsize=9,
                    fontweight="bold" if bold else "normal",
                    color="black" if val < 0.85 else "white")

    # divider line between category and task columns
    if CATEGORY_DIVIDER_AFTER < n_cols - 1:
        ax.axvline(CATEGORY_DIVIDER_AFTER + 0.5, color="black", linewidth=1.5,
                   linestyle="--", alpha=0.6)

    # top-level group labels
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    n_cat = CATEGORY_DIVIDER_AFTER + 1
    n_task = n_cols - n_cat
    cat_center = (n_cat - 1) / 2
    task_center = n_cat + (n_task - 1) / 2
    ax2.set_xticks([cat_center, task_center])
    ax2.set_xticklabels(["Category", "Task"], fontsize=10, fontweight="bold")
    ax2.tick_params(length=0, pad=28)

    ax.set_title(title, fontsize=12, pad=48)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diff",    default="pairwise_acceptance_diff.csv")
    parser.add_argument("--same",    default="pairwise_acceptance_same.csv")
    parser.add_argument("--out-dir", default="pairwise_acceptance_plots")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for csv_path, label, fname in [
        (args.diff, "Diff Set (Gemma × GPT × Krutrim × Human)", "pairwise_acceptance_diff.png"),
        (args.same, "Same Set (Gemma × GPT × Krutrim)",         "pairwise_acceptance_same.png"),
    ]:
        p = Path(csv_path)
        if not p.exists():
            print(f"  [SKIP] {csv_path} not found")
            continue
        df = pd.read_csv(p)
        # drop overall column
        df = df.drop(columns=["overall"], errors="ignore")
        plot_acceptance(df, title=f"Pairwise Judge Acceptance Rates — {label}\n(bold = column maximum)",
                        out_path=str(out_dir / fname))


if __name__ == "__main__":
    main()
