#!/usr/bin/env python3
"""
Plot refusal rates per task as annotated heatmaps.

One heatmap per source (Single-LLM, Self-Reflect, Debate Diff, Debate Same):
  rows    = models
  columns = tasks
  color   = refusal rate (0–1, YlOrRd: yellow=low, red=high)
  bold    = column-wise maximum (highest refusal rate per task)

Also produces a combined 2×2 subplot figure.

Usage:
  python plot_refusal_rates.py [--csv refusal_results/refusal_rates.csv]
                               [--out-dir refusal_results/plots]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Display names & ordering
# ---------------------------------------------------------------------------

TASK_DISPLAY = {
    "bias_judgement_negative":  "Bias Judg.\nNeg",
    "bias_judgement_positive":  "Bias Judg.\nPos",
    "bias_plausible_negative":  "Bias Plaus.\nNeg",
    "bias_plausible_positive":  "Bias Plaus.\nPos",
    "stereotype_judgement":     "Stereo.\nJudg.",
    "stereotype_plausible":     "Stereo.\nPlaus.",
}

TASK_ORDER = [
    "bias_judgement_negative",
    "bias_judgement_positive",
    "bias_plausible_negative",
    "bias_plausible_positive",
    "stereotype_judgement",
    "stereotype_plausible",
]

MODEL_ORDER = [
    "Airavata-7B",
    "Aya-23-8B",
    "BharatGPT-3B",
    "Gemma3-4B-IT",
    "Llama-3.2-3B",
    "Nemo-Hindi-4B",
    "Qwen3-4B",
]

# 12 pairs in multi_agent_diff (model1 vs model2 as they appear in filenames)
PAIR_ORDER = [
    "Aya-23-8B vs BharatGPT-3B",
    "Aya-23-8B vs Airavata-7B",
    "Aya-23-8B vs Nemo-Hindi-4B",
    "Qwen3-4B vs BharatGPT-3B",
    "Qwen3-4B vs Airavata-7B",
    "Qwen3-4B vs Nemo-Hindi-4B",
    "Gemma3-4B-IT vs BharatGPT-3B",
    "Gemma3-4B-IT vs Airavata-7B",
    "Gemma3-4B-IT vs Nemo-Hindi-4B",
    "Llama-3.2-3B vs BharatGPT-3B",
    "Llama-3.2-3B vs Airavata-7B",
    "Llama-3.2-3B vs Nemo-Hindi-4B",
]

SOURCE_ORDER = [
    ("single_llm",       "Single-LLM"),
    ("selfreflect",      "Self-Reflect"),
    ("multi_agent_diff", "Debate (Diff)"),
    ("multi_agent_same", "Debate (Same)"),
]

# Divider after the 4th column (between bias plausible pos and stereotype judg)
TASK_DIVIDER_AFTER = 3


# ---------------------------------------------------------------------------
# Core plot function
# ---------------------------------------------------------------------------

def _draw_heatmap(ax, mat, row_labels, col_labels, title,
                  show_yticklabels=True, vmin=0.0, vmax=1.0):
    """Draw one annotated heatmap onto ax. Returns the imshow object."""
    n_rows, n_cols = mat.shape

    # column-wise max mask
    col_max = np.nanmax(mat, axis=0)
    is_max = np.isclose(mat, col_max[np.newaxis, :])

    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("lightgrey")
    im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap,
                   vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=8, ha="center")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    ax.set_yticks(range(n_rows))
    if show_yticklabels:
        ax.set_yticklabels(row_labels, fontsize=8)
    else:
        ax.set_yticklabels([])

    # cell annotations
    for i in range(n_rows):
        for j in range(n_cols):
            val = mat[i, j]
            if np.isnan(val):
                continue
            bold = is_max[i, j]
            ax.text(j, i, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=7.5,
                    fontweight="bold" if bold else "normal",
                    color="white" if val > 0.65 else "black")

    # divider between bias and stereotype task groups
    if TASK_DIVIDER_AFTER < n_cols - 1:
        ax.axvline(TASK_DIVIDER_AFTER + 0.5, color="black",
                   linewidth=1.5, linestyle="--", alpha=0.6)

    ax.set_title(title, fontsize=10, pad=28, fontweight="bold")
    return im


# ---------------------------------------------------------------------------
# Build matrix for one source
# ---------------------------------------------------------------------------

def _source_matrix(df: pd.DataFrame, source_key: str):
    sub = df[df["source"] == source_key]
    pivot = sub.pivot_table(index="model", columns="task",
                            values="refusal_rate", aggfunc="mean")
    # diff uses pair rows; same/single use individual model rows
    if source_key == "multi_agent_diff":
        row_order = [p for p in PAIR_ORDER if p in pivot.index]
    else:
        row_order = [m for m in MODEL_ORDER if m in pivot.index]
    tasks  = [t for t in TASK_ORDER if t in pivot.columns]
    mat = pivot.reindex(index=row_order, columns=tasks).values.astype(float)
    col_labels = [TASK_DISPLAY.get(t, t) for t in tasks]
    return mat, row_order, col_labels


# ---------------------------------------------------------------------------
# Individual plots (one per source)
# ---------------------------------------------------------------------------

def plot_individual(df: pd.DataFrame, out_dir: Path):
    for source_key, source_label in SOURCE_ORDER:
        mat, row_labels, col_labels = _source_matrix(df, source_key)
        if mat.size == 0:
            continue

        n_rows, n_cols = mat.shape
        fig_w = max(9, 1.3 * n_cols)
        fig_h = max(4, 0.6 * n_rows)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        im = _draw_heatmap(ax, mat, row_labels, col_labels,
                           title=f"Refusal Rate — {source_label}\n(bold = task maximum)")

        cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
        cbar.set_label("Refusal Rate", fontsize=9)

        # group labels above column headers
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        n_bias = TASK_DIVIDER_AFTER + 1
        n_stereo = n_cols - n_bias
        ax2.set_xticks([(n_bias - 1) / 2, n_bias + (n_stereo - 1) / 2])
        ax2.set_xticklabels(["Bias", "Stereotype"], fontsize=9, fontweight="bold")
        ax2.tick_params(length=0, pad=28)

        plt.tight_layout()
        out_path = out_dir / f"refusal_rates_{source_key}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Combined 2×2 figure
# ---------------------------------------------------------------------------

def plot_combined(df: pd.DataFrame, out_dir: Path):
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.08)

    last_im = None
    for idx, (source_key, source_label) in enumerate(SOURCE_ORDER):
        mat, row_labels, col_labels = _source_matrix(df, source_key)
        if mat.size == 0:
            continue

        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        show_y = (idx % 2 == 0)   # only leftmost column shows row labels
        im = _draw_heatmap(ax, mat, row_labels, col_labels,
                           title=source_label,
                           show_yticklabels=show_y)
        last_im = im

        # group headers
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        n_cols = mat.shape[1]
        n_bias = TASK_DIVIDER_AFTER + 1
        n_stereo = n_cols - n_bias
        ax2.set_xticks([(n_bias - 1) / 2, n_bias + (n_stereo - 1) / 2])
        ax2.set_xticklabels(["Bias", "Stereotype"], fontsize=8, fontweight="bold")
        ax2.tick_params(length=0, pad=28)

    if last_im is not None:
        fig.colorbar(last_im, ax=fig.get_axes(), shrink=0.4,
                     location="right", pad=0.02, label="Refusal Rate")

    fig.suptitle("Refusal Rates per Model × Task  (bold = task maximum)",
                 fontsize=13, fontweight="bold", y=1.01)

    out_path = out_dir / "refusal_rates_combined.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",     default="refusal_results/refusal_rates.csv")
    parser.add_argument("--out-dir", default="refusal_results/plots")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)

    print("Generating individual plots …")
    plot_individual(df, out_dir)

    print("Generating combined 2×2 plot …")
    plot_combined(df, out_dir)


if __name__ == "__main__":
    main()
