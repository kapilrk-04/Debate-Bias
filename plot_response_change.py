#!/usr/bin/env python3
"""
Plot response change rates as annotated heatmaps.

Sources × agents:
  selfreflect      / agent   — 7 model rows
  multi_agent_diff / agent1  — 12 pair rows
  multi_agent_diff / agent2  — 12 pair rows
  multi_agent_same / agent1  — 7 model rows
  multi_agent_same / agent2  — 7 model rows

Rows    = models (or pairs for multi_agent_diff)
Columns = tasks
Color   = change_rate (0–1, YlOrRd: yellow=low, red=high)
Bold    = column-wise maximum (highest change rate per task)

Individual plots: one per (source, agent).
Combined figure:  selfreflect alone, then diff and same each as a 2-row panel.

Usage:
  python plot_response_change.py [--csv refusal_results/response_change_rates.csv]
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

# Divider after the 4th column (between bias_plausible_positive and stereotype_judgement)
TASK_DIVIDER_AFTER = 3

# All (source, agent) panels in display order
PANELS = [
    ("selfreflect",      "agent",  "Self-Reflect"),
    ("multi_agent_diff", "agent1", "Debate (Diff) — Agent 1"),
    ("multi_agent_diff", "agent2", "Debate (Diff) — Agent 2"),
    ("multi_agent_same", "agent1", "Debate (Same) — Agent 1"),
    ("multi_agent_same", "agent2", "Debate (Same) — Agent 2"),
]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _build_matrix(df: pd.DataFrame, source_key: str, agent_key: str):
    """Return (mat, row_labels, col_labels) for a given source×agent slice."""
    sub = df[(df["source"] == source_key) & (df["agent"] == agent_key)]
    if sub.empty:
        return None, [], []

    pivot = sub.pivot_table(
        index="model", columns="task",
        values="change_rate", aggfunc="mean",
    )

    if source_key == "multi_agent_diff":
        row_order = [p for p in PAIR_ORDER if p in pivot.index]
    else:
        row_order = [m for m in MODEL_ORDER if m in pivot.index]

    tasks = [t for t in TASK_ORDER if t in pivot.columns]
    mat = pivot.reindex(index=row_order, columns=tasks).values.astype(float)
    col_labels = [TASK_DISPLAY.get(t, t) for t in tasks]
    return mat, row_order, col_labels


def _draw_heatmap(ax, mat, row_labels, col_labels, title,
                  show_yticklabels=True, vmin=0.0, vmax=1.0):
    """Draw one annotated heatmap onto ax. Returns the imshow object."""
    n_rows, n_cols = mat.shape

    col_max = np.nanmax(mat, axis=0)
    is_max  = np.isclose(mat, col_max[np.newaxis, :])

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

    for i in range(n_rows):
        for j in range(n_cols):
            val = mat[i, j]
            if np.isnan(val):
                continue
            ax.text(j, i, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=7.5,
                    fontweight="bold" if is_max[i, j] else "normal",
                    color="white" if val > 0.65 else "black")

    if TASK_DIVIDER_AFTER < n_cols - 1:
        ax.axvline(TASK_DIVIDER_AFTER + 0.5, color="black",
                   linewidth=1.5, linestyle="--", alpha=0.6)

    ax.set_title(title, fontsize=10, pad=28, fontweight="bold")
    return im


def _add_group_headers(ax, n_cols):
    """Add Bias / Stereotype group labels above the column tick labels."""
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    n_bias   = TASK_DIVIDER_AFTER + 1
    n_stereo = n_cols - n_bias
    ax2.set_xticks([(n_bias - 1) / 2, n_bias + (n_stereo - 1) / 2])
    ax2.set_xticklabels(["Bias", "Stereotype"], fontsize=9, fontweight="bold")
    ax2.tick_params(length=0, pad=28)


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------

def plot_individual(df: pd.DataFrame, out_dir: Path):
    for source_key, agent_key, label in PANELS:
        mat, row_labels, col_labels = _build_matrix(df, source_key, agent_key)
        if mat is None or mat.size == 0:
            print(f"  [SKIP] {source_key}/{agent_key} — no data")
            continue

        n_rows, n_cols = mat.shape
        fig_w = max(9, 1.3 * n_cols)
        fig_h = max(4, 0.6 * n_rows)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        im = _draw_heatmap(
            ax, mat, row_labels, col_labels,
            title=f"Response Change Rate — {label}\n(bold = task maximum)",
        )

        cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
        cbar.set_label("Change Rate", fontsize=9)

        _add_group_headers(ax, n_cols)

        plt.tight_layout()
        fname = f"response_change_{source_key}_{agent_key}.png"
        out_path = out_dir / fname
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Combined figure (3-row layout: selfreflect, diff ×2, same ×2)
# ---------------------------------------------------------------------------

def plot_combined(df: pd.DataFrame, out_dir: Path):
    """
    Layout:
      Row 0 (1 panel):  selfreflect / agent
      Row 1 (2 panels): multi_agent_diff / agent1 | agent2
      Row 2 (2 panels): multi_agent_same / agent1 | agent2
    """
    fig = plt.figure(figsize=(22, 16))
    outer = gridspec.GridSpec(3, 1, figure=fig, hspace=0.55)

    last_im = None

    # ── Row 0: selfreflect ─────────────────────────────────────────────────
    gs0 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0])
    ax  = fig.add_subplot(gs0[0, 0])
    mat, row_labels, col_labels = _build_matrix(df, "selfreflect", "agent")
    if mat is not None and mat.size > 0:
        im = _draw_heatmap(ax, mat, row_labels, col_labels, "Self-Reflect")
        _add_group_headers(ax, mat.shape[1])
        last_im = im

    # ── Row 1: multi_agent_diff ───────────────────────────────────────────
    gs1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.08)
    for col_idx, agent_key in enumerate(["agent1", "agent2"]):
        ax = fig.add_subplot(gs1[0, col_idx])
        mat, row_labels, col_labels = _build_matrix(df, "multi_agent_diff", agent_key)
        if mat is None or mat.size == 0:
            continue
        show_y = (col_idx == 0)
        im = _draw_heatmap(ax, mat, row_labels, col_labels,
                           f"Debate (Diff) — {agent_key.capitalize()}",
                           show_yticklabels=show_y)
        _add_group_headers(ax, mat.shape[1])
        last_im = im

    # ── Row 2: multi_agent_same ───────────────────────────────────────────
    gs2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2], wspace=0.08)
    for col_idx, agent_key in enumerate(["agent1", "agent2"]):
        ax = fig.add_subplot(gs2[0, col_idx])
        mat, row_labels, col_labels = _build_matrix(df, "multi_agent_same", agent_key)
        if mat is None or mat.size == 0:
            continue
        show_y = (col_idx == 0)
        im = _draw_heatmap(ax, mat, row_labels, col_labels,
                           f"Debate (Same) — {agent_key.capitalize()}",
                           show_yticklabels=show_y)
        _add_group_headers(ax, mat.shape[1])
        last_im = im

    if last_im is not None:
        fig.colorbar(last_im, ax=fig.get_axes(), shrink=0.3,
                     location="right", pad=0.02, label="Change Rate")

    fig.suptitle(
        "Response Change Rates per Model × Task  (bold = task maximum)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    out_path = out_dir / "response_change_combined.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",     default="refusal_results/response_change_rates.csv")
    parser.add_argument("--out-dir", default="refusal_results/plots")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)

    print("Generating individual plots …")
    plot_individual(df, out_dir)

    print("Generating combined plot …")
    plot_combined(df, out_dir)


if __name__ == "__main__":
    main()
