#!/usr/bin/env python3
"""
Generate bias RSM heatmaps from rsm_results/.

Layout mirrors the reference script:
  - One plot per task_type (judgement / plausible)
  - Each plot combines two source groups side by side, separated by a vertical line:
      left  = "diff" sources  (gemma_diff, gpt_diff, indic_diff, single_llm)
      right = "same" sources  (gemma_same, gpt_same, indic_same, selfreflect)
  - rows = models (unique across all files in the group)
  - columns = language × source
  - colourmap = RdBu_r  (red = positive RSM, blue = negative RSM)
  - NaN cells shown in light grey

Usage:
  python plot_rsm_heatmaps.py [--rsm-dir rsm_results] [--out-dir rsm_heatmaps]
                               [--tasks judgement plausible]
"""

import json
import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Name mappings
# ---------------------------------------------------------------------------

FILE_TO_MODEL = {
    "CohereLabs_aya-23-8B": "Aya-23-8B",
    "CoRover_BharatGPT-3B-Indic": "BharatGPT-3B",
    "Qwen_Qwen3-4B-Instruct-2507": "Qwen3-4B",
    "ai4bharat_Airavata": "Airavata-7B",
    "google_gemma-3-4b-it": "Gemma3-4B-IT",
    "meta-llama_Llama-3.2-3B-Instruct": "Llama-3.2-3B",
    "nvidia_Nemotron-4-Mini-Hindi-4B-Instruct": "Nemo-Hindi-4B",
    "gemini-2.5-flash": "Gemini-2.5-Flash",
    "gemini-2.5-pro": "Gemini-2.5-Pro",
    "google_gemma-3-27b-it": "Gemma3-27B-IT",
    "krutrim-ai-labs_Krutrim-2-instruct": "Krutrim-2-12B",
    "Qwen_Qwen2.5-72B-Instruct": "Qwen-2.5-72B",
    "meta-llama_Llama-3.3-70B-Instruct": "Llama-3.3-70B",
    "sarvamai_sarvam-m": "Sarvam-M-24B",
    "Telugu-LLM-Labs_Indic-gemma-7b-finetuned-sft-Navarasa-2.0": "Indic-Gemma-7B",
    "GenVRadmin_AryaBhatta-GemmaOrca-Merged": "GemmaOrca-8.5B",
    "GenVRadmin_AryaBhatta-GemmaUltra-Merged": "GemmaUltra-8.5B",
    "MBZUAI_Llama-3-Nanda-10B-Chat": "Nanda-10B",
    "MBZUAI-IFM_Llama-3.1-Nanda-87B-Chat": "Nanda-87B",
    "meta-llama_Meta-Llama-3-8B-Instruct": "Llama-3-8B",
    "meta-llama_Meta-Llama-3-70B-Instruct": "Llama-3-70B",
    "meta-llama_Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "meta-llama_Llama-3.1-70B-Instruct": "Llama-3.1-70B",
    "large-traversaal_Qwen-2.5-14B-Hindi": "Qwen-2.5-14B-Hi",
}

LANG_TO_SHORT = {
    "Assamese": "as", "Bengali": "bn", "Gujarati": "gu", "Hindi": "hi",
    "Kannadiga": "kn", "Malayali": "ml", "Marathi": "mr", "Punjabi": "pa",
    "Tamil": "ta", "Telugu": "te",
    # lower-case fallbacks
    "assamese": "as", "bengali": "bn", "gujarati": "gu", "hindi": "hi",
    "kannadiga": "kn", "malayali": "ml", "marathi": "mr", "punjabi": "pa",
    "tamil": "ta", "telugu": "te",
}

# Source groups: (display_label, [sources])
SOURCE_GROUPS = [
    ("Diff",  ["gemma_diff", "gpt_diff", "indic_diff", "single_llm"]),
    ("Same",  ["gemma_same", "gpt_same", "indic_same", "selfreflect"]),
]

SOURCE_TO_DISPLAY = {
    "single_llm":  "Single-LLM",
    "selfreflect": "Self-Reflect",
    "gemma_diff":  "Gemma(D)",
    "gemma_same":  "Gemma(S)",
    "gpt_diff":    "GPT(D)",
    "gpt_same":    "GPT(S)",
    "indic_diff":  "Krutrim(D)",
    "indic_same":  "Krutrim(S)",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stem_to_model_key(stem: str) -> str:
    stem = re.sub(r"_rsm$", "", stem)
    stem = re.sub(r"_(responses|reflected|debate)$", "", stem)
    return stem


def _map_model(raw: str) -> str:
    if "_vs_" in raw:
        parts = raw.split("_vs_", 1)
        a = FILE_TO_MODEL.get(parts[0], parts[0])
        b = FILE_TO_MODEL.get(parts[1], parts[1])
        return f"{a} vs {b}"
    return FILE_TO_MODEL.get(raw, raw)


def load_rsm_source(rsm_dir: str, source: str, task_type: str) -> tuple:
    """
    Load all *_rsm.jsonl files for one (source, task_type).
    Returns (matrix, model_names, lang_short_labels).
    """
    data_dir = Path(rsm_dir) / source / task_type
    if not data_dir.exists():
        return None, None, None

    files = sorted(data_dir.glob("*_rsm.jsonl"))
    if not files:
        return None, None, None

    langs_set = set()
    file_records = {}
    for fp in files:
        records = {}
        with open(fp, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                lang = obj["identity"]
                records[lang] = obj["rsm"]
                langs_set.add(lang)
        file_records[fp] = records

    languages = sorted(langs_set, key=lambda l: l.lower())
    lang_labels = [LANG_TO_SHORT.get(l, l) for l in languages]

    models = []
    rsm_rows = []
    for fp in files:
        key = _stem_to_model_key(fp.stem)
        model_name = _map_model(key)
        records = file_records[fp]
        models.append(model_name)
        rsm_rows.append([records.get(l, np.nan) for l in languages])

    return np.array(rsm_rows), models, lang_labels


# ---------------------------------------------------------------------------
# Plot one task_type: combine two source groups side by side
# ---------------------------------------------------------------------------

def plot_rsm_task(rsm_dir: str, task_type: str, out_path: str):
    """
    One figure per task_type.  Left half = diff sources, right half = same sources.
    Each source contributes its own language columns (with source label at top).
    Rows are the union of all models.
    """
    # ---- collect data from each source in each group ----
    group_blocks = []   # list of (group_label, [(source, matrix, models, langs)])

    for group_label, sources in SOURCE_GROUPS:
        block = []
        for source in sources:
            mat, models, langs = load_rsm_source(rsm_dir, source, task_type)
            if mat is not None and len(models) > 0:
                block.append((source, mat, models, langs))
        if block:
            group_blocks.append((group_label, block))

    if not group_blocks:
        print(f"  [SKIP] no data for task_type={task_type}")
        return

    # ---- get union of all models (preserve order) ----
    all_models = []
    for _, block in group_blocks:
        for _, _, models, _ in block:
            for m in models:
                if m not in all_models:
                    all_models.append(m)

    # ---- build combined matrix: rows=models, cols=all source×lang ----
    col_labels   = []    # language short labels per column
    source_labels = []   # (source_display, start_col, end_col) for top axis
    group_ranges  = []   # (group_label, start_col, end_col) for group header axis
    group_dividers = []  # x positions for group separator lines

    big_matrix = np.full((len(all_models), 0), np.nan)

    for g_idx, (group_label, block) in enumerate(group_blocks):
        group_start = big_matrix.shape[1]

        for source, mat, models, langs in block:
            src_col = np.full((len(all_models), len(langs)), np.nan)
            for row_i, model in enumerate(all_models):
                if model in models:
                    m_idx = models.index(model)
                    src_col[row_i, :] = mat[m_idx, :]

            start_c = big_matrix.shape[1]
            big_matrix = np.hstack([big_matrix, src_col])
            end_c = big_matrix.shape[1]

            col_labels.extend(langs)
            source_labels.append((SOURCE_TO_DISPLAY.get(source, source), start_c, end_c))

        group_end = big_matrix.shape[1]
        group_ranges.append((group_label, group_start, group_end))

        # divider between groups (not after last)
        if g_idx < len(group_blocks) - 1:
            group_dividers.append(big_matrix.shape[1] - 0.5)

    # ---- colour scale ----
    max_abs = np.nanmax(np.abs(big_matrix))
    if max_abs == 0 or np.isnan(max_abs):
        max_abs = 1.0

    # ---- figure ----
    n_cols   = big_matrix.shape[1]
    n_models = len(all_models)
    fig_w = max(14, 0.65 * n_cols)
    fig_h = max(8,  0.45 * n_models)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    masked = np.ma.masked_invalid(big_matrix)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="lightgrey")
    im = ax.imshow(masked, cmap=cmap, vmin=-max_abs, vmax=max_abs, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.6, label="RSM (pos_elo − neg_elo)")

    # Bottom x-axis: language labels
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=0, fontsize=9)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(all_models, fontsize=9)

    # Top x-axis: source labels centred over their column span
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    tick_pos   = [(s + e) / 2 - 0.5 for _, s, e in source_labels]
    tick_names = [lbl for lbl, _, _ in source_labels]
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels(tick_names, rotation=0, ha="center", fontsize=9, weight="bold")
    ax2.tick_params(length=0)

    # Group dividers (thick black line between diff and same)
    for xpos in group_dividers:
        ax.axvline(xpos, color="black", linewidth=2.5)

    # Cell annotations
    for i in range(n_models):
        for j in range(n_cols):
            val = big_matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if abs(val) > 0.6 * max_abs else "black")

    # Group labels above source labels (using tracked group_ranges)
    if len(group_ranges) > 1:
        ax3 = ax.twiny()
        ax3.set_xlim(ax.get_xlim())
        centers = [(s + e) / 2 - 0.5 for _, s, e in group_ranges]
        names   = [lbl for lbl, _, _ in group_ranges]
        ax3.set_xticks(centers)
        ax3.set_xticklabels(names, fontsize=12, weight="bold")
        ax3.tick_params(length=0, pad=14)

    title = f"Bias RSM — {task_type.title()}"
    ax.set_title(title, fontsize=13, pad=30)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Per-source plots (simpler, one plot per source × task_type)
# ---------------------------------------------------------------------------

def plot_rsm_per_source(rsm_dir: str, source: str, task_type: str, out_path: str):
    mat, models, langs = load_rsm_source(rsm_dir, source, task_type)
    if mat is None:
        print(f"  [SKIP] {source} | {task_type}")
        return

    max_abs = np.nanmax(np.abs(mat))
    if max_abs == 0 or np.isnan(max_abs):
        max_abs = 1.0

    n_models, n_langs = mat.shape
    fig_w = max(10, 0.7 * n_langs)
    fig_h = max(6,  0.45 * n_models)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    masked = np.ma.masked_invalid(mat)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="lightgrey")
    im = ax.imshow(masked, cmap=cmap, vmin=-max_abs, vmax=max_abs, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.7, label="RSM")

    ax.set_xticks(range(n_langs))
    ax.set_xticklabels(langs, rotation=0, fontsize=11)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(models, fontsize=10)

    for i in range(n_models):
        for j in range(n_langs):
            val = mat[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=9,
                        color="white" if abs(val) > 0.6 * max_abs else "black")

    title = f"Bias RSM — {SOURCE_TO_DISPLAY.get(source, source)} | {task_type.title()}"
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rsm-dir",   default="rsm_results")
    parser.add_argument("--out-dir",   default="rsm_heatmaps")
    parser.add_argument("--tasks",     nargs="+", default=["judgement", "plausible"])
    parser.add_argument("--mode",      choices=["combined", "per-source", "both"],
                        default="both",
                        help="combined=diff+same side by side; per-source=one per source")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    all_sources = [s for _, grp in SOURCE_GROUPS for s in grp]

    for task_type in args.tasks:
        print(f"\n=== task_type: {task_type} ===")

        if args.mode in ("combined", "both"):
            out_path = os.path.join(args.out_dir, f"rsm_combined_{task_type}.png")
            print(f"  Combined diff+same plot …")
            plot_rsm_task(args.rsm_dir, task_type, out_path)

        if args.mode in ("per-source", "both"):
            for source in all_sources:
                out_path = os.path.join(args.out_dir, f"rsm_{source}_{task_type}.png")
                plot_rsm_per_source(args.rsm_dir, source, task_type, out_path)


if __name__ == "__main__":
    main()
