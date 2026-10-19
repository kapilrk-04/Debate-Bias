#!/usr/bin/env python3
"""
Generate SAR heatmaps from sar_results/.

One heatmap per (source, task) showing:
  rows    = models
  columns = language × {SAR | Tie} pairs  (orange = SAR, blue = Tie rate)

Usage:
  python plot_sar_heatmaps.py [--sar-dir sar_results] [--out-dir sar_heatmaps]
                               [--sources single_llm gemma_diff ...]
                               [--tasks stereotype_judgement stereotype_plausible]
"""

import json
import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Name mappings  (same as reference)
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
    "assamese": "as", "bengali": "bn", "gujarati": "gu", "hindi": "hi",
    "kannadiga": "kn", "malayali": "ml", "marathi": "mr", "punjabi": "pa",
    "tamil": "ta", "telugu": "te",
}

SOURCE_TO_DISPLAY = {
    "single_llm": "Single LLM",
    "selfreflect": "Self-Reflect",
    "gemma_diff": "Gemma-3-12B (Diff)",
    "gemma_same": "Gemma-3-12B (Same)",
    "gpt_diff":   "GPT-4o-nano (Diff)",
    "gpt_same":   "GPT-4o-nano (Same)",
    "indic_diff": "Krutrim-2 (Diff)",
    "indic_same": "Krutrim-2 (Same)",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _map_model(raw: str) -> str:
    """Map raw model key (possibly a vs-pair) to display name."""
    if "_vs_" in raw:
        # model pair: MODEL1_vs_MODEL2
        parts = raw.split("_vs_", 1)
        a = FILE_TO_MODEL.get(parts[0], parts[0])
        b = FILE_TO_MODEL.get(parts[1], parts[1])
        return f"{a} vs {b}"
    return FILE_TO_MODEL.get(raw, raw)


def _stem_to_model_key(stem: str) -> str:
    """Strip known suffixes and return the raw model key."""
    stem = re.sub(r"_sar$", "", stem)
    stem = re.sub(r"_(responses|reflected|debate)$", "", stem)
    return stem


def load_sar_data(data_dir: str) -> tuple:
    """
    Load all *_sar.jsonl files in data_dir.
    Returns (matrix, model_names, language_short_names).
    matrix shape: (n_models, n_langs).  Values = SAR.
    Also returns tie_matrix of same shape.
    """
    files = sorted(Path(data_dir).glob("*_sar.jsonl"))
    if not files:
        return None, None, None, None

    # First pass: collect all languages
    langs_set = set()
    file_records = {}
    for fp in files:
        records = {}
        with open(fp, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                lang = obj["identity"].lower()
                records[lang] = {"sar": obj["sar"], "tie_rate": obj["tie_rate"]}
                langs_set.add(lang)
        file_records[fp] = records

    languages = sorted(langs_set)
    lang_labels = [LANG_TO_SHORT.get(l, l) for l in languages]

    models = []
    sar_rows, tie_rows = [], []

    for fp in files:
        key = _stem_to_model_key(fp.stem)
        model_name = _map_model(key)
        records = file_records[fp]

        # skip models with all-zero / all-same SAR (no signal)
        sar_vals = [records.get(l, {}).get("sar", 0.0) for l in languages]
        if all(v == sar_vals[0] for v in sar_vals):
            continue

        models.append(model_name)
        sar_rows.append([records.get(l, {}).get("sar",      np.nan) for l in languages])
        tie_rows.append([records.get(l, {}).get("tie_rate", np.nan) for l in languages])

    if not models:
        return None, None, None, None

    return np.array(sar_rows), np.array(tie_rows), models, lang_labels


def create_sar_heatmap(data_dir: str, output_path: str, title: str):
    sar_mat, tie_mat, models, lang_labels = load_sar_data(data_dir)

    if sar_mat is None:
        print(f"  [SKIP] no data in {data_dir}")
        return

    n_models = len(models)
    n_langs  = len(lang_labels)

    # Interleave SAR | Tie columns: [sar_0, tie_0, sar_1, tie_1, ...]
    matrix = np.full((n_models, n_langs * 2), np.nan)
    for j in range(n_langs):
        matrix[:, 2 * j]     = sar_mat[:, j]
        matrix[:, 2 * j + 1] = tie_mat[:, j]

    fig_height = max(6, 0.55 * n_models)
    fig_width  = max(16, 1.8 * n_langs)
    plt.figure(figsize=(fig_width, fig_height))

    win_mask = np.ones_like(matrix, dtype=bool)
    tie_mask = np.ones_like(matrix, dtype=bool)
    win_mask[:, ::2]  = False   # show even cols (SAR)
    tie_mask[:, 1::2] = False   # show odd  cols (Tie)

    plt.imshow(np.ma.masked_where(win_mask, matrix), cmap="Oranges", vmin=0, vmax=1, aspect="auto")
    plt.imshow(np.ma.masked_where(tie_mask, matrix), cmap="Blues",   vmin=0, vmax=1, aspect="auto")

    # Bottom x-axis: Win / Tie per language
    subcol_labels = []
    for _ in lang_labels:
        subcol_labels.extend(["SAR", "Tie"])
    plt.xticks(range(len(subcol_labels)), subcol_labels, rotation=0, fontsize=10)

    # Top x-axis: language short codes centred over each pair
    ax = plt.gca()
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks([2 * i + 0.5 for i in range(n_langs)])
    ax_top.set_xticklabels(lang_labels, rotation=0, ha="center", fontsize=12)
    ax_top.xaxis.set_ticks_position("top")
    ax_top.spines["top"].set_visible(False)

    plt.yticks(range(n_models), models, fontsize=10)

    # Cell annotations
    for i in range(n_models):
        for j in range(n_langs * 2):
            val = matrix[i, j]
            if not np.isnan(val):
                plt.text(j, i, f"{val:.2f}", ha="center", va="center",
                         fontsize=9, color="black" if val < 0.6 else "white")

    plt.title(title, fontsize=13, pad=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sar-dir",  default="sar_results")
    parser.add_argument("--out-dir",  default="sar_heatmaps")
    parser.add_argument("--sources",  nargs="+",
                        default=list(SOURCE_TO_DISPLAY.keys()))
    parser.add_argument("--tasks",    nargs="+",
                        default=["stereotype_judgement", "stereotype_plausible"])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for source in args.sources:
        for task in args.tasks:
            data_dir = os.path.join(args.sar_dir, source, task)
            if not os.path.isdir(data_dir):
                print(f"  [SKIP] {data_dir} not found")
                continue

            title = f"SAR — {SOURCE_TO_DISPLAY.get(source, source)} | {task.replace('_', ' ').title()}"
            out_path = os.path.join(args.out_dir, f"sar_{source}_{task}.png")
            print(f"\n{title}")
            create_sar_heatmap(data_dir, out_path, title)


if __name__ == "__main__":
    main()
