#!/usr/bin/env python3
"""
Apply the Prioritarian Principle to find the fairest judge for RSM and SAR.

    Fairness score (per source × task × model) = max(values) - min(values)
                                                  across all identities

    Lower spread → more equal treatment → fairer.

For RSM  reads from: rsm_results/{source}/{task_type}/{model_stem}_rsm.jsonl
For SAR  reads from: sar_results/{source}/{task}/{model_stem}_sar.jsonl

Outputs:
  fairness_rsm_{task_type}.csv   — spread per (source, model); fairest source highlighted
  fairness_sar_{task}.csv        — spread per (source, model); fairest source highlighted
  fairness_summary.csv           — mean spread per source per task (overall ranking)
  fairness_heatmaps/             — optional heatmaps of spread × source × model
"""

import json
import argparse
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ---------------------------------------------------------------------------
# Source / task definitions
# ---------------------------------------------------------------------------

RSM_SOURCES = [
    "single_llm", "selfreflect",
    "gemma_diff", "gemma_same",
    "gpt_diff",   "gpt_same",
    "indic_diff", "indic_same",
]
RSM_TASKS = ["judgement", "plausible"]

SAR_SOURCES = RSM_SOURCES
SAR_TASKS   = ["stereotype_judgement", "stereotype_plausible"]

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

# Heatmap variants: subsets of sources for focused comparison
# "all"          — all 8 sources (default combined view)
# "single_model" — non-judge baselines: single model responses only
# "model_pairs"  — diff judge sources (model A vs model B comparisons)
# "judge_only"   — all 6 LLM judge sources (diff + same variants)
HEATMAP_VARIANTS = {
    "all":          RSM_SOURCES,  # resolved at runtime (same list)
    "single_model": ["single_llm", "selfreflect"],
    "model_pairs":  ["gemma_diff", "gpt_diff", "indic_diff"],
    "judge_only":   ["gemma_diff", "gemma_same", "gpt_diff", "gpt_same", "indic_diff", "indic_same"],
    "judge_same":   ["gemma_same", "gpt_same", "indic_same"],
}

FILE_TO_MODEL = {
    "CohereLabs_aya-23-8B": "Aya-23-8B",
    "CoRover_BharatGPT-3B-Indic": "BharatGPT-3B",
    "Qwen_Qwen3-4B-Instruct-2507": "Qwen3-4B",
    "ai4bharat_Airavata": "Airavata-7B",
    "google_gemma-3-4b-it": "Gemma3-4B-IT",
    "meta-llama_Llama-3.2-3B-Instruct": "Llama-3.2-3B",
    "nvidia_Nemotron-4-Mini-Hindi-4B-Instruct": "Nemo-Hindi-4B",
}

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

import re

def _stem_to_model(stem: str, suffix: str) -> str:
    """Strip suffix and map to display name (handles both single and pair)."""
    key = re.sub(rf"_{suffix}$", "", stem)
    key = re.sub(r"_(responses|reflected|debate)$", "", key)
    if "_vs_" in key:
        parts = key.split("_vs_", 1)
        a = FILE_TO_MODEL.get(parts[0], parts[0])
        b = FILE_TO_MODEL.get(parts[1], parts[1])
        return f"{a} vs {b}"
    return FILE_TO_MODEL.get(key, key)


def load_spreads(base_dir: str, sources: list, tasks: list,
                 value_key: str, file_suffix: str) -> pd.DataFrame:
    """
    For each (source, task, model_file), compute spread = max(values) - min(values).
    Returns a long-format DataFrame with columns:
      source, task, model, spread, mean_val, max_val, min_val, n_identities
    """
    rows = []
    for source in sources:
        for task in tasks:
            d = Path(base_dir) / source / task
            if not d.exists():
                continue
            for fp in sorted(d.glob(f"*_{file_suffix}.jsonl")):
                model = _stem_to_model(fp.stem, file_suffix)
                vals = []
                with open(fp, encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        obj = json.loads(line)
                        v = obj.get(value_key)
                        if v is not None:
                            vals.append(v)
                if len(vals) < 2:
                    continue
                rows.append({
                    "source":       source,
                    "source_display": SOURCE_TO_DISPLAY.get(source, source),
                    "task":         task,
                    "model":        model,
                    "spread":       round(max(vals) - min(vals), 4),
                    "mean_val":     round(float(np.mean(vals)), 4),
                    "max_val":      round(max(vals), 4),
                    "min_val":      round(min(vals), 4),
                    "n_identities": len(vals),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fairness analysis
# ---------------------------------------------------------------------------

def fairest_per_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (task, model), find the source with minimum spread.
    Returns a DataFrame with the winning source per (task, model) pair.
    """
    idx = df.groupby(["task", "model"])["spread"].idxmin()
    best = df.loc[idx, ["task", "model", "source_display", "spread"]].copy()
    best = best.rename(columns={"source_display": "fairest_source", "spread": "min_spread"})
    best = best.sort_values(["task", "model"]).reset_index(drop=True)
    return best


def mean_spread_per_source(df: pd.DataFrame) -> pd.DataFrame:
    """
    Overall ranking: mean spread per (source, task).
    Lower mean spread → fairer overall.
    """
    agg = (df.groupby(["source_display", "task"])["spread"]
             .agg(mean_spread="mean", std_spread="std", n_models="count")
             .reset_index()
             .round(4)
             .sort_values(["task", "mean_spread"]))
    return agg


def spread_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot: rows=models, columns=sources, values=spread (per task separately)."""
    return df.pivot_table(index="model", columns="source_display",
                          values="spread", aggfunc="mean").round(4)


def load_identity_spreads(base_dir: str, sources: list, tasks: list,
                          value_key: str, file_suffix: str) -> pd.DataFrame:
    """
    For each (source, task, identity), compute spread = max(values) - min(values)
    across all model files.
    Returns a long-format DataFrame with columns:
      source, source_display, task, identity, spread, mean_val, max_val, min_val, n_models
    """
    from collections import defaultdict
    data = defaultdict(list)  # (source, task, identity) -> [values]
    for source in sources:
        for task in tasks:
            d = Path(base_dir) / source / task
            if not d.exists():
                continue
            for fp in sorted(d.glob(f"*_{file_suffix}.jsonl")):
                with open(fp, encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        obj = json.loads(line)
                        identity = obj.get("identity")
                        v = obj.get(value_key)
                        if identity is not None and v is not None:
                            data[(source, task, identity)].append(v)

    rows = []
    for (source, task, identity), vals in data.items():
        if len(vals) < 2:
            continue
        rows.append({
            "source":         source,
            "source_display": SOURCE_TO_DISPLAY.get(source, source),
            "task":           task,
            "identity":       identity,
            "spread":         round(max(vals) - min(vals), 4),
            "mean_val":       round(float(np.mean(vals)), 4),
            "max_val":        round(max(vals), 4),
            "min_val":        round(min(vals), 4),
            "n_models":       len(vals),
        })
    return pd.DataFrame(rows)


def fairest_per_identity(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (task, identity), find the source with minimum spread across models.
    Returns a DataFrame with the winning source per (task, identity) pair.
    """
    idx = df.groupby(["task", "identity"])["spread"].idxmin()
    best = df.loc[idx, ["task", "identity", "source_display", "spread"]].copy()
    best = best.rename(columns={"source_display": "fairest_source", "spread": "min_spread"})
    best = best.sort_values(["task", "identity"]).reset_index(drop=True)
    return best


def spread_pivot_identity(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot: rows=identities, columns=sources, values=spread across models."""
    return df.pivot_table(index="identity", columns="source_display",
                          values="spread", aggfunc="mean").round(4)


# ---------------------------------------------------------------------------
# Heatmap helpers
# ---------------------------------------------------------------------------

def plot_spread_heatmap(pivot: pd.DataFrame, title: str, out_path: str,
                        fairest: pd.Series = None):
    """
    Heatmap: models × sources, colour = spread (lower = greener = fairer).
    Cells with the minimum spread per row are starred.
    """
    mat = pivot.values.astype(float)
    models  = list(pivot.index)
    sources = list(pivot.columns)

    n_models  = len(models)
    n_sources = len(sources)
    fig_w = max(8, 1.2 * n_sources)
    fig_h = max(5, 0.45 * n_models)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    max_val = np.nanmax(mat)
    min_val = max(0, np.nanmin(mat))
    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_bad("lightgrey")

    im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap,
                   vmin=min_val, vmax=max_val, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.7, label="Spread (max − min)")

    ax.set_xticks(range(n_sources))
    ax.set_xticklabels(sources, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(models, fontsize=9)

    for i in range(n_models):
        row_vals = mat[i]
        row_min  = np.nanmin(row_vals)
        for j in range(n_sources):
            val = mat[i, j]
            if np.isnan(val):
                continue
            is_min = np.isclose(val, row_min)
            txt    = f"{val:.2f}" + (" ★" if is_min else "")
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if val > 0.65 * max_val else "black",
                    fontweight="bold" if is_min else "normal")

    ax.set_title(title, fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prioritarian fairness analysis for RSM and SAR")
    parser.add_argument("--rsm-dir",   default="rsm_results")
    parser.add_argument("--sar-dir",   default="sar_results")
    parser.add_argument("--out-dir",   default="fairness_results")
    parser.add_argument("--heatmaps",  action="store_true", default=True,
                        help="Generate spread heatmaps (default: on)")
    parser.add_argument("--no-heatmaps", dest="heatmaps", action="store_false")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.heatmaps:
        hmap_dir = out_dir / "heatmaps"
        hmap_dir.mkdir(exist_ok=True)

    summary_rows = []

    # =========================================================
    # RSM
    # =========================================================
    print("\n" + "=" * 60)
    print("RSM FAIRNESS  (spread = max_rsm − min_rsm per identity set)")
    print("=" * 60)

    rsm_df = load_spreads(args.rsm_dir, RSM_SOURCES, RSM_TASKS, "rsm", "rsm")

    for task_type in RSM_TASKS:
        sub = rsm_df[rsm_df["task"] == task_type]
        if sub.empty:
            continue

        print(f"\n── RSM | {task_type} ──────────────────────────────────")

        # Overall ranking
        rank = mean_spread_per_source(sub)
        print("\nMean spread per source (↑ least fair → ↓ fairest):")
        print(rank.to_string(index=False))

        # Per-model fairest
        best = fairest_per_model(sub)
        counts = best["fairest_source"].value_counts()
        print(f"\nFairest source per model (wins):\n{counts.to_string()}")

        for _, row in rank.iterrows():
            summary_rows.append({
                "metric": "RSM", "task": task_type,
                "source": row["source_display"],
                "mean_spread": row["mean_spread"],
                "std_spread": row["std_spread"],
                "n_models": row["n_models"],
            })

        # Save CSVs
        sub.to_csv(out_dir / f"rsm_spreads_{task_type}.csv", index=False)
        best.to_csv(out_dir / f"rsm_fairest_per_model_{task_type}.csv", index=False)

        # Heatmaps (all variants)
        if args.heatmaps:
            for variant_name, variant_sources in HEATMAP_VARIANTS.items():
                v_sub = sub[sub["source"].isin(variant_sources)]
                if v_sub.empty:
                    continue
                pivot = spread_pivot(v_sub).dropna(how="all")
                if pivot.empty:
                    continue
                variant_label = variant_name.replace("_", " ").title()
                plot_spread_heatmap(
                    pivot,
                    title=f"RSM Spread (Prioritarian Fairness) — {task_type.title()} [{variant_label}]\n★ = fairest source per model",
                    out_path=str(hmap_dir / f"rsm_fairness_{task_type}_{variant_name}.png"),
                )

    # =========================================================
    # RSM — per identity (spread across models per source)
    # =========================================================
    print("\n" + "=" * 60)
    print("RSM FAIRNESS PER IDENTITY  (spread = max_rsm − min_rsm across models)")
    print("=" * 60)

    rsm_id_df = load_identity_spreads(args.rsm_dir, RSM_SOURCES, RSM_TASKS, "rsm", "rsm")

    for task_type in RSM_TASKS:
        sub = rsm_id_df[rsm_id_df["task"] == task_type]
        if sub.empty:
            continue

        print(f"\n── RSM per identity | {task_type} ──────────────────────────────────")

        rank = mean_spread_per_source(sub)
        print("\nMean spread per source (↑ least fair → ↓ fairest):")
        print(rank.to_string(index=False))

        best = fairest_per_identity(sub)
        counts = best["fairest_source"].value_counts()
        print(f"\nFairest source per identity (wins):\n{counts.to_string()}")

        best.to_csv(out_dir / f"rsm_fairest_per_identity_{task_type}.csv", index=False)

        if args.heatmaps:
            for variant_name, variant_sources in HEATMAP_VARIANTS.items():
                v_sub = sub[sub["source"].isin(variant_sources)]
                if v_sub.empty:
                    continue
                pivot = spread_pivot_identity(v_sub).dropna(how="all")
                if pivot.empty:
                    continue
                variant_label = variant_name.replace("_", " ").title()
                plot_spread_heatmap(
                    pivot,
                    title=f"RSM Spread per Identity — {task_type.title()} [{variant_label}]\n★ = fairest source per identity",
                    out_path=str(hmap_dir / f"rsm_fairness_per_identity_{task_type}_{variant_name}.png"),
                )

    # =========================================================
    # SAR
    # =========================================================
    print("\n" + "=" * 60)
    print("SAR FAIRNESS  (spread = max_sar − min_sar per identity set)")
    print("=" * 60)

    sar_df = load_spreads(args.sar_dir, SAR_SOURCES, SAR_TASKS, "sar", "sar")

    for task in SAR_TASKS:
        sub = sar_df[sar_df["task"] == task]
        if sub.empty:
            continue

        print(f"\n── SAR | {task} ──────────────────────────────────")

        rank = mean_spread_per_source(sub)
        print("\nMean spread per source (↑ least fair → ↓ fairest):")
        print(rank.to_string(index=False))

        best = fairest_per_model(sub)
        counts = best["fairest_source"].value_counts()
        print(f"\nFairest source per model (wins):\n{counts.to_string()}")

        for _, row in rank.iterrows():
            summary_rows.append({
                "metric": "SAR", "task": task,
                "source": row["source_display"],
                "mean_spread": row["mean_spread"],
                "std_spread": row["std_spread"],
                "n_models": row["n_models"],
            })

        sub.to_csv(out_dir / f"sar_spreads_{task}.csv", index=False)
        best.to_csv(out_dir / f"sar_fairest_per_model_{task}.csv", index=False)

        if args.heatmaps:
            for variant_name, variant_sources in HEATMAP_VARIANTS.items():
                v_sub = sub[sub["source"].isin(variant_sources)]
                if v_sub.empty:
                    continue
                pivot = spread_pivot(v_sub).dropna(how="all")
                if pivot.empty:
                    continue
                variant_label = variant_name.replace("_", " ").title()
                plot_spread_heatmap(
                    pivot,
                    title=f"SAR Spread (Prioritarian Fairness) — {task.replace('_', ' ').title()} [{variant_label}]\n★ = fairest source per model",
                    out_path=str(hmap_dir / f"sar_fairness_{task}_{variant_name}.png"),
                )

    # =========================================================
    # SAR — per identity (spread across models per source)
    # =========================================================
    print("\n" + "=" * 60)
    print("SAR FAIRNESS PER IDENTITY  (spread = max_sar − min_sar across models)")
    print("=" * 60)

    sar_id_df = load_identity_spreads(args.sar_dir, SAR_SOURCES, SAR_TASKS, "sar", "sar")

    for task in SAR_TASKS:
        sub = sar_id_df[sar_id_df["task"] == task]
        if sub.empty:
            continue

        print(f"\n── SAR per identity | {task} ──────────────────────────────────")

        rank = mean_spread_per_source(sub)
        print("\nMean spread per source (↑ least fair → ↓ fairest):")
        print(rank.to_string(index=False))

        best = fairest_per_identity(sub)
        counts = best["fairest_source"].value_counts()
        print(f"\nFairest source per identity (wins):\n{counts.to_string()}")

        best.to_csv(out_dir / f"sar_fairest_per_identity_{task}.csv", index=False)

        if args.heatmaps:
            for variant_name, variant_sources in HEATMAP_VARIANTS.items():
                v_sub = sub[sub["source"].isin(variant_sources)]
                if v_sub.empty:
                    continue
                pivot = spread_pivot_identity(v_sub).dropna(how="all")
                if pivot.empty:
                    continue
                variant_label = variant_name.replace("_", " ").title()
                plot_spread_heatmap(
                    pivot,
                    title=f"SAR Spread per Identity — {task.replace('_', ' ').title()} [{variant_label}]\n★ = fairest source per identity",
                    out_path=str(hmap_dir / f"sar_fairness_per_identity_{task}_{variant_name}.png"),
                )

    # =========================================================
    # Combined summary
    # =========================================================
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)

        # Rank within each (metric, task) group
        summary_df["rank"] = (summary_df
                              .groupby(["metric", "task"])["mean_spread"]
                              .rank(method="min")
                              .astype(int))

        summary_path = out_dir / "fairness_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        print("\n" + "=" * 60)
        print("OVERALL FAIRNESS RANKING  (rank 1 = fairest)")
        print("=" * 60)
        print(summary_df.sort_values(["metric", "task", "rank"]).to_string(index=False))
        print(f"\nSaved → {summary_path}")

        # Which source wins most often across all tasks?
        wins = summary_df[summary_df["rank"] == 1]["source"].value_counts()
        print(f"\nFairest source overall (# task-wins):\n{wins.to_string()}")


if __name__ == "__main__":
    main()
