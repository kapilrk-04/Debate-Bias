#!/usr/bin/env python3
"""
Compute Relative Score Metric (RSM) from per-subtask ELO scores.

    RSM(identity, model, source) = positive_elo - negative_elo

Sources and task types mirror compute_bias_elo.py.
Reads from:  elo_results/{source}/bias_{task_type}_{positive|negative}/{stem}_elo.json
Writes to:   rsm_results/{source}/{task_type}/{stem}_rsm.jsonl
             rsm_results/{source}/{task_type}_per_model.csv  ← identities × models
             rsm_results/summary_rsm.csv
"""

import json
import argparse
from pathlib import Path

import pandas as pd

SOURCES = [
    "single_llm",
    "selfreflect",
    "gemma_diff",
    "gemma_same",
    "gpt_diff",
    "gpt_same",
    "indic_diff",
    "indic_same",
]

TASK_TYPES = ["judgement", "plausible"]


def load_elo(path: Path) -> dict:
    """Load {identity: elo_score} from a _elo.json file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compute_rsm_for_source(source: str, elo_dir: Path, rsm_dir: Path) -> list:
    """
    For one source, compute RSM across both task types.
    Returns summary rows.
    """
    summary_rows = []
    src_elo = elo_dir / source
    src_rsm = rsm_dir / source
    src_rsm.mkdir(parents=True, exist_ok=True)

    for task_type in TASK_TYPES:
        pos_dir = src_elo / f"bias_{task_type}_positive"
        neg_dir = src_elo / f"bias_{task_type}_negative"

        if not pos_dir.exists() or not neg_dir.exists():
            print(f"  [SKIP] {source} | {task_type} — pos or neg dir missing")
            continue

        pos_stems = {f.stem.replace("_elo", "") for f in pos_dir.glob("*_elo.json")}
        neg_stems = {f.stem.replace("_elo", "") for f in neg_dir.glob("*_elo.json")}
        common = sorted(pos_stems & neg_stems)

        if not common:
            print(f"  [SKIP] {source} | {task_type} — no matching model stems")
            continue

        print(f"\n--- {source} | {task_type}  ({len(common)} models) ---")

        task_rsm_dir = src_rsm / task_type
        task_rsm_dir.mkdir(parents=True, exist_ok=True)

        per_model_rsm: dict[str, dict] = {}   # {model_stem: {identity: rsm}}

        for stem in common:
            pos_scores = load_elo(pos_dir / f"{stem}_elo.json")
            neg_scores = load_elo(neg_dir / f"{stem}_elo.json")

            identities = sorted(set(pos_scores) | set(neg_scores))
            records = []
            for identity in identities:
                pos_val = pos_scores.get(identity)
                neg_val = neg_scores.get(identity)
                rsm_val = round(pos_val - neg_val, 4) if (pos_val is not None and neg_val is not None) else None
                records.append({
                    "identity":     identity,
                    "positive_elo": round(pos_val, 4) if pos_val is not None else None,
                    "negative_elo": round(neg_val, 4) if neg_val is not None else None,
                    "rsm":          rsm_val,
                })

            # Save per-model JSONL
            out_path = task_rsm_dir / f"{stem}_rsm.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            per_model_rsm[stem] = {r["identity"]: r["rsm"] for r in records}

            rsm_vals = [r["rsm"] for r in records if r["rsm"] is not None]
            top_identity = max(per_model_rsm[stem], key=lambda k: per_model_rsm[stem][k] or float("-inf"))
            print(f"  {stem}: rsm range [{min(rsm_vals):.2f}, {max(rsm_vals):.2f}]  top={top_identity}")

            summary_rows.append({
                "source":       source,
                "task_type":    task_type,
                "model":        stem,
                "top_identity": top_identity,
                "top_rsm":      round(per_model_rsm[stem][top_identity], 4),
                "mean_rsm":     round(sum(rsm_vals) / len(rsm_vals), 4) if rsm_vals else None,
            })

        # Combined CSV: identities as rows, model stems as columns
        if per_model_rsm:
            combined = pd.DataFrame.from_dict(per_model_rsm, orient="columns").sort_index()
            combined_path = src_rsm / f"{task_type}_per_model.csv"
            combined.to_csv(combined_path)
            print(f"  Combined RSM table → {combined_path}")

    return summary_rows


def main():
    parser = argparse.ArgumentParser(description="Compute RSM = positive_elo - negative_elo")
    parser.add_argument("--elo-dir",  default="elo_results",  help="Input ELO results dir")
    parser.add_argument("--rsm-dir",  default="rsm_results",  help="Output RSM results dir")
    parser.add_argument("--sources",  nargs="+", default=SOURCES, choices=SOURCES)
    args = parser.parse_args()

    elo_dir = Path(args.elo_dir)
    rsm_dir = Path(args.rsm_dir)
    rsm_dir.mkdir(parents=True, exist_ok=True)

    all_summary = []
    for source in args.sources:
        rows = compute_rsm_for_source(source, elo_dir, rsm_dir)
        all_summary.extend(rows)

    if all_summary:
        summary_df = pd.DataFrame(all_summary)
        summary_path = rsm_dir / "summary_rsm.csv"
        summary_df.to_csv(summary_path, index=False)

        print("\n" + "=" * 70)
        print("RSM SUMMARY")
        print("=" * 70)
        print(summary_df.to_string(index=False))
        print(f"\nSaved → {summary_path}")


if __name__ == "__main__":
    main()
