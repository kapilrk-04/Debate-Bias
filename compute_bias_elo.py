#!/usr/bin/env python3
"""
Compute Bradley-Terry MLE ELO rankings for bias tasks.

Two modes:
  --aggregate   (default) one ELO per source × task category (all files pooled)
  --per-file              one ELO per individual model file × task category

Sources:
  single_llm   -> outputs/single_llm/         field: response
  selfreflect  -> outputs/selfreflect/         field: final_response
  gemma_diff   -> judge_outputs/multi_agent_diff/          field: judge_decision
  gemma_same   -> judge_outputs/multi_agent_same/          field: judge_decision
  gpt_diff     -> judge_outputs/gpt_nano/multi_agent_diff/ field: judge_decision
  gpt_same     -> judge_outputs/gpt_nano/multi_agent_same/ field: judge_decision
  indic_diff   -> indic_judge_outputs/Krutrim-2-instruct/multi_agent_diff/  field: judge_decision
  indic_same   -> indic_judge_outputs/Krutrim-2-instruct/multi_agent_same/  field: judge_decision

Task categories:
  judgement: bias_judgement_negative, bias_judgement_positive
  plausible:  bias_plausible_negative,  bias_plausible_positive
"""

import json
import argparse
from pathlib import Path

import pandas as pd

from elo_utils import (
    parse_judgement,
    parse_plausible_scenario,
    compute_elo_ranking_mle,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JUDGEMENT_TASKS = ["bias_judgement_negative", "bias_judgement_positive"]
PLAUSIBLE_TASKS = ["bias_plausible_negative",  "bias_plausible_positive"]

SOURCES = {
    "single_llm":  {"base": "outputs/single_llm",                                          "field": "response"},
    "selfreflect": {"base": "outputs/selfreflect",                                          "field": "final_response"},
    "gemma_diff":  {"base": "judge_outputs/multi_agent_diff",                               "field": "judge_decision"},
    "gemma_same":  {"base": "judge_outputs/multi_agent_same",                               "field": "judge_decision"},
    "gpt_diff":    {"base": "judge_outputs/gpt_nano/multi_agent_diff",                      "field": "judge_decision"},
    "gpt_same":    {"base": "judge_outputs/gpt_nano/multi_agent_same",                      "field": "judge_decision"},
    "indic_diff":  {"base": "indic_judge_outputs/Krutrim-2-instruct/multi_agent_diff",      "field": "judge_decision"},
    "indic_same":  {"base": "indic_judge_outputs/Krutrim-2-instruct/multi_agent_same",      "field": "judge_decision"},
}

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _read_jsonl(fpath: Path, response_field: str) -> list:
    """Read one JSONL file, normalising the response field to 'response'."""
    records = []
    with open(fpath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # Fix "identity:" key bug present in some records
            if "identity" not in rec and "identity:" in rec:
                rec["identity"] = rec.pop("identity:")
            rec["response"] = rec.get(response_field, "")
            records.append(rec)
    return records


def _parse_and_elo(records: list, label: str) -> tuple:
    """Run parse + ELO on a list of records.  Returns (elo_series, matches_df)."""
    if not records:
        return None, None

    parse_fn = parse_judgement if label == "judgement" else parse_plausible_scenario
    parsed = parse_fn(records, original_data=None)

    rows = []
    for item in parsed:
        identity = item["meta_data"]["identity"]
        rows.append({
            "identity_1": identity.get("identity_1", ""),
            "identity_2": identity.get("identity_2", ""),
            "winner":     item["winner"],
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return None, df

    try:
        elo = compute_elo_ranking_mle(df)
        return elo, df
    except Exception as e:
        print(f"    ELO failed: {e}")
        return None, df


# ---------------------------------------------------------------------------
# Aggregate mode  (one ELO per source × task category)
# ---------------------------------------------------------------------------

def run_aggregate(sources: list, task_map: dict, out_dir: Path):
    summary_rows = []
    all_elo = {}   # {identity: {col: score}}

    for source_name in sources:
        cfg = SOURCES[source_name]
        src_dir = out_dir / source_name
        src_dir.mkdir(parents=True, exist_ok=True)

        for label, tasks in task_map.items():
            print(f"\n--- {source_name} | {label} ---")

            # Pool all files across both tasks for this category
            records = []
            for task in tasks:
                task_dir = Path(cfg["base"]) / task
                if not task_dir.exists():
                    print(f"  [SKIP] {task_dir} not found")
                    continue
                for fpath in sorted(task_dir.glob("*.jsonl")):
                    records.extend(_read_jsonl(fpath, cfg["field"]))

            print(f"  Loaded {len(records)} records")
            elo, df = _parse_and_elo(records, label)

            if elo is None:
                summary_rows.append({"source": source_name, "task": label,
                                     "n_records": len(records), "n_identities": 0,
                                     "top_identity": "N/A", "top_elo": None})
                continue

            print(f"  ELO computed for {len(elo)} identities")

            # Save per-source ELO JSON
            elo_path = src_dir / f"{label}_elo.json"
            with open(elo_path, "w", encoding="utf-8") as f:
                json.dump(elo.to_dict(), f, indent=2, ensure_ascii=False)

            df.to_csv(src_dir / f"{label}_matches.csv", index=False)

            col = f"{source_name}_{label}"
            for identity, score in elo.items():
                all_elo.setdefault(identity, {})[col] = round(float(score), 2)

            summary_rows.append({
                "source": source_name, "task": label,
                "n_records": len(df), "n_identities": len(elo),
                "top_identity": elo.index[0], "top_elo": round(float(elo.iloc[0]), 2),
            })

    _save_summary(summary_rows, out_dir, "summary.csv")
    _save_combined(all_elo, out_dir, "all_elo_combined.csv")


# ---------------------------------------------------------------------------
# Per-file mode  (one ELO per model file × task category)
# ---------------------------------------------------------------------------

def run_per_file(sources: list, task_map: dict, out_dir: Path,
                 per_subtask: bool = False):
    """
    For each source, compute one ELO per individual model file.

    per_subtask=False (default):
      Pool neg+pos for each label category (judgement / plausible).
      Output: {out_dir}/{source}/{label}/{model_stem}_elo.json
              {out_dir}/{source}/{label}_per_model.csv

    per_subtask=True:
      Each of the 4 tasks is computed individually.
      Output: {out_dir}/{source}/{task}/{model_stem}_elo.json
              {out_dir}/{source}/{task}_per_model.csv
    """
    summary_rows = []

    # Build the iteration units: {unit_name: (tasks_list, parse_label)}
    if per_subtask:
        units = {}
        for label, tasks in task_map.items():
            for task in tasks:
                units[task] = (task, label)   # (single_task, judgement|plausible)
    else:
        units = {label: (None, label) for label in task_map}  # pool neg+pos

    for source_name in sources:
        cfg = SOURCES[source_name]
        src_dir = out_dir / source_name
        src_dir.mkdir(parents=True, exist_ok=True)

        for unit_name, (single_task, parse_label) in units.items():
            # tasks to pool together for this unit
            if per_subtask:
                tasks_to_load = [single_task]
            else:
                tasks_to_load = task_map[parse_label]

            unit_dir = src_dir / unit_name
            unit_dir.mkdir(parents=True, exist_ok=True)

            # Collect all file stems across tasks_to_load
            stems: set[str] = set()
            for task in tasks_to_load:
                task_dir = Path(cfg["base"]) / task
                if task_dir.exists():
                    for f in task_dir.glob("*.jsonl"):
                        stems.add(f.stem)

            if not stems:
                print(f"\n  [SKIP] no files for {source_name} | {unit_name}")
                continue

            print(f"\n=== {source_name} | {unit_name}  ({len(stems)} model files) ===")

            per_model_elo: dict[str, dict] = {}

            for stem in sorted(stems):
                records = []
                for task in tasks_to_load:
                    fpath = Path(cfg["base"]) / task / f"{stem}.jsonl"
                    if fpath.exists():
                        records.extend(_read_jsonl(fpath, cfg["field"]))

                if not records:
                    continue

                elo, df = _parse_and_elo(records, parse_label)

                win_dist = df["winner"].value_counts().to_dict() if df is not None else {}
                print(f"  {stem}: {len(records)} records  winners={win_dist}")

                if elo is None:
                    summary_rows.append({
                        "source": source_name, "task": unit_name, "model": stem,
                        "n_records": len(records), "n_identities": 0,
                        "top_identity": "N/A", "top_elo": None,
                    })
                    continue

                elo_path = unit_dir / f"{stem}_elo.json"
                with open(elo_path, "w", encoding="utf-8") as f:
                    json.dump(elo.to_dict(), f, indent=2, ensure_ascii=False)

                per_model_elo[stem] = {identity: round(float(score), 2)
                                       for identity, score in elo.items()}

                summary_rows.append({
                    "source": source_name, "task": unit_name, "model": stem,
                    "n_records": len(records), "n_identities": len(elo),
                    "top_identity": elo.index[0], "top_elo": round(float(elo.iloc[0]), 2),
                })

            if per_model_elo:
                combined = pd.DataFrame.from_dict(per_model_elo, orient="columns").sort_index()
                combined_path = src_dir / f"{unit_name}_per_model.csv"
                combined.to_csv(combined_path)
                print(f"  Combined table → {combined_path}")
                print(combined.to_string())

    _save_summary(summary_rows, out_dir, "summary_per_file.csv",
                  extra_cols=["model"])


# ---------------------------------------------------------------------------
# Shared output helpers
# ---------------------------------------------------------------------------

def _save_summary(rows: list, out_dir: Path, fname: str, extra_cols: list = None):
    if not rows:
        return
    df = pd.DataFrame(rows)
    path = out_dir / fname
    df.to_csv(path, index=False)
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(df.to_string(index=False))
    print(f"\nSaved → {path}")


def _save_combined(all_elo: dict, out_dir: Path, fname: str):
    if not all_elo:
        return
    df = pd.DataFrame.from_dict(all_elo, orient="index").sort_index()
    path = out_dir / fname
    df.to_csv(path)
    print(f"\nCombined ELO table → {path}")
    print(df.to_string())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute bias ELO rankings")
    parser.add_argument("--output-dir", default="elo_results",
                        help="Root output directory (default: elo_results)")
    parser.add_argument("--sources", nargs="+", default=list(SOURCES.keys()),
                        choices=list(SOURCES.keys()),
                        help="Sources to include (default: all 8)")
    parser.add_argument("--tasks", nargs="+", default=["judgement", "plausible"],
                        choices=["judgement", "plausible"],
                        help="Task categories (default: both)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--aggregate", dest="mode", action="store_const", const="aggregate",
                      help="One ELO per source × task category (pool all files)")
    mode.add_argument("--per-file", dest="mode", action="store_const", const="per_file",
                      help="One ELO per model file, neg+pos pooled per category")
    mode.add_argument("--per-subtask", dest="mode", action="store_const", const="per_subtask",
                      help="One ELO per model file × individual subtask (neg/pos separately)")
    parser.set_defaults(mode="per_subtask")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_map = {}
    if "judgement" in args.tasks:
        task_map["judgement"] = JUDGEMENT_TASKS
    if "plausible" in args.tasks:
        task_map["plausible"] = PLAUSIBLE_TASKS

    if args.mode == "aggregate":
        run_aggregate(args.sources, task_map, out_dir)
    elif args.mode == "per_subtask":
        run_per_file(args.sources, task_map, out_dir, per_subtask=True)
    else:
        run_per_file(args.sources, task_map, out_dir, per_subtask=False)


if __name__ == "__main__":
    main()
