#!/usr/bin/env python3
"""
Compute pairwise acceptance (agreement) rates across judges.

comparison_set_diff  → 4 judges: gemma-3-12b-it, gpt-5-nano, krutrim-2, human
comparison_set_same  → 3 judges: gemma-3-12b-it, gpt-5-nano, krutrim-2

Acceptance rate for a pair (A, B) = fraction of rows where A and B agree.
Reported overall and broken down by task_category and individual task.

LLM judges store '1' / '2' / 'Neither'.
Human judge stores identity names → normalised to '1' / '2' / 'Neither'
using identity_1 / identity_2 from the diff JSONL.

Output:
  pairwise_acceptance_diff.csv
  pairwise_acceptance_same.csv
"""

import json
import csv
import argparse
from itertools import combinations
from pathlib import Path
from collections import defaultdict

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LLM_JUDGES = {
    "gemma":   "judge_gemma_3_12b_it",
    "gpt":     "judge_gpt_5_nano",
    "krutrim": "judge_krutrim_2",
}
HUMAN_JUDGE = "judge_human"

NEITHER_ALIASES = {"neither", "none", "n/a", "", "id_missing", "file_missing"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_label(value: str, id1: str, id2: str) -> str:
    """Map any judge output → '1', '2', or 'Neither'."""
    v = str(value).strip()
    if v in ("1", "2", "Neither"):
        return v
    vl = v.lower()
    if vl in NEITHER_ALIASES:
        return "Neither"
    if id1 and vl == id1.lower():
        return "1"
    if id2 and vl == id2.lower():
        return "2"
    return "Neither"


def load_diff(jsonl_path: str, human_csv_path: str) -> list[dict]:
    """
    Load comparison_set_diff records joined with human annotations.
    Returns list of dicts with normalised judge columns added.
    """
    with open(jsonl_path, encoding="utf-8") as f:
        records = {json.loads(l)["id"]: json.loads(l) for l in f if l.strip()}

    human = {}
    with open(human_csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            human[row["id"]] = row[HUMAN_JUDGE]

    rows = []
    for rid, rec in records.items():
        id1 = rec.get("identity_1", "")
        id2 = rec.get("identity_2", "")

        row = {
            "id":            rid,
            "task":          rec.get("task", ""),
            "task_category": rec.get("task_category", ""),
        }
        for short, col in LLM_JUDGES.items():
            row[short] = normalize_label(rec.get(col, ""), id1, id2)

        if rid in human:
            row["human"] = normalize_label(human[rid], id1, id2)
        else:
            row["human"] = None   # missing annotation

        rows.append(row)

    missing = sum(1 for r in rows if r["human"] is None)
    if missing:
        print(f"  Warning: {missing} diff records have no human annotation")
    return rows


def load_same(jsonl_path: str) -> list[dict]:
    """Load comparison_set_same records (LLM judges only)."""
    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            id1 = rec.get("identity_1", "")
            id2 = rec.get("identity_2", "")
            row = {
                "id":            rec["id"],
                "task":          rec.get("task", ""),
                "task_category": rec.get("task_category", ""),
            }
            for short, col in LLM_JUDGES.items():
                row[short] = normalize_label(rec.get(col, ""), id1, id2)
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Acceptance rate computation
# ---------------------------------------------------------------------------

def acceptance_rate(rows: list[dict], judge_a: str, judge_b: str,
                    filter_key: str = None, filter_val: str = None) -> float | None:
    """
    Fraction of rows where judge_a and judge_b agree.
    Optionally filter to rows where row[filter_key] == filter_val.
    Rows where either judge is None are excluded.
    """
    subset = rows
    if filter_key and filter_val:
        subset = [r for r in rows if r.get(filter_key) == filter_val]

    valid = [r for r in subset if r.get(judge_a) is not None and r.get(judge_b) is not None]
    if not valid:
        return None
    agree = sum(1 for r in valid if r[judge_a] == r[judge_b])
    return round(agree / len(valid), 4)


def build_table(rows: list[dict], judge_names: list[str]) -> pd.DataFrame:
    """
    Build acceptance-rate table.
    Rows = judge pairs.
    Columns = overall + each task_category + each task.
    """
    pairs = list(combinations(judge_names, 2))

    tasks = sorted({r["task"] for r in rows if r["task"]})
    categories = sorted({r["task_category"] for r in rows if r["task_category"]})

    result_rows = []
    for a, b in pairs:
        entry = {"judge_a": a, "judge_b": b}
        entry["overall"] = acceptance_rate(rows, a, b)

        for cat in categories:
            entry[f"cat_{cat}"] = acceptance_rate(rows, a, b, "task_category", cat)

        for task in tasks:
            entry[f"task_{task}"] = acceptance_rate(rows, a, b, "task", task)

        result_rows.append(entry)

    return pd.DataFrame(result_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute pairwise judge acceptance rates")
    parser.add_argument("--diff-jsonl",  default="comparison_set_diff.jsonl")
    parser.add_argument("--diff-human",  default="comparison_set_diff_human.csv")
    parser.add_argument("--same-jsonl",  default="comparison_set_same.jsonl")
    parser.add_argument("--output-dir",  default=".")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── DIFF: 4 judges ──────────────────────────────────────────────────────
    print("Loading comparison_set_diff …")
    diff_rows = load_diff(args.diff_jsonl, args.diff_human)
    print(f"  {len(diff_rows)} rows loaded")

    diff_judges = ["gemma", "gpt", "krutrim", "human"]
    diff_table  = build_table(diff_rows, diff_judges)

    diff_path = out / "pairwise_acceptance_diff.csv"
    diff_table.to_csv(diff_path, index=False)

    print("\n=== DIFF pairwise acceptance rates ===")
    print(diff_table.to_string(index=False))
    print(f"\nSaved → {diff_path}")

    # ── SAME: 3 LLM judges ──────────────────────────────────────────────────
    print("\nLoading comparison_set_same …")
    same_rows   = load_same(args.same_jsonl)
    print(f"  {len(same_rows)} rows loaded")

    same_judges = ["gemma", "gpt", "krutrim"]
    same_table  = build_table(same_rows, same_judges)

    same_path = out / "pairwise_acceptance_same.csv"
    same_table.to_csv(same_path, index=False)

    print("\n=== SAME pairwise acceptance rates ===")
    print(same_table.to_string(index=False))
    print(f"\nSaved → {same_path}")

    # ── Quick summary: overall rates side by side ────────────────────────────
    print("\n=== OVERALL acceptance rates (side by side) ===")
    merged = pd.merge(
        diff_table[["judge_a", "judge_b", "overall"]].rename(columns={"overall": "diff_overall"}),
        same_table[["judge_a", "judge_b", "overall"]].rename(columns={"overall": "same_overall"}),
        on=["judge_a", "judge_b"], how="outer"
    )
    print(merged.to_string(index=False))


if __name__ == "__main__":
    main()
