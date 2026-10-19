#!/usr/bin/env python3
"""
Extract a cross-judge comparison dataset.

Three judge systems compared side by side:
  - gemma-3-12b-it : judge_outputs/multi_agent_{diff,same}/
  - gpt-5-nano     : judge_outputs/gpt_nano/multi_agent_{diff,same}/
  - krutrim-2      : indic_judge_outputs/Krutrim-2-instruct/multi_agent_{diff,same}/

Output: one row per selected record with all three judge decisions.

Sampling (seeded, reproducible):
  - 180 bias rows   — 45 per bias task (× 4 tasks), split 23 diff + 22 same per task
  - 180 stereotype  — 90 per stereo task (× 2 tasks), split 45 diff + 45 same per task
  - Prompt IDs UNIQUE across all rows (no id reused across diff/same or tasks).

Strategy: use ONE representative model-pair file per (task, format) — the same
filename across all three judge bases — so we load ~12 files instead of ~300.
"""

import json
import csv
import os
import random
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42

BIAS_TASKS  = ["bias_judgement_negative", "bias_judgement_positive",
                "bias_plausible_negative",  "bias_plausible_positive"]
STEREO_TASKS = ["stereotype_judgement", "stereotype_plausible"]

# Base directories for each judge (format → base_path)
JUDGE_BASES = {
    "gemma-3-12b-it": {
        "diff": "judge_outputs/multi_agent_diff",
        "same": "judge_outputs/multi_agent_same",
    },
    "gpt-5-nano": {
        "diff": "judge_outputs/gpt_nano/multi_agent_diff",
        "same": "judge_outputs/gpt_nano/multi_agent_same",
    },
    "krutrim-2": {
        "diff": "indic_judge_outputs/Krutrim-2-instruct/multi_agent_diff",
        "same": "indic_judge_outputs/Krutrim-2-instruct/multi_agent_same",
    },
}

REFERENCE_JUDGE = "gemma-3-12b-it"   # used to drive sampling

BIAS_PER_TASK   = 62   # 4 × 62 = 248 bias rows
STEREO_PER_TASK = 126  # 2 × 126 = 252 stereo rows  → 500 total
BIAS_DIFF       = 31; BIAS_SAME   = 31   # 4×31 = 124 diff + 124 same from bias
STEREO_DIFF     = 63; STEREO_SAME = 63   # 2×63 = 126 diff + 126 same from stereo
# diff total = 124+126 = 250,  same total = 124+126 = 250


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def all_files(base: str, task: str) -> list[Path]:
    """Return all .jsonl files for a task dir, sorted alphabetically."""
    d = Path(base) / task
    files = sorted(d.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No .jsonl files in {d}")
    return files


def load_file_index(fpath: Path) -> dict[str, dict]:
    """Load a .jsonl file, return {id: record}."""
    idx = {}
    with open(fpath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            idx[rec["id"]] = rec
    return idx


def get_identity(rec: dict) -> tuple[str, str]:
    identity = rec.get("identity") or rec.get("identity:") or {}
    return identity.get("identity_1", ""), identity.get("identity_2", "")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract a cross-judge comparison dataset"
    )
    parser.add_argument("--output",       default="comparison_set.csv",
                        help="Base CSV name — _diff/_same suffixes added (default: comparison_set.csv)")
    parser.add_argument("--output-jsonl", default="comparison_set.jsonl",
                        help="Base JSONL name — _diff/_same suffixes added (default: comparison_set.jsonl)")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"Random seed (default: {SEED})")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = []
    used_ids: set[str] = set()   # global uniqueness guard across all strata

    # ---- For each task + format, pick the reference file (same name used
    #      in all three judge systems).  Load reference once for sampling,
    #      then load the same file from the other two judges for decisions. ----

    def fetch_record(rid: str, fpath: Path) -> dict:
        """Stream-search a file for a record by id."""
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec["id"] == rid:
                    return rec
        return {}

    def process_task(task: str, n_diff: int, n_same: int):
        """
        Sample n_diff records from diff and n_same from same for a given task.
        IDs are sampled from a single reference file (all files share the same
        IDs), then each ID is assigned a different model-pair file by cycling
        through the full file list for diversity.
        """
        ref_base_diff = JUDGE_BASES[REFERENCE_JUDGE]["diff"]
        ref_base_same = JUDGE_BASES[REFERENCE_JUDGE]["same"]

        # All model-pair files for this task (same names exist in all judges)
        diff_files = all_files(ref_base_diff, task)   # e.g. 12 files for diff
        same_files = all_files(ref_base_same, task)   # e.g.  7 files for same

        # Load one file just to get the universe of IDs
        all_ids_diff = sorted(load_file_index(diff_files[0]).keys())
        all_ids_same = sorted(load_file_index(same_files[0]).keys())

        # Available ids (exclude already used)
        avail_diff = [i for i in all_ids_diff if i not in used_ids]
        diff_sampled = rng.sample(avail_diff, n_diff)
        diff_set = set(diff_sampled)

        avail_same = [i for i in all_ids_same if i not in used_ids and i not in diff_set]
        same_sampled = rng.sample(avail_same, n_same)

        def make_row(rid: str, fmt: str, assigned_file: Path) -> dict:
            fname = assigned_file.name
            # Fetch base record (agent responses live here)
            base_rec = fetch_record(rid, assigned_file)
            id1, id2 = get_identity(base_rec)

            # Fetch judge decision from each system using the same filename
            decisions = {}
            for judge, bases in JUDGE_BASES.items():
                judge_path = Path(bases[fmt]) / task / fname
                if not judge_path.exists():
                    decisions[judge] = "FILE_MISSING"
                    continue
                rec = fetch_record(rid, judge_path)
                decisions[judge] = rec.get("judge_decision", "ID_MISSING") if rec else "ID_MISSING"

            return {
                "id":                        base_rec.get("id", ""),
                "template_id":               base_rec.get("template_id", ""),
                "task":                      task,
                "task_category":             "bias" if task in BIAS_TASKS else "stereotype",
                "format":                    fmt,
                "model_pair_file":           fname,
                "prompt":                    base_rec.get("prompt", "").strip(),
                "identity_1":                id1,
                "identity_2":                id2,
                "model1":                    base_rec.get("model1", ""),
                "model2":                    base_rec.get("model2", ""),
                # Round 1 responses
                "agent1_response_1":         base_rec.get("agent1_response_1", "").strip(),
                "agent2_response_1":         base_rec.get("agent2_response_1", "").strip(),
                # Feedback (cross-agent critique)
                "agent1_feedback":           base_rec.get("agent1_feedback", "").strip(),
                "agent2_feedback":           base_rec.get("agent2_feedback", "").strip(),
                # Final responses after reflection
                "agent1_final_response":     base_rec.get("agent1_final_response", "").strip(),
                "agent2_final_response":     base_rec.get("agent2_final_response", "").strip(),
                # Judge decisions
                "judge_gemma_3_12b_it":      decisions.get("gemma-3-12b-it", ""),
                "judge_gpt_5_nano":          decisions.get("gpt-5-nano", ""),
                "judge_krutrim_2":           decisions.get("krutrim-2", ""),
            }

        # Cycle through model-pair files to spread diversity
        for i, rid in enumerate(diff_sampled):
            assigned = diff_files[i % len(diff_files)]
            rows.append(make_row(rid, "diff", assigned))
            used_ids.add(rid)

        for i, rid in enumerate(same_sampled):
            assigned = same_files[i % len(same_files)]
            rows.append(make_row(rid, "same", assigned))
            used_ids.add(rid)

        diff_pairs = {diff_files[i % len(diff_files)].name for i in range(n_diff)}
        same_pairs = {same_files[i % len(same_files)].name for i in range(n_same)}
        print(f"  {task}: {n_diff} diff ({len(diff_pairs)} model pairs) + "
              f"{n_same} same ({len(same_pairs)} models)")

    print("Sampling bias tasks ...")
    for task in BIAS_TASKS:
        process_task(task, BIAS_DIFF, BIAS_SAME)

    print("\nSampling stereotype tasks ...")
    for task in STEREO_TASKS:
        process_task(task, STEREO_DIFF, STEREO_SAME)

    # ---- verify ----
    all_ids = [r["id"] for r in rows]
    assert len(all_ids) == len(set(all_ids)), "Duplicate ids in output!"
    bias_n   = sum(1 for r in rows if r["task_category"] == "bias")
    stereo_n = sum(1 for r in rows if r["task_category"] == "stereotype")
    print(f"\nTotal rows : {len(rows)}  (bias={bias_n}, stereotype={stereo_n})")
    print(f"Unique ids : {len(set(all_ids))}  ✓")

    # ---- split by format ----
    diff_rows  = [r for r in rows if r["format"] == "diff"]
    same_rows  = [r for r in rows if r["format"] == "same"]

    def stem(path: str) -> tuple[str, str]:
        """Return (base, ext) splitting on first dot."""
        p = Path(path)
        return str(p.parent / p.stem), p.suffix

    def write_pair(subset: list[dict], csv_path: str, jsonl_path: str, label: str):
        if not subset:
            return
        fieldnames = list(subset[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(subset)
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for row in subset:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  [{label}] {len(subset)} rows → {csv_path}  |  {jsonl_path}")

    csv_stem,  csv_ext  = stem(args.output)
    jl_stem,   jl_ext   = stem(args.output_jsonl)

    print("\nWriting output files ...")
    write_pair(diff_rows,
               f"{csv_stem}_diff{csv_ext}", f"{jl_stem}_diff{jl_ext}", "diff")
    write_pair(same_rows,
               f"{csv_stem}_same{csv_ext}", f"{jl_stem}_same{jl_ext}", "same")


if __name__ == "__main__":
    main()
