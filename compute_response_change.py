#!/usr/bin/env python3
"""
Compute response change rates between initial and final agent responses.

  selfreflect      — outputs/selfreflect/{task}/*_reflected.jsonl
                     response_1_formatted  →  final_response   (1 agent)

  multi_agent_diff — judge_outputs/multi_agent_diff/{task}/*_debate.jsonl
  multi_agent_same — judge_outputs/multi_agent_same/{task}/*_debate.jsonl
                     agent{1,2}_initial_response_formatted
                       →  agent{1,2}_final_response_formatted  (2 agents)

  Diff row key = "M1 vs M2" pair label.
  Same row key = individual model name.

Change  = initial.strip() != final.strip()
Rate    = n_changed / n_total

Output:
  refusal_results/response_change_rates.csv
  refusal_results/response_change_pivot.csv
"""

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd

# ---------------------------------------------------------------------------
# Name mappings
# ---------------------------------------------------------------------------

FILE_TO_MODEL = {
    "CohereLabs_aya-23-8B":                        "Aya-23-8B",
    "CoRover_BharatGPT-3B-Indic":                  "BharatGPT-3B",
    "Qwen_Qwen3-4B-Instruct-2507":                 "Qwen3-4B",
    "ai4bharat_Airavata":                          "Airavata-7B",
    "google_gemma-3-4b-it":                        "Gemma3-4B-IT",
    "meta-llama_Llama-3.2-3B-Instruct":            "Llama-3.2-3B",
    "nvidia_Nemotron-4-Mini-Hindi-4B-Instruct":    "Nemo-Hindi-4B",
}

MODEL_ID_TO_DISPLAY = {k.replace("_", "/"): v for k, v in FILE_TO_MODEL.items()}

TASKS = [
    "bias_judgement_negative",
    "bias_judgement_positive",
    "bias_plausible_negative",
    "bias_plausible_positive",
    "stereotype_judgement",
    "stereotype_plausible",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stem_to_model(stem: str, suffix: str) -> str:
    key = re.sub(rf"_{suffix}$", "", stem)
    return FILE_TO_MODEL.get(key, key)


def _changed(a, b) -> bool:
    return str(a).strip() != str(b).strip()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_selfreflect(base_dir: str, tasks: list) -> list:
    """
    Compare response_1_formatted (initial) vs final_response (final).
    One row per (task, model).
    """
    rows = []
    for task in tasks:
        d = Path(base_dir) / task
        if not d.exists():
            continue
        for fp in sorted(d.glob("*_reflected.jsonl")):
            model = _stem_to_model(fp.stem, "reflected")
            total = changed = 0
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    r_init  = obj.get("response_1_formatted")
                    r_final = obj.get("final_response")
                    if r_init is None or r_final is None:
                        continue
                    total += 1
                    if _changed(r_init, r_final):
                        changed += 1
            if total > 0:
                rows.append({
                    "source":      "selfreflect",
                    "task":        task,
                    "model":       model,
                    "agent":       "agent",
                    "total":       total,
                    "changed":     changed,
                    "change_rate": round(changed / total, 4),
                })
    return rows


def load_multi_agent(base_dir: str, tasks: list, source_key: str, same: bool) -> list:
    """
    Compare agent{1,2}_initial_response_formatted vs agent{1,2}_final_response_formatted.
    Diff: row key = "M1 vs M2".  Same: row key = individual model name.
    Returns two rows per (task, key): one for agent1, one for agent2.
    """
    counts: dict = defaultdict(lambda: [0, 0])   # (task, key, agent) -> [total, changed]

    for task in tasks:
        d = Path(base_dir) / task
        if not d.exists():
            continue
        for fp in sorted(d.glob("*_debate.jsonl")):
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    m1 = MODEL_ID_TO_DISPLAY.get(obj.get("model1", ""), obj.get("model1", ""))
                    m2 = MODEL_ID_TO_DISPLAY.get(obj.get("model2", ""), obj.get("model2", ""))

                    row_key = f"{m1} vs {m2}" if not same else m1

                    for agent_label, init_field, final_field in [
                        ("agent1",
                         "agent1_initial_response_formatted",
                         "agent1_final_response_formatted"),
                        ("agent2",
                         "agent2_initial_response_formatted",
                         "agent2_final_response_formatted"),
                    ]:
                        r_init  = obj.get(init_field)
                        r_final = obj.get(final_field)
                        if r_init is None or r_final is None:
                            continue
                        counts[(task, row_key, agent_label)][0] += 1
                        if _changed(r_init, r_final):
                            counts[(task, row_key, agent_label)][1] += 1

    rows = []
    for (task, model, agent), (total, changed) in counts.items():
        if total > 0:
            rows.append({
                "source":      source_key,
                "task":        task,
                "model":       model,
                "agent":       agent,
                "total":       total,
                "changed":     changed,
                "change_rate": round(changed / total, 4),
            })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute response change rates")
    parser.add_argument("--selfreflect-dir",     default="outputs/selfreflect")
    parser.add_argument("--judge-outputs-dir",   default="judge_outputs")
    parser.add_argument("--out-dir",             default="refusal_results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    judge_base = Path(args.judge_outputs_dir)

    all_rows = []
    all_rows += load_selfreflect(args.selfreflect_dir, TASKS)
    all_rows += load_multi_agent(str(judge_base / "multi_agent_diff"), TASKS,
                                 "multi_agent_diff", same=False)
    all_rows += load_multi_agent(str(judge_base / "multi_agent_same"), TASKS,
                                 "multi_agent_same", same=True)

    df = pd.DataFrame(all_rows, columns=[
        "source", "task", "model", "agent", "total", "changed", "change_rate",
    ]).sort_values(["source", "task", "model", "agent"]).reset_index(drop=True)

    # ── long CSV ──────────────────────────────────────────────────────────
    long_path = out_dir / "response_change_rates.csv"
    df.to_csv(long_path, index=False)
    print(f"Saved → {long_path}")

    # ── printed summary ───────────────────────────────────────────────────
    TASK_ORDER = [
        "bias_judgement_negative", "bias_judgement_positive",
        "bias_plausible_negative", "bias_plausible_positive",
        "stereotype_judgement",    "stereotype_plausible",
    ]

    for source_key, source_label in [
        ("selfreflect",      "Self-Reflect  (response_1_formatted → final_response)"),
        ("multi_agent_diff", "Debate Diff   (initial_formatted → final_formatted, by pair)"),
        ("multi_agent_same", "Debate Same   (initial_formatted → final_formatted, by model)"),
    ]:
        sub = df[df["source"] == source_key]
        if sub.empty:
            continue
        print(f"\n{'='*70}")
        print(f"{source_label}")
        print(f"{'='*70}")
        for agent in sorted(sub["agent"].unique()):
            a_sub = sub[sub["agent"] == agent]
            pivot = a_sub.pivot_table(
                index="model", columns="task",
                values="change_rate", aggfunc="mean"
            ).round(4)
            task_cols = [t for t in TASK_ORDER if t in pivot.columns]
            pivot = pivot[task_cols]
            print(f"\n  [{agent}]")
            print(pivot.to_string())

    # ── wide pivot CSV ─────────────────────────────────────────────────────
    pivot_df = df.pivot_table(
        index="model",
        columns=["source", "agent", "task"],
        values="change_rate",
        aggfunc="mean",
    ).round(4)
    pivot_path = out_dir / "response_change_pivot.csv"
    pivot_df.to_csv(pivot_path)
    print(f"\nSaved pivot → {pivot_path}")


if __name__ == "__main__":
    main()
