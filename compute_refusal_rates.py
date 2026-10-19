#!/usr/bin/env python3
"""
Compute "Neither" refusal rates for each model × subtask × source.

Refusal = response is exactly "Neither" (case-sensitive).

Sources:
  single_llm       — outputs/single_llm/{task}/*_responses.jsonl   → field: "response"
  selfreflect      — outputs/selfreflect/{task}/*_reflected.jsonl  → field: "final_response"
  multi_agent_diff — outputs/multi_agent_diff/{task}/*_debate.jsonl
                     → agent1_final_response (model1) + agent2_final_response (model2)
  multi_agent_same — outputs/multi_agent_same/{task}/*_debate.jsonl
                     → agent1_final_response + agent2_final_response (both = same model)

Outputs:
  refusal_rates.csv              — long format: source, task, model, total, neither_count, refusal_rate
  refusal_rates_pivot.csv        — pivot: model × task, refusal_rate per source (multi-index columns)
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

# model1/model2 fields in multi_agent files use "/" separator
MODEL_ID_TO_DISPLAY = {k.replace("_", "/"): v for k, v in FILE_TO_MODEL.items()}

TASKS = [
    "bias_judgement_negative",
    "bias_judgement_positive",
    "bias_plausible_negative",
    "bias_plausible_positive",
    "stereotype_judgement",
    "stereotype_plausible",
]

SOURCE_TO_DISPLAY = {
    "single_llm":       "Single-LLM",
    "selfreflect":      "Self-Reflect",
    "multi_agent_diff": "Debate (Diff)",
    "multi_agent_same": "Debate (Same)",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stem_to_model(stem: str, suffix: str) -> str:
    key = re.sub(rf"_{suffix}$", "", stem)
    return FILE_TO_MODEL.get(key, key)


def is_neither(response: str) -> bool:
    return str(response).strip() == "Neither"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_single(base_dir: str, tasks: list, response_field: str, stem_suffix: str) -> list:
    """Load single-model response files (single_llm or selfreflect)."""
    rows = []
    for task in tasks:
        d = Path(base_dir) / task
        if not d.exists():
            continue
        for fp in sorted(d.glob(f"*_{stem_suffix}.jsonl")):
            model = _stem_to_model(fp.stem, stem_suffix)
            total = neither = 0
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    resp = obj.get(response_field, "")
                    if resp is None:
                        continue
                    total += 1
                    if is_neither(resp):
                        neither += 1
            if total > 0:
                rows.append({
                    "task":          task,
                    "model":         model,
                    "total":         total,
                    "neither_count": neither,
                    "refusal_rate":  round(neither / total, 4),
                })
    return rows


def load_multi_agent(base_dir: str, tasks: list, same: bool) -> list:
    """
    Load multi-agent debate files.
    Diff files (same=False): track by model pair "M1 vs M2" — both agent
      responses pooled together as one observation unit per pair.
    Same files (same=True): both agents are the same model — track per model.
    """
    # accumulate counts: (task, model_or_pair) -> [total, neither]
    counts: dict = defaultdict(lambda: [0, 0])

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
                    m1_id = obj.get("model1", "")
                    m2_id = obj.get("model2", "")
                    m1 = MODEL_ID_TO_DISPLAY.get(m1_id, m1_id)
                    m2 = MODEL_ID_TO_DISPLAY.get(m2_id, m2_id)
                    r1 = obj.get("agent1_final_response", "")
                    r2 = obj.get("agent2_final_response", "")

                    if not same:
                        # diff: key is the pair label; pool both responses
                        pair = f"{m1} vs {m2}"
                        if r1 is not None:
                            counts[(task, pair)][0] += 1
                            if is_neither(r1):
                                counts[(task, pair)][1] += 1
                        if r2 is not None:
                            counts[(task, pair)][0] += 1
                            if is_neither(r2):
                                counts[(task, pair)][1] += 1
                    else:
                        # same: key is individual model (m1 == m2)
                        if r1 is not None:
                            counts[(task, m1)][0] += 1
                            if is_neither(r1):
                                counts[(task, m1)][1] += 1
                        if r2 is not None:
                            counts[(task, m2)][0] += 1
                            if is_neither(r2):
                                counts[(task, m2)][1] += 1

    rows = []
    for (task, model), (total, neither) in counts.items():
        if total > 0:
            rows.append({
                "task":          task,
                "model":         model,
                "total":         total,
                "neither_count": neither,
                "refusal_rate":  round(neither / total, 4),
            })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute Neither refusal rates per model × subtask")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--out-dir",     default="refusal_results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = Path(args.outputs_dir)

    all_rows = []

    source_configs = [
        ("single_llm",       base / "single_llm",       "response",        "responses"),
        ("selfreflect",      base / "selfreflect",       "final_response",  "reflected"),
    ]

    for source_key, src_base, field, suffix in source_configs:
        rows = load_single(str(src_base), TASKS, field, suffix)
        for r in rows:
            r["source"]         = source_key
            r["source_display"] = SOURCE_TO_DISPLAY[source_key]
        all_rows.extend(rows)

    for source_key, src_base, same in [
        ("multi_agent_diff", base / "multi_agent_diff", False),
        ("multi_agent_same", base / "multi_agent_same", True),
    ]:
        rows = load_multi_agent(str(src_base), TASKS, same)
        for r in rows:
            r["source"]         = source_key
            r["source_display"] = SOURCE_TO_DISPLAY[source_key]
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows, columns=[
        "source", "source_display", "task", "model",
        "total", "neither_count", "refusal_rate",
    ]).sort_values(["source", "task", "model"]).reset_index(drop=True)

    # ── long CSV ──────────────────────────────────────────────────────────
    long_path = out_dir / "refusal_rates.csv"
    df.to_csv(long_path, index=False)
    print(f"Saved → {long_path}")

    # ── print summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("REFUSAL RATES (% 'Neither' responses per model × subtask × source)")
    print("=" * 70)
    for source_key in ["single_llm", "selfreflect", "multi_agent_diff", "multi_agent_same"]:
        sub = df[df["source"] == source_key]
        if sub.empty:
            continue
        label = SOURCE_TO_DISPLAY[source_key]
        print(f"\n── {label} ──")
        pivot = sub.pivot_table(index="model", columns="task",
                                values="refusal_rate", aggfunc="mean").round(4)
        # reorder columns to canonical task order
        task_order = [t for t in TASKS if t in pivot.columns]
        pivot = pivot[task_order]
        print(pivot.to_string())

    # ── pivot CSV: model × (source, task) ─────────────────────────────────
    pivot_df = df.pivot_table(
        index="model",
        columns=["source_display", "task"],
        values="refusal_rate",
        aggfunc="mean",
    ).round(4)
    pivot_path = out_dir / "refusal_rates_pivot.csv"
    pivot_df.to_csv(pivot_path)
    print(f"\nSaved pivot → {pivot_path}")


if __name__ == "__main__":
    main()
