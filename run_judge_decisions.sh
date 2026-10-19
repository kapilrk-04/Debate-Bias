#!/usr/bin/env bash
# Run generate_judge_decisions.py for each judge model over all input variants.
# Outputs are written to judge_outputs/<model-slug>/ to keep judge results separate.

set -euo pipefail

SCRIPT="generate_judge_decisions.py"
INPUT_BASE="outputs"
OUTPUT_BASE="indic_judge_outputs"

JUDGE_MODELS=(
    # "google/gemma-3-12b-it"
    "krutrim-ai-labs/Krutrim-2-instruct"
)

INPUT_DIRS=(
    "${INPUT_BASE}/multi_agent_diff"
    "${INPUT_BASE}/multi_agent_same"
)

for JUDGE_MODEL in "${JUDGE_MODELS[@]}"; do
    MODEL_SLUG=$(basename "${JUDGE_MODEL}")
    OUTPUT_DIR="${OUTPUT_BASE}/${MODEL_SLUG}"

    echo "=========================================="
    echo "Judge Model : ${JUDGE_MODEL}"
    echo "Output Dir  : ${OUTPUT_DIR}"
    echo "=========================================="

    python "${SCRIPT}" \
        --judge-model "${JUDGE_MODEL}" \
        --input-dirs  "${INPUT_DIRS[@]}" \
        --output-dir  "${OUTPUT_DIR}"

    echo ""
done

echo "All judge decisions complete."
