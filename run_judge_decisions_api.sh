#!/usr/bin/env bash
# Run API-based judge decision generation over all JSONL files in
# outputs/multi_agent_diff and outputs/multi_agent_same.
#
# Usage:
#   ./run_judge_decisions_api.sh            # run GPT-5-Nano judge
#   ./run_judge_decisions_api.sh gpt        # run GPT-5-Nano judge
#   ./run_judge_decisions_api.sh --debug    # debug mode (first 10 prompts per file)

set -euo pipefail

JUDGE="${1:-gpt}"
DEBUG_FLAG=""

# Allow --debug as first or second arg
for arg in "$@"; do
    if [[ "$arg" == "--debug" ]]; then
        DEBUG_FLAG="--debug"
        [[ "$1" == "--debug" ]] && JUDGE="gpt"
    fi
done

INPUT_DIRS="outputs/multi_agent_diff outputs/multi_agent_same"

run_gpt() {
    echo "=========================================="
    echo "Running GPT-5-Nano judge"
    echo "  Input dirs : ${INPUT_DIRS}"
    echo "  Output dir : judge_outputs/gpt_nano"
    echo "  Batch dir  : batches/judge_gpt"
    echo "=========================================="
    python generate_judge_decisions_gpt.py \
        --input-dirs ${INPUT_DIRS} \
        --output-dir judge_outputs/gpt_nano \
        --batch-dir  batches/judge_gpt \
        ${DEBUG_FLAG}
    echo "GPT-5-Nano judge complete."
}

case "${JUDGE}" in
    gpt)
        run_gpt
        ;;
    *)
        echo "Unknown argument: ${JUDGE}"
        echo "Usage: $0 [gpt] [--debug]"
        exit 1
        ;;
esac

echo "All judge decisions complete."
