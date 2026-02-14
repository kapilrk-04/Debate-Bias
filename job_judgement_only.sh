#!/bin/bash

# Response Generation Job Script - Judgment Tasks Only
# Runs single LLM, self-reflection, and multi-agent debate scenarios on judgment tasks

set -e  # Exit on error

# Define models to test
MODELS=(
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "CohereLabs/aya-23-8B"
    "google/gemma-3-4b-it"
    "nvidia/Nemotron-4-Mini-Hindi-4B-Instruct"
    "MBZUAI/Llama-3-Nanda-10B-Chat"
)

# Configuration
BACKEND="vllm"  # or "transformers"
MAX_TOKENS=512
BATCH_SIZE=16
TEMPERATURE=0.0
TOP_P=1.0
TENSOR_PARALLEL_SIZE=4
GPU_MEMORY_UTIL=0.85

# Create output directories
mkdir -p outputs/single_llm
mkdir -p outputs/selfreflect
mkdir -p outputs/multi_agent_same
mkdir -p logs

echo "========================================"
echo "Starting Judgment Task Pipeline"
echo "Backend: $BACKEND"
echo "========================================"

# Process only judgment files
for PROMPT_FILE in processed_prompts/*judgement*.jsonl; do
    if [ ! -f "$PROMPT_FILE" ]; then
        continue
    fi

    PROMPT_NAME=$(basename "$PROMPT_FILE" .jsonl)
    echo ""
    echo "=========================================="
    echo "Processing: $PROMPT_NAME"
    echo "=========================================="

    # ============================================================================
    # PART 1: Single LLM Baseline
    # ============================================================================
    echo ""
    echo "PART 1: Single LLM Baseline"

    for MODEL in "${MODELS[@]}"; do
        MODEL_SAFE="${MODEL//\//_}"
        OUTPUT_FILE="outputs/single_llm/${PROMPT_NAME}/${MODEL_SAFE}_responses.jsonl"

        mkdir -p "outputs/single_llm/${PROMPT_NAME}"

        if [ -f "$OUTPUT_FILE" ]; then
            echo "  Skipping $MODEL (already exists)"
            continue
        fi

        echo "  Running: $MODEL"

        python generate_single_llm.py \
            --model-path "$MODEL" \
            --input-file "$PROMPT_FILE" \
            --output-file "$OUTPUT_FILE" \
            --backend "$BACKEND" \
            --max-tokens $MAX_TOKENS \
            --batch-size $BATCH_SIZE \
            --temperature $TEMPERATURE \
            --top-p $TOP_P \
            --prompt-type direct

        echo "  ✓ Completed"
    done

    # ============================================================================
    # PART 2: Self-Reflection
    # ============================================================================
    echo ""
    echo "PART 2: Self-Reflection"

    for MODEL in "${MODELS[@]}"; do
        MODEL_SAFE="${MODEL//\//_}"
        OUTPUT_FILE="outputs/selfreflect/${PROMPT_NAME}/${MODEL_SAFE}_reflected.jsonl"

        mkdir -p "outputs/selfreflect/${PROMPT_NAME}"

        if [ -f "$OUTPUT_FILE" ]; then
            echo "  Skipping $MODEL (already exists)"
            continue
        fi

        echo "  Running: $MODEL"

        python generate_selfreflect_llm.py \
            --model-path "$MODEL" \
            --input-file "$PROMPT_FILE" \
            --output-file "$OUTPUT_FILE" \
            --backend "$BACKEND" \
            --max-tokens $MAX_TOKENS \
            --temperature $TEMPERATURE \
            --top-p $TOP_P

        echo "  ✓ Completed"
    done

    # ============================================================================
    # PART 3: Multi-Agent Debate (Same Model)
    # ============================================================================
    echo ""
    echo "PART 3: Multi-Agent Debate (Same Model)"

    for MODEL in "${MODELS[@]}"; do
        MODEL_SAFE="${MODEL//\//_}"
        OUTPUT_FILE="outputs/multi_agent_same/${PROMPT_NAME}/${MODEL_SAFE}_debate.jsonl"

        mkdir -p "outputs/multi_agent_same/${PROMPT_NAME}"

        if [ -f "$OUTPUT_FILE" ]; then
            echo "  Skipping $MODEL (already exists)"
            continue
        fi

        echo "  Running: $MODEL"

        python generate_multi_agent.py \
            --model1-path "$MODEL" \
            --same-model \
            --input-file "$PROMPT_FILE" \
            --output-file "$OUTPUT_FILE" \
            --backend "$BACKEND" \
            --max-tokens $MAX_TOKENS \
            --temperature $TEMPERATURE \
            --top-p $TOP_P

        echo "  ✓ Completed"
    done
done

echo ""
echo "========================================"
echo "Judgment Task Pipeline Complete!"
echo "========================================"
