#!/bin/bash

# Response Generation Job Script
# Runs single LLM, self-reflection, and multi-agent debate scenarios
# Per-model vLLM settings (max_model_len, gpu_memory_utilization, etc.)
# are loaded from model_configs.yaml by each Python script.

set -e  # Exit on error

# Define models to test
MODELS=(
    "Qwen/Qwen3-4B-Instruct-2507"
    "google/gemma-3-4b-it"
    "CohereLabs/aya-23-8B"
    "nvidia/Nemotron-4-Mini-Hindi-4B-Instruct"
    "CoRover/BharatGPT-3B-Indic"
    "ai4bharat/Airavata"
    "meta-llama/Llama-3.2-3B-Instruct"
)

# Define prompt files to process
PROMPT_FILES=(
    "processed_prompts/bias_judgement_negative.jsonl"
    "processed_prompts/bias_judgement_positive.jsonl"
    "processed_prompts/bias_plausible_negative.jsonl"
    "processed_prompts/bias_plausible_positive.jsonl"
    "processed_prompts/stereotype_judgement.jsonl"
    "processed_prompts/stereotype_plausible.jsonl"
)

# Configuration
BACKEND="vllm"  # or "transformers"
MAX_TOKENS=512
BATCH_SIZE=16
TEMPERATURE=0.0
TOP_P=1.0

# Create output directories
mkdir -p outputs/single_llm
mkdir -p outputs/selfreflect
mkdir -p outputs/multi_agent_same
mkdir -p outputs/multi_agent_diff
mkdir -p logs

echo "========================================"
echo "Starting Response Generation Pipeline"
echo "Backend: $BACKEND"
echo "========================================"
echo "Initial GPU state:"
nvidia-smi || true

# ============================================================================
# PART 1: Single LLM Baseline
# ============================================================================
echo ""
echo "=========================================="
echo "PART 1: Single LLM Baseline"
echo "=========================================="

for MODEL in "${MODELS[@]}"; do
    MODEL_SAFE="${MODEL//\//_}"

    for PROMPT_FILE in "${PROMPT_FILES[@]}"; do
        PROMPT_NAME=$(basename "$PROMPT_FILE" .jsonl)
        OUTPUT_FILE="outputs/single_llm/${PROMPT_NAME}/${MODEL_SAFE}_responses.jsonl"

        # Create output directory
        mkdir -p "outputs/single_llm/${PROMPT_NAME}"

        # Skip if already exists
        if [ -f "$OUTPUT_FILE" ]; then
            echo "Skipping $MODEL on $PROMPT_NAME (already exists)"
            continue
        fi

        echo ""
        echo "Running Single LLM: $MODEL"
        echo "Prompt File: $PROMPT_FILE"
        echo "Output: $OUTPUT_FILE"

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

        echo "✓ Completed: $MODEL on $PROMPT_NAME"
    done
done

# ============================================================================
# PART 2: Self-Reflection
# ============================================================================
# echo ""
# echo "=========================================="
# echo "PART 2: Self-Reflection"
# echo "=========================================="

# for MODEL in "${MODELS[@]}"; do
#     MODEL_SAFE="${MODEL//\//_}"

#     for PROMPT_FILE in "${PROMPT_FILES[@]}"; do
#         PROMPT_NAME=$(basename "$PROMPT_FILE" .jsonl)
#         OUTPUT_FILE="outputs/selfreflect/${PROMPT_NAME}/${MODEL_SAFE}_reflected.jsonl"

#         # Create output directory
#         mkdir -p "outputs/selfreflect/${PROMPT_NAME}"

#         # Skip if already exists
#         if [ -f "$OUTPUT_FILE" ]; then
#             echo "Skipping $MODEL on $PROMPT_NAME (already exists)"
#             continue
#         fi

#         echo ""
#         echo "Running Self-Reflection: $MODEL"
#         echo "Prompt File: $PROMPT_FILE"
#         echo "Output: $OUTPUT_FILE"

#         python generate_selfreflect_llm.py \
#             --model-path "$MODEL" \
#             --input-file "$PROMPT_FILE" \
#             --output-file "$OUTPUT_FILE" \
#             --backend "$BACKEND" \
#             --max-tokens $MAX_TOKENS \
#             --temperature $TEMPERATURE \
#             --top-p $TOP_P

#         echo "✓ Completed: $MODEL self-reflection on $PROMPT_NAME"
#     done
# done

# # ============================================================================
# # PART 3: Multi-Agent Debate (Same Model)
# # ============================================================================
# echo ""
# echo "=========================================="
# echo "PART 3: Multi-Agent Debate (Same Model)"
# echo "=========================================="

# for MODEL in "${MODELS[@]}"; do
#     MODEL_SAFE="${MODEL//\//_}"

#     for PROMPT_FILE in "${PROMPT_FILES[@]}"; do
#         PROMPT_NAME=$(basename "$PROMPT_FILE" .jsonl)
#         OUTPUT_FILE="outputs/multi_agent_same/${PROMPT_NAME}/${MODEL_SAFE}_debate.jsonl"

#         # Create output directory
#         mkdir -p "outputs/multi_agent_same/${PROMPT_NAME}"

#         # Skip if already exists
#         if [ -f "$OUTPUT_FILE" ]; then
#             echo "Skipping $MODEL on $PROMPT_NAME (already exists)"
#             continue
#         fi

#         echo ""
#         echo "Running Multi-Agent Debate (Same Model): $MODEL"
#         echo "Prompt File: $PROMPT_FILE"
#         echo "Output: $OUTPUT_FILE"
#         echo "GPU state before run:"
#         nvidia-smi || true

#         python generate_multi_agent_sequential.py \
#             --model1-path "$MODEL" \
#             --same-model \
#             --input-file "$PROMPT_FILE" \
#             --output-file "$OUTPUT_FILE" \
#             --backend "$BACKEND" \
#             --max-tokens $MAX_TOKENS \
#             --temperature $TEMPERATURE \
#             --top-p $TOP_P

#         echo "✓ Completed: $MODEL same-model debate on $PROMPT_NAME"
#     done
# done

# # ============================================================================
# # PART 4: Multi-Agent Debate (Different Models)
# ============================================================================
# echo ""
# echo "=========================================="
# echo "PART 4: Multi-Agent Debate (Different Models)"
# echo "=========================================="

# # Define model pairs for debate
# MODEL_PAIRS=(
#     "Qwen/Qwen3-4B-Instruct-2507:ai4bharat/Airavata"
#     "Qwen/Qwen3-4B-Instruct-2507:CoRover/BharatGPT-3B-Indic"
#     "Qwen/Qwen3-4B-Instruct-2507:nvidia/Nemotron-4-Mini-Hindi-4B-Instruct"
#     "meta-llama/Llama-3.2-3B-Instruct:ai4bharat/Airavata"
#     "meta-llama/Llama-3.2-3B-Instruct:CoRover/BharatGPT-3B-Indic"
#     "meta-llama/Llama-3.2-3B-Instruct:nvidia/Nemotron-4-Mini-Hindi-4B-Instruct"
#     "google/gemma-3-4b-it:ai4bharat/Airavata"
#     "google/gemma-3-4b-it:CoRover/BharatGPT-3B-Indic"
#     "google/gemma-3-4b-it:nvidia/Nemotron-4-Mini-Hindi-4B-Instruct"
#     "CohereLabs/aya-23-8B:ai4bharat/Airavata"
#     "CohereLabs/aya-23-8B:CoRover/BharatGPT-3B-Indic"
#     "CohereLabs/aya-23-8B:nvidia/Nemotron-4-Mini-Hindi-4B-Instruct"
# )

# for PAIR in "${MODEL_PAIRS[@]}"; do
#     MODEL1="${PAIR%%:*}"
#     MODEL2="${PAIR##*:}"
#     MODEL1_SAFE="${MODEL1//\//_}"
#     MODEL2_SAFE="${MODEL2//\//_}"

#     for PROMPT_FILE in "${PROMPT_FILES[@]}"; do
#         PROMPT_NAME=$(basename "$PROMPT_FILE" .jsonl)
#         OUTPUT_FILE="outputs/multi_agent_diff/${PROMPT_NAME}/${MODEL1_SAFE}_vs_${MODEL2_SAFE}_debate.jsonl"

#         # Create output directory
#         mkdir -p "outputs/multi_agent_diff/${PROMPT_NAME}"

#         # Skip if already exists
#         if [ -f "$OUTPUT_FILE" ]; then
#             echo "Skipping $MODEL1 vs $MODEL2 on $PROMPT_NAME (already exists)"
#             continue
#         fi

#         echo ""
#         echo "Running Multi-Agent Debate: $MODEL1 vs $MODEL2"
#         echo "Prompt File: $PROMPT_FILE"
#         echo "Output: $OUTPUT_FILE"

#         python generate_multi_agent_sequential.py \
#             --model1-path "$MODEL1" \
#             --model2-path "$MODEL2" \
#             --input-file "$PROMPT_FILE" \
#             --output-file "$OUTPUT_FILE" \
#             --backend "$BACKEND" \
#             --max-tokens $MAX_TOKENS \
#             --temperature $TEMPERATURE \
#             --top-p $TOP_P

#         echo "✓ Completed: $MODEL1 vs $MODEL2 debate on $PROMPT_NAME"
#     done
# done

echo ""
echo "========================================"
echo "Response Generation Pipeline Complete!"
echo "========================================"
echo "Results saved in:"
echo "  - outputs/single_llm/"
# echo "  - outputs/selfreflect/"
# echo "  - outputs/multi_agent_same/"
# echo "  - outputs/multi_agent_diff/"
echo "========================================"
