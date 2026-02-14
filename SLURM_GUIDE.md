# SLURM Job Submission Guide

## Overview

This guide explains how to run the response generation pipeline on SLURM clusters.

## Files Created

### 1. Main Job Script
- **`job_response_generation.sh`** - Comprehensive pipeline that runs all scenarios on all prompt files

### 2. Judgment-Only Script
- **`job_judgement_only.sh`** - Runs only on judgment tasks (files containing "judgement" in filename)

### 3. SLURM Submission Script
- **`train.slurm`** - Updated to run the response generation pipeline

## Configuration

### Models to Test

Both scripts are configured to run with these models (edit the `MODELS` array in the scripts):

```bash
MODELS=(
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "CohereLabs/aya-23-8B"
    "google/gemma-3-4b-it"
    "nvidia/Nemotron-4-Mini-Hindi-4B-Instruct"
    "MBZUAI/Llama-3-Nanda-10B-Chat"
)
```

### Prompt Files

The scripts automatically process files from `processed_prompts/`:
- `bias_judgement_negative.jsonl`
- `bias_judgement_positive.jsonl`
- `bias_plausible_negative.jsonl`
- `bias_plausible_positive.jsonl`
- `stereotype_judgement.jsonl`
- `stereotype_plausible.jsonl`

### Backend Configuration

Default settings in the scripts:
```bash
BACKEND="vllm"           # or "transformers"
MAX_TOKENS=512
BATCH_SIZE=16
TEMPERATURE=0.0
TOP_P=1.0
```

## Usage

### 1. Make Scripts Executable

```bash
chmod +x job_response_generation.sh
chmod +x job_judgement_only.sh
```

### 2. Submit SLURM Job

```bash
sbatch train.slurm
```

### 3. Or Run Scripts Directly (if not using SLURM)

```bash
# Run all scenarios on all prompts
./job_response_generation.sh

# Run only judgment tasks
./job_judgement_only.sh
```

## Pipeline Stages

### Stage 1: Single LLM Baseline
- Each model generates responses directly without any deliberation
- Output: `outputs/single_llm/{prompt_name}/{model}_responses.jsonl`

### Stage 2: Self-Reflection
- Model generates initial response, reflects on it, then provides final answer
- Output: `outputs/selfreflect/{prompt_name}/{model}_reflected.jsonl`

### Stage 3: Multi-Agent Debate (Same Model)
- Two instances of the same model debate before providing answers
- Output: `outputs/multi_agent_same/{prompt_name}/{model}_debate.jsonl`

### Stage 4: Multi-Agent Debate (Different Models)
- Two different models debate before providing answers
- Only in `job_response_generation.sh`
- Output: `outputs/multi_agent_diff/{prompt_name}/{model1}_vs_{model2}_debate.jsonl`

## Output Structure

```
outputs/
├── single_llm/
│   ├── bias_judgement_negative/
│   │   ├── meta-llama_Meta-Llama-3-8B-Instruct_responses.jsonl
│   │   └── ...
│   └── stereotype_judgement/
│       └── ...
├── selfreflect/
│   ├── bias_judgement_negative/
│   │   ├── meta-llama_Meta-Llama-3-8B-Instruct_reflected.jsonl
│   │   └── ...
│   └── ...
├── multi_agent_same/
│   ├── bias_judgement_negative/
│   │   ├── meta-llama_Meta-Llama-3-8B-Instruct_debate.jsonl
│   │   └── ...
│   └── ...
└── multi_agent_diff/
    ├── bias_judgement_negative/
    │   ├── meta-llama_Meta-Llama-3-8B-Instruct_vs_CohereLabs_aya-23-8B_debate.jsonl
    │   └── ...
    └── ...
```

## Logs

Logs are saved in:
- SLURM logs: `logs/surakshaeval_vllm_responses_{job_id}.out`
- Python logs: `logs/single_llm_*.log`, `logs/selfreflect_llm_*.log`, `logs/multi_agent_*.log`

## Customization

### Adding More Models

Edit the `MODELS` array in the bash scripts:

```bash
MODELS=(
    # Add your models here
    "your-org/your-model"
)
```

### Adding More Model Pairs (for different model debates)

In `job_response_generation.sh`, edit the `MODEL_PAIRS` array:

```bash
MODEL_PAIRS=(
    "model1:model2"
    "model3:model4"
)
```

### Changing Backend

Edit the `BACKEND` variable:

```bash
BACKEND="transformers"  # or "vllm"
```

### Processing Only Specific Files

To process only plausible tasks, modify the loop:

```bash
# In job_response_generation.sh or job_judgement_only.sh
for PROMPT_FILE in processed_prompts/*plausible*.jsonl; do
    ...
done
```

## Resume Support

Both scripts check if output files already exist and skip them:

```bash
if [ -f "$OUTPUT_FILE" ]; then
    echo "Skipping (already exists)"
    continue
fi
```

This allows you to:
- Resume interrupted jobs
- Add new models without re-running existing ones
- Process new prompt files without re-processing old ones

## Resource Requirements

### SLURM Configuration (from train.slurm)

```bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia-rtx-5000-ada-generation:1
#SBATCH --mem=80G
#SBATCH --time=20:00:00
```

### For Larger Models

If running models larger than 10B parameters, you may need:
- More GPU memory: `#SBATCH --mem=160G`
- Multiple GPUs: `#SBATCH --gres=gpu:4`
- Longer time: `#SBATCH --time=96:00:00`

Adjust `TENSOR_PARALLEL_SIZE` in the scripts for multi-GPU:

```bash
# For 4 GPUs
python generate_single_llm.py \
    ... \
    --tensor-parallel-size 4
```

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# Check specific job
squeue -j JOB_ID

# View real-time output
tail -f logs/surakshaeval_vllm_responses_*.out

# Cancel job if needed
scancel JOB_ID
```

## Troubleshooting

### Out of Memory

- Reduce `--batch-size`
- Lower `--gpu-memory-utilization` (e.g., from 0.85 to 0.7)
- Use `BACKEND="transformers"` instead of vLLM
- Request more GPU memory in SLURM

### Job Timeout

- Increase `#SBATCH --time=`
- Process fewer models per job
- Split into multiple SLURM jobs

### Import Errors

Make sure your conda environment is activated in `train.slurm`:
```bash
source /path/to/miniconda3/bin/activate your_env
```

## Example: Custom Run

To run only a specific model on specific prompts:

```bash
#!/bin/bash

MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
PROMPT="processed_prompts/bias_judgement_negative.jsonl"

# Single LLM
python generate_single_llm.py \
    --model-path "$MODEL" \
    --input-file "$PROMPT" \
    --output-file "outputs/custom_single.jsonl" \
    --backend vllm

# Self-reflection
python generate_selfreflect_llm.py \
    --model-path "$MODEL" \
    --input-file "$PROMPT" \
    --output-file "outputs/custom_reflect.jsonl" \
    --backend vllm

# Multi-agent (same model)
python generate_multi_agent.py \
    --model1-path "$MODEL" \
    --same-model \
    --input-file "$PROMPT" \
    --output-file "outputs/custom_debate.jsonl" \
    --backend vllm
```
