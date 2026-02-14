# Usage Guide for Response Generation Scripts

This guide explains how to use the new generalized response generation scripts for judgment and plausibility tasks.

## Overview

Three new scripts have been created to handle different response generation scenarios:

1. **`generate_single_llm.py`** - Single LLM response generation
2. **`generate_selfreflect_llm.py`** - Self-reflection with a single LLM
3. **`generate_multi_agent.py`** - Multi-agent debate (2 LLMs or 2 instances of same LLM)

Each script supports both **vLLM** and **transformers** backends.

## Prompt Files

Three new prompt template files have been created:

- **`prompts_single_new.py`** - Templates for single-turn responses
- **`prompts_selfreflect_new.py`** - Templates for self-reflection (3 turns)
- **`prompts_multi_new.py`** - Templates for multi-agent debate (6 turns total)

## 1. Single LLM Response Generation

### Basic Usage

```bash
python generate_single_llm.py \
  --model-path "meta-llama/Meta-Llama-3-8B-Instruct" \
  --input-file "processed_prompts/bias_judgement_negative.jsonl" \
  --output-file "outputs/single_llm/bias_judgement_negative_responses.jsonl" \
  --backend transformers \
  --max-tokens 512
```

### With vLLM Backend

```bash
python generate_single_llm.py \
  --model-path "meta-llama/Meta-Llama-3-8B-Instruct" \
  --input-file "processed_prompts/stereotype_judgement.jsonl" \
  --output-file "outputs/single_llm/stereotype_judgement_responses.jsonl" \
  --backend vllm \
  --batch-size 16 \
  --max-tokens 512
```

### With Reasoning

```bash
python generate_single_llm.py \
  --model-path "meta-llama/Meta-Llama-3-8B-Instruct" \
  --input-file "processed_prompts/bias_plausible_negative.jsonl" \
  --output-file "outputs/single_llm/bias_plausible_negative_with_reasoning.jsonl" \
  --backend transformers \
  --prompt-type judgment_with_reasoning \
  --max-tokens 512
```

### Arguments

- `--model-path`: HuggingFace model ID or local path
- `--input-file`: Path to input JSONL file
- `--output-file`: Path to output JSONL file
- `--backend`: `vllm` or `transformers` (default: `transformers`)
- `--max-tokens`: Maximum tokens to generate (default: 512)
- `--prompt-type`: `direct`, `judgment_single`, `plausibility_single`, `judgment_with_reasoning`, `plausibility_with_reasoning` (default: `direct`)
- `--batch-size`: Batch size for vLLM (default: 8)
- `--temperature`: Sampling temperature (default: 0.0)
- `--top-p`: Top-p sampling (default: 1.0)

## 2. Self-Reflection Response Generation

The model generates an initial response, reflects on it, and then provides a final answer.

### Basic Usage

```bash
python generate_selfreflect_llm.py \
  --model-path "meta-llama/Meta-Llama-3-8B-Instruct" \
  --input-file "processed_prompts/bias_judgement_negative.jsonl" \
  --output-file "outputs/selfreflect/bias_judgement_negative_reflected.jsonl" \
  --backend transformers \
  --max-tokens 512
```

### With vLLM Backend

```bash
python generate_selfreflect_llm.py \
  --model-path "meta-llama/Meta-Llama-3-8B-Instruct" \
  --input-file "processed_prompts/stereotype_plausible.jsonl" \
  --output-file "outputs/selfreflect/stereotype_plausible_reflected.jsonl" \
  --backend vllm \
  --max-tokens 512
```

### Arguments

- `--model-path`: HuggingFace model ID or local path
- `--input-file`: Path to input JSONL file
- `--output-file`: Path to output JSONL file
- `--backend`: `vllm` or `transformers` (default: `transformers`)
- `--max-tokens`: Maximum tokens to generate (default: 512)
- `--temperature`: Sampling temperature (default: 0.0)
- `--top-p`: Top-p sampling (default: 1.0)

### Output Format

The output JSONL file contains:
- `response_1`: Initial response
- `reflection`: Self-reflection on initial response
- `final_response`: Final answer after reflection

## 3. Multi-Agent Debate

Two agents (LLMs) debate before providing their final answers.

### Two Different LLMs

```bash
python generate_multi_agent.py \
  --model1-path "meta-llama/Meta-Llama-3-8B-Instruct" \
  --model2-path "SeaLLMs/SeaLLMs-v3-7B-Chat" \
  --input-file "processed_prompts/bias_judgement_negative.jsonl" \
  --output-file "outputs/multi_agent/bias_judgement_llama_vs_seallm.jsonl" \
  --backend transformers \
  --max-tokens 512
```

### Same LLM, Two Instances

```bash
python generate_multi_agent.py \
  --model1-path "meta-llama/Meta-Llama-3-8B-Instruct" \
  --same-model \
  --input-file "processed_prompts/stereotype_judgement.jsonl" \
  --output-file "outputs/multi_agent/stereotype_judgement_llama_debate.jsonl" \
  --backend transformers \
  --max-tokens 512
```

### With vLLM Backend

```bash
python generate_multi_agent.py \
  --model1-path "meta-llama/Meta-Llama-3-8B-Instruct" \
  --model2-path "SeaLLMs/SeaLLMs-v3-7B-Chat" \
  --input-file "processed_prompts/bias_plausible_positive.jsonl" \
  --output-file "outputs/multi_agent/bias_plausible_positive_debate.jsonl" \
  --backend vllm \
  --max-tokens 512
```

### Arguments

- `--model1-path`: First model's HuggingFace ID or local path
- `--model2-path`: Second model's HuggingFace ID or local path (optional)
- `--same-model`: Use two instances of the same model (flag)
- `--input-file`: Path to input JSONL file
- `--output-file`: Path to output JSONL file
- `--backend`: `vllm` or `transformers` (default: `transformers`)
- `--max-tokens`: Maximum tokens to generate (default: 512)
- `--temperature`: Sampling temperature (default: 0.0)
- `--top-p`: Top-p sampling (default: 1.0)

### Output Format

The output JSONL file contains:
- `agent1_response_1`: Agent 1's initial response
- `agent2_response_1`: Agent 2's initial response
- `agent1_feedback`: Agent 1's feedback on Agent 2
- `agent2_feedback`: Agent 2's feedback on Agent 1
- `agent1_final_response`: Agent 1's final answer
- `agent2_final_response`: Agent 2's final answer

## Environment Setup

Make sure you have a `.env` file with:

```
HF_TOKEN=your_huggingface_token_here
HF_HOME=/path/to/your/huggingface/cache
```

## Input File Format

All scripts expect JSONL files with at least a `prompt` field:

```json
{"id": "...", "prompt": "Your question here...", ...}
```

## Output Files

All scripts generate JSONL files with:
- Original fields from input
- Generated responses and formatted prompts
- Model information

## Logs

Logs are saved in the `logs/` directory with timestamps:
- `logs/single_llm_<timestamp>.log`
- `logs/selfreflect_llm_<timestamp>.log`
- `logs/multi_agent_<timestamp>.log`

## Tips

1. **Backend Choice**:
   - Use `transformers` for smaller batches and more control
   - Use `vllm` for faster batch processing

2. **Memory Management**:
   - For multi-agent with different models, ensure you have enough GPU memory
   - The script uses `gpu_memory_utilization=0.45` for each model in vLLM mode

3. **Same Model Debate**:
   - With `transformers` backend, truly separate instances are used
   - With `vllm` backend, the same instance is reused (less true separation but more memory efficient)

## Example Workflow

```bash
# 1. Single LLM baseline
python generate_single_llm.py \
  --model-path "meta-llama/Meta-Llama-3-8B-Instruct" \
  --input-file "processed_prompts/bias_judgement_negative.jsonl" \
  --output-file "outputs/single/baseline.jsonl" \
  --backend transformers

# 2. Self-reflection
python generate_selfreflect_llm.py \
  --model-path "meta-llama/Meta-Llama-3-8B-Instruct" \
  --input-file "processed_prompts/bias_judgement_negative.jsonl" \
  --output-file "outputs/selfreflect/reflected.jsonl" \
  --backend transformers

# 3. Multi-agent debate (same model)
python generate_multi_agent.py \
  --model1-path "meta-llama/Meta-Llama-3-8B-Instruct" \
  --same-model \
  --input-file "processed_prompts/bias_judgement_negative.jsonl" \
  --output-file "outputs/multi/debate_same.jsonl" \
  --backend transformers

# 4. Multi-agent debate (different models)
python generate_multi_agent.py \
  --model1-path "meta-llama/Meta-Llama-3-8B-Instruct" \
  --model2-path "SeaLLMs/SeaLLMs-v3-7B-Chat" \
  --input-file "processed_prompts/bias_judgement_negative.jsonl" \
  --output-file "outputs/multi/debate_diff.jsonl" \
  --backend transformers
```
