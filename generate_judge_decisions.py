# -*- coding: utf-8 -*-
"""
Judge evaluation script using vLLM to evaluate multi-agent debate outcomes.

Processes debate results from multi_agent_same or multi_agent_diff outputs
and generates judge's final decisions using Google Gemma-3-12B-IT model via vLLM.
"""

import torch
from datetime import datetime
import json
import os
import argparse
import logging
import gc
from huggingface_hub import login
import yaml
import judge_prompts as judge_module
from model_config_loader import load_model_config
from chat_template_utils import format_messages_for_model

# Load secrets from secrets.yaml
def load_secrets():
    """Load secrets from secrets.yaml file."""
    secrets_path = "secrets.yaml"
    if not os.path.exists(secrets_path):
        raise FileNotFoundError(
            f"secrets.yaml not found. Please copy secrets.yaml.example to secrets.yaml "
            f"and fill in your credentials."
        )
    with open(secrets_path, "r") as f:
        return yaml.safe_load(f)

secrets = load_secrets()
hf_token = secrets.get("HF_TOKEN")

# Set HF_HOME environment variable if specified
if "HF_HOME" in secrets:
    os.environ["HF_HOME"] = secrets["HF_HOME"]

# Setup logging
os.makedirs("logs", exist_ok=True)
now_str = datetime.now().strftime("%d-%m-%y_%H-%M-%S")
logging.basicConfig(
    level=logging.INFO,
    filename=f"logs/judge_evaluation_{now_str}.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

if hf_token:
    login(token=hf_token)
    print("Successfully logged in to Hugging Face Hub.")
else:
    print("Warning: HF_TOKEN environment variable is not set.")


def generate_judge_decisions_batched(llm, tokenizer, judge_prompts_list, model_config):
    """Generate judge decisions for all prompts in a single batched vLLM call."""
    from vllm import SamplingParams

    formatted_prompts = [
        format_messages_for_model(
            tokenizer=tokenizer,
            model_path=model_config.get("model_path", ""),
            messages=[{"role": "user", "content": p}],
            add_generation_prompt=True,
            enable_thinking=model_config.get("enable_thinking"),
        )
        for p in judge_prompts_list
    ]

    sampling_params = SamplingParams(
        temperature=0.3,
        top_p=1.0,
        max_tokens=256
    )

    outputs = llm.generate(formatted_prompts, sampling_params)
    return [o.outputs[0].text.strip() for o in outputs]


def load_model_and_tokenizer(model_path, hf_home, mcfg):
    """Load model and tokenizer for vLLM."""
    trust_remote_code = mcfg.get("trust_remote_code", False)
    
    attn_backend = mcfg.get("attention_backend")
    if attn_backend:
        os.environ["VLLM_ATTENTION_BACKEND"] = attn_backend
        logging.info(f"Using attention backend override for {model_path}: {attn_backend}")
        print(f"Using attention backend override for {model_path}: {attn_backend}")
    
    from vllm import LLM
    from transformers import AutoTokenizer
    
    logging.info(f"Loading {model_path} with vLLM")
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        dtype=mcfg.get("dtype", "auto"),
        tensor_parallel_size=1,
        max_num_seqs=mcfg.get("max_num_seqs", 16),
        gpu_memory_utilization=mcfg["gpu_memory_utilization"],
        max_model_len=mcfg["max_model_len"],
        download_dir=hf_home,
        trust_remote_code=trust_remote_code,
        enforce_eager=mcfg.get("enforce_eager", False),
        hf_overrides=mcfg.get("hf_overrides", {})
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=hf_home,
        trust_remote_code=trust_remote_code
    )
    return llm, tokenizer


def unload_model(model):
    """Unload model and free GPU memory."""
    try:
        model.llm_engine.shutdown()
    except Exception:
        pass
    
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    logging.info("Model unloaded and GPU memory freed")
    print("  Model unloaded, GPU memory freed.")


def collect_input_files(input_dirs, output_base):
    """
    Walk each input directory recursively and return a list of
    (input_path, output_path) pairs for every .jsonl file found.
    Output paths mirror the input directory structure under output_base.
    Already-processed files (output exists) are skipped.
    """
    pairs = []
    for input_dir in input_dirs:
        input_dir = os.path.abspath(input_dir)
        for root, _, files in os.walk(input_dir):
            for fname in sorted(files):
                if not fname.endswith(".jsonl"):
                    continue
                input_path = os.path.join(root, fname)
                rel = os.path.relpath(input_path, os.path.dirname(input_dir))
                output_path = os.path.join(output_base, rel)
                if os.path.exists(output_path):
                    print(f"Skipping (already done): {input_path}")
                    continue
                pairs.append((input_path, output_path))
    return pairs


def process_file(input_path, output_path, judge_llm, judge_tokenizer,
                 judge_config, judge_model_path):
    """Load one JSONL file, run batched judge inference, save results."""
    logging.info(f"Processing {input_path}")
    print(f"\n  Input : {input_path}")
    print(f"  Output: {output_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        debates = [json.loads(line) for line in f]
    print(f"  Loaded {len(debates)} debate records")

    judge_prompt_template = judge_module.judge_prompts["judge_decision"]
    judge_prompts_list = []
    for debate in debates:
        p = judge_prompt_template
        p = p.replace("{{prompt}}", debate.get("prompt", ""))
        p = p.replace("{{model1_response}}", debate.get("agent1_response_1", ""))
        p = p.replace("{{model2_response}}", debate.get("agent2_response_1", ""))
        p = p.replace("{{model1_feedback}}", debate.get("agent1_feedback", ""))
        p = p.replace("{{model2_feedback}}", debate.get("agent2_feedback", ""))
        p = p.replace("{{model1_final}}", debate.get("agent1_final_response", ""))
        p = p.replace("{{model2_final}}", debate.get("agent2_final_response", ""))
        judge_prompts_list.append(p)

    decisions = generate_judge_decisions_batched(
        judge_llm, judge_tokenizer, judge_prompts_list, judge_config
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for debate, decision in zip(debates, decisions):
            debate["judge_decision"] = decision
            debate["judge_model"] = judge_model_path
            f.write(json.dumps(debate, ensure_ascii=False) + "\n")

    print(f"  Saved {len(debates)} records.")
    logging.info(f"Saved {len(debates)} records to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Judge Evaluation for Multi-Agent Debates")
    parser.add_argument(
        "--judge-model",
        type=str,
        default="google/gemma-3-12b-it",
        help=(
            "Judge model path or HuggingFace model ID. "
            "Supported: google/gemma-3-12b-it, krutrim-ai-labs/Krutrim-2-instruct, "
            "or any model defined in model_configs.yaml."
        )
    )
    parser.add_argument(
        "--input-dirs",
        type=str,
        nargs="+",
        default=["outputs/multi_agent_diff", "outputs/multi_agent_same"],
        help="One or more directories to scan recursively for input JSONL files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="judge_outputs",
        help="Base output directory (mirrors input directory structure)"
    )

    args = parser.parse_args()
    logging.info(f"\nCLI Arguments:\n{vars(args)}")

    judge_model_path = args.judge_model
    print(f"Judge Model: {judge_model_path}")

    # Load judge model config
    judge_config = load_model_config(judge_model_path)
    judge_config["model_path"] = judge_model_path
    logging.info(f"Judge model config: {judge_config}")
    print(f"Judge model config: {judge_config}")

    hf_home = os.getenv("HF_HOME", None)

    # Collect all input/output file pairs
    file_pairs = collect_input_files(args.input_dirs, args.output_dir)
    if not file_pairs:
        print("No new files to process. Exiting.")
        return
    print(f"\nFound {len(file_pairs)} file(s) to process.")

    # Load judge model once
    print(f"\n{'='*60}")
    print(f"Loading judge model: {judge_model_path}")
    print(f"{'='*60}")
    judge_llm, judge_tokenizer = load_model_and_tokenizer(
        judge_model_path, hf_home, judge_config
    )

    # Process all files with the single loaded model
    for i, (input_path, output_path) in enumerate(file_pairs, 1):
        print(f"\n[{i}/{len(file_pairs)}]")
        process_file(input_path, output_path, judge_llm, judge_tokenizer,
                     judge_config, judge_model_path)

    unload_model(judge_llm)
    print(f"\nAll done. Processed {len(file_pairs)} file(s).")


if __name__ == "__main__":
    main()
