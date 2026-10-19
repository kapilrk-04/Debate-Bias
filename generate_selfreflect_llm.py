# -*- coding: utf-8 -*-
"""
Self-reflection LLM response generation for judgment and plausibility tasks.
Supports both vLLM and transformers backends.
"""

import torch
from datetime import datetime
import json
import os
import argparse
from tqdm import tqdm
import logging
from huggingface_hub import login
import yaml
import prompts_selfreflect_new as prompts_module
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
    filename=f"logs/selfreflect_llm_{now_str}.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

if hf_token:
    login(token=hf_token)
    print("Successfully logged in to Hugging Face Hub.")
else:
    print("Warning: HF_TOKEN environment variable is not set.")


def format_chat_prompts(tokenizer, prompts, model_config):
    """Format raw prompts using chat template."""
    formatted_prompts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        formatted = format_messages_for_model(
            tokenizer=tokenizer,
            model_path=model_config.get("model_path", ""),
            messages=messages,
            add_generation_prompt=True,
            enable_thinking=model_config.get("enable_thinking"),
        )
        formatted_prompts.append(formatted)
    return formatted_prompts


def generate_batch_vllm(llm, tokenizer, prompts, sampling_params, model_config):
    """Generate a batch of responses using vLLM."""
    formatted_prompts = format_chat_prompts(tokenizer, prompts, model_config)
    outputs = llm.generate(formatted_prompts, sampling_params)
    responses = [o.outputs[0].text.strip() for o in outputs]
    return formatted_prompts, responses


def generate_batch_transformers(model, tokenizer, prompts, max_new_tokens, model_config):
    """Generate a batch of responses using transformers (sequential)."""
    formatted_prompts = []
    responses = []

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        formatted = format_messages_for_model(
            tokenizer=tokenizer,
            model_path=model_config.get("model_path", ""),
            messages=messages,
            add_generation_prompt=True,
            enable_thinking=model_config.get("enable_thinking"),
        )
        input_ids = tokenizer(formatted, return_tensors="pt").input_ids.to(model.device)

        terminators = [tokenizer.eos_token_id]
        if hasattr(tokenizer, 'convert_tokens_to_ids'):
            eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if eot_id is not None:
                terminators.append(eot_id)

        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=False,
            temperature=0.0,
        )

        response = outputs[0][input_ids.shape[-1]:]
        generation = tokenizer.decode(response, skip_special_tokens=True)
        formatted_prompts.append(formatted)
        responses.append(generation.strip())

    return formatted_prompts, responses


def initialize_vllm_engine(model_path, model_config):
    """Initialize vLLM with backend fallback for incompatible FlashAttention builds."""
    from vllm import LLM

    requested_backend = model_config.get("attention_backend")
    backend_candidates = [requested_backend] if requested_backend else []
    if requested_backend == "FLASH_ATTN":
        backend_candidates.append("TRITON_ATTN")
    if not backend_candidates:
        backend_candidates = [None]

    last_exc = None
    for backend in backend_candidates:
        if backend:
            os.environ["VLLM_ATTENTION_BACKEND"] = backend
            logging.info(f"Using attention backend override: {backend}")
            print(f"Using attention backend override: {backend}")
        try:
            llm = LLM(
                model=model_path,
                tokenizer=model_path,
                dtype=model_config.get("dtype", "auto"),
                tensor_parallel_size=1,
                max_num_seqs=model_config.get("max_num_seqs", 16),
                gpu_memory_utilization=model_config["gpu_memory_utilization"],
                max_model_len=model_config["max_model_len"],
                download_dir=os.getenv("HF_HOME", None),
                trust_remote_code=model_config.get("trust_remote_code", False),
                enforce_eager=model_config.get("enforce_eager", False),
                hf_overrides=model_config.get("hf_overrides", {})
            )
            return llm
        except RuntimeError as e:
            last_exc = e
            msg = str(e)
            if backend == "FLASH_ATTN" and "headdim not being a multiple of 32" in msg:
                logging.warning(
                    "FLASH_ATTN unsupported for %s (%s). Falling back to TRITON_ATTN.",
                    model_path, msg
                )
                print("FLASH_ATTN unsupported for this model; retrying with TRITON_ATTN.")
                continue
            raise

    raise last_exc


def main():
    parser = argparse.ArgumentParser(description="Self-Reflection LLM Response Generation")
    parser.add_argument("--model-path", type=str, required=True, help="Model path or HuggingFace model ID")
    parser.add_argument("--input-file", type=str, required=True, help="Input JSONL file path")
    parser.add_argument("--output-file", type=str, required=True, help="Output JSONL file path")
    parser.add_argument("--backend", type=str, choices=["vllm", "transformers"], default="transformers",
                        help="Backend to use for generation")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")

    args = parser.parse_args()
    logging.info(f"\nCLI Arguments:\n{vars(args)}")

    # Load per-model config from model_configs.yaml
    model_config = load_model_config(args.model_path)
    model_config["model_path"] = args.model_path
    logging.info(f"Model config: {model_config}")
    print(f"Model config for {args.model_path}: {model_config}")

    # Create output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Initialize model and tokenizer based on backend
    if args.backend == "vllm":
        from vllm import SamplingParams
        from transformers import AutoTokenizer

        logging.info(f"Loading {args.model_path} with vLLM")
        llm = initialize_vllm_engine(args.model_path, model_config)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            cache_dir=os.getenv("HF_HOME", None),
            trust_remote_code=model_config.get("trust_remote_code", False)
        )

        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens
        )

    else:  # transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM

        logging.info(f"Loading {args.model_path} with transformers")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            cache_dir=os.getenv("HF_HOME", None),
            trust_remote_code=model_config.get("trust_remote_code", False)
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            cache_dir=os.getenv("HF_HOME", None),
            device_map="auto",
            trust_remote_code=model_config.get("trust_remote_code", False)
        )

    # Load input data
    logging.info(f"Loading input from {args.input_file}")
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    logging.info(f"Loaded {len(data)} samples")

    # Phase-wise self-reflection (batch all prompts per phase).
    logging.info("[Phase 1/3] Initial decision for all samples")
    prompts_1 = [
        prompts_module.prompts["prompt_1"].replace("{{prompt}}", item["prompt"])
        for item in data
    ]
    if args.backend == "vllm":
        formatted_1_list, response_1_list = generate_batch_vllm(
            llm, tokenizer, prompts_1, sampling_params, model_config
        )
    else:
        formatted_1_list, response_1_list = generate_batch_transformers(
            model, tokenizer, prompts_1, args.max_tokens, model_config
        )

    logging.info("[Phase 2/3] Self-reflection for all samples")
    prompts_2 = [
        prompts_module.prompts["prompt_2"]
        .replace("{{prompt}}", item["prompt"])
        .replace("{{response}}", response_1_list[i])
        for i, item in enumerate(data)
    ]
    if args.backend == "vllm":
        formatted_2_list, response_2_list = generate_batch_vllm(
            llm, tokenizer, prompts_2, sampling_params, model_config
        )
    else:
        formatted_2_list, response_2_list = generate_batch_transformers(
            model, tokenizer, prompts_2, args.max_tokens, model_config
        )

    logging.info("[Phase 3/3] Final decision for all samples")
    prompts_3 = [
        prompts_module.prompts["prompt_3"]
        .replace("{{prompt}}", item["prompt"])
        .replace("{{response}}", response_1_list[i])
        .replace("{{reflection}}", response_2_list[i])
        for i, item in enumerate(data)
    ]
    if args.backend == "vllm":
        formatted_3_list, response_3_list = generate_batch_vllm(
            llm, tokenizer, prompts_3, sampling_params, model_config
        )
    else:
        formatted_3_list, response_3_list = generate_batch_transformers(
            model, tokenizer, prompts_3, args.max_tokens, model_config
        )

    results = []
    for i, item in enumerate(data):
        item["formatted_prompt_1"] = formatted_1_list[i]
        item["response_1"] = response_1_list[i]
        item["formatted_prompt_2"] = formatted_2_list[i]
        item["reflection"] = response_2_list[i]
        item["formatted_prompt_3"] = formatted_3_list[i]
        item["final_response"] = response_3_list[i]
        item["model"] = args.model_path
        results.append(item)

    # Save results
    logging.info(f"Saving results to {args.output_file}")
    with open(args.output_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logging.info(f"Completed! Saved {len(results)} responses to {args.output_file}")
    print(f"Completed! Saved {len(results)} responses to {args.output_file}")


if __name__ == "__main__":
    main()
