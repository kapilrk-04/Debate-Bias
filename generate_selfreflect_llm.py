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


def generate_single_vllm(llm, tokenizer, prompt, sampling_params, model_config):
    """Generate a single response using vLLM."""
    messages = [{"role": "user", "content": prompt}]

    if "enable_thinking" in model_config:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=model_config["enable_thinking"]
        )
    else:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    outputs = llm.generate([formatted], sampling_params)
    response = outputs[0].outputs[0].text.strip()
    return formatted, response


def generate_single_transformers(model, tokenizer, prompt, max_new_tokens):
    """Generate a single response using transformers."""
    messages = [{"role": "user", "content": prompt}]

    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    # Determine terminators
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
    formatted = tokenizer.decode(input_ids[0], skip_special_tokens=False)

    return formatted, generation.strip()


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

    # Create output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Model configuration (simplified)
    model_config = {}

    # Initialize model and tokenizer based on backend
    if args.backend == "vllm":
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        logging.info(f"Loading {args.model_path} with vLLM")
        llm = LLM(
            model=args.model_path,
            tokenizer=args.model_path,
            dtype="auto",
            tensor_parallel_size=1,
            max_num_seqs=1,  # Process one at a time for self-reflection
            gpu_memory_utilization=0.9,
            download_dir=os.getenv("HF_HOME", None)
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            cache_dir=os.getenv("HF_HOME", None),
            trust_remote_code=True
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
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            cache_dir=os.getenv("HF_HOME", None),
            device_map="auto",
            trust_remote_code=True
        )

    # Load input data
    logging.info(f"Loading input from {args.input_file}")
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    logging.info(f"Loaded {len(data)} samples")

    # Process each sample with 3-turn self-reflection
    results = []
    for item in tqdm(data, desc="Processing"):
        prompt_text = item["prompt"]

        # Turn 1: Initial decision
        prompt_1 = prompts_module.prompts["prompt_1"].replace("{{prompt}}", prompt_text)
        logging.info(f"[1] Making initial decision")

        if args.backend == "vllm":
            formatted_1, response_1 = generate_single_vllm(
                llm, tokenizer, prompt_1, sampling_params, model_config
            )
        else:
            formatted_1, response_1 = generate_single_transformers(
                model, tokenizer, prompt_1, args.max_tokens
            )

        # Turn 2: Self-reflection
        prompt_2 = prompts_module.prompts["prompt_2"].replace("{{prompt}}", prompt_text).replace("{{response}}", response_1)
        logging.info(f"[2] Generating self-reflection")

        if args.backend == "vllm":
            formatted_2, response_2 = generate_single_vllm(
                llm, tokenizer, prompt_2, sampling_params, model_config
            )
        else:
            formatted_2, response_2 = generate_single_transformers(
                model, tokenizer, prompt_2, args.max_tokens
            )

        # Turn 3: Final decision
        prompt_3 = prompts_module.prompts["prompt_3"].replace("{{prompt}}", prompt_text).replace("{{response}}", response_1).replace("{{reflection}}", response_2)
        logging.info(f"[3] Making final decision")

        if args.backend == "vllm":
            formatted_3, response_3 = generate_single_vllm(
                llm, tokenizer, prompt_3, sampling_params, model_config
            )
        else:
            formatted_3, response_3 = generate_single_transformers(
                model, tokenizer, prompt_3, args.max_tokens
            )

        # Store results
        item["formatted_prompt_1"] = formatted_1
        item["response_1"] = response_1
        item["formatted_prompt_2"] = formatted_2
        item["reflection"] = response_2
        item["formatted_prompt_3"] = formatted_3
        item["final_response"] = response_3
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
