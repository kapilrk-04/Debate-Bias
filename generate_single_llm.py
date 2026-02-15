# -*- coding: utf-8 -*-
"""
Single LLM response generation for judgment and plausibility tasks.
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
from dotenv import load_dotenv

load_dotenv()

# Get the token from environment variable
hf_token = os.getenv("HF_TOKEN")

# Setup logging
os.makedirs("logs", exist_ok=True)
now_str = datetime.now().strftime("%d-%m-%y_%H-%M-%S")
logging.basicConfig(
    level=logging.INFO,
    filename=f"logs/single_llm_{now_str}.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

if hf_token:
    login(token=hf_token)
    print("Successfully logged in to Hugging Face Hub.")
else:
    print("Warning: HF_TOKEN environment variable is not set.")


def build_prompt_single(prompt_text, prompt_type, prompts_module):
    """Build prompt for single LLM."""
    if prompt_type in ["judgment_single", "plausibility_single"]:
        return prompts_module.prompts[prompt_type].replace("{{prompt}}", prompt_text)
    elif prompt_type in ["judgment_with_reasoning", "plausibility_with_reasoning"]:
        return prompts_module.prompts[prompt_type].replace("{{prompt}}", prompt_text)
    else:
        # Default: just use the prompt as-is
        return prompt_text


def generate_vllm(llm, tokenizer, prompts, sampling_params, model_config):
    """Generate responses using vLLM."""
    from vllm import SamplingParams

    # Format prompts for the model
    formatted_prompts = []
    for prompt in prompts:
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
        formatted_prompts.append(formatted)

    # Generate
    outputs = llm.generate(formatted_prompts, sampling_params)

    # Extract responses
    responses = [o.outputs[0].text.strip() for o in outputs]
    return formatted_prompts, responses


def generate_transformers(model, tokenizer, prompts, max_new_tokens, model_config):
    """Generate responses using transformers."""
    formatted_prompts = []
    responses = []

    for prompt in tqdm(prompts, desc="Generating"):
        messages = [{"role": "user", "content": prompt}]

        # Format with chat template
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        # Determine terminators
        terminators = [tokenizer.eos_token_id]
        if hasattr(tokenizer, 'convert_tokens_to_ids'):
            eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if eot_id is not None:
                terminators.append(eot_id)

        # Generate
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=False,
            temperature=0.0,
        )

        # Decode
        response = outputs[0][input_ids.shape[-1]:]
        generation = tokenizer.decode(response, skip_special_tokens=True)

        formatted_prompts.append(tokenizer.decode(input_ids[0], skip_special_tokens=False))
        responses.append(generation.strip())

    return formatted_prompts, responses


def main():
    parser = argparse.ArgumentParser(description="Single LLM Response Generation")
    parser.add_argument("--model-path", type=str, required=True, help="Model path or HuggingFace model ID")
    parser.add_argument("--input-file", type=str, required=True, help="Input JSONL file path")
    parser.add_argument("--output-file", type=str, required=True, help="Output JSONL file path")
    parser.add_argument("--backend", type=str, choices=["vllm", "transformers"], default="transformers",
                        help="Backend to use for generation")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum number of tokens to generate")
    parser.add_argument("--prompt-type", type=str,
                        choices=["judgment_single", "plausibility_single", "judgment_with_reasoning", "plausibility_with_reasoning", "direct"],
                        default="direct", help="Type of prompt to use")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for vLLM")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")

    args = parser.parse_args()
    logging.info(f"\nCLI Arguments:\n{vars(args)}")

    # Import prompts module
    if args.prompt_type != "direct":
        import prompts_single_new as prompts_module
    else:
        prompts_module = None

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
            max_num_seqs=args.batch_size,
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

    # Process in batches or individually
    if args.backend == "vllm":
        # Batch processing with vLLM
        all_prompts = []
        for item in data:
            if args.prompt_type == "direct":
                prompt = item["prompt"]
            else:
                prompt = build_prompt_single(item["prompt"], args.prompt_type, prompts_module)
            all_prompts.append(prompt)

        # Generate all at once
        formatted_prompts, responses = generate_vllm(
            llm, tokenizer, all_prompts, sampling_params, model_config
        )

    else:  # transformers - process individually
        all_prompts = []
        for item in data:
            if args.prompt_type == "direct":
                prompt = item["prompt"]
            else:
                prompt = build_prompt_single(item["prompt"], args.prompt_type, prompts_module)
            all_prompts.append(prompt)

        formatted_prompts, responses = generate_transformers(
            model, tokenizer, all_prompts, args.max_tokens, model_config
        )

    # Save results
    logging.info(f"Saving results to {args.output_file}")
    with open(args.output_file, "w", encoding="utf-8") as f:
        for item, formatted_prompt, response in zip(data, formatted_prompts, responses):
            item["formatted_prompt"] = formatted_prompt
            item["response"] = response
            item["model"] = args.model_path
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logging.info(f"Completed! Saved {len(data)} responses to {args.output_file}")
    print(f"Completed! Saved {len(data)} responses to {args.output_file}")


if __name__ == "__main__":
    main()
