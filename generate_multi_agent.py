# -*- coding: utf-8 -*-
"""
Multi-agent debate response generation for judgment and plausibility tasks.
Supports:
- Two different LLMs debating
- Two instances of the same LLM debating (same model, different random seeds or separate instances)
- Both vLLM and transformers backends
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
import prompts_multi_new as prompts_module

load_dotenv()

# Get the token from environment variable
hf_token = os.getenv("HF_TOKEN")

# Setup logging
os.makedirs("logs", exist_ok=True)
now_str = datetime.now().strftime("%d-%m-%y_%H-%M-%S")
logging.basicConfig(
    level=logging.INFO,
    filename=f"logs/multi_agent_{now_str}.log",
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


def load_model_and_tokenizer(model_path, backend, hf_home):
    """Load model and tokenizer based on backend."""
    if backend == "vllm":
        from vllm import LLM
        from transformers import AutoTokenizer

        logging.info(f"Loading {model_path} with vLLM")
        llm = LLM(
            model=model_path,
            tokenizer=model_path,
            dtype="auto",
            tensor_parallel_size=1,
            max_num_seqs=1,
            gpu_memory_utilization=0.45,  # Use less memory to fit 2 models
            download_dir=hf_home
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=hf_home,
            trust_remote_code=True
        )
        return llm, tokenizer

    else:  # transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM

        logging.info(f"Loading {model_path} with transformers")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=hf_home,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            cache_dir=hf_home,
            device_map="auto",
            trust_remote_code=True
        )
        return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Debate Response Generation")
    parser.add_argument("--model1-path", type=str, required=True, help="First model path or HuggingFace model ID")
    parser.add_argument("--model2-path", type=str, default=None,
                        help="Second model path (if None, uses same as model1)")
    parser.add_argument("--input-file", type=str, required=True, help="Input JSONL file path")
    parser.add_argument("--output-file", type=str, required=True, help="Output JSONL file path")
    parser.add_argument("--backend", type=str, choices=["vllm", "transformers"], default="transformers",
                        help="Backend to use for generation")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--same-model", action="store_true",
                        help="Use two instances of the same model (useful for debate with same model)")

    args = parser.parse_args()
    logging.info(f"\nCLI Arguments:\n{vars(args)}")

    # Determine model paths
    model1_path = args.model1_path
    if args.same_model or args.model2_path is None:
        model2_path = args.model1_path
        print(f"Using two instances of the same model: {model1_path}")
    else:
        model2_path = args.model2_path
        print(f"Using two different models: {model1_path} and {model2_path}")

    # Create output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Model configuration (simplified)
    model_config = {}
    hf_home = os.getenv("HF_HOME", None)

    # Initialize models and tokenizers
    if args.backend == "vllm":
        from vllm import SamplingParams

        llm1, tokenizer1 = load_model_and_tokenizer(model1_path, "vllm", hf_home)

        # Load second model if different
        if model1_path == model2_path:
            llm2, tokenizer2 = llm1, tokenizer1
            print("Note: Using the same vLLM instance for both agents (for true separation, use transformers backend)")
        else:
            llm2, tokenizer2 = load_model_and_tokenizer(model2_path, "vllm", hf_home)

        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens
        )

    else:  # transformers
        model1, tokenizer1 = load_model_and_tokenizer(model1_path, "transformers", hf_home)

        # Load second model if different, otherwise use same model
        if model1_path == model2_path:
            model2, tokenizer2 = model1, tokenizer1
        else:
            model2, tokenizer2 = load_model_and_tokenizer(model2_path, "transformers", hf_home)

    # Load input data
    logging.info(f"Loading input from {args.input_file}")
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    logging.info(f"Loaded {len(data)} samples")

    # Process each sample with multi-agent debate
    results = []
    for item in tqdm(data, desc="Processing"):
        prompt_text = item["prompt"]

        # Turn 1: Agent 1 makes initial decision
        prompt_1_agent1 = prompts_module.prompts["prompt_1"].replace("{{prompt}}", prompt_text)
        logging.info(f"[1] Agent 1 making initial decision")

        if args.backend == "vllm":
            formatted_1_agent1, response_1_agent1 = generate_single_vllm(
                llm1, tokenizer1, prompt_1_agent1, sampling_params, model_config
            )
        else:
            formatted_1_agent1, response_1_agent1 = generate_single_transformers(
                model1, tokenizer1, prompt_1_agent1, args.max_tokens
            )

        # Turn 2: Agent 2 makes initial decision
        prompt_1_agent2 = prompts_module.prompts["prompt_1"].replace("{{prompt}}", prompt_text)
        logging.info(f"[2] Agent 2 making initial decision")

        if args.backend == "vllm":
            formatted_1_agent2, response_1_agent2 = generate_single_vllm(
                llm2, tokenizer2, prompt_1_agent2, sampling_params, model_config
            )
        else:
            formatted_1_agent2, response_1_agent2 = generate_single_transformers(
                model2, tokenizer2, prompt_1_agent2, args.max_tokens
            )

        # Turn 3: Agent 1 gives feedback to Agent 2
        prompt_2_agent1 = prompts_module.prompts["prompt_2"].replace("{{prompt}}", prompt_text).replace("{{your_response}}", response_1_agent1).replace("{{other_response}}", response_1_agent2)
        logging.info(f"[3] Agent 1 giving feedback")

        if args.backend == "vllm":
            formatted_2_agent1, response_2_agent1 = generate_single_vllm(
                llm1, tokenizer1, prompt_2_agent1, sampling_params, model_config
            )
        else:
            formatted_2_agent1, response_2_agent1 = generate_single_transformers(
                model1, tokenizer1, prompt_2_agent1, args.max_tokens
            )

        # Turn 4: Agent 2 gives feedback to Agent 1
        prompt_2_agent2 = prompts_module.prompts["prompt_2"].replace("{{prompt}}", prompt_text).replace("{{your_response}}", response_1_agent2).replace("{{other_response}}", response_1_agent1)
        logging.info(f"[4] Agent 2 giving feedback")

        if args.backend == "vllm":
            formatted_2_agent2, response_2_agent2 = generate_single_vllm(
                llm2, tokenizer2, prompt_2_agent2, sampling_params, model_config
            )
        else:
            formatted_2_agent2, response_2_agent2 = generate_single_transformers(
                model2, tokenizer2, prompt_2_agent2, args.max_tokens
            )

        # Turn 5: Agent 1 makes final decision
        prompt_3_agent1 = prompts_module.prompts["prompt_3"].replace("{{prompt}}", prompt_text).replace("{{your_response}}", response_1_agent1).replace("{{other_response}}", response_1_agent2).replace("{{your_feedback}}", response_2_agent1).replace("{{other_feedback}}", response_2_agent2)
        logging.info(f"[5] Agent 1 making final decision")

        if args.backend == "vllm":
            formatted_3_agent1, response_3_agent1 = generate_single_vllm(
                llm1, tokenizer1, prompt_3_agent1, sampling_params, model_config
            )
        else:
            formatted_3_agent1, response_3_agent1 = generate_single_transformers(
                model1, tokenizer1, prompt_3_agent1, args.max_tokens
            )

        # Turn 6: Agent 2 makes final decision
        prompt_3_agent2 = prompts_module.prompts["prompt_3"].replace("{{prompt}}", prompt_text).replace("{{your_response}}", response_1_agent2).replace("{{other_response}}", response_1_agent1).replace("{{your_feedback}}", response_2_agent2).replace("{{other_feedback}}", response_2_agent1)
        logging.info(f"[6] Agent 2 making final decision")

        if args.backend == "vllm":
            formatted_3_agent2, response_3_agent2 = generate_single_vllm(
                llm2, tokenizer2, prompt_3_agent2, sampling_params, model_config
            )
        else:
            formatted_3_agent2, response_3_agent2 = generate_single_transformers(
                model2, tokenizer2, prompt_3_agent2, args.max_tokens
            )

        # Store results
        item["agent1_formatted_prompt_1"] = formatted_1_agent1
        item["agent1_response_1"] = response_1_agent1
        item["agent1_formatted_prompt_2"] = formatted_2_agent1
        item["agent1_feedback"] = response_2_agent1
        item["agent1_formatted_prompt_3"] = formatted_3_agent1
        item["agent1_final_response"] = response_3_agent1

        item["agent2_formatted_prompt_1"] = formatted_1_agent2
        item["agent2_response_1"] = response_1_agent2
        item["agent2_formatted_prompt_2"] = formatted_2_agent2
        item["agent2_feedback"] = response_2_agent2
        item["agent2_formatted_prompt_3"] = formatted_3_agent2
        item["agent2_final_response"] = response_3_agent2

        item["model1"] = model1_path
        item["model2"] = model2_path
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
