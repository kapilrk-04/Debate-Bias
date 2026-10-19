# -*- coding: utf-8 -*-
"""
Multi-agent debate response generation - SEQUENTIAL VERSION.

Instead of loading both models simultaneously, this version processes
one agent at a time across all prompts, unloading the model between turns.
This allows full GPU memory utilization for a single model at a time.

Flow (for different models):
  Phase 1: Load model1 → Agent1 Turn1 (initial decision) for ALL prompts → unload
  Phase 2: Load model2 → Agent2 Turn2 (initial decision) for ALL prompts → unload
  Phase 3: Load model1 → Agent1 Turn3 (feedback) for ALL prompts → unload
  Phase 4: Load model2 → Agent2 Turn4 (feedback) for ALL prompts → unload
  Phase 5: Load model1 → Agent1 Turn5 (final decision) for ALL prompts → unload
  Phase 6: Load model2 → Agent2 Turn6 (final decision) for ALL prompts → unload

For same-model mode, the model is loaded once and all turns are processed
sequentially without reloading.
"""

import torch
from datetime import datetime
import json
import os
import argparse
from tqdm import tqdm
import logging
import gc
from huggingface_hub import login
import yaml
import prompts_multi_new as prompts_module
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
    filename=f"logs/multi_agent_seq_{now_str}.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

if hf_token:
    login(token=hf_token)
    print("Successfully logged in to Hugging Face Hub.")
else:
    print("Warning: HF_TOKEN environment variable is not set.")


def generate_batch_vllm(llm, tokenizer, prompts_list, sampling_params, model_config):
    """Generate responses for a batch of prompts using vLLM."""
    formatted_list = []
    for prompt in prompts_list:
        messages = [{"role": "user", "content": prompt}]
        formatted = format_messages_for_model(
            tokenizer=tokenizer,
            model_path=model_config.get("model_path", ""),
            messages=messages,
            add_generation_prompt=True,
            enable_thinking=model_config.get("enable_thinking"),
        )
        formatted_list.append(formatted)

    outputs = llm.generate(formatted_list, sampling_params)
    results = []
    for i, output in enumerate(outputs):
        response = output.outputs[0].text.strip()
        results.append((formatted_list[i], response))
    return results


def generate_batch_transformers(model, tokenizer, prompts_list, max_new_tokens, temperature=0.0):
    """Generate responses for a batch of prompts using transformers (one at a time)."""
    results = []
    for prompt in tqdm(prompts_list, desc="  Generating", leave=False):
        messages = [{"role": "user", "content": prompt}]
        formatted = format_messages_for_model(
            tokenizer=tokenizer,
            model_path=getattr(tokenizer, "name_or_path", ""),
            messages=messages,
            add_generation_prompt=True,
            enable_thinking=None,
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
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
        )

        response = outputs[0][input_ids.shape[-1]:]
        generation = tokenizer.decode(response, skip_special_tokens=True)
        results.append((formatted, generation.strip()))
    return results


def load_model_and_tokenizer(model_path, backend, hf_home, mcfg):
    """Load model and tokenizer based on backend.

    Args:
        mcfg: Model config dict from model_configs.yaml (via load_model_config).
    """
    trust_remote_code = mcfg.get("trust_remote_code", False)

    if backend == "vllm":
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

    else:  # transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM

        logging.info(f"Loading {model_path} with transformers")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=hf_home,
            trust_remote_code=trust_remote_code
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            cache_dir=hf_home,
            device_map="auto",
            trust_remote_code=trust_remote_code
        )
        return model, tokenizer


def unload_model(model, backend):
    """Unload model and free GPU memory."""
    if backend == "vllm":
        # Shut down the V1 engine core subprocess cleanly
        try:
            model.llm_engine.shutdown()
        except Exception:
            pass
        try:
            from vllm.distributed.parallel_state import (
                destroy_model_parallel, destroy_distributed_environment
            )
            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:
            pass

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    logging.info("Model unloaded and GPU memory freed")
    print("  Model unloaded, GPU memory freed.")


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Debate Response Generation (Sequential)")
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
        same_model = True
        print(f"Using same model for both agents: {model1_path}")
    else:
        model2_path = args.model2_path
        same_model = False
        print(f"Using two different models: {model1_path} and {model2_path}")

    # Load per-model configs from model_configs.yaml
    model1_config = load_model_config(model1_path)
    model2_config = load_model_config(model2_path)
    model1_config["model_path"] = model1_path
    model2_config["model_path"] = model2_path
    logging.info(f"Model1 config: {model1_config}")
    logging.info(f"Model2 config: {model2_config}")
    print(f"Model1 config: {model1_config}")
    print(f"Model2 config: {model2_config}")

    # Create output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    hf_home = os.getenv("HF_HOME", None)

    # Load input data
    logging.info(f"Loading input from {args.input_file}")
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    logging.info(f"Loaded {len(data)} samples")
    print(f"Loaded {len(data)} samples")

    # Initialize per-item storage
    n = len(data)
    agent1_formatted_1 = [None] * n
    agent1_response_1 = [None] * n
    agent2_formatted_1 = [None] * n
    agent2_response_1 = [None] * n
    agent1_formatted_2 = [None] * n
    agent1_feedback = [None] * n
    agent2_formatted_2 = [None] * n
    agent2_feedback = [None] * n
    agent1_formatted_3 = [None] * n
    agent1_final = [None] * n
    agent2_formatted_3 = [None] * n
    agent2_final = [None] * n

    if args.backend == "vllm":
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens
        )

    if same_model:
        # ---- SAME MODEL: load once, process all 6 turns sequentially ----
        print(f"\n{'='*60}")
        print(f"Loading model: {model1_path}")
        print(f"{'='*60}")
        model, tokenizer = load_model_and_tokenizer(
            model1_path, args.backend, hf_home, model1_config
        )

        # Turn 1: Agent 1 initial decisions (all prompts)
        print(f"\n[Phase 1/6] Agent 1 - Initial decisions ({n} prompts)")
        turn1_prompts = [
            prompts_module.prompts["prompt_1"].replace("{{prompt}}", item["prompt"])
            for item in data
        ]
        if args.backend == "vllm":
            results = generate_batch_vllm(model, tokenizer, turn1_prompts, sampling_params, model1_config)
        else:
            results = generate_batch_transformers(model, tokenizer, turn1_prompts, args.max_tokens, args.temperature)
        for i, (fmt, resp) in enumerate(results):
            agent1_formatted_1[i] = fmt
            agent1_response_1[i] = resp

        # Turn 2: Agent 2 initial decisions (all prompts)
        print(f"[Phase 2/6] Agent 2 - Initial decisions ({n} prompts)")
        turn2_prompts = [
            prompts_module.prompts["prompt_1"].replace("{{prompt}}", item["prompt"])
            for item in data
        ]
        if args.backend == "vllm":
            results = generate_batch_vllm(model, tokenizer, turn2_prompts, sampling_params, model1_config)
        else:
            results = generate_batch_transformers(model, tokenizer, turn2_prompts, args.max_tokens, args.temperature)
        for i, (fmt, resp) in enumerate(results):
            agent2_formatted_1[i] = fmt
            agent2_response_1[i] = resp

        # Turn 3: Agent 1 feedback (all prompts)
        print(f"[Phase 3/6] Agent 1 - Feedback ({n} prompts)")
        turn3_prompts = [
            prompts_module.prompts["prompt_2"]
            .replace("{{prompt}}", data[i]["prompt"])
            .replace("{{your_response}}", agent1_response_1[i])
            .replace("{{other_response}}", agent2_response_1[i])
            for i in range(n)
        ]
        if args.backend == "vllm":
            results = generate_batch_vllm(model, tokenizer, turn3_prompts, sampling_params, model1_config)
        else:
            results = generate_batch_transformers(model, tokenizer, turn3_prompts, args.max_tokens, args.temperature)
        for i, (fmt, resp) in enumerate(results):
            agent1_formatted_2[i] = fmt
            agent1_feedback[i] = resp

        # Turn 4: Agent 2 feedback (all prompts)
        print(f"[Phase 4/6] Agent 2 - Feedback ({n} prompts)")
        turn4_prompts = [
            prompts_module.prompts["prompt_2"]
            .replace("{{prompt}}", data[i]["prompt"])
            .replace("{{your_response}}", agent2_response_1[i])
            .replace("{{other_response}}", agent1_response_1[i])
            for i in range(n)
        ]
        if args.backend == "vllm":
            results = generate_batch_vllm(model, tokenizer, turn4_prompts, sampling_params, model1_config)
        else:
            results = generate_batch_transformers(model, tokenizer, turn4_prompts, args.max_tokens, args.temperature)
        for i, (fmt, resp) in enumerate(results):
            agent2_formatted_2[i] = fmt
            agent2_feedback[i] = resp

        # Turn 5: Agent 1 final decision (all prompts)
        print(f"[Phase 5/6] Agent 1 - Final decisions ({n} prompts)")
        turn5_prompts = [
            prompts_module.prompts["prompt_3"]
            .replace("{{prompt}}", data[i]["prompt"])
            .replace("{{your_response}}", agent1_response_1[i])
            .replace("{{other_response}}", agent2_response_1[i])
            .replace("{{your_feedback}}", agent1_feedback[i])
            .replace("{{other_feedback}}", agent2_feedback[i])
            for i in range(n)
        ]
        if args.backend == "vllm":
            results = generate_batch_vllm(model, tokenizer, turn5_prompts, sampling_params, model1_config)
        else:
            results = generate_batch_transformers(model, tokenizer, turn5_prompts, args.max_tokens, args.temperature)
        for i, (fmt, resp) in enumerate(results):
            agent1_formatted_3[i] = fmt
            agent1_final[i] = resp

        # Turn 6: Agent 2 final decision (all prompts)
        print(f"[Phase 6/6] Agent 2 - Final decisions ({n} prompts)")
        turn6_prompts = [
            prompts_module.prompts["prompt_3"]
            .replace("{{prompt}}", data[i]["prompt"])
            .replace("{{your_response}}", agent2_response_1[i])
            .replace("{{other_response}}", agent1_response_1[i])
            .replace("{{your_feedback}}", agent2_feedback[i])
            .replace("{{other_feedback}}", agent1_feedback[i])
            for i in range(n)
        ]
        if args.backend == "vllm":
            results = generate_batch_vllm(model, tokenizer, turn6_prompts, sampling_params, model1_config)
        else:
            results = generate_batch_transformers(model, tokenizer, turn6_prompts, args.max_tokens, args.temperature)
        for i, (fmt, resp) in enumerate(results):
            agent2_formatted_3[i] = fmt
            agent2_final[i] = resp

        unload_model(model, args.backend)

    else:
        # ---- DIFFERENT MODELS: load/unload alternating between models ----
        # Phase 1: Load model1 → Agent1 Turn1
        print(f"\n{'='*60}")
        print(f"[Phase 1/6] Loading model1: {model1_path}")
        print(f"{'='*60}")
        model, tokenizer = load_model_and_tokenizer(
            model1_path, args.backend, hf_home, model1_config
        )
        print(f"  Agent 1 - Initial decisions ({n} prompts)")
        turn1_prompts = [
            prompts_module.prompts["prompt_1"].replace("{{prompt}}", item["prompt"])
            for item in data
        ]
        if args.backend == "vllm":
            results = generate_batch_vllm(model, tokenizer, turn1_prompts, sampling_params, model1_config)
        else:
            results = generate_batch_transformers(model, tokenizer, turn1_prompts, args.max_tokens)
        for i, (fmt, resp) in enumerate(results):
            agent1_formatted_1[i] = fmt
            agent1_response_1[i] = resp
        unload_model(model, args.backend)
        del model, tokenizer

        # Phase 2: Load model2 → Agent2 Turn2
        print(f"\n{'='*60}")
        print(f"[Phase 2/6] Loading model2: {model2_path}")
        print(f"{'='*60}")
        model, tokenizer = load_model_and_tokenizer(
            model2_path, args.backend, hf_home, model2_config
        )
        print(f"  Agent 2 - Initial decisions ({n} prompts)")
        turn2_prompts = [
            prompts_module.prompts["prompt_1"].replace("{{prompt}}", item["prompt"])
            for item in data
        ]
        if args.backend == "vllm":
            results = generate_batch_vllm(model, tokenizer, turn2_prompts, sampling_params, model2_config)
        else:
            results = generate_batch_transformers(model, tokenizer, turn2_prompts, args.max_tokens)
        for i, (fmt, resp) in enumerate(results):
            agent2_formatted_1[i] = fmt
            agent2_response_1[i] = resp
        unload_model(model, args.backend)
        del model, tokenizer

        # Phase 3: Load model1 → Agent1 Feedback
        print(f"\n{'='*60}")
        print(f"[Phase 3/6] Loading model1: {model1_path}")
        print(f"{'='*60}")
        model, tokenizer = load_model_and_tokenizer(
            model1_path, args.backend, hf_home, model1_config
        )
        print(f"  Agent 1 - Feedback ({n} prompts)")
        turn3_prompts = [
            prompts_module.prompts["prompt_2"]
            .replace("{{prompt}}", data[i]["prompt"])
            .replace("{{your_response}}", agent1_response_1[i])
            .replace("{{other_response}}", agent2_response_1[i])
            for i in range(n)
        ]
        if args.backend == "vllm":
            results = generate_batch_vllm(model, tokenizer, turn3_prompts, sampling_params, model1_config)
        else:
            results = generate_batch_transformers(model, tokenizer, turn3_prompts, args.max_tokens)
        for i, (fmt, resp) in enumerate(results):
            agent1_formatted_2[i] = fmt
            agent1_feedback[i] = resp
        unload_model(model, args.backend)
        del model, tokenizer

        # Phase 4: Load model2 → Agent2 Feedback
        print(f"\n{'='*60}")
        print(f"[Phase 4/6] Loading model2: {model2_path}")
        print(f"{'='*60}")
        model, tokenizer = load_model_and_tokenizer(
            model2_path, args.backend, hf_home, model2_config
        )
        print(f"  Agent 2 - Feedback ({n} prompts)")
        turn4_prompts = [
            prompts_module.prompts["prompt_2"]
            .replace("{{prompt}}", data[i]["prompt"])
            .replace("{{your_response}}", agent2_response_1[i])
            .replace("{{other_response}}", agent1_response_1[i])
            for i in range(n)
        ]
        if args.backend == "vllm":
            results = generate_batch_vllm(model, tokenizer, turn4_prompts, sampling_params, model2_config)
        else:
            results = generate_batch_transformers(model, tokenizer, turn4_prompts, args.max_tokens)
        for i, (fmt, resp) in enumerate(results):
            agent2_formatted_2[i] = fmt
            agent2_feedback[i] = resp
        unload_model(model, args.backend)
        del model, tokenizer

        # Phase 5: Load model1 → Agent1 Final
        print(f"\n{'='*60}")
        print(f"[Phase 5/6] Loading model1: {model1_path}")
        print(f"{'='*60}")
        model, tokenizer = load_model_and_tokenizer(
            model1_path, args.backend, hf_home, model1_config
        )
        print(f"  Agent 1 - Final decisions ({n} prompts)")
        turn5_prompts = [
            prompts_module.prompts["prompt_3"]
            .replace("{{prompt}}", data[i]["prompt"])
            .replace("{{your_response}}", agent1_response_1[i])
            .replace("{{other_response}}", agent2_response_1[i])
            .replace("{{your_feedback}}", agent1_feedback[i])
            .replace("{{other_feedback}}", agent2_feedback[i])
            for i in range(n)
        ]
        if args.backend == "vllm":
            results = generate_batch_vllm(model, tokenizer, turn5_prompts, sampling_params, model1_config)
        else:
            results = generate_batch_transformers(model, tokenizer, turn5_prompts, args.max_tokens)
        for i, (fmt, resp) in enumerate(results):
            agent1_formatted_3[i] = fmt
            agent1_final[i] = resp
        unload_model(model, args.backend)
        del model, tokenizer

        # Phase 6: Load model2 → Agent2 Final
        print(f"\n{'='*60}")
        print(f"[Phase 6/6] Loading model2: {model2_path}")
        print(f"{'='*60}")
        model, tokenizer = load_model_and_tokenizer(
            model2_path, args.backend, hf_home, model2_config
        )
        print(f"  Agent 2 - Final decisions ({n} prompts)")
        turn6_prompts = [
            prompts_module.prompts["prompt_3"]
            .replace("{{prompt}}", data[i]["prompt"])
            .replace("{{your_response}}", agent2_response_1[i])
            .replace("{{other_response}}", agent1_response_1[i])
            .replace("{{your_feedback}}", agent2_feedback[i])
            .replace("{{other_feedback}}", agent1_feedback[i])
            for i in range(n)
        ]
        if args.backend == "vllm":
            results = generate_batch_vllm(model, tokenizer, turn6_prompts, sampling_params, model2_config)
        else:
            results = generate_batch_transformers(model, tokenizer, turn6_prompts, args.max_tokens)
        for i, (fmt, resp) in enumerate(results):
            agent2_formatted_3[i] = fmt
            agent2_final[i] = resp
        unload_model(model, args.backend)

    # Assemble results
    results_out = []
    for i, item in enumerate(data):
        item["agent1_formatted_prompt_1"] = agent1_formatted_1[i]
        item["agent1_response_1"] = agent1_response_1[i]
        item["agent1_formatted_prompt_2"] = agent1_formatted_2[i]
        item["agent1_feedback"] = agent1_feedback[i]
        item["agent1_formatted_prompt_3"] = agent1_formatted_3[i]
        item["agent1_final_response"] = agent1_final[i]

        item["agent2_formatted_prompt_1"] = agent2_formatted_1[i]
        item["agent2_response_1"] = agent2_response_1[i]
        item["agent2_formatted_prompt_2"] = agent2_formatted_2[i]
        item["agent2_feedback"] = agent2_feedback[i]
        item["agent2_formatted_prompt_3"] = agent2_formatted_3[i]
        item["agent2_final_response"] = agent2_final[i]

        item["model1"] = model1_path
        item["model2"] = model2_path
        results_out.append(item)

    # Save results
    logging.info(f"Saving results to {args.output_file}")
    with open(args.output_file, "w", encoding="utf-8") as f:
        for item in results_out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logging.info(f"Completed! Saved {len(results_out)} responses to {args.output_file}")
    print(f"\nCompleted! Saved {len(results_out)} responses to {args.output_file}")


if __name__ == "__main__":
    main()
