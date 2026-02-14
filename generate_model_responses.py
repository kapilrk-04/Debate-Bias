# -*- coding: utf-8 -*-

from vllm import LLM, SamplingParams
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import json
import os
from tqdm import tqdm
import logging
from datetime import datetime
from huggingface_hub import login
import yaml

from dotenv import load_dotenv

load_dotenv()

DEFAULT_SYSTEM_PROMPT = ""

# Get the token from environment variable
hf_token = os.getenv("HF_TOKEN")

os.makedirs("logs", exist_ok=True)
now_str = datetime.now().strftime("%d-%m-%y_%H-%M-%S")
logging.basicConfig(
    level=logging.INFO,
    filename=f"logs/model_responses_{now_str}.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

if hf_token:
    login(token=hf_token)
    print("Successfully logged in to Hugging Face Hub.")
else:
    raise ValueError("HUGGINGFACE_TOKEN environment variable is not set.")

def create_prompt_with_chat_format(question, system_prompt, model_path, model_config):
    messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
    formatted_text = ""
    for message in messages:
        if message["role"] == "system":
            if system_prompt not in ["", None]:
                formatted_text += model_config["prompt_template"]["system_prompt_template"].format(system_prompt=system_prompt)
        elif message["role"] == "user":
            formatted_text += model_config["prompt_template"]["user_prompt_template"].format(question=question)
    if "bos" in model_config["prompt_template"]:
        formatted_text = model_config["prompt_template"]["bos"] + formatted_text
 
    return formatted_text

def build_prompt(tokenizer, model_path, model_config, question: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    """Construct a model-specific prompt format.

    Parameters
    ----------
    question : str
        The user's question in any language.
    system_prompt : str, optional
        A system level instruction that guides the model's behaviour, by default
        `DEFAULT_SYSTEM_PROMPT`. An empty string will omit the system prompt.

    Returns
    -------
    str
        The complete formatted prompt string ready for tokenisation.
    """
    prompt_format = model_config["prompt_format"]
    formatted_prompt = ""

    if prompt_format == "custom_chat_template":
        formatted_prompt = create_prompt_with_chat_format(question, system_prompt, model_path, model_config)

    elif prompt_format == "chat_template":
        if system_prompt not in ["", None]:            
            chat_prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
        else:
            chat_prompt = [
                {"role": "user", "content": question}
            ]
        # Apply chat template
        if "enable_thinking" in model_config:
            formatted_prompt = tokenizer.apply_chat_template(chat_prompt, tokenize=False, add_generation_prompt=True, enable_thinking=model_config["enable_thinking"])    
        else:
            formatted_prompt = tokenizer.apply_chat_template(chat_prompt, tokenize=False, add_generation_prompt=True)

    return formatted_prompt

# Core batch generation function
def batch_generate(llm, tokenizer, model_path, model_config, sampling_params, questions: list, system_prompt: str=DEFAULT_SYSTEM_PROMPT):
    prompts = []
    logging.info("Formatting prompt batch")
    for question in questions:
        prompts.append(build_prompt(tokenizer, model_path, model_config, question, system_prompt=system_prompt))

    outputs = llm.generate(prompts, sampling_params)

    # Extract generated completions
    return prompts, [
        o.outputs[0].text.strip()
        for o in outputs
    ]

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Model Response Generation for Safety Questions")
    parser.add_argument("--model-path", type=str, default="MBZUAI-IFM/Llama-3.1-Nanda-87B-Chat", help="Model Path")
    parser.add_argument("--config-filepath", type=str, default="./config.yaml", help="Config File Path")
    parser.add_argument("--ablations", action="store_true", help="Enable Ablations")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum Number of tokens")
    
    args = parser.parse_args()
    logging.info(f"\nCLI Arguments:\n{vars(args)}")

    # Path to your YAML file
    yaml_path = "./config.yaml"

    # Load the YAML content
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    vllm_config = config["model_response_generation"]["vllm_config"]
    logging.info(f"vLLM Config: {vllm_config}")

    model_config = config["eval_models"][args.model_path]
    logging.info(f"{args.model_path} Model Config: {model_config}")
    
    logging.info(f"Loading {args.model_path} from: {os.getenv('HF_HOME')}/hub")
    # Initialize vLLM model
    llm = LLM(model=args.model_path, 
              tokenizer=args.model_path,
                dtype=vllm_config["dtype"],                                 # Match your original setup
                tensor_parallel_size=vllm_config["tensor_parallel_size"],       # Change to >1 if using multi-GPU
                max_num_seqs=vllm_config["max_num_seqs"], 
                gpu_memory_utilization=vllm_config["gpu_memory_utilization"],
                download_dir=os.getenv("HF_HOME")+"/hub")                       # HuggingFace Cache Directory for Models

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, cache_dir=os.getenv("HF_HOME")+"/hub", trust_remote_code=True)

    
    # Sampling config
    sampling_params = SamplingParams(
        temperature=model_config["sampling_params"]["temperature"],
        top_p=model_config["sampling_params"]["top_p"],
        max_tokens=args.max_tokens
    )
    
    if args.ablations:
        system_prompt_config = model_config["system_prompts"]["ablations"]
    else:
        system_prompt_config = model_config["system_prompts"]["default"]
    
    logging.info(system_prompt_config)

    checkpoint_str = args.model_path.replace("/", "-").replace(".", "p")
    for region in config["regions"]:
        indic_lang_code = config['language_codes'][region]
        for lang_code in [indic_lang_code, "en"]:
            if lang_code in model_config["supported_languages"]:
                # if region == "malayalam" and lang_code == "in-ml":
                for system_prompt_type in system_prompt_config:
                    for turn in range(1, config["model_response_generation"]["n_turns"] + 1):
                        # if (args.model_path == "krutrim-ai-labs/Krutrim-2-instruct" and turn == 4 and system_prompt_type == "empty" and region == "telugu" and lang_code == "in-te") or \
                        #     (args.model_path == "google/gemma-3-27b-it" and turn == 3 and system_prompt_type == "empty" and region == "tamil" and lang_code == "en"):
                            
                        input_filepath = f"{config['input_dir']}/filtered_and_updated/{region}/{lang_code}_questions.jsonl"
                
                        logging.info(f"{args.model_path}, Region: {region}, Language Code: {lang_code}, System Prompt: {system_prompt_type}, Turn: {turn}")
                        output_dir = f"{config['output_dir']}/model_responses/{system_prompt_type}/{turn}/{checkpoint_str}"

                        # now_str = datetime.now().strftime("%d-%m-%y_%H-%M-%S")

                        output_filepath = f"{output_dir}/{region}/{lang_code}_responses_{args.max_tokens}.jsonl"
                        os.makedirs(f"{output_dir}/{region}", exist_ok=True)
                    
                        with open(f"{input_filepath}", "r", encoding="utf-8") as rf:
                            
                            system_prompt = system_prompt_config[system_prompt_type]
                            content = [json.loads(row) for row in rf.read().strip().split("\n")]
                            
                            questions = []
                            with open(f"{output_filepath}", "w", encoding="utf-8") as wf:
                                for row in tqdm(content):
                                    questions.append(row["question"])
                                formatted_prompts, responses = batch_generate(llm, tokenizer, args.model_path, model_config, sampling_params, questions, system_prompt=system_prompt)
                                for f_prompt, row, output in zip(formatted_prompts, content, responses):
                                    row["output"] = output
                                    row["system_instruction"] = system_prompt
                                    row["formatted_prompt"] = f_prompt
                                    wf.write(json.dumps(row, ensure_ascii=False) + "\n")
                                    wf.flush()
                            logging.info(f"Model responses saved at {output_filepath}")

if __name__ == "__main__":
    main()
