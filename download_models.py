#!/usr/bin/env python3
"""Pre-download all model weights to the HF cache so compute nodes don't need internet."""

import yaml
import os
from huggingface_hub import login, snapshot_download

# Load secrets
with open("secrets.yaml", "r") as f:
    secrets = yaml.safe_load(f)

hf_token = secrets.get("HF_TOKEN")
hf_home = secrets.get("HF_HOME")

if hf_token:
    login(token=hf_token)

# Models to download
MODELS = [
    "Qwen/Qwen3-4B-Instruct-2507",
    "google/gemma-3-4b-it",
    "CohereLabs/aya-23-8B",
    "nvidia/Nemotron-4-Mini-Hindi-4B-Instruct",
    "CoRover/BharatGPT-3B-Indic",
    "ai4bharat/Airavata",
    # "meta-llama/Llama-3.2-3B-Instruct",
]

for model in MODELS:
    print(f"\n{'='*60}")
    print(f"Downloading: {model}")
    print(f"{'='*60}")
    try:
        snapshot_download(
            model,
            cache_dir=os.path.join(hf_home, "hub") if hf_home else None,
            token=hf_token,
        )
        print(f"Done: {model}")
    except Exception as e:
        print(f"FAILED: {model} — {e}")
