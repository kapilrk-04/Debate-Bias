# Setup Guide

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (for GPU inference)
- HuggingFace account and token

## Installation

### 1. Install Dependencies

```bash
pip install torch transformers vllm huggingface-hub pyyaml python-dotenv tqdm
```

Or if you have a `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2. Configure Secrets

Copy the example secrets file and fill in your credentials:

```bash
cp secrets.yaml.example secrets.yaml
```

Edit `secrets.yaml` with your credentials:

```yaml
HF_TOKEN: "hf_your_actual_token_here"
HF_HOME: "/path/to/your/huggingface/cache"
```

**Important:**
- Get your HuggingFace token from: https://huggingface.co/settings/tokens
- The `secrets.yaml` file is already in `.gitignore` and will NOT be committed to git
- Never share or commit your `secrets.yaml` file

### 3. Verify Setup

Test that everything is configured correctly:

```bash
python -c "import yaml; secrets = yaml.safe_load(open('secrets.yaml')); print('HF_TOKEN:', 'configured' if secrets.get('HF_TOKEN') else 'missing')"
```

## Directory Structure

After setup, your directory should look like:

```
.
├── secrets.yaml                    # Your credentials (NOT in git)
├── secrets.yaml.example            # Template file
├── generate_single_llm.py          # Single LLM script
├── generate_selfreflect_llm.py     # Self-reflection script
├── generate_multi_agent.py         # Multi-agent debate script
├── prompts_single_new.py           # Prompt templates
├── prompts_selfreflect_new.py      # Prompt templates
├── prompts_multi_new.py            # Prompt templates
├── job_response_generation.sh      # Bash script for all scenarios
├── job_judgement_only.sh           # Bash script for judgments only
├── train.slurm                     # SLURM submission script
├── processed_prompts/              # Input prompt files
│   ├── bias_judgement_negative.jsonl
│   ├── bias_judgement_positive.jsonl
│   ├── bias_plausible_negative.jsonl
│   ├── bias_plausible_positive.jsonl
│   ├── stereotype_judgement.jsonl
│   └── stereotype_plausible.jsonl
└── outputs/                        # Generated responses (created automatically)
    ├── single_llm/
    ├── selfreflect/
    ├── multi_agent_same/
    └── multi_agent_diff/
```

## HuggingFace Token Setup

### Getting Your Token

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Give it a name (e.g., "research-project")
4. Select "Read" permission (or "Write" if you plan to upload models)
5. Click "Generate token"
6. Copy the token and paste it in `secrets.yaml`

### Model Access

Some models require explicit access approval:

1. **Meta LLama models**: Visit the model page and request access
   - https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
   - https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

2. After approval, the models will be accessible with your token

## Cache Directory

The `HF_HOME` directory stores downloaded models to avoid re-downloading:

```yaml
HF_HOME: "/scratch/your_username/huggingface_cache"  # Example for HPC
# or
HF_HOME: "/home/your_username/.cache/huggingface"    # Example for local
```

**Tips:**
- Use a location with sufficient storage (models can be 10GB+ each)
- On HPC systems, use scratch or project storage, not home directory
- The directory will be created automatically if it doesn't exist

## Testing Your Setup

### Quick Test

Run a single generation to verify everything works:

```bash
python generate_single_llm.py \
    --model-path "meta-llama/Meta-Llama-3-8B-Instruct" \
    --input-file "processed_prompts/bias_judgement_negative.jsonl" \
    --output-file "test_output.jsonl" \
    --backend transformers \
    --max-tokens 128
```

If successful, you should see:
```
Successfully logged in to Hugging Face Hub.
Loading meta-llama/Meta-Llama-3-8B-Instruct with transformers
Loaded X samples
Generating: 100%|████████████| X/X
Completed! Saved X responses to test_output.jsonl
```

## Troubleshooting

### "secrets.yaml not found"

```bash
cp secrets.yaml.example secrets.yaml
# Then edit secrets.yaml with your credentials
```

### "Invalid token" or authentication errors

- Check that your token is correct in `secrets.yaml`
- Verify the token has "Read" permissions
- For gated models (LLama, etc.), ensure you've been granted access on HuggingFace

### "Out of memory" errors

- Reduce `--batch-size` (for vLLM)
- Use a smaller model
- Use `--backend transformers` instead of vLLM
- Lower `gpu_memory_utilization` in the scripts

### Module not found errors

```bash
pip install pyyaml transformers torch vllm huggingface-hub tqdm
```

## Next Steps

Once setup is complete, see:
- [USAGE_GUIDE.md](USAGE_GUIDE.md) - How to run the generation scripts
- [SLURM_GUIDE.md](SLURM_GUIDE.md) - How to submit SLURM jobs

## Security Best Practices

1. ✅ Never commit `secrets.yaml` to git
2. ✅ Use read-only tokens when possible
3. ✅ Rotate tokens periodically
4. ✅ Don't share tokens in chat/email
5. ✅ Keep `secrets.yaml` file permissions restricted: `chmod 600 secrets.yaml`

## Environment Variables Alternative

If you prefer to use environment variables instead of `secrets.yaml`, you can:

1. Set environment variables:
```bash
export HF_TOKEN="your_token_here"
export HF_HOME="/path/to/cache"
```

2. Or create a `.env` file (also in `.gitignore`):
```
HF_TOKEN=your_token_here
HF_HOME=/path/to/cache
```

The scripts will work with either method, but `secrets.yaml` is recommended for better organization.
