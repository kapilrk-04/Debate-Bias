#!/bin/bash
#SBATCH --job-name=fsga_vllm_responses
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia-rtx-5000-ada-generation:1
#SBATCH --mem=80G
#SBATCH --time=20:00:00

# Activate conda environment
source /home/kapil.kavitha/miniconda3/bin/activate new_pytorch_env

# Set CUDA_HOME so PyTorch JIT can compile C extensions (e.g. torch-c-dlpack-ext)
export CUDA_HOME=/usr/local/cuda-12.2
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_EXTENSIONS_DIR=/tmp/$USER/torch_extensions
mkdir -p "$TORCH_EXTENSIONS_DIR"

# Enable verbose vLLM logging
export VLLM_LOGGING_LEVEL=DEBUG

# Optional: Print environment info for debugging
echo "CUDA_HOME: $CUDA_HOME"
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")')"
echo "nvidia-smi snapshot:"
nvidia-smi || true

# Run training scripts
#./job_fairi_tales_vllm_responses.sh
#./job_fairi_tales_translated_vllm_responses.sh
#./job_fairi_tales_parse_judgements.sh

# Run new response generation pipeline
# ./job_response_generation.sh
./run_judge_decisions.sh