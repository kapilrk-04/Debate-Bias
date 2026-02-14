#!/bin/bash
#SBATCH --job-name=surakshaeval_vllm_responses
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

# Load CUDA 12.1 or higher (try different module names)
# Remove the old cuda-11.6 line and try one of these:
# module load cuda/12.1 || module load cuda-toolkit/12.1 || module load cudatoolkit/12.1 || module load cuda/12.4

# Alternative: If no CUDA 12 module exists, use PyTorch's bundled CUDA
# Just comment out or remove the module load line entirely
# PyTorch installed with cu121 includes its own CUDA runtime

# Optional: Print environment info for debugging
echo "CUDA_HOME: $CUDA_HOME"
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")')"

# Run training scripts
#./job_fairi_tales_vllm_responses.sh
#./job_fairi_tales_translated_vllm_responses.sh
#./job_fairi_tales_parse_judgements.sh

# Run new response generation pipeline
./job_response_generation.sh

