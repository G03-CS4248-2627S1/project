#!/bin/bash
# Checks the environment on a GPU node: PyTorch sees the GPU, and the GPU supports bf16 natively.
# Writes results/<date>-smoke-j<job id>/smoke.json.
# Submit from the repo root after `uv sync`: sbatch slurm/smoke_test.sbatch
#SBATCH --job-name=smoke
#SBATCH --partition=gpu
#SBATCH --gpus=a100-40
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set -a; source .env; set +a
source .venv/bin/activate
export RUN_ID="$(date +%F)-smoke-j$SLURM_JOB_ID"

nvidia-smi
python - <<'EOF'
import json, os, torch
import peft, scipy, transformers   # fails here if the environment lacks a locked dependency

assert torch.cuda.is_available(), "PyTorch cannot see a GPU"
name = torch.cuda.get_device_name()
cap = torch.cuda.get_device_capability()
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}) on {name}, compute capability {cap}")
# is_bf16_supported() also reports emulated bf16, so check the GPU generation instead.
assert cap >= (8, 0), f"{name} has no native bf16; request an A100 or H100"
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
mean = (x @ x).float().abs().mean().item()
print("bf16 matmul ok:", mean)
print(f"transformers {transformers.__version__}, peft {peft.__version__}, scipy {scipy.__version__}")

run_dir = os.path.join("results", os.environ["RUN_ID"])
os.makedirs(run_dir)
with open(os.path.join(run_dir, "smoke.json"), "w") as f:
    json.dump({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": name,
               "capability": list(cap), "bf16_matmul_mean": mean, "transformers": transformers.__version__,
               "peft": peft.__version__, "scipy": scipy.__version__}, f, indent=2)
print("wrote", run_dir)
EOF
