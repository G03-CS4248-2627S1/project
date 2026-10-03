#!/bin/bash
# Quick sanity fine-tune (sandbox, not the spec's E0): DeBERTa-v3-large, token-level start/end head, 1 epoch.
# Submit from the repo root:  sbatch sandbox/bruhmiuz/quick_finetune.sbatch [script args]
# Other seeds (default 0):    sbatch --export=ALL,SEED=1 sandbox/bruhmiuz/quick_finetune.sbatch
# Resume after a timeout by resubmitting with the same run id:
#   sbatch --export=ALL,RUN_ID=<run id> sandbox/bruhmiuz/quick_finetune.sbatch
#SBATCH --job-name=quick-ft
#SBATCH --partition=gpu
#SBATCH --gpus=a100-40
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set -a; source .env; set +a
source .venv/bin/activate
export PYTHONPATH="$PWD"
SEED="${SEED:-0}"
RUN_ID="${RUN_ID:-$(date +%F)-quickft-s$SEED-j$SLURM_JOB_ID}"
python sandbox/bruhmiuz/quick_finetune.py --run-id "$RUN_ID" --seed "$SEED" "$@"
