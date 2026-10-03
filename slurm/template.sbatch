#!/bin/bash
# Template for job scripts. Copy it to slurm/<name>.sbatch, replace each TODO, and don't submit this file.
# Submit from your own machine; soc.sh checks that the cluster clone is at your commit and avoids broken nodes:
#   bash .claude/skills/soc-cluster/soc.sh submit slurm/<name>.sbatch [script args]
# Another seed:          soc.sh submit --export=ALL,SEED=1 slurm/<name>.sbatch
# Resume a run:          soc.sh submit --export=ALL,RUN_ID=<run id> slurm/<name>.sbatch
# Other GPU or time:     soc.sh submit --gpus=a100-80 --time=06:00:00 --partition=gpu-long slurm/<name>.sbatch
#
# The log goes to logs/<job name>-<job id>.out. Partitions: gpu (up to 3 hours) or gpu-long (up to 3 days).
# Request a bf16 GPU: a100-40, a100-80, h100-47, h100-96 or h200-141.
#SBATCH --job-name=TODO
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
export HF_HUB_OFFLINE=1   # a missing model file raises an error instead of downloading an unpinned one

EXPERIMENT=TODO           # the file name in configs/experiments/, without .yaml
SEED="${SEED:-0}"
RUN_ID="${RUN_ID:-$(date +%F)-$EXPERIMENT-s$SEED-j$SLURM_JOB_ID}"
echo "run $RUN_ID: commit $(git rev-parse --short HEAD) on $(hostname)"

python -m squadqa.TODO --config "configs/experiments/$EXPERIMENT.yaml" --seed "$SEED" --run-id "$RUN_ID" "$@"
