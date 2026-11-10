#!/bin/bash
# Build the one shared span-list set and freeze its hashes; based on template.sbatch.
#SBATCH --job-name=verifier_lists
#SBATCH --partition=gpu
#SBATCH --gpus=a100-40
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set -a; source .env; set +a
source .venv/bin/activate
export HF_HUB_OFFLINE=1
[ "$(git rev-parse HEAD)" = "${SQUADQA_COMMIT:?Submit with soc.sh submit}" ] || exit 1
reader=2026-10-23-row1-joint-s0-j998373
for split in train validation; do
  if [[ ! -e data/verifier/$reader/candidates_$split.json ]]; then
    python -m squadqa.candidates --run-id "$reader" --split "$split" --seed 0 \
      --config configs/experiments/verifier-separate.yaml --batch-size 32
  fi
done
python sandbox/heshi/verifier_lists.py freeze
