#!/bin/bash
# Full-span inference diagnostics (#95/#96); no training or test-set evaluation.
#SBATCH --job-name=answer-dist-smoke
#SBATCH --partition=gpu
#SBATCH --gpus=a100-40
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set -a; source .env; set +a
source .venv/bin/activate
export HF_HUB_OFFLINE=1   # a missing model file raises an error instead of downloading an unpinned one
# A job reads code from the clone when it starts, so stop if the clone moved after soc.sh submit checked it.
[ "$(git rev-parse HEAD)" = "${SQUADQA_COMMIT:?Submit with soc.sh submit, which sets SQUADQA_COMMIT}" ] ||
  { echo "The clone is at $(git rev-parse --short HEAD), not the submitted ${SQUADQA_COMMIT:0:7}." >&2; exit 1; }

echo "answer distribution: commit $(git rev-parse --short HEAD) on $(hostname)"
python -m squadqa.answer_distribution shape --run-id 2026-10-23-row1-joint-s0-j998373 2026-10-05-row0-baseline-s0-j910132 --limit 8 --output-root "local/smoke-$SLURM_JOB_ID"
python -m squadqa.answer_distribution multi-answer --run-id 2026-10-23-row1-joint-s0-j998373 --limit 8 --output-root "local/smoke-$SLURM_JOB_ID"
