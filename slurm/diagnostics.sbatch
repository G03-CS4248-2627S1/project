#!/bin/bash
# Cross-span diagnostic (plan E3, squadqa.diagnostics) of trained readers, on the validation split.
#   bash .claude/skills/soc-cluster/soc.sh submit slurm/diagnostics.sbatch --run-id <run_id> [<run_id> ...]
# Each run needs results/<run_id>/ (config.yaml, calibration.json) in the clone and its checkpoint in
# $SQUADQA_SCRATCH/runs/<run_id>/. Writes results/<run_id>/cross_span_validation{,_questions}.json.
# The log goes to logs/<job name>-<job id>.out. Partitions: gpu (up to 3 hours) or gpu-long (up to 3 days).
#SBATCH --job-name=diagnostics
#SBATCH --partition=gpu
#SBATCH --gpus=a100-40
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set -a; source .env; set +a
source .venv/bin/activate
export HF_HUB_OFFLINE=1   # a missing model file raises an error instead of downloading an unpinned one
# A job reads code from the clone when it starts, so stop if the clone moved after soc.sh submit checked it.
[ "$(git rev-parse HEAD)" = "${SQUADQA_COMMIT:?Submit with soc.sh submit, which sets SQUADQA_COMMIT}" ] ||
  { echo "The clone is at $(git rev-parse --short HEAD), not the submitted ${SQUADQA_COMMIT:0:7}." >&2; exit 1; }
echo "diagnostics: commit $(git rev-parse --short HEAD) on $(hostname)"

python -m squadqa.diagnostics "$@"
