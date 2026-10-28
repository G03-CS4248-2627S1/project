#!/bin/bash
# Verifier candidate lists (squadqa.candidates, spec §5.1, #66) from a trained reader, one split per job.
#   bash .claude/skills/soc-cluster/soc.sh submit slurm/candidates.sbatch --run-id <reader run_id> --split validation
#   bash .claude/skills/soc-cluster/soc.sh submit slurm/candidates.sbatch --run-id <reader run_id> --split train
# Other list settings: add e.g. --set verifier.sample_temperature=1.5 (and --seed for the sampling seed).
# The reader run needs results/<run_id>/ (config.yaml, calibration.json) in the clone and its checkpoint in
# $SQUADQA_SCRATCH/runs/<run_id>/. Writes $SQUADQA_SCRATCH/candidates/<run_id>/candidates_<split>.json and
# summary_<split>.json; nothing goes to results/, since the lists are large (workflow.md §2.2).
# Validation (7,708 questions) is one reader pass; train (79,577) about ten times that, in chunks of 4,096.
# The log goes to logs/<job name>-<job id>.out. Partitions: gpu (up to 3 hours) or gpu-long (up to 3 days).
#SBATCH --job-name=candidates
#SBATCH --partition=gpu
#SBATCH --gpus=a100-40
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set -a; source .env; set +a
source .venv/bin/activate
export HF_HUB_OFFLINE=1   # a missing model file raises an error instead of downloading an unpinned one
# A job reads code from the clone when it starts, so stop if the clone moved after soc.sh submit checked it.
[ "$(git rev-parse HEAD)" = "${SQUADQA_COMMIT:?Submit with soc.sh submit, which sets SQUADQA_COMMIT}" ] ||
  { echo "The clone is at $(git rev-parse --short HEAD), not the submitted ${SQUADQA_COMMIT:0:7}." >&2; exit 1; }
echo "candidates: commit $(git rev-parse --short HEAD) on $(hostname)"

python -m squadqa.candidates "$@"
