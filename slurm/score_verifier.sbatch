#!/bin/bash
# Score a finished verifier on an evaluation set from its checkpoint (squadqa.train_verifier --score, #113).
# Nothing is trained or fitted: lambda and the temperatures come from the run's metrics.json.
# First the reader's lists for the set (stage 5 only, workflow.md §6.1):
#   soc.sh submit slurm/candidates.sbatch --run-id <reader run_id> --split dev --final
# Then each verifier seed:
#   soc.sh submit slurm/score_verifier.sbatch --score <verifier run_id> --eval-set dev --final
# Smoke run (spec §9), which checks the path on validation and writes nothing:
#   soc.sh submit slurm/score_verifier.sbatch --score <verifier run_id> --eval-set validation
# Needs results/<run_id>/ (config, metrics) and the reader's results/<reader>/ in the clone, the lists in
# data/verifier/<reader>/ (or --lists DIR), and the checkpoint in $SQUADQA_SCRATCH/runs/<run_id>/.
# Writes verifier_scores_<set>.json and predictions_<set>_*.json to results/<run_id>/, and adds <set> to metrics.json.
# The log goes to logs/<job name>-<job id>.out. Partitions: gpu (up to 3 hours) or gpu-long (up to 3 days).
#SBATCH --job-name=score_verifier
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
echo "score_verifier: commit $(git rev-parse --short HEAD) on $(hostname)"

python -m squadqa.train_verifier "$@"
