#!/bin/bash
# Sequential MAP loss search. Submit PHASE=e first, then PHASE=c. Each phase has
# six full one-epoch, seed-0 trainings and a fetchable results/<SEARCH_ID>/summary.json.
# Example (after asking the user):
#   soc.sh submit --gpus=a100-40 --export=ALL,PHASE=e slurm/map_loss_search.sbatch
#   soc.sh submit --gpus=a100-40 --export=ALL,PHASE=c slurm/map_loss_search.sbatch
# To resume an interrupted job: --export=ALL,PHASE=e,SEARCH_ID=<original search id>
# Based on slurm/template.sbatch.
#SBATCH --job-name=map_loss_search
#SBATCH --partition=gpu
#SBATCH --gpus=a100-40
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set -a; source .env; set +a
source .venv/bin/activate
export HF_HUB_OFFLINE=1
[ "$(git rev-parse HEAD)" = "${SQUADQA_COMMIT:?Submit with soc.sh submit}" ] ||
  { echo "Clone moved after submission" >&2; exit 1; }

: "${PHASE:?Set PHASE=e or PHASE=c}"
case "$PHASE" in e|c) ;; *) echo "PHASE must be e or c" >&2; exit 2 ;; esac
SEARCH_ID="${SEARCH_ID:-$(date +%F)-map-loss-bo-$PHASE-s0-j$SLURM_JOB_ID}"
echo "Search $SEARCH_ID, phase $PHASE, commit $(git rev-parse --short HEAD)"
python -m squadqa.loss_search --phase "$PHASE" --search-id "$SEARCH_ID"
