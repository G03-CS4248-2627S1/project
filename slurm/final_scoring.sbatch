#!/bin/bash
# Final scoring (stage 5, workflow.md §6.1, #115): one run on all seven final sets. Nothing is fitted: a reader
# reuses its validation temperature, a verifier its lambda and temperatures (spec §6, rule 1).
#   soc.sh submit --gpus=flexible slurm/final_scoring.sbatch reader <reader run_id>
#   soc.sh submit --gpus=flexible slurm/final_scoring.sbatch lists <reader run_id>
#   soc.sh submit --gpus=flexible slurm/final_scoring.sbatch verifier <verifier run_id>     # once lists has finished
# reader:   squadqa.predict; writes predictions_<set>_{map,mbr}.json and candidates_<set>.json to results/<run_id>/.
# lists:    squadqa.candidates; writes data/verifier/<run_id>/candidates_<set>.json, which verifier needs.
# verifier: squadqa.train_verifier --score; first checks the run's validation predictions (it stops if over 1% of
#           answers differ), then writes predictions_<set>_*.json and verifier_scores_<set>.json to results/<run_id>/.
# reader and verifier add each set to metrics.json. Each run needs its results/<run_id>/ in the clone and its
# checkpoint in $SQUADQA_SCRATCH/runs/<run_id>/ (a link is fine, as for row 5's readers).
# The log goes to logs/<job name>-<job id>.out. Partitions: gpu (up to 3 hours) or gpu-long (up to 3 days).
#SBATCH --job-name=final_scoring
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
echo "final_scoring: commit $(git rev-parse --short HEAD) on $(hostname)"

mode=${1:?reader, lists or verifier}
run=${2:?a run id}
sets="dev addsent addonesent squadshifts-new_wiki squadshifts-nyt squadshifts-reddit squadshifts-amazon_reviews"
[ "$mode" = verifier ] && python -m squadqa.train_verifier --score "$run" --eval-set validation
for s in $sets; do
  echo "=== $mode $run $s"
  case "$mode" in
    reader)   python -m squadqa.predict --run-id "$run" --eval-set "$s" --final ;;
    lists)    python -m squadqa.candidates --run-id "$run" --split "$s" --final ;;
    verifier) python -m squadqa.train_verifier --score "$run" --eval-set "$s" --final ;;
    *)        echo "unknown mode $mode: reader, lists or verifier" >&2; exit 2 ;;
  esac
done
