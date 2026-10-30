#!/bin/bash
# A verifier (squadqa.train_verifier) on a reader's candidate lists (squadqa.candidates): separate
# (#68, ModernBERT-large, ladder row 4 since the E2 freeze; the default) or shared_lora (#67):
#   bash .claude/skills/soc-cluster/soc.sh submit slurm/train_verifier.sbatch --reader <reader run_id>
#   soc.sh submit --export=ALL,EXPERIMENT=verifier-shared-lora slurm/train_verifier.sbatch --reader <reader run_id>
# Row 4's runs: configs/experiments/verifier-separate.yaml.
# Smoke run (spec §9), then the resume with the same RUN_ID and without --stop-after:
#   soc.sh submit slurm/train_verifier.sbatch --reader <run> --max-steps 40 --stop-after 20
#   soc.sh submit --export=ALL,RUN_ID=<run id> slurm/train_verifier.sbatch --reader <run> --max-steps 40
# Another seed: --export=ALL,SEED=1. Settings: --set verifier.train.lr=2.0e-4 (and so on).
# Needs results/<reader>/ (config, calibration) and the reader's lists in data/verifier/<reader>/ in
# the clone (or --lists DIR), and for shared_lora the reader's checkpoint in $SQUADQA_SCRATCH/runs/<reader>/.
# Writes results/$RUN_ID/ and checkpoints to $SQUADQA_SCRATCH/runs/$RUN_ID/.
# The log goes to logs/<job name>-<job id>.out. Partitions: gpu (up to 3 hours) or gpu-long (up to 3 days).
#SBATCH --job-name=train_verifier
#SBATCH --partition=gpu
#SBATCH --gpus=a100-40
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set -a; source .env; set +a
source .venv/bin/activate
export HF_HUB_OFFLINE=1   # a missing model file raises an error instead of downloading an unpinned one
# A job reads code from the clone when it starts, so stop if the clone moved after soc.sh submit checked it.
[ "$(git rev-parse HEAD)" = "${SQUADQA_COMMIT:?Submit with soc.sh submit, which sets SQUADQA_COMMIT}" ] ||
  { echo "The clone is at $(git rev-parse --short HEAD), not the submitted ${SQUADQA_COMMIT:0:7}." >&2; exit 1; }

EXPERIMENT="${EXPERIMENT:-verifier-separate}"       # the file name in configs/experiments/, without .yaml
SEED="${SEED:-0}"
TAG="${TAG:+-$TAG}"                                 # optional variant name for --set runs (e.g. TAG=base)
RUN_ID="${RUN_ID:-$(date +%F)-$EXPERIMENT$TAG-s$SEED-j$SLURM_JOB_ID}"
echo "run $RUN_ID: commit $(git rev-parse --short HEAD) on $(hostname)"

python -m squadqa.train_verifier --config "configs/experiments/$EXPERIMENT.yaml" --seed "$SEED" --run-id "$RUN_ID" "$@"
