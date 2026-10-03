#!/bin/bash
# Train a reader (squadqa.train_reader), then predict and score the validation split.
#   bash .claude/skills/soc-cluster/soc.sh submit slurm/train_reader.sbatch [train_reader args]
# Experiment:       soc.sh submit --export=ALL,EXPERIMENT=row1-compound slurm/train_reader.sbatch
# Another seed:     soc.sh submit --export=ALL,SEED=1 slurm/train_reader.sbatch
# Resume a run:     soc.sh submit --export=ALL,RUN_ID=<run id> slurm/train_reader.sbatch (same args)
# Smoke run (spec §9): soc.sh submit slurm/train_reader.sbatch --max-steps 40 --stop-after 30 --set train.checkpoint_every=20
#   then resubmit with RUN_ID=<that run id> and the same args; it resumes at step 20 and predicts.
# One large epoch takes about 15 minutes on an A100 (plan §12.1), plus about 2 minutes of featurizing.
# The log goes to logs/<job name>-<job id>.out. Partitions: gpu (up to 3 hours) or gpu-long (up to 3 days).
#SBATCH --job-name=train_reader
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
# A job reads code from the clone when it starts, so stop if the clone moved after soc.sh submit checked it.
[ "$(git rev-parse HEAD)" = "${SQUADQA_COMMIT:?Submit with soc.sh submit, which sets SQUADQA_COMMIT}" ] ||
  { echo "The clone is at $(git rev-parse --short HEAD), not the submitted ${SQUADQA_COMMIT:0:7}." >&2; exit 1; }

EXPERIMENT="${EXPERIMENT:-row0-baseline}"   # the file name in configs/experiments/, without .yaml
SEED="${SEED:-0}"
RUN_ID="${RUN_ID:-$(date +%F)-$EXPERIMENT-s$SEED-j$SLURM_JOB_ID}"
echo "run $RUN_ID: commit $(git rev-parse --short HEAD) on $(hostname)"

python -m squadqa.train_reader --config "configs/experiments/$EXPERIMENT.yaml" --seed "$SEED" --run-id "$RUN_ID" "$@"
