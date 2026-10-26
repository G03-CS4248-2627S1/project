#!/bin/bash
# Pure-risk checkpoint/resume smoke in one allocation; screens depend on its success.
# Started from slurm/template.sbatch. Submit through soc.sh submit.
#SBATCH --job-name=ce_risk_smoke
#SBATCH --partition=gpu
#SBATCH --gpus=a100-40
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set -a; source .env; set +a
source .venv/bin/activate
export HF_HUB_OFFLINE=1
[ "$(git rev-parse HEAD)" = "${SQUADQA_COMMIT:?Submit with soc.sh submit}" ] ||
  { echo "Checkout differs from submitted commit" >&2; exit 1; }
RUN_ID="${RUN_ID:-$(date +%F)-row1-ce-risk-pure-smoke-s0-j$SLURM_JOB_ID}"
args=(--config configs/experiments/row1-ce-risk.yaml --seed 0 --run-id "$RUN_ID"
      --per-device-batch 8 --max-steps 40 --set train.checkpoint_every=20
      --set loss_weights.group=0.0 --set loss_weights.f1_soft=0.0
      --set loss_weights.risk=1.0 --set risk_em_weight=1.0)
python -m squadqa.train_reader "${args[@]}" --stop-after 30
python -m squadqa.train_reader "${args[@]}"
