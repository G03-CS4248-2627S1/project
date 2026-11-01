#!/bin/bash
# Check stop/resume/evaluation for all four settings in one GPU allocation; template.sbatch.
#SBATCH --job-name=verifier_smoke
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
export HF_HUB_OFFLINE=1
[ "$(git rev-parse HEAD)" = "${SQUADQA_COMMIT:?Submit with soc.sh submit}" ] || exit 1
python sandbox/heshi/verifier_lists.py audit
reader=2026-10-23-row1-joint-s0-j998373
for exp in verifier-separate verifier-gold-group verifier-ce-margin verifier-deberta-independent; do
  run="$(date +%F)-$exp-smoke-s0-j$SLURM_JOB_ID"
  args=(--config "configs/experiments/$exp.yaml" --reader "$reader" --seed 0 --run-id "$run"
        --per-device-batch 16 --max-steps 40 --set verifier.train.checkpoint_every=20)
  python -m squadqa.train_verifier "${args[@]}" --stop-after 20
  python -m squadqa.train_verifier "${args[@]}"
done
python sandbox/heshi/verifier_lists.py audit
