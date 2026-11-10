#!/bin/bash
# Full one-epoch complement/control, with safe resume; based on template.sbatch.
#SBATCH --job-name=verifier_trial
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
export HF_HUB_OFFLINE=1
[ "$(git rev-parse HEAD)" = "${SQUADQA_COMMIT:?Submit with soc.sh submit}" ] || exit 1
experiment="${EXPERIMENT:?Name one of the verifier complement configs}"
case "$experiment" in
  verifier-separate|verifier-gold-group|verifier-ce-margin|verifier-deberta-independent) ;;
  *) echo "Unsupported experiment: $experiment" >&2; exit 1 ;;
esac
if [[ -n ${RESUME_JOB_ID:-} ]]; then
  matches=(results/*-s0-j"$RESUME_JOB_ID")
  [[ ${#matches[@]} == 1 && -d ${matches[0]} ]] || { echo "Cannot uniquely locate run for $RESUME_JOB_ID" >&2; exit 1; }
  run="${matches[0]#results/}"
else
  run="${RUN_ID:-$(date +%F)-$experiment-s0-j$SLURM_JOB_ID}"
fi
# A chained resume must not rescore a finished run or overwrite its training-time metadata.
if [[ -s results/$run/metrics.json ]] && python -c \
  'import json,sys; m=json.load(open(sys.argv[1])); sys.exit(0 if m.get("steps",0)>0 and len(m.get("validation",{}))==6 else 1)' \
  "results/$run/metrics.json"; then
  echo "Run already completed: $run; skipping resume."
  exit 0
fi
python sandbox/heshi/verifier_lists.py audit
python -m squadqa.train_verifier --config "configs/experiments/$experiment.yaml" \
  --reader 2026-10-23-row1-joint-s0-j998373 --seed 0 --run-id "$run" --per-device-batch 16
python sandbox/heshi/verifier_lists.py audit
