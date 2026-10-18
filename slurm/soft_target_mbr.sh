#!/usr/bin/env bash
# Repeat the F1 soft-target E1 setting on one seed, then compare MAP and MBR on validation.
# Temperature is fitted by squadqa.predict after training; no dev/test data is read.
# Run from the repository root after the cluster tunnel is open and this branch is on the cluster.
#
#   bash slurm/soft_target_mbr.sh show
#   bash slurm/soft_target_mbr.sh smoke
#   bash slurm/soft_target_mbr.sh train
#   bash .claude/skills/soc-cluster/soc.sh fetch <run-id>
#   bash slurm/soft_target_mbr.sh report <run-id>
set -euo pipefail
cd "$(dirname "$0")/.."

SOC=.claude/skills/soc-cluster/soc.sh
EXPERIMENT=1a-soft
SEED=0
TAG=t0.3-f0.0
GPU=a100-40
# alpha=0.5 comes from configs/experiments/1a-soft.yaml; base.yaml sets MBR gamma=1.
ARGS=(--set f1_target_temperature=0.3 --set f1_target_floor=0.0)

submit() {
  local output
  output=$(bash "$SOC" submit "$@") || { printf '%s\n' "$output" >&2; return 1; }
  printf '%s\n' "$output"
}

case "${1:-}" in
  show)
    printf 'Model: DeBERTa-v3-base, joint span head, group target\n'
    printf 'Training: F1 soft target alpha=0.5, temperature tau=0.3, floor=0.0\n'
    printf 'Post-training: fit probability temperature on validation; decode MAP and MBR (gamma=1)\n'
    printf 'GPU: %s; seed: %s; smoke: two 20-minute jobs; full run: 45 minutes\n' "$GPU" "$SEED"
    printf 'Training command: python -m squadqa.train_reader --config configs/experiments/%s.yaml --seed %s %s\n' \
      "$EXPERIMENT" "$SEED" "${ARGS[*]}" ;;
  smoke)
    # The first job saves at step 20 and exits at 30. The second job resumes at 20,
    # finishes at 40, and writes validation predictions and official-scorer metrics.
    run_id="$(date +%F)-soft-mbr-smoke-s${SEED}-t$(date +%H%M%S)"
    common=(--max-steps 40 --set train.checkpoint_every=20 "${ARGS[@]}")
    first=$(submit --gpus="$GPU" --time=00:20:00 \
      --export="ALL,EXPERIMENT=$EXPERIMENT,SEED=$SEED,RUN_ID=$run_id" \
      slurm/train_reader.sbatch "${common[@]}" --stop-after 30)
    printf '%s\n' "$first"
    first_id=${first##* }
    [[ $first_id =~ ^[0-9]+$ ]] || { echo "Could not read Slurm job id: $first" >&2; exit 1; }
    second=$(submit --gpus="$GPU" --time=00:20:00 --dependency="afterany:$first_id" \
      --export="ALL,EXPERIMENT=$EXPERIMENT,SEED=$SEED,RUN_ID=$run_id" \
      slurm/train_reader.sbatch "${common[@]}")
    printf '%s\n' "$second"
    printf 'Smoke run: %s (resume must log step 20)\n' "$run_id" ;;
  train)
    submit --gpus="$GPU" --time=00:45:00 \
      --export="ALL,EXPERIMENT=$EXPERIMENT,SEED=$SEED,TAG=$TAG" \
      slurm/train_reader.sbatch "${ARGS[@]}" ;;
  report)
    run_id="${2:?Usage: bash slurm/soft_target_mbr.sh report <run-id>}"
    case "$run_id" in */*|.*) echo "Invalid run id: $run_id" >&2; exit 1 ;; esac
    python3 - "$run_id" <<'PY'
import json
import sys
from pathlib import Path

run = Path("results") / sys.argv[1]
metrics = json.loads((run / "metrics.json").read_text())
scores = metrics["validation"]
for metric in ("exact", "f1"):
    a, b = scores["map"][metric], scores["mbr"][metric]
    print(f"{metric.upper()}: MAP {a:.3f}, MBR {b:.3f}, MBR - MAP {b-a:+.3f}")
cal = metrics["calibration"]
print(f"Validation temperature: {cal['temperature']:.3f}; NLL: {cal['nll_at_1']:.3f} -> {cal['nll']:.3f}")
report = run / "report_validation.json"
if report.exists():
    data = json.loads(report.read_text())
    print(f"MAP ECE: {data['ece_map']:.4f}; top-10 recall: {data['topk_recall']['10']:.3f}")
PY
    ;;
  *)
    echo "Usage: bash slurm/soft_target_mbr.sh {show|smoke|train|report <run-id>}" >&2
    exit 2 ;;
esac
