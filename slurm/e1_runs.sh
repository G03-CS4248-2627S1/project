#!/bin/bash
# E1 (#49): the tuning runs of the six-cell comparison, deberta-v3-base, seed 0, scored on the validation
# split. Every cell uses the joint-plain head per group of boundary variants and the shared EM:F1 ratio
# r = 1 (alpha = 0.5, gamma = 1, gamma_r = 1); configs/experiments/1a-{onehot,soft,risk}.yaml.
#
# Run from the repo root on your own machine (Git Bash on Windows), with the soc.sh tunnel open and the
# cluster clone at this commit. Jobs use shared GPU time: agents ask the user first (AGENTS.md).
#
#   bash slurm/e1_runs.sh list                 # the runs, and the command each would submit
#   bash slurm/e1_runs.sh smoke                # the new losses first (spec §9): stop at 30, resume, predict
#   POOL_BRANCH=<results branch> bash slurm/e1_runs.sh pool-json > pool.json   # the runs as a task pool
#   bash slurm/e1_runs.sh report [-o e1.json]  # after the runs, on CPU: calibration, top-k recall, the six cells
#
# GPUS (default a100-40) and TIME (default 00:45:00; a base run takes about 15 minutes) set the request.
set -euo pipefail
SOC=.claude/skills/soc-cluster/soc.sh
[ -f "$SOC" ] || { echo "Run from the repo root." >&2; exit 1; }
GPUS="${GPUS:-a100-40}"
TIME="${TIME:-00:45:00}"

# name|experiment|tag|train_reader args
RUNS=("onehot|1a-onehot||")
for t in 0.1 0.3 1.0; do
  for f in 0.0 0.5; do
    RUNS+=("soft-t$t-f$f|1a-soft|t$t-f$f|--set f1_target_temperature=$t --set f1_target_floor=$f")
  done
done
# The runs added beta x the risk loss (old key risk_weight); risk_share = beta / (1 + beta) is the same mix
# up to the loss scale, so the names and tags keep beta.
for bs in "0.3 0.23076923076923075" "1.0 0.5" "3.0 0.75"; do
  read -r b s <<<"$bs"
  RUNS+=("risk-b$b|1a-risk|b$b|--set risk_share=$s")
done

field() { cut -d'|' -f"$2" <<<"$1"; }

submit() {            # submit [sbatch options] -- [train_reader args]; prints the job id
  local opts=() out id
  while [ "$1" != "--" ]; do opts+=("$1"); shift; done
  shift
  out=$(bash "$SOC" submit --gpus="$GPUS" "${opts[@]}" slurm/train_reader.sbatch "$@") || { echo "$out" >&2; exit 1; }
  id=$(sed -n 's/^Submitted batch job \([0-9][0-9]*\).*/\1/p' <<<"$out")
  [ -n "$id" ] || { echo "No job id in: $out" >&2; exit 1; }
  echo "$id"
}

case "${1:-}" in
  list)
    for r in "${RUNS[@]}"; do
      tag=$(field "$r" 3)
      printf '%-14s soc.sh submit --gpus=%s --time=%s --export=ALL,EXPERIMENT=%s%s slurm/train_reader.sbatch %s\n' \
        "$(field "$r" 1)" "$GPUS" "$TIME" "$(field "$r" 2)" "${tag:+,TAG=$tag}" "$(field "$r" 4)"
    done ;;
  smoke)
    # One chain per new loss: the soft target and the risk term. Job 1 stops after step 30 with a
    # checkpoint at 20; job 2 resumes it and predicts. The resume drops --stop-after, which is not
    # part of the config, so the resume check passes (AGENTS.md lessons).
    for spec in "1a-soft|--set f1_target_temperature=0.3 --set f1_target_floor=0.5" "1a-risk|"; do
      exp=$(field "$spec" 1)
      read -ra extra <<<"$(field "$spec" 2)"
      run_id="$(date +%F)-$exp-smoke-s0-t$(date +%H%M%S)"
      args=(--max-steps 40 --set train.checkpoint_every=20 "${extra[@]}")
      first=$(submit --time=00:20:00 --export=ALL,EXPERIMENT="$exp",RUN_ID="$run_id" -- "${args[@]}" --stop-after 30)
      second=$(submit --time=00:20:00 --dependency=afterany:"$first" --export=ALL,EXPERIMENT="$exp",RUN_ID="$run_id" -- "${args[@]}")
      echo "$run_id: jobs $first (stops at 30) and $second (resumes at 20, predicts)"
      sleep 1
    done
    echo "Check: logs/train_reader-<job id>.out says 'resumed from step 20' and logs 't_F1 expected F1' (soft)." ;;
  pool-json)
    # The runs as a task pool (workflow.md §5.4), pinned to this checkout's commit:
    #   POOL_BRANCH=<results branch> bash slurm/e1_runs.sh pool-json > pool.json && python -m squadqa.pool publish 49 pool.json
    printf '{"pool": 1, "branch": "%s", "commit": "%s", "script": "slurm/train_reader.sbatch",\n' \
      "${POOL_BRANCH:-$(git branch --show-current)}" "$(git rev-parse HEAD)"
    printf ' "sbatch": ["--gpus=%s", "--time=%s"], "tasks": {\n' "$GPUS" "$TIME"
    n=${#RUNS[@]}; k=0
    for r in "${RUNS[@]}"; do
      k=$((k + 1)); tag=$(field "$r" 3)
      read -ra args <<<"$(field "$r" 4)"
      if [ ${#args[@]} -eq 0 ]; then args_json="[]"; else args_json=$(printf '"%s", ' "${args[@]}"); args_json="[${args_json%, }]"; fi
      printf '  "%s": {"export": {"EXPERIMENT": "%s"%s}, "args": %s}%s\n' "$(field "$r" 1)" "$(field "$r" 2)" \
        "${tag:+, \"TAG\": \"$tag\"}" "$args_json" "$([ "$k" -lt "$n" ] && echo ,)"
    done
    printf '}}\n' ;;
  report)
    # Decode-only, on this machine, with every run's candidates_validation.json in results/ (they
    # are not committed): a report per E1 run (T, NLL, ECE, top-k recall), then the six cells on
    # the confirmation seeds 1-3 with the contrasts for H1-H3 (squadqa.report). The soft-target
    # and risk settings are the ones seed 0 picked (#49). Extra arguments go to compare (-o out.json).
    shift
    cells() { ls -d results/*-"$1"-s[123]-j* | xargs -n1 basename | paste -sd, -; }
    onehot=$(cells 1a-onehot); soft=$(cells 1a-soft-t0.3-f0.0); risk=$(cells 1a-risk-b1.0)
    python -m squadqa.report run $(ls -d results/*-1a-*-s[0-9]-j* | xargs -n1 basename)
    python -m squadqa.report compare A=map:"$onehot" B=mbr:"$onehot" C=map:"$soft" D=mbr:"$soft" \
      E=map:"$risk" F=mbr:"$risk" --contrast B-A --contrast C-A --contrast E-A --contrast C-B \
      --contrast E-B --contrast F-B --contrast D+A-B-C --contrast F+A-B-E "$@" ;;
  *)
    sed -n '2,14p' "$0"; exit 1 ;;
esac
