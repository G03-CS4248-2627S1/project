#!/bin/bash
# Milestone 1b, joint span head: development runs on deberta-v3-base, seed 0, scored on the validation
# split (workflow.md §5.1). Every 1b run has the no-answer option; the runs compare the four objectives
# of Fajcik et al. (2021) and, for the bilinear ones, the length bias and the position encodings
# added right before the bilinear term (spec §4.1, §4.3).
#
# Run from the repo root on your own machine (Git Bash on Windows), with the soc.sh tunnel open and the
# cluster clone at this commit (soc.sh submit refuses otherwise). Jobs use shared GPU time: agents
# ask the user before submitting (AGENTS.md, "Submitting jobs").
#
#   bash slurm/joint_head_tests.sh list                  # the runs, and the commands each would submit
#   bash slurm/joint_head_tests.sh smoke                 # new heads first (spec §9): stop at 30, resume, predict
#   bash slurm/joint_head_tests.sh grid [name ...]       # the comparison runs: all, or the named ones
#   bash slurm/joint_head_tests.sh e3 <run_id> ...       # cross-span diagnostic (plan E3) of finished runs
#   bash slurm/joint_head_tests.sh summary               # table of the fetched runs (soc.sh fetch <run_id> first)
#
# GPUS (default a100-40) and TIME (default 00:45:00; a base run takes about 15 minutes) set the request.
set -euo pipefail
SOC=.claude/skills/soc-cluster/soc.sh
[ -f "$SOC" ] || { echo "Run from the repo root." >&2; exit 1; }
GPUS="${GPUS:-a100-40}"
TIME="${TIME:-00:45:00}"
BASE="--set model=microsoft/deberta-v3-base"

# name|experiment|tag|train_reader args. The reference is row 0 itself on base: the standard head
# without the no-answer option, so the 1b control (independent) shows what the option alone does.
RUNS=(
  "row0-base|row0-baseline|base|$BASE"
  "independent|1b-independent||"
  "conditional|1b-conditional||"
  "joint-lenbias|1b-joint|lenbias|"
  "joint-posenc|1b-joint|posenc|--set length_bias=false --set bilinear_pos_enc=true"
  "joint-lenbias-posenc|1b-joint|lenbias-posenc|--set bilinear_pos_enc=true"
  "joint-plain|1b-joint|plain|--set length_bias=false"
  "compound-lenbias|1b-compound|lenbias|"
  "compound-posenc|1b-compound|posenc|--set length_bias=false --set bilinear_pos_enc=true"
  "compound-lenbias-posenc|1b-compound|lenbias-posenc|--set bilinear_pos_enc=true"
  "compound-plain|1b-compound|plain|--set length_bias=false"
)

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
      printf '%-24s soc.sh submit --gpus=%s --time=%s --export=ALL,EXPERIMENT=%s%s slurm/train_reader.sbatch %s\n' \
        "$(field "$r" 1)" "$GPUS" "$TIME" "$(field "$r" 2)" "${tag:+,TAG=$tag}" "$(field "$r" 4)"
    done ;;
  smoke)
    # The two new code paths: the compound head with every option (joint, auxiliary, length bias,
    # position encodings, null) and the conditional head. Job 1 stops after step 30 with a checkpoint
    # at 20; job 2 resumes it and predicts. The resume drops --stop-after, which is not part of the
    # config, so the resume check passes (AGENTS.md lessons).
    for spec in "1b-compound|--set bilinear_pos_enc=true" "1b-conditional|"; do
      exp=$(field "$spec" 1)
      read -ra extra <<<"$(field "$spec" 2)"
      run_id="$(date +%F)-$exp-smoke-s0-t$(date +%H%M%S)"
      args=(--max-steps 40 --set train.checkpoint_every=20 "${extra[@]}")
      first=$(submit --time=00:20:00 --export=ALL,EXPERIMENT="$exp",RUN_ID="$run_id" -- "${args[@]}" --stop-after 30)
      second=$(submit --time=00:20:00 --dependency=afterany:"$first" --export=ALL,EXPERIMENT="$exp",RUN_ID="$run_id" -- "${args[@]}")
      echo "$run_id: jobs $first (stops at 30) and $second (resumes at 20, predicts)"
    done
    echo "Check: logs/train_reader-<job id>.out says 'resumed from step 20', and soc.sh fetch <run_id> has metrics.json." ;;
  grid)
    shift
    for r in "${RUNS[@]}"; do
      name=$(field "$r" 1)
      if [ $# -gt 0 ] && ! printf '%s\n' "$@" | grep -qx "$name"; then continue; fi
      exp=$(field "$r" 2); tag=$(field "$r" 3)
      read -ra args <<<"$(field "$r" 4)"
      id=$(submit --time="$TIME" --export=ALL,EXPERIMENT="$exp"${tag:+,TAG=$tag} -- "${args[@]}")
      echo "$name: job $id, run id $(date +%F)-$exp${tag:+-$tag}-s0-j$id"
    done ;;
  e3)
    shift
    [ $# -gt 0 ] || { echo "usage: joint_head_tests.sh e3 <run_id> ..." >&2; exit 1; }
    bash "$SOC" submit --gpus="$GPUS" --time=00:30:00 slurm/diagnostics.sbatch --run-id "$@" ;;
  summary)
    PYTHONIOENCODING=utf-8 python - <<'PY'
import json, re
from pathlib import Path

def scalar(cfg, key):
    m = re.search(rf"^{key}: (.*)$", cfg, re.M)
    return m.group(1).strip() if m else "-"

rows = []
for d in sorted(Path("results").iterdir()):
    run, metrics = d / "run.json", d / "metrics.json"
    if not (run.exists() and metrics.exists()):
        continue
    info = json.loads(run.read_text(encoding="utf-8"))
    if not (info["experiment"].startswith("1b-") or "-row0-baseline-base-" in d.name) or "-smoke-" in d.name:
        continue
    cfg, m = (d / "config.yaml").read_text(encoding="utf-8"), json.loads(metrics.read_text(encoding="utf-8"))
    head = scalar(cfg, "span_head")
    bil = head in ("joint", "compound")
    na = m.get("no_answer_validation")
    e3 = d / "cross_span_validation.json"
    e3 = json.loads(e3.read_text(encoding="utf-8"))["ratio"] if e3.exists() else None
    v = m["validation"]
    rows.append([d.name, head, scalar(cfg, "length_bias") if bil else "", scalar(cfg, "bilinear_pos_enc") if bil else "",
                 scalar(cfg, "no_answer"), f"{v['map']['exact']:.2f} / {v['map']['f1']:.2f}",
                 f"{v['mbr']['exact']:.2f} / {v['mbr']['f1']:.2f}", f"{m['calibration']['temperature']:.2f}",
                 f"{na['mean_p']:.4f}" if na else "", f"{e3['<0.5']['mean']:.3f}" if e3 else ""])
head = ["run", "span_head", "length_bias", "pos_enc", "no_answer", "MAP EM / F1", "MBR EM / F1", "T", "mean p(null)",
        "E3 ratio, top-1 < 0.5"]
print("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
for r in rows:
    print("| " + " | ".join(r) + " |")
if not rows:
    print("(no 1b runs in results/; fetch them with soc.sh fetch <run_id>)")
PY
    ;;
  *)
    awk 'NR > 1 && !/^#/ { exit } NR > 1 { sub(/^# ?/, ""); print }' "$0"
    exit 1 ;;
esac
