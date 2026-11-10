#!/bin/bash
# CPU preflight, pinned model cache, checkpoint visibility and tests; based on template.sbatch.
#SBATCH --job-name=verifier_setup
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set -a; source .env; set +a
source .venv/bin/activate
[ "$(git rev-parse HEAD)" = "${SQUADQA_COMMIT:?Submit with soc.sh submit}" ] || exit 1
reader=2026-10-23-row1-joint-s0-j998373
target="${SQUADQA_SCRATCH:?}/runs/$reader"
source_dir="${READER_CHECKPOINT_DIR:-/home/l/leong616/cs4248-scratch/runs/$reader}"
mkdir -p "$SQUADQA_SCRATCH/runs" "$SQUADQA_SCRATCH/tmp"
export TMPDIR="$SQUADQA_SCRATCH/tmp" OMP_NUM_THREADS=4
if [[ ! -e $target && ! -L $target ]]; then
  test -r "$source_dir/checkpoint.pt" || { echo "Reader checkpoint not readable: $source_dir" >&2; exit 1; }
  ln -s "$source_dir" "$target"
fi
test -r "$target/checkpoint.pt" || { echo "Reader checkpoint not readable: $target" >&2; exit 1; }
HF_HUB_OFFLINE=0 python -m squadqa.download_model answerdotai/ModernBERT-large microsoft/deberta-v3-large
HF_HUB_OFFLINE=0 python -m squadqa.download_model answerdotai/ModernBERT-base microsoft/deberta-v3-base --tokenizer-only
export HF_HUB_OFFLINE=1
python -m pytest -q
