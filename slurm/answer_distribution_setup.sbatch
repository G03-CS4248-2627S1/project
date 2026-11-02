#!/bin/bash
# CPU preparation for #95/#96: pinned E6 data, tokenizer/config cache, tests.
#SBATCH --job-name=answer-dist-setup
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set -a; source .env; set +a
[ "$(git rev-parse HEAD)" = "${SQUADQA_COMMIT:?Submit with soc.sh submit}" ]
export PATH="$HOME/.local/bin:$PATH"
uv sync --frozen
source .venv/bin/activate
# These fetch only pinned tokenizer/config files, never additional encoder weights.
python -m squadqa.download_model microsoft/deberta-v3-large --tokenizer-only
python -m squadqa.download_model microsoft/deberta-v3-base --tokenizer-only
python -m squadqa.multi_answer
export HF_HUB_OFFLINE=1
python -m pytest -q
