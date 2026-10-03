#!/usr/bin/env bash
# Run commands on the SoC cluster and copy results back, through an SSH tunnel.
# The jump host (stujump) always asks for a password, so `connect` opens a tunnel through it once;
# every other command logs in to xlogin through that tunnel with your SSH key, without a password.
# Needs the NUS Cisco VPN (NUS Wi-Fi is not enough) and SOC_USER / SOC_REPO in .env.
#
#   soc.sh connect            open the tunnel (asks for your SoC password; keep that terminal open)
#   soc.sh setup-key          install your SSH key on the cluster, once (needs the tunnel)
#   soc.sh check              test that the tunnel and key login work
#   soc.sh run '<command>'    run a command in your cluster clone
#   soc.sh submit [sbatch options] <script> [script args]
#                             submit a job from your cluster clone, which must be at this checkout's commit
#   soc.sh jobs               list your queued and running jobs
#   soc.sh sync               build or update the cluster clone's environment (uv sync, in a CPU job)
#   soc.sh list [n]           list the n most recent runs in your cluster clone (default 20)
#   soc.sh fetch <run_id>...  copy results/<run_id>/ from the cluster into local results/
#   soc.sh log <file>         copy logs/<file> from the cluster into local logs/
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

[ -f .env ] || { echo "No .env: copy .env.example to .env and set SOC_USER and SOC_REPO." >&2; exit 1; }
set -a; source <(tr -d '\r' < .env); set +a
: "${SOC_USER:?Set SOC_USER (your SoC unix id) in .env}"
: "${SOC_REPO:?Set SOC_REPO (your repo clone on the cluster) in .env}"
case "$SOC_REPO" in "~"*) echo "SOC_REPO must be relative to your home directory or absolute, without ~." >&2; exit 1 ;; esac

PORT=2222   # local end of the tunnel
HOST="$SOC_USER@localhost"
# Nodes that fail our jobs (see SKILL.md, "Cluster facts"). `submit` keeps every job off them.
BROKEN_NODES='xgph[10-15],xgpj0'
# HostKeyAlias records xlogin's host key under its real name rather than localhost.
SSH_OPTS=(-o Port=$PORT -o HostKeyAlias=xlogin.comp.nus.edu.sg -o BatchMode=yes -o ConnectTimeout=15)

need_tunnel() {
  local err
  if ! err=$(ssh -n "${SSH_OPTS[@]}" "$HOST" true 2>&1); then
    case "$err" in
      *"Connection refused"*|*"timed out"*)
        echo "The tunnel is not open. Run in your own terminal and keep it open: bash .claude/skills/soc-cluster/soc.sh connect" >&2 ;;
      *"Permission denied"*)
        echo "Key login to xlogin is not set up. Run in your own terminal, with the tunnel open: bash .claude/skills/soc-cluster/soc.sh setup-key" >&2 ;;
      *) echo "$err" >&2 ;;
    esac
    exit 1
  fi
}
remote() { ssh "${SSH_OPTS[@]}" "$HOST" "cd '$SOC_REPO' && $1"; }

case "${1:-}" in
  connect)
    echo "Opening the tunnel through stujump. Enter your SoC password, then leave this terminal open (Ctrl+C closes it)."
    exec ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=60 \
      -L "$PORT:xlogin.comp.nus.edu.sg:22" "$SOC_USER@stujump.comp.nus.edu.sg" ;;
  setup-key)
    exec ssh-copy-id -o Port=$PORT -o HostKeyAlias=xlogin.comp.nus.edu.sg "$HOST" ;;
  check)
    need_tunnel
    echo "Connected to xlogin as $SOC_USER." ;;
  run)
    shift
    [ $# -gt 0 ] || { echo "usage: soc.sh run '<command>'" >&2; exit 1; }
    need_tunnel
    remote "$*" ;;
  submit)
    shift
    [ $# -gt 0 ] || { echo "usage: soc.sh submit [sbatch options] <script> [script args]" >&2; exit 1; }
    need_tunnel
    # The job runs the cluster clone's code, so that code must be the commit checked out here.
    head=$(git rev-parse HEAD)
    read -r cluster_head cluster_edits < <(remote 'echo "$(git rev-parse HEAD) $(git status --porcelain --untracked-files=no | wc -l)"') || true
    [ -n "${cluster_head:-}" ] || { echo "Could not read the commit of the cluster clone ($SOC_REPO)." >&2; exit 1; }
    if [ "$cluster_head" != "$head" ]; then
      echo "The cluster clone is at ${cluster_head:0:7}, but this checkout is at ${head:0:7}." >&2
      echo "Push this branch, then update the clone: soc.sh run 'git fetch && git checkout $(git branch --show-current) && git pull --ff-only'" >&2
      exit 1
    fi
    [ "$cluster_edits" = 0 ] || { echo "The cluster clone has uncommitted changes (soc.sh run 'git status'). Make code changes here, not on the cluster." >&2; exit 1; }
    git diff --quiet HEAD || echo "Note: this checkout has uncommitted changes. The job runs commit ${head:0:7} without them." >&2
    opts=(--exclude="$BROKEN_NODES")
    [ -z "${SOC_MAIL_TYPE:-}" ] || opts+=(--mail-type="$SOC_MAIL_TYPE")
    # Options given here come after the defaults, so they override them.
    remote "$(printf '%q ' sbatch "${opts[@]}" "$@")" ;;
  jobs)
    need_tunnel
    remote 'squeue -u "$USER" -o "%.10i %.16j %.9T %.10M %.10l %.14b %R"' ;;
  sync)
    need_tunnel
    # The login node caps each process at 1 GB of virtual memory and /tmp at a tiny quota,
    # so uv runs in a CPU job with its temp files under the cluster .env's scratch directory.
    remote 'set -a && . ./.env && set +a && mkdir -p "$SQUADQA_SCRATCH/tmp" &&
      srun --partition=normal --time=00:30:00 --cpus-per-task=4 --mem=8G --job-name=uv-sync \
        env TMPDIR="$SQUADQA_SCRATCH/tmp" ~/.local/bin/uv sync' ;;
  list)
    need_tunnel
    remote "ls -t results | head -n ${2:-20}" ;;
  fetch)
    shift
    [ $# -gt 0 ] || { echo "usage: soc.sh fetch <run_id>..." >&2; exit 1; }
    need_tunnel
    for run in "$@"; do
      case "$run" in */*|.*) echo "Invalid run id: $run" >&2; exit 1 ;; esac
      if [ -e "results/$run" ]; then echo "results/$run already exists locally; skipped." >&2; continue; fi
      scp -q -r "${SSH_OPTS[@]}" "$HOST:$SOC_REPO/results/$run" results/
      echo "Fetched results/$run"
    done ;;
  log)
    [ -n "${2:-}" ] || { echo "usage: soc.sh log <file>" >&2; exit 1; }
    case "$2" in */*|.*) echo "Invalid log file name: $2" >&2; exit 1 ;; esac
    need_tunnel
    scp -q "${SSH_OPTS[@]}" "$HOST:$SOC_REPO/logs/$2" logs/
    echo "Fetched logs/$2" ;;
  *)
    sed -n '7,17p' "$0" >&2
    exit 1 ;;
esac
