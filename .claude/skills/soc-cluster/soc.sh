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
# HostKeyAlias records xlogin's host key under its real name rather than localhost.
SSH_OPTS=(-o Port=$PORT -o HostKeyAlias=xlogin.comp.nus.edu.sg -o BatchMode=yes -o ConnectTimeout=15)

need_tunnel() {
  local err
  if ! err=$(ssh "${SSH_OPTS[@]}" "$HOST" true 2>&1); then
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
    sed -n '7,14p' "$0" >&2
    exit 1 ;;
esac
