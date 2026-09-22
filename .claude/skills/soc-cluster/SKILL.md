---
name: soc-cluster
description: Connect to the NUS SoC Slurm cluster to run commands, check or submit jobs, and read job output. Use when the user asks to do anything on the cluster, such as checking the queue, submitting or cancelling a job, updating the cluster clone or its environment, or looking at a running job's log.
allowed-tools: Bash(bash .claude/skills/soc-cluster/soc.sh check)
---

# SoC cluster

The cluster is reachable only from the **NUS Cisco VPN**; being on NUS Wi-Fi is not enough. Logins go through the jump host `stujump`, which always asks for a password. So the helper script works through a tunnel:

- **Once per session,** the user runs `bash .claude/skills/soc-cluster/soc.sh connect` in their own terminal, enters their SoC password and leaves the terminal open.
- **Once per machine,** with the tunnel open, the user runs `bash .claude/skills/soc-cluster/soc.sh setup-key`, so that logins through the tunnel use their SSH key.

Both commands ask for the password, so never run them yourself; ask the user. Every other command runs from the repo root and reads `SOC_USER` and `SOC_REPO` (the user's repo clone on the cluster) from `.env`:

| Command | Does |
|---|---|
| `bash .claude/skills/soc-cluster/soc.sh check` | Tests the tunnel and key login |
| `bash .claude/skills/soc-cluster/soc.sh run '<command>'` | Runs a command inside the cluster clone |
| `bash .claude/skills/soc-cluster/soc.sh sync` | Builds or updates the cluster clone's environment (`uv sync` in a CPU job) |
| `bash .claude/skills/soc-cluster/soc.sh list [n]` | Lists the most recent runs in `results/` |
| `bash .claude/skills/soc-cluster/soc.sh log <file>` | Copies `logs/<file>` to local `logs/` |

Keep `$USER` and `$HOME` inside single quotes so they expand on the cluster, for example `run 'squeue -u $USER'`.

## Cluster facts (checked September 2026)

- **The login node is too small for real work.** Each process gets at most 1 GB of virtual memory, each user at most 64 processes, and `/tmp` has a tiny per-user quota. uv and PyTorch fail there, so build environments and run Python inside jobs:
  - CPU work: `srun --partition=normal --time=00:30:00 --cpus-per-task=4 --mem=8G <command>`.
  - Rebuild the environment with `sync`, for example after `pyproject.toml` or `uv.lock` changes and a `git pull`.
- **GPU jobs need a GPU partition.** Use `--partition=gpu` (up to 3 hours) or `--partition=gpu-long` (up to 3 days); the default partition `normal` has no GPUs.
  - Request a GPU with bf16 support: `--gpus=a100-40`, `a100-80`, `h100-96`, `h100-47` or `h200-141`.
- **uv lives at `~/.local/bin/uv`** and is not on `PATH` in non-interactive shells.
- **Compute nodes have internet access** (PyPI and Hugging Face).
- **Storage:** there is no `/temp` on the login node. Checkpoints and caches go under `$HOME` (`$SQUADQA_SCRATCH`, `$HF_HOME` in the cluster `.env`).
- **SSH to GitHub is blocked,** so the clone must use HTTPS.

## Steps

1. **Run `check` first.**
   - If `.env` is missing `SOC_USER` or `SOC_REPO`, ask the user for them.
   - If the check fails, pass on the script's message; it says whether to run `connect` or `setup-key`. Don't retry in a loop.
2. **Do the task** with `run`.
   - To submit a job and wait for it, use `run 'sbatch --wait slurm/<script>.sbatch'` in the background.
   - Fetch its output afterwards with `log <job-name>-<job-id>.out`.

## Rules

- **Ask before `sbatch` or `scancel`.** Jobs use shared GPU time. Only cancel the user's own jobs.
- **Don't edit code on the cluster.** Change it locally and push, then run `git pull` in the cluster clone.
- **To copy results to this machine,** use the `fetch-results` skill.
