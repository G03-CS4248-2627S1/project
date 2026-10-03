---
name: fetch-results
description: Copy run results (config, metrics, validation predictions) or Slurm logs from the user's repo clone on the NUS SoC cluster to this machine. Use when the user asks to fetch, pull, download or sync results or logs from the cluster.
argument-hint: "[run_id ...]"
allowed-tools: Bash(bash .claude/skills/soc-cluster/soc.sh check) Bash(bash .claude/skills/soc-cluster/soc.sh list *) Bash(bash .claude/skills/soc-cluster/soc.sh fetch *) Bash(bash .claude/skills/soc-cluster/soc.sh log *)
---

# Fetch results from the SoC cluster

Run ids requested: $ARGUMENTS

This skill uses the helper script from the `soc-cluster` skill; run it from the repo root. It needs the NUS Cisco VPN (being on NUS Wi-Fi is not enough), the tunnel the user opens with `soc.sh connect`, and `SOC_USER` / `SOC_REPO` in `.env`.

1. **Check the connection:** `bash .claude/skills/soc-cluster/soc.sh check`. If it fails, pass on the script's message. It tells the user either to open the tunnel (`soc.sh connect`) or to set up key login (`soc.sh setup-key`). Both ask for the SoC password, so the user must run them in their own terminal. Don't retry in a loop.
2. **Pick the runs.** If no run ids were given above, list recent runs with `bash .claude/skills/soc-cluster/soc.sh list`. Choose the runs the user described (for example "today's runs" or "the baseline seeds"). If that's ambiguous, ask.
3. **Copy them:** `bash .claude/skills/soc-cluster/soc.sh fetch <run_id>...` copies each `results/<run_id>/` into local `results/`. Runs that already exist locally are skipped. Don't overwrite them without asking.
4. **Summarize** each fetched run: its experiment, its seed, and the EM and F1 from its metrics file.
5. **Remind the user about committing.** Fetched runs are committed on the branch of the code that produced them and go into its PR (workflow.md §4.2 and §5). Commit only if the user asks.

## Notes

- **Checkpoints are never fetched.** They stay in `$SQUADQA_SCRATCH` on the cluster.
- **For a job's Slurm output,** use `bash .claude/skills/soc-cluster/soc.sh log <job-name>-<job-id>.out`.
- **This reaches only the user's own cluster clone.** For a teammate's runs, `git pull` after they commit them.
