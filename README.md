# CS4248 Extractive QA (SQuAD v1.1)

Group G03's CS4248 final project: an extractive question answering system on SQuAD v1.1. Given a passage and a question, it returns a span of the passage as the answer.

- Plan (what and why): `Extractive_QA_Design_Plan.docx`
- Implementation spec (how): [implementation_details.md](implementation_details.md)
- Schedule (when): [timeline.md](timeline.md): stages, freeze dates and what blocks what
- Guidance for coding agents: [AGENTS.md](AGENTS.md)

**Status:** repository skeleton only. There is no model code yet.

## Setup

1. Install [uv](https://docs.astral.sh/uv/).
2. Copy `.env.example` to `.env` and fill in the paths for this machine. This file is your profile, and it is gitignored.
3. Run `uv sync`. It creates `.venv` with Python 3.12 and the dependencies.

### SoC cluster

Connect to the NUS Cisco VPN first. Being on NUS Wi-Fi is not enough. Then log in through the SoC jump host:

```
ssh -J <soc unix id>@stujump.comp.nus.edu.sg <soc unix id>@xlogin.comp.nus.edu.sg
```

Optionally, add this to `~/.ssh/config` so that `ssh soc` works. It also works for `scp` and VS Code Remote-SSH.

```
Host soc
    HostName xlogin.comp.nus.edu.sg
    User <soc unix id>
    ProxyJump <soc unix id>@stujump.comp.nus.edu.sg
```

#### One-time setup on the cluster

Run these in a login session on the cluster.

1. **Install uv.** `/tmp` on the login node has a tiny per-user quota, so point the installer at a folder in your home:
   ```
   export TMPDIR=$HOME/cs4248-scratch/tmp && mkdir -p $TMPDIR
   curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
   ```
2. **Clone the repo over HTTPS**, because SSH to GitHub is blocked from the cluster. If git asks for a password, use a GitHub personal access token. Clone into the folder named by `SOC_REPO` in your local `.env`:
   ```
   git clone https://github.com/G03-CS4248-2627S1/project.git cs4248 && cd cs4248
   ```
   So that your assistant can `git pull` there without you, store the token first with `git config --global credential.helper store`. A fine-grained, read-only token for this repo is enough.
3. **Create `.env`** from `.env.example`, with `SQUADQA_SCRATCH` and `HF_HOME` under your home, for example `$HOME/cs4248-scratch` and `$HOME/.cache/huggingface`. There is no `/temp` on the login node.
4. **Build the environment in a CPU job.** The login node limits each process to 1 GB of virtual memory, which uv and PyTorch exceed:
   ```
   srun --partition=normal --time=00:30:00 --cpus-per-task=4 --mem=8G ~/.local/bin/uv sync
   ```
5. **Download the pinned model and the external datasets,** also in a CPU job. Jobs load models offline, so they fail without this:
   ```
   srun --partition=normal --time=00:30:00 --cpus-per-task=4 --mem=8G .venv/bin/python -m squadqa.download_model
   srun --partition=normal --time=00:30:00 --cpus-per-task=4 --mem=8G .venv/bin/python -m squadqa.download_external
   ```
6. **Check the GPU setup.** The job writes `results/<date>-smoke-j<job id>/smoke.json`:
   ```
   sbatch slurm/smoke_test.sbatch
   ```

GPU jobs need `--partition=gpu` (up to 3 hours) or `--partition=gpu-long` (up to 3 days). The default partition has no GPUs.

### Working with the cluster from your own machine

`.claude/skills/soc-cluster/soc.sh` runs commands in your cluster clone, rebuilds its environment, and copies results and logs back. Claude Code uses it through the `soc-cluster` and `fetch-results` skills, and you can also run it yourself.

It needs the NUS Cisco VPN and `SOC_USER` / `SOC_REPO` in your local `.env`. The jump host always asks for a password, so the script works through a tunnel. On Windows, use a Git Bash terminal for these.

1. **Each session, open the tunnel and leave that terminal open.** It asks for your SoC password:
   ```
   bash .claude/skills/soc-cluster/soc.sh connect
   ```
2. **Once per machine, install your SSH key on the cluster** while the tunnel is open. It asks for the password once:
   ```
   bash .claude/skills/soc-cluster/soc.sh setup-key
   ```

After that, the other commands don't ask for a password. For example:

```
bash .claude/skills/soc-cluster/soc.sh fetch <run_id>
```

Run the script without arguments to list its commands.

## Team workflow

Every teammate works with an AI assistant. Each person's assistant keeps a personal memory in `local/MEMORY.md`, which is gitignored. On your first session, the assistant sets up `local/` by following `local.example/README.md`.

**Each session**

1. Run `git pull`. Your assistant reads AGENTS.md and your `local/MEMORY.md`.
2. For cluster work, connect to the NUS Cisco VPN, then run `bash .claude/skills/soc-cluster/soc.sh connect` in a Git Bash terminal.
3. Work on a branch named `<name>/<topic>`.
4. Before you stop, your assistant updates `local/MEMORY.md`, and you push your branch.

**Running an experiment**

1. Add or edit a config in `configs/experiments/` along with the code it needs, then commit and push. Each run records the commit it came from.
2. On the cluster, run `git pull`. If `pyproject.toml` or `uv.lock` changed, also run `soc.sh sync`.
3. Write the job script from `slurm/template.sbatch`, then submit it with `soc.sh submit slurm/<name>.sbatch` (AGENTS.md, "Submitting jobs"). It refuses if the cluster clone isn't at your commit. Use the `gpu` partition (up to 3 hours) or `gpu-long` (up to 3 days). Checkpoints go to `$SQUADQA_SCRATCH`.
4. Run `soc.sh fetch <run_id>`, then commit `results/<run_id>/` on the same branch as the code that produced it.

**Merging and sharing**

- **Pull requests:** open a PR into `main` and have a teammate review it. Roles are flexible, so ask whoever knows that area best. Keep `main` runnable.
- **Headline numbers:** they come only from code on `main` whose results are committed.
- **Dependencies:** edit `pyproject.toml`, run `uv lock`, and pass the smoke test on the cluster before committing `uv.lock`. Everyone then runs `uv sync` locally and `soc.sh sync` for the cluster.
- **Shared knowledge,** such as a cluster quirk, goes into AGENTS.md through a PR. Personal notes stay in `local/MEMORY.md`.
- **The plan:** whoever edits `Extractive_QA_Design_Plan.docx` bumps its version, adds a change-log row and tells the team.

## Evaluating predictions

The official scorer takes a JSON object mapping each question id to an answer string:

```
uv run python qa/evaluate-v2.0.py qa/dev-v1.1.json predictions.json
```

## Layout

| Path | Contents |
|---|---|
| `qa/` | Course-provided data (`train-v1.1.json`, `dev-v1.1.json`) and the official scorer |
| `data/splits/` | Validation split, cross-fitting folds, skipped training ids (committed) |
| `data/external/` | Downloaded datasets such as Adversarial SQuAD and MRQA (gitignored) |
| `configs/` | `base.yaml` defaults and one file per experiment in `experiments/` |
| `squadqa/` | Project code, one module per component |
| `slurm/` | Job scripts for the SoC cluster |
| `tests/` | Tests. `tests/reference/` holds human-written or human-reviewed reference tests. |
| `results/<run_id>/` | Config, metrics and dev predictions for each run (committed) |
| `sandbox/<name>/` | Personal notebooks and scratch work |
| `local/` | Your assistant's personal memory (gitignored); the template and first-session guide are in `local.example/` |
