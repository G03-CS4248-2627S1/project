# First-session setup

This is for AI assistants and their users. If `local/` does not exist, this is the user's first session in this repo. Work through these steps with the user before doing anything else, so that every teammate's setup ends up the same.

Find out what you can by exploring, and ask only for what you can't discover.

1. **Create `local/`** and copy this folder's `MEMORY.md` into it as `local/MEMORY.md`. `local/` is gitignored.
2. **Get to know the user.** Ask for their name, SoC unix id, what they plan to work on and which AI assistants they use. Roles are flexible; §10 of `Extractive_QA_Design_Plan.docx` lists the areas. Record these under "Me".
3. **Explore the machine without changing anything.** Record under "Machines":
   - the OS and shell, and the git user;
   - the Python and uv versions;
   - `ssh`, `ssh-copy-id` and any existing keys in `~/.ssh`;
   - the local GPU, if any.

   On Windows, run the repo's scripts in Git Bash. In Git Bash, write paths as `/c/Users/...`, because the colon in `C:/...` breaks `PATH`-like variables.
4. **Create `.env`** from `.env.example`, filling in `SOC_USER`, `SOC_REPO` (the clone folder on the cluster, for example `cs4248`) and the local paths.
5. **Connect the assistant to the memory file.**
   - **Claude Code:** create `CLAUDE.local.md` (gitignored) containing the single line `@local/MEMORY.md`, so the memory loads automatically. The project skills in `.claude/skills/` load on their own.
   - **Assistants that read AGENTS.md** (Codex, Cursor, GitHub Copilot and others): nothing to do. AGENTS.md tells them to read `local/MEMORY.md`.
   - **Assistants that read a different file** (for example Gemini CLI, which reads `GEMINI.md`): configure them to read AGENTS.md.
6. **Set up cluster access** (README.md, "Working with the cluster from your own machine").
   - The user connects to the NUS Cisco VPN, then runs `bash .claude/skills/soc-cluster/soc.sh connect` and `bash .claude/skills/soc-cluster/soc.sh setup-key` in their own terminal. Both ask for the SoC password.
   - Then you run `soc.sh check`.
7. **Set up the cluster clone** (README.md, "One-time setup on the cluster"): install uv, clone over HTTPS, create the cluster `.env`, then run `soc.sh sync`. Ask before submitting the smoke test (`sbatch slurm/smoke_test.sbatch`), then fetch its result with `soc.sh fetch`.
8. **Create `sandbox/<name>/`** for the user's scratch work.
9. **Record what's done** under "Cluster status" and anything unfinished under "Open threads", with dates.
