# AGENTS.md

CS4248 (Introduction to Large Language Models, AY2026/27 Sem 1) group project: an **extractive question answering** system on SQuAD v1.1. Given a passage and a question, the system returns a span of the passage as the answer.

The data pipeline, evaluation, decoding and their tests are on `main`; the reader and the rest of the method are being built (workflow.md §2.1 lists the modules).

## Personal memory and first session

Each teammate has a gitignored `local/` folder. Its `local/MEMORY.md` holds what is specific to them: identity and current focus, machine setup, cluster status, personal procedures and open threads.

- **At the start of every session,** read `local/MEMORY.md` before other work and follow it.
- **Before finishing,** update it if the user's setup, preferences or open threads changed.
  - Record facts with dates.
  - Never store passwords or tokens in it.
- **If there is no `local/` folder,** this is the user's first session. Follow `local.example/README.md` with them before other work, so no teammate's setup lags behind.
- **Shared or personal?** Facts that help everyone, such as a cluster quirk, belong in AGENTS.md; propose the edit to the user. Facts about this user belong in `local/MEMORY.md`.
- **Mistakes.** When you make a mistake (a wasted job, a wrong claim, a defect a reviewer catches, a command that failed on a wrong assumption), record it with its lesson in `local/MEMORY.md` under "Mistakes and lessons" in the same session. If a teammate's assistant could repeat it, also add the lesson to "Lessons from past mistakes" below through a PR. Read that section before similar work.

**Session workflow:**
1. Run `git pull` and read `local/MEMORY.md`.
2. Work on a branch named `<name>/<topic>`, made from an up-to-date `main`, and open a PR when workflow.md §4.2 says to.
3. Before stopping, update `local/MEMORY.md` and push the branch.

workflow.md has the full version: issues, when to open a PR, review, and how experiments move from config to committed results (§5).

## Spec documents

- **[implementation_details.md](implementation_details.md)** says *how* to build the system: hard rules, config defaults, data splits, the definition of a word, the joint span head, the verifier, and the testing workflow for custom attention code. **Read it before writing any data, model or evaluation code.** Its §1 hard rules are binding.
- **[Extractive_QA_Design_Plan.docx](Extractive_QA_Design_Plan.docx)** (the plan) says *what* is being built and *why*. It is a living document that the team edits between sessions.
  - Before any task that depends on it, re-read the current file and check its version line and change log (§19). Don't rely on an earlier summary, and don't copy plan content into this file.
  - A docx is a zip archive, and the text is in `word/document.xml`. If python-docx and pandoc aren't available, read that file with Python's `zipfile` module.
- **[workflow.md](workflow.md)** says *how the team works*: where code goes, issues and stage kickoffs, branches, when to open a PR, review, and how experiments and stage freezes run. **Read it before starting a task.** Assistants never push to `main` or merge PRs.
- **[timeline.md](timeline.md)** says *when*: the project's stages, freeze dates, the critical path, and what to cut if the team falls behind. Plan weeks W1–W7 are course weeks 7–13. Check it before choosing what to work on next, and update it when a freeze date or a stage's scope changes.
- If the spec and the plan disagree, stop and ask a human.

## Layout and conventions

- **`qa/`** holds the course-provided data and the official scorer.
- **`data/`:**
  - Generated split files go in `data/splits/` (spec §3.1–3.2).
  - Downloaded datasets go in `data/external/`, which is gitignored as a whole. `python -m squadqa.download_external` re-creates it from fixed URLs with SHA-256 checks; it needs only the standard library. Its docstring says which files are **evaluation-only** because their passages come from `dev-v1.1.json` (AdversarialQA `dev.json`, Adversarial SQuAD, Paraphrased SQuAD). Training files built on `train-v1.1.json` passages (SQuAD 2.0 train, AdversarialQA `train.json`) must have the validation articles removed. The script also removes duplicate answers and questions, both within a file and across files (including questions already in `qa/`, which it never changes), and logs the removed ids to `data/external/duplicates.json`. To check new data such as generated questions against existing files, run `python -m squadqa.dedup <files...>`; it only reports, and earlier files win.
  - Pretrained model files go in the Hugging Face cache (`$HF_HOME`). `python -m squadqa.download_model` downloads them, pinned to a repo commit and checked with SHA-256. Load models with the same pin (`revision=MODELS[name]["revision"]` from that module).
  - Data the project generates and should keep, such as augmented training passages (plan §4.5), goes in its own `data/<name>/` folder and is committed.
- **`configs/base.yaml`** mirrors spec §2, so change both together. Each file in `configs/experiments/` (one per ladder row or ablation) holds only the keys that differ from base.
- **`squadqa/`** has one module per component. Split a module into a package only when it grows. Run entry points from the repo root as `python -m squadqa.<module>`; the project is not installed as a package.
- **Profiles:** each machine has a gitignored `.env`, created from `.env.example`. It contains paths, plus the SoC login details (`SOC_USER`, `SOC_REPO`) on machines that reach the cluster remotely.
  - Hyperparameters never go in `.env`.
  - A missing variable must raise an error, not fall back to a default.
  - Per-device batch size is a command-line flag. Gradient accumulation is adjusted so the effective batch size stays fixed.
- **`results/<run_id>/`**, where `run_id` = `<date>-<experiment>-s<seed>-j<Slurm job id>` (use a timestamp in place of the job id for local runs):
  - Slurm job ids are unique across the cluster, so run ids never collide between teammates, and tracked paths contain no personal ids.
  - Commit the run's resolved config, commit hash, seed, metrics and validation predictions. Dev and robustness predictions come only from the stage-5 final scoring (workflow.md §6.1).
  - Checkpoints go under `$SQUADQA_SCRATCH`, never in the repo.
- **`tests/reference/`** holds reference tests written or reviewed by a human.
  - Claude Code asks before editing them, and edits to `qa/evaluate-v2.0.py` are denied (see `.claude/settings.json`).
  - Other agents must not edit either location.
  - Other tests go directly in `tests/`.
- **`sandbox/<name>/`** is personal scratch space. Nothing in `squadqa/` may import from it, and notebook outputs should be cleared before committing.
- **Environment:**
  - Dependencies are managed by uv (`uv sync`). `uv.lock` pins the exact versions that passed the cluster smoke test, including torch 2.14.0+cu126 and transformers 5.17.0.
  - **Reaching the cluster:**
    - It needs the NUS Cisco VPN; being on NUS Wi-Fi is not enough.
    - Log in with `ssh -J <soc unix id>@stujump.comp.nus.edu.sg <soc unix id>@xlogin.comp.nus.edu.sg`. The jump host always asks for a password.
  - **`.claude/skills/soc-cluster/soc.sh`** runs commands in the user's cluster clone and copies results and logs back.
    - It works through a tunnel the user opens with `soc.sh connect`, plus SSH key login set up once with `soc.sh setup-key`. Both ask for the SoC password, so the user runs them; agents can't type passwords.
    - The `soc-cluster` and `fetch-results` skills wrap the script. Other agents can call it directly; run it without arguments for usage.
  - **Running repo scripts on Windows.** They need Git Bash.
    - In PowerShell, `bash` is WSL's bash, which has its own home and SSH keys. Call `"C:/Program Files/Git/bin/bash.exe" .claude/skills/soc-cluster/soc.sh ...` instead.
    - Codex's sandbox runs as a different Windows account: it can't read `~/.ssh`, and git reports "dubious ownership". Run `soc.sh` and `git pull` with escalated permissions.
  - **Submitting jobs.** Every agent follows these steps, whatever tool it is. First read `.claude/skills/soc-cluster/SKILL.md`: its cluster facts and rules apply to every agent, not only Claude Code.
    1. **Ask the user first,** naming the script, GPU and time limit. Jobs use shared GPU time. Do the same before `scancel`, and only cancel the user's own jobs.
    2. **Write the job script** by copying `slurm/template.sbatch`. It sets up `.env`, the venv, offline model loading and the run id.
       - The code must write the run's config, metrics and predictions to `results/<run_id>/` in the clone, which is what `soc.sh fetch` copies. Checkpoints go to `$SQUADQA_SCRATCH/runs/<run_id>/`.
       - A new training script gets a smoke run first (spec §9): a few dozen steps with a checkpoint, then a rerun with the same run id to test the resume.
       - Commit and push the script with the code it runs.
    3. **Update the cluster clone** to the same commit: `soc.sh run 'git fetch && git checkout <branch> && git pull --ff-only'`.
       - Don't change the clone's checkout while `soc.sh jobs` lists your jobs. A job reads code from the clone when it starts, so the change would alter what queued jobs run. Jobs built from the template stop if the clone's commit differs from the one they were submitted at.
       - The pull needs a GitHub token stored on the cluster (README, "One-time setup on the cluster"). If it fails with `could not read Username`, ask the user to pull in their own ssh session.
       - If `pyproject.toml` or `uv.lock` changed, run `soc.sh sync` afterwards.
       - Committing fetched results moves your local commit ahead of the clone, so push and pull again before the next submit.
    4. **Submit** with `soc.sh submit [sbatch options] slurm/<name>.sbatch [script args]`, never with a raw `sbatch`.
       - It refuses if the clone isn't at your local commit or has uncommitted edits.
       - It keeps jobs off broken nodes. The list is `BROKEN_NODES` in `soc.sh`; update it there when a node breaks or is fixed.
       - If `SOC_MAIL_TYPE` is set in the local `.env` (for example `END,FAIL`), Slurm emails `<soc unix id>@comp.nus.edu.sg`. Delivery hasn't been confirmed yet.
       - Before a GPU type's first use, run the smoke test on it. H100 and H200 haven't been tested yet.
    5. **Watch and collect.** `soc.sh jobs` shows the queue, `soc.sh log <job name>-<job id>.out` copies a log, and `soc.sh fetch <run_id>` copies results. To change a pending job, see SKILL.md.
    6. **Runs longer than 3 hours** must checkpoint and resume (spec rule 8). Queue the resume behind the running job: `soc.sh submit --dependency=afterany:<job id> --export=ALL,RUN_ID=<run id> slurm/<name>.sbatch`. Ask the team before using `gpu-long` (up to 3 days), because the spec assumes 3-hour jobs.
  - **Cluster limits** (checked September 2026; details in `.claude/skills/soc-cluster/SKILL.md`):
    - The login node allows 1 GB of virtual memory and 300 s of CPU time per process and has a tiny `/tmp` quota. Build environments and run Python in jobs, not on the login node.
    - GPU jobs need `--partition=gpu` (up to 3 hours) or `--partition=gpu-long` (up to 3 days). The default partition has no GPUs.
    - Request A100, H100 or H200 GPUs, which support bf16.
    - SSH to GitHub is blocked, so clone and pull over HTTPS.
    - `soc.sh submit slurm/smoke_test.sbatch` checks the environment.

## Lessons from past mistakes

Mistakes assistants made in this project, kept so nobody repeats them. Add to this list through a PR; keep each entry to the lesson.

- **Trace every job before asking to submit it.** Check what each job will do with its exact arguments. A smoke-run resume must drop `--stop-after` (it is a command-line flag, not config, so the resume check still passes); passing it again stops the resumed run at the same step and wastes a job (2026-10-04).
- **`Closes #N` only when the PR finishes the issue.** Read the issue's "Done when" first. If part of it depends on another issue's work, write `Part of #N` and note on both issues which PR will close it. Reviewers caught two premature closures (2026-10-04).
- **GitHub closing keywords work anywhere in a PR body,** even mid-sentence: "#21's PR closes #22" linked #22 for closing. Before creating a PR whose issue must stay open, search the body for `close`, `fix` and `resolve` followed by `#`, including in instructions or quoted examples (describe them in words instead), and make a match stop the command (`! grep ... && gh pr create ...`); after creating or editing it, check `gh pr view <n> --json closingIssuesReferences` (it happened twice, 2026-10-04).
- **Verify spec claims against a primary source or the data before writing them.** Three spec statements were wrong and caught later: the Adversarial SQuAD grouping (checked against the authors' `eval_squad.py`), the validation-split size (count it), and pooling of spans without content words.
- **The spec and the plan move together.** When a number or rule changes in the spec, search the plan for it too. A plan version bump touches the title line, the page header (`word/header1.xml`) and the change log; the header was once left at an old version.
- **Load large checkpoints into CPU memory:** `torch.load(path, map_location="cpu")`, then `load_state_dict` copies them to the GPU. Loading a 5.2 GB checkpoint straight onto an A100-40 fragmented memory and caused out-of-memory retries (2026-10-04).
- **Check end-to-end invariants, not only local ones.** Checking that tokens lie inside words missed that `return_overflowing_tokens` (tokenizers 0.23) drops the end of long passages; check that every passage token reaches some window (spec §3.3).
- **Tool gotchas:**
  - transformers 5 removed `build_inputs_with_special_tokens`; use `squadqa.data.pair_layout`.
  - `load_dotenv()` with no path fails in code run from stdin (`python - <<EOF`); pass `".env"`.
  - On Windows, set `PYTHONIOENCODING=utf-8` before printing non-ASCII text, or Python raises `UnicodeEncodeError`.
  - YAML 1.1 reads `3e-5` as a string. Write `3.0e-5` in YAML files; `--set train.lr=3e-5` is converted for you, and a wrong type raises.
  - In tests, compute hand-made word indices from the text, and compare float32 values with `pytest.approx`.

## Data (`qa/`)

- `train-v1.1.json`: training set, 87,599 questions with exactly one answer each.
- `dev-v1.1.json`: the course brief designates this as the **test set** (10,570 questions). Never train or tune on it. Tuning uses a validation split carved from train by article; see spec §3.1.
- Schema: `data[] → {title, paragraphs[] → {context, qas[] → {id, question, answers[] → {text, answer_start}}}}`. `answer_start` is a character offset into `context`.

Both files differ from the official release (the originals are in commit `8a3a52b`):
- They are pretty-printed with 2-space indentation. The originals were a single line each.
- In dev, answers repeated with the same `text` **and** `answer_start` were removed, keeping the first copy. `answers[0]` and each question's set of distinct answers are unchanged, so EM/F1 are identical. The one thing lost is how many annotators chose each answer, so a majority-vote answer can no longer be recovered.
- As a result, dev now averages **1.72 reference answers per question**, not the official 3.29 (plan §2.1 rounds it to 3.3). Statistics based on the first answer are unaffected.
- **Validation scores are not comparable with dev scores.** Validation has one reference per question, so the same model scores much higher on dev. Measured on row 0 (3 seeds, deberta-v3-large, MAP; job 911434, an approved early exception to workflow.md §6.1, 2026-10-05):

  | Scored on | EM | F1 |
  |---|---|---|
  | validation | 76.89 | 89.62 |
  | dev, first reference only | 74.81 | 87.77 |
  | dev, all references | 90.03 | 95.30 |

  The extra references add about 15 EM and 7.5 F1; dev questions scored with one reference are about 2 points harder than validation. Compare systems with each other on validation, never with published dev numbers.

Keep these JSON files ASCII-only by writing them with `json.dump`'s default `ensure_ascii=True`. The scorer opens files without an encoding, which on Windows means cp1252.

## Evaluation

The brief requires the official scorer, `qa/evaluate-v2.0.py`. Do not modify it.

```
python qa/evaluate-v2.0.py qa/dev-v1.1.json predictions.json [-o eval.json]
```

- `predictions.json` is a flat object `{question_id: answer_text}`.
- The scorer imports numpy at the top even though it only uses it for plotting. numpy is listed in `pyproject.toml`.
- **Every question id in the data file needs a non-empty prediction.** A missing id crashes the scorer with `KeyError` in the HasAns breakdown, and an empty string scores 0. To score a subset, trim the data file to match. Extra ids in the predictions are ignored.
- **Write predictions ASCII-only** (plain `json.dump`) or run with `python -X utf8`. Raw UTF-8 predictions raise `UnicodeDecodeError` on Windows, or worse, get misread and lower the score silently.
- Scoring: both strings are lowercased, and punctuation, `a`/`an`/`the` and extra whitespace are removed. EM is exact equality. F1 is bag-of-tokens overlap. Both take the maximum over a question's gold answers and are averaged over questions.
- The script is written for SQuAD 2.0. Its unanswerable-question features (`-n`, `-t`, `-p`, `NoAns_*`) don't apply to v1.1, and `HasAns_*` just repeats the overall scores.

## Course rules that constrain the code

These are summarised from the two PDFs in the repo root. The PDFs are gitignored, so a fresh clone won't have them.

- **Closed-source LLMs and API calls (ChatGPT, Gemini, etc.) are not allowed** ("not allowed in your project", per the brief). Spec rule 3 bans them from training and inference code. Using them for data generation is pending a ruling (plan §2), so don't add such use until that ruling is made. Open-source pretrained models and embeddings are allowed.
- External open-source code and libraries are allowed. However, the report must state which parts the group implemented and which came from others, with citations. When borrowing code, record where it came from, for example in a comment with the source URL.
- The submission must include all source code, any external code and data, and instructions for running the code. Keep training and evaluation reproducible from the command line.
- The report is an ACL-style short paper (at most 4 pages of content plus 1 page of references, Times Roman 11pt). The group also submits slides and a 15-minute video presentation.
