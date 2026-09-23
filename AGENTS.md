# AGENTS.md

CS4248 (Introduction to Large Language Models, AY2026/27 Sem 1) group project: an **extractive question answering** system on SQuAD v1.1. Given a passage and a question, the system returns a span of the passage as the answer.

The directory skeleton and the cluster environment are in place, but there is no model code or test yet.

## Personal memory and first session

Each teammate has a gitignored `local/` folder. Its `local/MEMORY.md` holds what is specific to them: identity and current focus, machine setup, cluster status, personal procedures and open threads.

- **At the start of every session,** read `local/MEMORY.md` before other work and follow it.
- **Before finishing,** update it if the user's setup, preferences or open threads changed.
  - Record facts with dates.
  - Never store passwords or tokens in it.
- **If there is no `local/` folder,** this is the user's first session. Follow `local.example/README.md` with them before other work, so no teammate's setup lags behind.
- **Shared or personal?** Facts that help everyone, such as a cluster quirk, belong in AGENTS.md; propose the edit to the user. Facts about this user belong in `local/MEMORY.md`.

**Session workflow:**
1. Run `git pull` and read `local/MEMORY.md`.
2. Work on a branch named `<name>/<topic>` and open a PR into `main`.
3. Before stopping, update `local/MEMORY.md` and push the branch.

README.md, "Team workflow", has the full version, including how experiments move from config to committed results.

## Spec documents

- **[implementation_details.md](implementation_details.md)** says *how* to build the system: hard rules, config defaults, data splits, the definition of a word, the joint span head, the verifier, and the testing workflow for custom attention code. **Read it before writing any data, model or evaluation code.** Its §1 hard rules are binding.
- **[Extractive_QA_Design_Plan.docx](Extractive_QA_Design_Plan.docx)** (the plan) says *what* is being built and *why*. It is a living document that the team edits between sessions.
  - Before any task that depends on it, re-read the current file and check its version line and change log (§16). Don't rely on an earlier summary, and don't copy plan content into this file.
  - A docx is a zip archive, and the text is in `word/document.xml`. If python-docx and pandoc aren't available, read that file with Python's `zipfile` module.
- If the spec and the plan disagree, stop and ask a human.

## Layout and conventions

- **`qa/`** holds the course-provided data and the official scorer.
- **`data/`:**
  - Generated split files go in `data/splits/` (spec §3.1–3.2).
  - Downloaded datasets go in `data/external/`, which is gitignored as a whole. `python -m squadqa.download_external` re-creates it from fixed URLs with SHA-256 checks; it needs only the standard library. Its docstring says which files are **evaluation-only** because their passages come from `dev-v1.1.json` (AdversarialQA `dev.json`, Paraphrased SQuAD). Training files built on `train-v1.1.json` passages (SQuAD 2.0 train, AdversarialQA `train.json`) must have the validation articles removed. The script also removes duplicate answers and questions, both within a file and across files (including questions already in `qa/`, which it never changes), and logs the removed ids to `data/external/duplicates.json`. To check new data such as generated questions against existing files, run `python -m squadqa.dedup <files...>`; it only reports, and earlier files win.
  - Pretrained model files go in the Hugging Face cache (`$HF_HOME`). `python -m squadqa.download_model` downloads them, pinned to a repo commit and checked with SHA-256. Load models with the same pin (`revision=MODELS[name]["revision"]` from that module).
  - Data the project generates and should keep, such as E7 distractors or E8 synthetic questions, goes in its own `data/<name>/` folder and is committed.
- **`configs/base.yaml`** mirrors spec §2, so change both together. Each file in `configs/experiments/` (one per ladder row or ablation) holds only the keys that differ from base.
- **`squadqa/`** has one module per component. Split a module into a package only when it grows. Run entry points from the repo root as `python -m squadqa.<module>`; the project is not installed as a package.
- **Profiles:** each machine has a gitignored `.env`, created from `.env.example`. It contains paths, plus the SoC login details (`SOC_USER`, `SOC_REPO`) on machines that reach the cluster remotely.
  - Hyperparameters never go in `.env`.
  - A missing variable must raise an error, not fall back to a default.
  - Per-device batch size is a command-line flag. Gradient accumulation is adjusted so the effective batch size stays fixed.
- **`results/<run_id>/`**, where `run_id` = `<date>-<experiment>-s<seed>-j<Slurm job id>` (use a timestamp in place of the job id for local runs):
  - Slurm job ids are unique across the cluster, so run ids never collide between teammates, and tracked paths contain no personal ids.
  - Commit the run's config, metrics and dev predictions.
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
  - **Cluster limits** (checked September 2026; details in `.claude/skills/soc-cluster/SKILL.md`):
    - The login node allows 1 GB of virtual memory per process and has a tiny `/tmp` quota. Build environments and run Python in jobs, not on the login node.
    - GPU jobs need `--partition=gpu` (up to 3 hours) or `--partition=gpu-long` (up to 3 days). The default partition has no GPUs.
    - Request A100, H100 or H200 GPUs, which support bf16.
    - SSH to GitHub is blocked, so clone and pull over HTTPS.
    - `sbatch slurm/smoke_test.sbatch` checks the environment.

## Data (`qa/`)

- `train-v1.1.json`: training set, 87,599 questions with exactly one answer each.
- `dev-v1.1.json`: the course brief designates this as the **test set** (10,570 questions). Never train or tune on it. Tuning uses a validation split carved from train by article; see spec §3.1.
- Schema: `data[] → {title, paragraphs[] → {context, qas[] → {id, question, answers[] → {text, answer_start}}}}`. `answer_start` is a character offset into `context`.

Both files differ from the official release (the originals are in commit `8a3a52b`):
- They are pretty-printed with 2-space indentation. The originals were a single line each.
- In dev, answers repeated with the same `text` **and** `answer_start` were removed, keeping the first copy. `answers[0]` and each question's set of distinct answers are unchanged, so EM/F1 are identical. The one thing lost is how many annotators chose each answer, so a majority-vote answer can no longer be recovered.
- As a result, dev now averages **1.72 reference answers per question**, not the official 3.29 (plan §2.1 rounds it to 3.3). Statistics based on the first answer are unaffected.

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
