# workflow.md — how the team works

How each stage and experiment is carried out: where code goes, how work is split and reviewed, and when to open a pull request.

- **What and why:** `Extractive_QA_Design_Plan.docx` (plan v1.1). **How to build it:** [implementation_details.md](implementation_details.md). **When:** [timeline.md](timeline.md). **How we work:** this file.
- Humans and AI assistants follow the same rules. Assistants read this file with AGENTS.md before starting a task.
- Change this file through a PR, like any other shared rule.

---

## 1. Principles

1. **`main` always runs.** Anything merged into `main` passes its tests and doesn't break someone else's entry point.
2. **Every number is reproducible.** A result is a commit, a config and a seed, stored in `results/<run_id>/`. Headline numbers come only from code on `main`.
3. **Dev is touched once.** Development and tuning use the validation split. Dev, Adversarial SQuAD and SQuAD-Shifts are scored only in stage 5, on frozen systems, by one scoring run (§6).
4. **Small PRs, merged often.** A branch should live for days, not weeks. Shared code goes in early so others can build on it.
5. **Decisions are written down.** A design decision goes in an issue or PR, and into the plan or spec if it changes them. A chat message isn't a record.

---

## 2. Where things go

### 2.1 Code

| Path | What | Who changes it |
|---|---|---|
| `squadqa/` | Shared, reviewed code: one module per component | Owner of the module; others through a PR the owner reviews |
| `sandbox/<name>/` | Personal experiments and throwaway scripts | Only you. Nothing in `squadqa/` imports from it |
| `tests/` | Unit tests, run by `uv run pytest` | Whoever changes the code under test |
| `tests/reference/` | Reference tests written or reviewed by a human (spec §8) | Humans only; assistants ask first |
| `configs/base.yaml` | The current default system, mirroring spec §2 | Only through a freeze PR (§5.3) |
| `configs/experiments/` | One file per ladder row or ablation, holding only the keys that differ from base | Whoever runs the experiment |
| `slurm/` | Job scripts, each started from `slurm/template.sbatch` | Whoever runs the job |
| `results/<run_id>/` | Config, metrics and predictions of each run (committed) | Created by runs; never edited by hand |
| `data/<name>/` | Data the project generates and keeps, such as augmented passages | The generator's owner |

Planned modules, so two people don't build the same thing twice. Split a module into a package only when it grows.

| Module | Contents | Spec | Built in |
|---|---|---|---|
| `squadqa/config.py` | Load `base.yaml` plus an experiment file plus `--set key=value` overrides; read `.env`; fail on missing values | §2 | Stage 1 |
| `squadqa/data.py` | SQuAD loading, words, windows, offset maps, validation filtering, row-5 loaders with the dev-passage check | §3 | Stage 1 |
| `squadqa/evaluate.py` | Official-scorer wrapper: per-question scores, Adversarial SQuAD worst case, bootstrap | §6 | Stage 1 |
| `squadqa/heads.py` | Standard, joint and compound heads and their losses, including F1-weighted targets | §4.1–4.3 | Stages 1–2 |
| `squadqa/train_reader.py` | Entry point: train a reader from a config | §9 | Stage 1 |
| `squadqa/decode.py` | Top-k extraction, calibration, MAP and MBR | §4.4 | Stage 1 |
| `squadqa/predict.py` | Run a checkpoint over a dataset; write `predictions.json` and span candidates | §4.4 | Stage 1 |
| `squadqa/diagnostics.py` | Cross-span mass (E3), ECE, fix types | §4.5, §6 | Stages 1–4 |
| `squadqa/candidates.py` | Verifier training candidates: sampling, cross-fitting | §5.1–5.2 | Stage 3 |
| `squadqa/verifier.py` | Marking, both variants, listwise loss | §5.3–5.4 | Stages 2–4 |
| `squadqa/train_verifier.py` | Entry point: train a verifier from a config | §5 | Stage 3 |
| `squadqa/report.py` | Tables and figures for the paper, built from `results/` | §6 | Stage 5 |

Entry points run from the repo root as `python -m squadqa.<module>`, with the config file, `--seed` and `--run-id` as arguments.

### 2.2 Runs, checkpoints and data

- **Run ids:** `<date>-<experiment>-s<seed>-j<Slurm job id>`, as in AGENTS.md. `<experiment>` is the config file name, for example `row1-compound`.
- **Each run writes** `results/<run_id>/` with the resolved config (after overrides), the git commit, the seed, `metrics.json`, and validation predictions. Final scoring adds the dev and robustness predictions (§6).
- **Checkpoints** stay in your own `$SQUADQA_SCRATCH/runs/<run_id>/` on the cluster and are never committed.
- **Need someone else's checkpoint?** Retrain it from the frozen config on `main` in your own clone. A large reader epoch takes about 15 minutes (plan §12.1), which is simpler and safer than copying files between accounts.
- **`data/external/`** is rebuilt in each clone with `python -m squadqa.download_external`, locally and on the cluster (in a CPU job). It is never committed.

---

## 3. Coordinating the team

### 3.1 Issues are the task list

- **One GitHub issue per task,** such as "F1-weighted target loss" or "E1 grid on base", assigned to its owner and labelled with its stage (`stage-1` … `stage-6`).
- **One GitHub milestone per stage,** due on its Thursday (timeline.md §1). The `stage-1` … `stage-6` labels and the six milestones already exist on GitHub.
- **Open an issue before starting work** that someone else might also start. If an issue already exists, comment on it instead of starting in parallel.
- **PRs close issues** with `Closes #<n>` in the description.

### 3.2 Stage kickoff and milestone check

- **Kickoff (start of each stage, about 15 minutes):** split the stage's work into issues, assign owners as in the proposal's breakdown (plan §10), and note who depends on whom.
- **Milestone check (Thursday, about 15 minutes):** each owner reports done, late, or blocked. Anything late uses the Sunday buffer. Anything still late on Sunday goes to the cut list (timeline.md §5), decided together.
- **Work no topic owner covers** (pipelines, cluster tooling, run tracking, paper assembly) is assigned at the week-8 setup, also as issues.

### 3.3 Communication

- **Quick questions:** the team chat.
- **Decisions:** the issue or PR. If a decision changes the plan or spec, the same PR updates them. If the plan and spec disagree, stop and ask (AGENTS.md).
- **Blocked on someone else's code?** Say so on their issue or PR right away, rather than working around it in your own branch.

### 3.4 Sharing the GPU queue

- **Critical-path jobs first** (timeline.md §3.2). Development ablations wait if the queue is tight.
- **Development ablations run on base,** 1 seed, on the validation split. Large runs are for final numbers and confirmations.
- **Before a batch over about 4 GPU-hours,** post in the chat what you're submitting and why.
- **Everyone submits through `soc.sh submit`** (AGENTS.md, "Submitting jobs"), from their own cluster clone.

---

## 4. Branches and pull requests

### 4.1 Branches

- **Name:** `<name>/<topic>`, branched from an up-to-date `main`.
- **One topic per branch.** A second topic gets a second branch.
- **Never push to `main` directly.** GitHub can't enforce this on our plan, so it's a team rule.
- **Never keep committing to a branch after its PR merges.** Start a new branch from `main`. A commit pushed to `bruhmiuz/quick-baseline` after its PR merged went missing from `main` for days.
- **Stacked branches** (one branched from another unmerged branch) are allowed when needed. Say so in the PR, open it against the lower branch, and merge them in order.
  - Squash merging rewrites the lower branch's commits, so after it merges, rebase the upper branch onto `main` before merging it: `git fetch && git rebase --onto origin/main <old lower commit> <upper branch>`, then `git push --force-with-lease`. GitHub retargets the upper PR to `main` on its own.

### 4.2 When to open a PR

| Situation | PR |
|---|---|
| Starting work others should know about | A **draft** PR early, so the plan is visible. Mark it ready when done |
| Shared code in `squadqa/` that others will build on | As soon as it works and has tests. Don't hold it back for your own experiments |
| An experiment's results | Code, config and `results/<run_id>/` together, in one PR |
| A design is chosen (stage freeze) | A **freeze PR** (§5.3) |
| Plan, spec, timeline or workflow change | Its own PR; the plan's change log gets a row |
| New dependency | Its own PR, after `uv lock` and the cluster smoke test (README, "Dependencies") |
| Sandbox code | Optional. Push to your branch any time; merge only if others benefit |

### 4.3 Before opening a PR

- **`uv run pytest` passes,** including `tests/reference/`.
- **New training code passed a smoke run** on the cluster: a few dozen steps, a checkpoint, a resume, and an official-scorer run (spec §9).
- **The spec and plan still match the code.** If not, update them in the same PR.
- **No dev results** before stage 5 (principle 3).
- The PR template (`.github/pull_request_template.md`) lists these checks.

### 4.4 Review and merge

- **One approving review from another teammate** before merging. For shared modules, the reviewer should be someone who builds on that module.
- **Reviewers check:** correctness against the spec, tests, that results come from the PR's own commit, and that no dev data was used.
- **Respond within a working day.** If you can't review in time, say so, so the author can ask someone else.
- **The author merges** after approval. The repository allows only **Squash and merge**, which uses the PR's title and description as the commit message, so write them for the history. Merged branches are deleted automatically.
- **Assistants never merge.** They may open a PR when their user asks.

---

## 5. Carrying out an experiment

### 5.1 Development experiment (choosing a design)

1. **Issue:** state the question, for example "does α = 0.3 beat the one-hot loss on base?"
2. **Branch** from `main`; write or change the code and its tests.
3. **Config:** add `configs/experiments/<stage>-<topic>.yaml` with only the changed keys. Use `--set key=value` for grid points instead of one file per point.
4. **Smoke run** on the cluster if the training code is new (§4.3).
5. **Runs:** base, 1 seed, scored on the validation split. Submit with `soc.sh submit`.
6. **Fetch** with `soc.sh fetch <run_id>` and commit `results/<run_id>/` on the same branch.
7. **PR** with the code, config and results. Summarize the outcome in the PR description: what won, by how much, against the seed spread (plan §12.3).

### 5.2 Final run (a ladder row)

1. The row's design is frozen and merged into `main` (§5.3).
2. Update your cluster clone to that `main` commit, so the runs come from `main` (principle 2).
3. Submit 3 seeds on large: `soc.sh submit --export=ALL,SEED=<s> slurm/<row>.sbatch`.
4. Fetch the results and commit them in a results PR, which records the run ids.

### 5.3 Freeze PR (end of a stage)

The owners of a stage open one PR by the Thursday milestone that:

- sets the chosen settings in `configs/base.yaml` and spec §2 together;
- adds or updates the row's config in `configs/experiments/` and its job script in `slurm/`;
- records what was chosen and why in the PR description, linking the development PRs as evidence;
- updates the plan, spec or timeline if the design changed.

Once it merges, the row's final runs start (§5.2), and later stages build on the frozen setting.

---

## 6. Stage by stage

Owners follow the proposal's breakdown (plan §10); names are assigned at each kickoff. Each stage ends with its freeze PR merged by its Thursday.

| Stage (milestone) | Work and owners | Main PRs | Done when |
|---|---|---|---|
| 1. Setup and baseline (Thu 8 Oct) | Everyone: onboarding and cluster smoke test. Owners for `config`, `data` with the offset test, `evaluate`, `train_reader`, `decode` and `predict` | One PR per module; baseline results PR; freeze PR for row 0 | Offset test passes; pipeline baseline matches the sandbox (about 89 validation F1); first MAP vs. MBR numbers |
| 2. Reader (Thu 15 Oct) | Compound objective (2 people); F1-weighted targets (2); decoding and E3 (1); verifier groundwork in parallel | Development PRs for each; freeze PR for rows 1–2 | Reader frozen in `base.yaml`; rows 1–2 running on large |
| 3. Decoding and verifier start (Thu 22 Oct) | E1 and calibration (1); candidate sampling; shared verifier (3); separate verifier (2); row 5 reader | E1 results PR; `candidates` and `verifier` PRs; row 5 reader results | Candidates for all of train; both variants trained on base |
| 4. Verifier (Thu 29 Oct) | E2 across both variant teams | E2 results PR; freeze PR for row 4 | Variant chosen; row 4 running on large; row 5 verifier running |
| 5. Final runs and writing (Thu 5 Nov) | One person runs final scoring; everyone writes | Final scoring PR; `report` PR | All numbers committed; paper and slides drafted |
| 6. Submission (Thu 12 Nov) | Everyone: video; one person checks the code release | Release PR (README run instructions, citations) | Submitted |

### 6.1 Final scoring (stage 5)

- **One person, one PR, from `main`:** score every frozen system on dev, Adversarial SQuAD and SQuAD-Shifts with the official scorer (spec §6). This is the only time dev is used.
- **Tables and figures come from `squadqa/report.py`,** reading `results/`. Nobody copies numbers into the paper by hand.

---

## 7. Writing

- **The paper** lives in a shared Overleaf project with the ACL template; its link goes in the team chat and the README. Text isn't kept in the repo.
- **Every number and figure** in the paper and slides comes from `squadqa/report.py` output, so it can be regenerated.
- **Intro and related work start in week 9,** from `literature_review.md`. Method sections follow each freeze; results follow stage 5.
- **Slides and video** reuse the paper's figures. Every member speaks in the video.

---

## 8. AI assistants

- Each teammate's assistant follows AGENTS.md and this file, and keeps `local/MEMORY.md` up to date.
- **Assistants may:** write code, tests and configs on a branch; submit cluster jobs after asking; fetch results; draft PR descriptions.
- **Assistants may not:** push to `main`, merge PRs, edit `tests/reference/` without asking, or use dev before stage 5.
- **The human who opens a PR owns it,** including code their assistant wrote. Reviewers review the code, not the assistant.
