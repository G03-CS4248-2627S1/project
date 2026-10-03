# timeline.md — CS4248 Extractive QA schedule

When each part of the project happens, what blocks what, and what to cut if we fall behind.

- **What and why:** `Extractive_QA_Design_Plan.docx` (plan v1.1). **How:** [implementation_details.md](implementation_details.md). **When:** this file. **How we work:** [workflow.md](workflow.md).
- This file expands plan §9 into stages. The plan's W1–W7 are course weeks 7–13. Section numbers below (§4.3, E3, …) refer to the plan.
- If this file and the plan disagree on scope, stop and ask a human.
- Drafted 2026-09-28 (week 7). Moved to the submitted proposal's schedule on 2026-10-03 (plan v1.0). Whoever changes a milestone or a stage's scope updates this file and tells the team.

---

## 1. Key dates

As in the proposal, milestones fall on Thursdays, and the following Sunday is a buffer for any stage that slips.

| Course week | Plan week | Dates (2026) | Milestone (Thursday) | Buffer (Sunday) |
|---|---|---|---|---|
| 7 | W1 | 28 Sep – 4 Oct | **Proposal submitted Fri 2 Oct** | — |
| 8 | W2 | 5 – 11 Oct | Thu 8 Oct: setup done, baseline trained, first decoding comparison | Sun 11 Oct |
| 9 | W3 | 12 – 18 Oct | **Thu 15 Oct: reader design frozen** | Sun 18 Oct |
| 10 | W4 | 19 – 25 Oct | Thu 22 Oct: decoding comparison on stage-1 models; both verifier variants started; P3 decision | Sun 25 Oct |
| 11 | W5 | 26 Oct – 1 Nov | **Thu 29 Oct: verifier variant chosen and frozen** | Sun 1 Nov |
| 12 | W6 | 2 – 8 Nov | **Thu 5 Nov: experiments finished; paper and slides written** | Sun 8 Nov |
| 13 | W7 | 9 – 12 Nov | **Project due Thu 12 Nov** | — |

---

## 2. Stages at a glance

| Stage | Covers | Milestone | Buffer |
|---|---|---|---|
| 1. Setup and baseline | Everyone's setup, data and evaluation pipelines, hyperparameter tuning, baseline, first decoding comparison, E3 | Thu 8 Oct | Sun 11 Oct |
| 2. Reader improvements | Compound objective and F1-weighted targets, each against the baseline, then combined if both beat it | Thu 15 Oct | Sun 18 Oct |
| 3. Decoding and verifier start | E1 on the stage-1 models, calibration, verifier training candidates, both verifier variants begin, row 5 reader | Thu 22 Oct | Sun 25 Oct |
| 4. Verifier | Variants compared (E2), the better one frozen, row 4 | Thu 29 Oct | Sun 1 Nov |
| 5. Final experiments and writing | Final ablations, full-system runs, scoring, significance tests; paper and slides | Thu 5 Nov | Sun 8 Nov |
| 6. Submission | Video, code and instructions | Thu 12 Nov | — |

The Sunday buffers are the only slack. The deadline doesn't move, so any slip past a buffer comes out of writing time. Drafting the paper from week 9 protects it.

---

## 3. How the work is ordered

### 3.1 Two kinds of ablation

| | Development ablations | Final ablations |
|---|---|---|
| Purpose | Choose the design (α, γ, τ, W init, verifier variant, …) | Produce the paper's numbers |
| Model | DeBERTa-v3-base (ModernBERT-base for the separate verifier), 1 seed | Large, 3 seeds |
| Scored on | Validation split | Dev (test set), Adversarial SQuAD and SQuAD-Shifts, with bootstrap tests |
| When | Inside each stage | As soon as that component is frozen |

Each ladder row (plan §6.1) depends only on the rows above it. Launch a row's large 3-seed runs as soon as its stage freezes; don't wait for stage 5. Dev is scored only once, in stage 5, on frozen systems.

| Stage | Development ablations (base) | Final large runs launched |
|---|---|---|
| 1 | First MAP vs. MBR comparison on the baseline | Row 0 (baseline) |
| 2 | Head: W init, length bonus, compound vs. joint only. Loss: α, τ. Each against the baseline, then combined | Rows 1–2 |
| 3 | E1 grid with α, γ and τ; calibration; k | Row 3 (decoding only, no retraining); row 5 reader |
| 4 | E2: shared vs. separate, pointwise vs. listwise, in-sample vs. cross-fitted candidates | Row 4 |
| 5 | P2 ablations if time (sibling negatives, augmentation, each data source) | Row 5 verifier; leftover seeds |

### 3.2 Critical path

Each step waits for the one before:

```
pipeline + offset test ─► baseline ─► reader frozen ─► final reader on large ─► candidates sampled ─► verifier training ─► E2, variant chosen ─► row 4 on large ─► final scoring ─► results
      (1.3–1.4)            (1.6)          (2.5)                (2.5)                  (3.4)                 (4.1)               (4.4)                (4.4)             (5.2)
```

**The reader freeze (2.5, Thu 15 Oct) is the main blocker.** Both verifier variants train on candidates sampled from the final reader, so they need the frozen configuration. Each week the freeze slips delays the verifier and everything after it by a week. Ways around it:

- Keep the freeze date and accept the best configuration found by then.
- Build and test the verifier on baseline-reader candidates during stage 2 (2.7), then regenerate candidates once the reader is frozen.
- In the worst case, sample the candidates from the baseline reader. The verifier then sees slightly different mistakes, which costs a little accuracy but keeps the schedule.

Critical-path jobs (2.5 → 3.4 → 4.1 → 4.4) get GPU priority over development ablations.

### 3.3 Work that can start now

None of these waits for a result. All of them can be built against the baseline. Each becomes a GitHub issue with an owner at the week-8 kickoff (workflow.md §3), and the code goes in the modules listed in workflow.md §2.1:

| Work | Needs |
|---|---|
| Compound-objective and F1-target loss code | Baseline code |
| Cross-span diagnostic (E3) | Baseline predictions |
| Calibration and MBR decoding code (F1 + γ·EM on the scorer's normalized strings, spec §4.4) | Baseline span probabilities; retuned later |
| Adversarial SQuAD (worst-case score, spec §6) and SQuAD-Shifts evaluation | Nothing |
| Bootstrap test script | Nothing |
| Pins for `deberta-v3-base` and ModernBERT-large; `peft` and `scipy` in `pyproject.toml` (with `uv lock` and the cluster smoke test) | Nothing |
| Verifier marking code and acceptance test V5, for both tokenizers | Baseline candidates |
| Loaders for AdversarialQA and the SQuAD 2.0 unanswerable questions (row 5) | Nothing |
| Paper intro and related work; Scholar check on MBR in extractive QA | Nothing |

These wait on a result, but are short and off the critical path:

- How much effort the joint head gets depends on E3 (1.7).
- The calibration temperature, γ, λ and k are refit on the validation split once the final reader and verifier exist. This takes minutes.
- Ladder rows 1–3 on large each wait only for their own stage to freeze.
- The paper's method section waits for designs to freeze; its results wait for stage 5.

"Parallel" means nobody waits on anybody else's result. All jobs still share one GPU queue.

---

## 4. Stages in detail

### Stage 1: Setup and baseline (week 8; milestone Thu 8 Oct, buffer Sun 11 Oct)

**Goal:** everyone set up, and a proper pipeline with a baseline trained in it.

- **1.1 Proposal.** Submitted Fri 2 Oct.
- **1.2 Team setup (all members).** GitHub access, repository, environment and cluster smoke test, through the onboarding in `local.example/README.md`. Stage issues and milestones are set up at the kickoff (workflow.md §3).
- **1.3 Data pipeline.** 384-token windows with stride 128; offset map between characters, subword tokens and words; dynamic padding and length bucketing (spec §3.2–3.4). The offset unit test must pass before any training.
- **1.4 Evaluation pipeline.** Model outputs → top span → `predictions.json` → official scorer. Windows of one passage share a softmax. Includes Adversarial SQuAD (worst-case score, spec §6) and SQuAD-Shifts, both from `download_external`.
- **1.5 Hyperparameter tuning and baseline (ladder row 0).** Base for development, large for results. Start the 3 large seeds.
- **1.6 First decoding comparison.** MAP vs. MBR on the baseline (plan §4.5), with MBR on the scorer's normalized strings (spec §4.4). Also measure the **normalization gap**: on validation, how often the prediction matches the gold span exactly by position vs. by the official EM. It bounds what normalized training targets (2.2) can gain.
- **1.7 Cross-span diagnostic (E3) on the baseline.** It decides how much effort the joint head deserves (plan §4.3; recipe in spec §4.5).
- **1.8 Dependencies and pins** for stage 2 and the verifier (§3.3).

**Done when:** the offset test passes, the pipeline baseline matches the sandbox quick fine-tune (about 89 validation F1), and the first MAP vs. MBR numbers exist.

**Progress (2026-10-03):** the validation split, cluster setup and tooling (`soc.sh submit`, job template), checkpoint and resume, and the sandbox baseline over 3 seeds (validation EM 76.50 ± 0.13, F1 89.29 ± 0.06, about 13–15 min of training each on an A100) are done. The offset test and the proper pipeline are not.

### Stage 2: Reader improvements (week 9; milestone Thu 15 Oct, buffer Sun 18 Oct)

**Goal:** settle the reader design. This is the main blocker (§3.2).

- **2.1 Compound objective** (plan §4.3; spec §4.3). Joint head plus the auxiliary independent loss, against the baseline on base. Ablate W init, the length bonus, and joint only.
- **2.2 F1-weighted targets** (plan §4.4), against the baseline on base. Sweep α (the F1:EM loss ratio) and τ, and compare positional with normalized F1 in the targets (spec §4.3), sized by the gap measured in 1.6.
- **2.3 Combine** the two only if both beat the baseline on the validation split. Otherwise keep the one that does, and report the other as an ablation.
- **2.4 Rerun E3 on the new head.** The before-and-after comparison is the E3 result.
- **2.5 Freeze the reader** (Thu 15 Oct) through a freeze PR (workflow.md §5.3). Then launch ladder rows 1–2 on large, 3 seeds, from `main`.
- **2.6 Report and slides** draft the model and data choices.
- **2.7 Verifier groundwork,** off the critical path: marking code and test V5, both backbones loading, a first run on baseline candidates.

**Done when:** the reader configuration is frozen and its large runs are queued.

### Stage 3: Decoding and verifier start (week 10; milestone Thu 22 Oct, buffer Sun 25 Oct)

**Goal:** metric-aware decoding on the stage-1 models, and both verifier variants under way.

- **3.1 Candidate extraction.** Top-k spans per question (k ≈ 10), merged across windows.
- **3.2 Calibration.** One temperature fit on the validation split; report ECE.
- **3.3 E1 on the stage-1 models.** {one-hot, F1-soft targets} × {MAP, MBR}, with the γ (F1:EM decoding ratio) and τ sweeps. Ladder row 3 needs no new training.
- **3.4 Verifier training candidates (critical path).** Sample spans from the final reader's span distribution on its own training questions, at a temperature above 1 (plan §4.6). Check on the validation split that they resemble the reader's real mistakes, and measure top-k recall, the verifier's ceiling. The cross-fitting readers for the E2 ablation (3 folds, on base) are off the critical path.
- **3.5 Both verifier variants begin** (plan §4.6; spec §5.3–5.4): shared DeBERTa + LoRA, and separate ModernBERT.
- **3.6 Row 5 reader.** Retrain the frozen reader configuration with AdversarialQA added (plan §4.8), large, 3 seeds.
- **3.7 Decide on P3** (levitated verifier, E4, E5), based on the schedule.

**Done when:** sampled candidates exist for all of train, decoding works end to end, and both variants have trained on base.

### Stage 4: Verifier (week 11; milestone Thu 29 Oct, buffer Sun 1 Nov)

**Goal:** choose and freeze the verifier. The hardest part.

- **4.1 Training.** On the sampled candidates, base first, then large. The spec §5.7 acceptance tests for the variant must pass first.
- **4.2 Combining.** Reader log-probability + λ × verifier score, softmax over candidates, recalibrate, decode. Tune λ on the validation split.
- **4.3 E2.** Shared vs. separate; listwise vs. pointwise; in-sample (sampled) vs. cross-fitted candidates; fix types (boundary fix vs. switch to a different answer).
- **4.4 Freeze the better variant** (Thu 29 Oct) through a freeze PR. Then launch ladder row 4 on large, 3 seeds, from `main`.
- **4.5 Row 5 verifier.** The frozen variant, trained with the SQuAD 2.0 unanswerable negatives (spec §5.1), on top of the row 5 reader.

**Done when:** row 4 is trained and row 5 is running. **Fallback:** drop the cross-fitting ablation before either variant.

### Stage 5: Final experiments and writing (week 12; milestone Thu 5 Nov, buffer Sun 8 Nov)

**Goal:** the paper's numbers, and the paper. Experiments stop at the milestone.

- **5.1 Final ablations and full-system runs;** leftover seeds, mostly the verifier's.
- **5.2 Score all frozen systems** on dev, Adversarial SQuAD and SQuAD-Shifts with the official scorer: one person, one PR, from `main` (workflow.md §6.1). This is the first time dev is used.
- **5.3 Statistics.** Mean ± standard deviation over 3 seeds; paired bootstrap tests with 95% confidence intervals (SciPy).
- **5.4 Analysis.** Final E3 numbers, verifier fix types, ECE, worked error examples for the paper and video.
- **5.5 P2 extras if time allows:** augmentation, sibling negatives, each data source on its own.
- **5.6 Paper and slides written.** ACL short format, 4 pages plus 1 page of references. What goes in the paper, the appendix or only the video is decided in week 11.

### Stage 6: Submission (week 13; due Thu 12 Nov)

- **6.1 Video:** 15 minutes, every member speaks.
- **6.2 Code release.** README with run instructions, citations for borrowed code, notebook outputs cleared, generated data committed.
- **6.3 Submit.**

---

## 5. If we fall behind

Cut in this order:

1. P3 extensions: levitated verifier and E2b, E4, E5, FlexAttention.
2. P2 ablations: augmentation, sibling negatives, each data source on its own, joint only.
3. The E2 cross-fitting ablation (keep in-sample candidates).
4. Large runs of the losing verifier variant (compare the two on base only).
5. Row 5 (additional data).

Keep the main ladder (rows 0–4), E1, E3 and the two-variant comparison. They are the core of the paper.
