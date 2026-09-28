# timeline.md — CS4248 Extractive QA schedule

When each part of the project happens, what blocks what, and what to cut if we fall behind.

- **What and why:** `Extractive_QA_Design_Plan.docx` (plan v0.5). **How:** [implementation_details.md](implementation_details.md). **When:** this file.
- This file expands plan §9 into stages. The plan's W1–W7 are course weeks 7–13. Section numbers below (§4.3, E3, …) refer to the plan.
- If this file and the plan disagree on scope, stop and ask a human.
- Drafted 2026-09-28 (week 7). Whoever changes a freeze date or a stage's scope updates this file and tells the team.

---

## 1. Key dates

| Course week | Plan week | Dates (2026) | Milestone |
|---|---|---|---|
| 7 | W1 | 28 Sep – 4 Oct | **Proposal due Fri 2 Oct** |
| 8 | W2 | 5 – 11 Oct | **Reader design frozen** (end of week) |
| 9 | W3 | 12 – 18 Oct | Decide whether to attempt P3 extensions |
| 10 | W4 | 19 – 25 Oct | Start writing intro and related work |
| 11 | W5 | 26 Oct – 1 Nov | **Experiments frozen** (expected) |
| 12 | W6 | 2 – 8 Nov | Experiments frozen at the latest by mid-week |
| 13 | W7 | 9 – 12 Nov | **Project due Thu 12 Nov** |

---

## 2. Stages at a glance

| Stage | Covers | Expected | Worst case |
|---|---|---|---|
| 1. Setup and baseline | Proposal, data and scoring pipeline, validation split, baseline | Week 7 | Week 8 |
| 2. Reader improvements | Joint span head, F1-weighted targets, cross-span diagnostic (E3) | Week 8 | Week 9 |
| 3. Decoding and evaluation | Calibration, MAP vs. expected-F1 decoding (E1), robustness sets, verifier training candidates | Week 9 | Week 10 |
| 4. Verifier | Candidate re-scoring model and its ablations (E2) | Weeks 9–10 | Week 11 |
| 5. Final experiments and analysis | Leftover seeds, scoring on dev and Adversarial SQuAD, significance tests, analysis | Week 11 | Mid-week 12 |
| 6. Report and presentation | Paper, slides, video, code release | Weeks 12–13 | Weeks 12–13, compressed |

The worst case is about one week behind expected, and that week comes out of writing time. Starting the paper in week 10 protects it.

---

## 3. How the work is ordered

### 3.1 Two kinds of ablation

| | Development ablations | Final ablations |
|---|---|---|
| Purpose | Choose the design (α, τ, W init, listwise vs. pointwise, …) | Produce the paper's numbers |
| Model | DeBERTa-v3-base, 1 seed | DeBERTa-v3-large, 3 seeds |
| Scored on | Validation split | Dev (test set) and Adversarial SQuAD, with bootstrap tests |
| When | Inside each stage | As soon as that component is frozen |

Each ladder row (plan §6.1) depends only on the rows above it. Launch a row's large 3-seed runs as soon as its stage freezes; don't wait for stage 5. Dev is scored only once, in stage 5, on frozen systems.

| Stage | Development ablations (base) | Final large runs launched |
|---|---|---|
| 1 | — | Row 0 (baseline) |
| 2 | Head: W init, length bonus. Loss: α, τ | Rows 1–2 |
| 3 | E1 grid, calibration, k | Row 3 (decoding only, no retraining) |
| 4 | E2: pointwise vs. listwise, in-sample vs. cross-fitted candidates, sibling negatives | Row 4 |
| 5 | P2 ablations if time | Leftover seeds only |

### 3.2 Critical path

Each step waits for the one before:

```
pipeline + offset test ─► baseline ─► reader frozen ─► final reader on large ─► candidates sampled ─► verifier training ─► verifier ablations ─► row 4 on large ─► final scoring ─► results
      (1.2–1.3)            (1.5)          (2.5)                (2.5)                  (3.5)                 (4.4)                (4.6)               (4.7)             (5.2)
```

**The reader freeze (2.5) is the main blocker.** The verifier trains on candidates sampled from the final reader, so it needs the frozen configuration. Since plan v0.5 there are no cross-fitting readers on this path: the verifier waits for one large reader run, not three. Each week the freeze slips still delays the verifier and everything after it by a week. Ways around it:

- Keep the hard freeze date (end of week 8) and accept the best configuration found by then.
- Build and test the verifier on baseline-reader candidates while stage 2 runs, then regenerate candidates once the reader is frozen.
- In the worst case, sample the candidates from the baseline reader. The verifier then sees slightly different mistakes, which costs a little accuracy but keeps the schedule.

Critical-path jobs (2.5 → 3.5 → 4.4 → 4.7) get GPU priority over development ablations.

### 3.3 Work that can start now

None of these waits for a result. All of them can be built against the baseline:

| Work | Needs |
|---|---|
| Joint head and F1-target loss code | Baseline code |
| Cross-span diagnostic (E3) | Baseline predictions |
| Calibration and expected-F1 decoding code | Baseline span probabilities; retuned later |
| Adversarial SQuAD and MRQA evaluation pipelines | Nothing |
| Bootstrap test and selective-QA (E5) scripts | Nothing |
| Verifier code and unit tests; DeBERTa mask and position check | Baseline candidates |
| Job scripts, checkpoint and resume, dynamic padding | Nothing |
| Paper intro and related work; novelty literature check | Nothing |

These wait on a result, but are short and off the critical path:

- How much effort the joint head gets depends on E3 (2.1). Stage 2 still freezes on time.
- The calibration temperature, λ and k are refit on the validation split once the final reader and verifier exist. This takes minutes.
- Ladder rows 1–3 on large each wait only for their own stage to freeze.
- The paper's method section waits for designs to freeze; its results wait for stage 5.

"Parallel" means nobody waits on anybody else's result. All jobs still share one GPU queue.

---

## 4. Stages in detail

### Stage 1: Setup and baseline (week 7; worst case week 8)

**Goal:** a proper pipeline with a baseline trained in it.

- **1.1 Proposal.** Due Fri 2 Oct. Email the lecturer about using a closed LLM for data generation (plan §2).
- **1.2 Data pipeline.** Validation split; 384-token windows with stride 128; offset map between characters, subword tokens and words; dynamic padding and length bucketing (spec §3.2–3.4). The offset unit test must pass before any training.
- **1.3 Evaluation pipeline.** Model outputs → top span → `predictions.json` → official scorer. Windows of one passage share a softmax.
- **1.4 Cluster infrastructure.** Job scripts, checkpoint and resume, run folders under `results/<run_id>/`.
- **1.5 Baseline (ladder row 0).** Base for development, large for results. Start the 3 large seeds.
- **1.6 Early checks.** Can DeBERTa take a custom 3D attention mask and position ids (plan §4.7)? Literature check on prior QA uses of MBR decoding, soft targets and cross-fitting (plan §5): done 2026-09-28, see [literature_review.md](literature_review.md). Still to do: a Scholar check on MBR in extractive QA, and reading the entries checked only from abstracts.

**Done when:** the offset test passes and the pipeline baseline matches the sandbox quick fine-tune (about 89 validation F1).

**Progress (2026-09-28):** the validation split, cluster setup, checkpoint and resume, and a timed 1-epoch large run (sandbox quick fine-tune: validation EM 76.5 / F1 89.3, about 13 min on an A100-80GB) are done. The offset test and the proper pipeline are not.

### Stage 2: Reader improvements (week 8; worst case week 9)

**Goal:** settle the reader design. This is the main blocker (§3.2).

- **2.1 Cross-span diagnostic (E3) on the baseline.** Probability on cross spans, by confidence level. It decides how much effort the joint head deserves (plan §4.3; recipe in spec §4.5).
- **2.2 Joint span head.** score(i, j) with W initialized to zero, length bonus, L = 30. Ablate on base: W init zero vs. random; length bonus on vs. off.
- **2.3 F1-weighted targets.** Soft target over spans that overlap the gold answer (plan §4.4). Sweep α and τ on base.
- **2.4 Rerun E3 on the joint head.** The before-and-after comparison is the E3 result.
- **2.5 Freeze the reader** (end of week 8). Launch ladder rows 1–2 on large, 3 seeds.

**Done when:** the reader configuration is frozen and its large runs are queued.

### Stage 3: Decoding and evaluation (week 9; worst case week 10)

**Goal:** metric-aware decoding, and training candidates for the verifier.

- **3.1 Candidate extraction.** Top-k spans per question (k ≈ 10), merged across windows.
- **3.2 Calibration.** One temperature fit on the validation split; report ECE.
- **3.3 Decoding rules.** MAP vs. expected-F1 (MBR) decoding (plan §4.5).
- **3.4 E1 grid on base.** {one-hot, F1-soft targets} × {MAP, MBR}, plus a τ sweep. Ladder row 3 needs no new training.
- **3.5 Verifier training candidates (critical path).** Sample spans from the final reader's span distribution on its own training questions, at a temperature above 1 (plan §4.6). Check on the validation split that they resemble the reader's real mistakes, and measure top-k recall, the verifier's ceiling. The cross-fitting readers for the E2 ablation (3 folds, on base) are off the critical path.
- **3.6 Robustness pipelines.** Adversarial SQuAD (AddSent, AddOneSent); MRQA out-of-domain sets (P2).
- **3.7 Decide on P3** (E7, E8, E9), based on the schedule.

**Done when:** sampled candidates exist for all of train and decoding works end to end.

### Stage 4: Verifier (weeks 9–10; worst case week 11)

**Goal:** a second model that re-scores the top candidates. The hardest part.

- **4.1 Marking.** Add `<a>` and `</a>` tokens, resize the embedding table, and wrap each candidate in the passage. Built early against baseline candidates.
- **4.2 Training candidates.** The sampled candidates from 3.5, plus the gold answer if missing, near-misses one word longer or shorter, and sibling answers (gold answers of other questions on the same passage).
- **4.3 Model and loss.** Score head on `<a>`, `</a>` and `[CLS]`; listwise loss with F1-weighted targets.
- **4.4 Training.** On the sampled candidates, base first, then large. The spec §5.6 acceptance tests must pass first.
- **4.5 Combining.** Reader log-probability + λ × verifier score, softmax over candidates, recalibrate, decode. Tune λ on the validation split.
- **4.6 E2 ablations on base.** In-sample (sampled) vs. cross-fitted candidates; listwise vs. pointwise; sibling negatives on vs. off; fix types (boundary fix vs. switch to a different answer).
- **4.7 Freeze the verifier.** Launch ladder row 4 on large, 3 seeds.
- **4.8 (P2) Levitated-marker verifier and cost study (E2b),** only if time allows.

**Done when:** row 4 is trained. **Fallback:** drop the cross-fitting ablation before dropping the verifier.

### Stage 5: Final experiments and analysis (week 11; worst case mid-week 12)

**Goal:** the paper's numbers. Experiments stop at the end of this stage.

- **5.1 Leftover seeds,** mostly the verifier's.
- **5.2 Score all frozen systems** on dev and Adversarial SQuAD with the official scorer. This is the first time dev is used.
- **5.3 Statistics.** Mean ± standard deviation over 3 seeds; paired bootstrap tests with 95% confidence intervals.
- **5.4 Analysis.** Final E3 numbers, verifier fix types, ECE, worked error examples for the paper and video.
- **5.5 P2 extras if time allows:** training-data fractions (E4), selective QA (E5), verifier cost study (E2b).
- **5.6 Decide what goes where:** the 4-page paper, the appendix, or only the video.

### Stage 6: Report and presentation (weeks 12–13)

**Goal:** the submission, due Thu 12 Nov.

- **6.1 Paper.** ACL short format, 4 pages plus 1 page of references. Intro and related work from week 10, method once designs freeze, results after stage 5.
- **6.2 Slides and a 15-minute video** in which every member speaks.
- **6.3 Code release.** README with run instructions, citations for borrowed code, notebook outputs cleared, generated data committed.
- **6.4 Submit,** with whatever buffer remains.

---

## 5. If we fall behind

Cut in this order:

1. P3 extensions (E7, E8, E9).
2. P2 extensions and ablations (E2b, E4, E5, MRQA).
3. The levitated verifier (keep the in-place one).
4. The E2 cross-fitting ablation (keep in-sample candidates).

Keep the main ladder (rows 0–4), E1 and E3. They are the core of the paper.
