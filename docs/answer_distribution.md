# Full-span diagnostics: issues #95 and #96

These are frozen-checkpoint inference diagnostics, not new training, final test-set scoring,
calibration fitting, or issue #97's decoder sweep. Existing predictions/metrics/calibration
are read only. New summary/per-question files are added to each producing reader's result folder.

## Commands

- #95: `python -m squadqa.answer_distribution multi-answer`
- #96: `python -m squadqa.answer_distribution shape --include-baseline`
- Smoke: add `--limit 8 --output-root local/answer-distribution-smoke` and explicit `--run-id`.
- Submit through `soc.sh submit slurm/answer_distribution.sbatch <arguments>`.
- CPU preparation: `slurm/answer_distribution_setup.sbatch` builds the pinned NQ data,
  populates only the pinned tokenizer/config caches and runs the full test suite.

Defaults: row 1 seeds 0–2 from #73; #96 optionally includes row 0 seeds 0–2.
Every checkpoint is loaded sequentially, from its complete trained state_dict using strict
loading and the pinned model architecture (no redundant pretrained weight download).
Both T=1 and the existing calibrated T use the same encoder scores. Inference is chunked
64 questions at a time and batched 16 questions per forward pass. All windows are handled
by the existing `predict.score_questions` and `decode.combine_windows` implementation.

## Definitions and interpretation

Full-span/group statistics are computed before top-k truncation. Boundary groups pool
all spans with the same content core. Empty cores remain separate, as in `decode`.
The null, if present, remains in the full probability denominator and entropy as its
own outcome. Group concentration counts are null if answer groups cannot reach a target.

For #96, modes are the reproducible greedy set obtained by visiting content cores in
probability order and accepting nonoverlapping cores with group mass >=5%. This is a
specified operational statistic, not a unique mathematical density-mode decomposition.
The second-disjoint-group mass is the largest nonempty core disjoint from the top group,
even when below 5%. Gold overlap is overlap at counted content positions; unaligned golds
are recorded, not silently omitted. Outcome splits use the existing top-10 MAP decoder
and official string EM/F1, while shape statistics use the full distribution. At full
validation size, reproduced MAP EM/F1 must be within 0.1 points of saved metrics or the
job raises before writing a summary (small bf16/hardware differences are possible).

For #95, merge overlapping annotated content cores (not merely adjacent spans). q is
uniform over distinct annotated groups, summed into regions. p(region) sums every group
whose nonempty core is wholly contained by that region; crossing/partial/outside spans
are in the rest bucket. Primary collapse/divergence statistics use questions with at
least two disjoint regions, with the total/eligible counts explicit. Report all questions,
held-out 250, and training-derived 1000 separately. The second answer region is the
second-highest region by model mass; its best group's rank is over ALL groups. The q
reference is constructed and must not be interpreted as human annotation frequencies.
KL and square-root JS use natural logs, a rest bucket, and p floor 1e-15 followed by
renormalization. Splits report first-annotated status, min region gap <=10/>10 words and
max region length <=5/>5 words, with per-question data enabling other summaries later.

Outputs: `distribution_validation{,_questions}.json` and `multi_answer{,_questions}.json`.
Each includes analysis commit/job/device, checkpoint path/size/step, original training
provenance, data/config/calibration hashes and all definitions. Existing outputs are
refused unless `--overwrite` is deliberately given. Smoke outputs cannot use `results/`.
A summary is written after its per-question artifact. No candidate dumps are committed.

## Review/results workflow

Commit code/config scripts first; use that exact commit on the cluster. CPU tests,
GPU smoke, then independent #95/#96 jobs gated by successful smoke. Fetch artifacts,
check counts and validation score agreement, summarize across seeds without confusing
question uncertainty with seed spread, and submit a results PR. Post issue-specific
summary comments with job IDs, commit, caveats and PR link. Assistants never merge.

## Collected results

See `reports/answer_distribution_2026-10-31.md`. Rebuild/audit the numerical report with
`python -m squadqa.report_answer_distribution`; this uses committed JSON only, without
a GPU or model checkpoint.
