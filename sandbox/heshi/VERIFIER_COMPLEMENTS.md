# Verifier complements — V1, V2, V3 and a matched V0 control

All runs use the same frozen Row 1 seed-0 reader and exact span candidate lists. One verifier epoch,
seed 0, learning rate 3e-5 and effective batch 16 are fixed. The three experiments are:

| ID | Config | Encoder | Loss |
| --- | --- | --- | --- |
| V0, control | `verifier-separate.yaml` | ModernBERT-large | Original one-hot listwise CE |
| V1 | `verifier-gold-group.yaml` | ModernBERT-large | `-log(sum probability of positional normalized EM-1 candidates)` |
| V2 | `verifier-ce-margin.yaml` | ModernBERT-large | .9 CE + .1 margin loss; strongest EM-0 negative, margin 1 |
| V3 | `verifier-deberta-independent.yaml` | Independently pretrained DeBERTa-v3-large | Original CE; full fine-tuning, explicit eager attention and gradient checkpointing |

Gold-group loss changes only training credit, not the candidate list or the words normalized away.
Correct article/punctuation variants are never negatives for V2's margin term. V3 never loads trained
reader weights as its verifier initialization. The default CE implementation and ModernBERT SDPA stay
unchanged for V0. New losses reject soft targets or group lists rather than silently mixing settings.

## Files and execution

The three configs are in `configs/experiments/`. Losses are in `squadqa/verifier.py`, with data-label
validation and training dispatch in `squadqa/train_verifier.py`. Tests include hand-calculated values
and gradients, boundary twins, absent negatives, invalid labels, tiny independent DeBERTa backward,
and end-to-end stop/resume/evaluation/rescore for both new losses.

Run these submission commands on the Mac from the **isolated verifier-complements checkout**. The
cluster checkout must be at the same commit, with a working local `.env`, SSH tunnel and cluster `.venv`:

```bash
# Print the exact job graph. This default mode submits nothing.
python3 sandbox/heshi/verifier_complements.py plan

# CPU model-cache, checkpoint-readability and test preflight only.
python3 sandbox/heshi/verifier_complements.py submit --phase setup

# Also create/audit one frozen candidate build and run GPU checkpoint/resume checks.
python3 sandbox/heshi/verifier_complements.py submit --phase smoke

# Queue control plus three full one-epoch trials, all gated by successful checks.
python3 sandbox/heshi/verifier_complements.py submit --phase full
```

The controller records each accepted job immediately and skips roles already recorded, so extending
setup to smoke to full does not duplicate jobs. A failed dependency cancels downstream jobs. A failed
setup/smoke role must be diagnosed before using its existing record; do not blindly reuse a failed
dependency. Never change the checkout commit while its jobs are pending.

Jobs, based on `slurm/template.sbatch`:

1. `verifier_complement_setup.sbatch`: normal CPU, 45-minute limit. Checks the shared final reader
   checkpoint, links it into the user's scratch if absent, downloads hash-checked pinned model files
   to the cluster cache and runs the CPU suite. It does not copy the reader checkpoint to the laptop.
2. `verifier_complement_candidates.sbatch`: A100-40, 45-minute limit. Creates the train and validation
   lists once, then validates labels/settings/counts and freezes SHA-256 hashes. Existing files are
   audited, never silently overwritten.
3. `verifier_complement_smoke.sbatch`: A100-40, 1-hour limit. Each of the four settings stops at step 20,
   resumes to step 40 and runs validation evaluation. These smoke scores are execution checks, not
   experiment results. Lists are audited before and after.
4. `verifier_complement_train.sbatch`: A100-40, up to 3 hours per allocation. Each real trial trains
   one full verifier epoch and writes reader-only, verifier-only and combined MAP/MBR predictions,
   metrics and candidate-file hashes. Lambda and temperature use the existing joint fit.

The independent DeBERTa run may exceed three hours, so full mode queues one checkpoint-resume
allocation behind it. That allocation exits immediately if valid completed metrics already exist,
avoiding the historical problem of a resume overwriting training time with zero. Additional resumes
can use the same script and `RESUME_JOB_ID=<original job id>` if needed. ModernBERT timeouts also
retain resumable checkpoints; inspect them before submitting a continuation.

All three-hour limits are limits, not runtime predictions. The historical ModernBERT run took about
88 minutes on H100 MIG at a different per-device batch. DeBERTa may be slower. Full mode can consume
more than four GPU-hours; coordinate this extra ablation batch with the team before full submission.
This script does not send team messages or push a GitHub branch.

## Result collection

The controller's default job record is in the original project's
`reports/2026-11-01-verifier-complement-jobs.json`; override with `--record` if running elsewhere.
Each full run ID ends in its primary Slurm job ID. Use `soc.sh list`, `soc.sh log` and `soc.sh fetch` to
collect results. Checkpoints stay under cluster scratch. The candidate manifest stays at
`data/verifier/2026-10-23-row1-joint-s0-j998373/complement_manifest.json`.

Compare each combined result with its paired V0/reader-only result on these same lists, not older
top-10 reader scores. Report separate EM/F1 and sums. The automatic metrics fit lambda/T on the scored
validation set; article-held-out fitting and uncertainty checks are required before a strong gain
claim. Hold out dev/test for final scoring. Only a promising seed-0 result should be confirmed with
additional verifier seeds; keep the reader fixed throughout.
