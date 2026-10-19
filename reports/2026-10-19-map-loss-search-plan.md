# MAP loss tuning job

Two sequential Slurm jobs use the existing DeBERTa-v3-base joint-span trainer. Each requests
one A100-40 GPU, 48 GB RAM and 2 h 30 min. Each runs six full, seed-0, one-epoch trials.
The first job (`PHASE=e`) tests the one-hot plus expected-risk loss at β=0, 1 and 3,
then chooses three β values by Gaussian-process expected improvement in [0, 3].
The β=0 trial is the matched one-hot baseline; its MBR score is cell B.

The second job (`PHASE=c`) tests F1 soft targets at (α, τ)=(0.25, 0.1), (0.5, 0.3)
and (0.75, 1.0), then chooses three points by expected improvement over
α∈[0.1, 0.9] and log τ∈[log 0.05, log 1]. The F1 floor remains φ=0.
The wider α range is a follow-up performance search, not the original E1 fixed-ratio comparison.

All trials use the current code commit, same seed and training-derived validation split.
The objective is **official validation MAP EM + F1**. MBR scores and fitted calibration
temperature are recorded for diagnosis but do not steer the search. Each job writes
`results/<SEARCH_ID>/summary.json` after every trial. Interrupted jobs can resume with
the same `SEARCH_ID`; completed trials' optimizer checkpoints are deleted on the cluster
to save space, while their metrics and predictions remain there. No checkpoint is
downloaded to this Mac.

Once the user requests retrieval, fetch the two summary folders and render a Markdown
table with `python -m squadqa.loss_search_report <E-summary> <C-summary> -o reports/map-loss-search.md`.
The table includes phase, trial number, selection method, α/τ/β, MAP EM, MAP F1,
MAP EM+F1, MBR EM+F1, MAP difference from B, and the trial run ID.
The selected candidate then needs seeds 1–3 against the matched B baseline before
claiming an improvement. Neither job reads the held-out SQuAD dev/test file.

Submission, once approved under `AGENTS.md` and the SoC cluster instructions:

```bash
bash .claude/skills/soc-cluster/soc.sh submit --gpus=a100-40 --export=ALL,PHASE=e slurm/map_loss_search.sbatch
bash .claude/skills/soc-cluster/soc.sh submit --gpus=a100-40 --export=ALL,PHASE=c slurm/map_loss_search.sbatch
```
