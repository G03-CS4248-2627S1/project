# Row 1 boundary-group loss test

## Question

Does partial training credit for dropping selected boundary prepositions improve
row 1's official validation EM and F1? The official scorer is unchanged.

## Matched settings

All systems use the row 1 joint head, even windows, two epochs, seed 0,
the same training split and the same MAP and MBR decoders. The changed word set is
`a, an, the, of, in, on, at, to, for, by, with, from, as`.

| System | Training group loss | MBR word set | Source |
|---|---|---|---|
| Current row 1 | Article-only group NLL | Article-only | Existing row1-joint seed 0 run |
| Expanded group | Expanded group NLL | Expanded | `row1-function-words.yaml`; smoke job 1009343 is unverified, full run not submitted |
| Mixed group | 0.5 article-only group NLL + 0.5 expanded group NLL | Expanded | `row1-boundary-mix.yaml`; not submitted |

The expanded and mixed configs differ only in `expanded_group_weight`.
They therefore isolate the training loss mixture. Comparing either to current
row 1 changes both training and MBR grouping; MAP is unaffected by the MBR
word-set setting. Each group's probability sums the spans at the annotated
passage position that share its content-word core. No other occurrence of the
same text receives training credit.

## Run and decision

1. Verify the existing smoke job and resume from its checkpoint. Do not claim
   that the code works until the resume and official validation scoring finish.
2. Smoke-test `row1-boundary-mix` with checkpoint and resume, because it adds a
   new training-loss path.
3. Train the expanded and mixed configs for two epochs on the same seed. Use
   `slurm/train_reader.sbatch --per-device-batch 8 --eval-each-epoch` and an
   A100-40 GPU with a 90-minute limit. Keep checkpoints on the cluster.
4. Compare official validation EM, F1 and EM+F1 for both MAP and MBR against
   existing row 1 seed 0. If a setting has a useful gain, confirm on seeds 1–2.

The held-out dev set is not used for selection. Report the resolved config,
commit, seed, run id, GPU and scores for every completed run.
