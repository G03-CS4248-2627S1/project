# Decoding sweep: γ, calibration, grouping and the normalization gap

Issue #25, plan §9.2. Validation (7,708 questions, one reference each), 3 seeds per reader, on DeBERTa-v3-large. Decode-only, from each run's saved top-10 candidates, scored with the official scorer. ± is the sample standard deviation over seeds; brackets are 95% paired bootstrap intervals over questions.

```
python -m squadqa.decode_sweep \
  row1=2026-10-23-row1-joint-s0-j998373,2026-10-23-row1-joint-s1-j998384,2026-10-23-row1-joint-s2-j998397 \
  row0=2026-10-05-row0-baseline-s0-j910132,2026-10-05-row0-baseline-s1-j910133,2026-10-05-row0-baseline-s2-j910134 \
  -o reports/decode_sweep_2026-11-05.json
```

## Decision

**γ stays at 1, so final scoring can use the frozen decoders as they are.** No setting beats it under plan §15.3's selection rule on either reader, and on row 1 γ = 1 has the highest EM + F1 of the sweep's γ values.

## Findings

Differences are in EM + F1 against MBR at γ = 1 on the same reader, unless stated.

1. **γ traces the expected trade-off, and 1 is the best point on row 1.** MAP is −0.18 [−0.31, −0.04], γ = 2 is −0.10 [−0.18, −0.02], γ = 0.5 is −0.06 [−0.16, +0.04], and γ = 0 is −0.26 [−0.50, −0.02]. Expected F1 alone (γ = 0) raises F1 by 0.10 and lowers EM by 0.36.
2. **On row 0, more weight on EM helps, and MAP stays ahead of MBR.** MAP is +0.18 [+0.02, +0.36], but that is below the rule's threshold of 0.26. This matches #108: the standard head's MBR loses by hedging.
3. **Calibration barely changes the answers.** On row 1, T = 1 is −0.07 [−0.24, +0.10] with MBR, and MAP falls by 0.02. The fitted T of 1.29 is mild, and the top ten stay the same.
4. **Grouping boundary variants matters more than γ on row 1.** Without it, MBR is −0.18 [−0.40, +0.05], and MAP is 0.48 below its grouped score (165.95 against 166.43; the mean difference, no interval). A reader trained on groups splits a group's mass among its variants, so each variant alone looks weaker.
5. **Spec rule 4 costs nothing at decoding.** Grouping by the scorer's normalization at any position ties on row 1 (+0.02 [−0.06, +0.10]). On row 0 it gains +0.17 [+0.09, +0.26] with MBR, still below the rule's threshold.
6. **Normalization gap.** Row 1's MAP answers score official EM 1 on 76.97% of questions: 66.48 points on the identical span, 9.85 on a boundary variant, and 0.62 on a different span the scorer also accepts. Row 0's are 68.93, 7.35 and 0.58. So scoring by position alone would understate EM by about 10 points, and spec rule 4 forgoes at most about 0.6. Row 1's larger boundary-variant share is what training on groups should produce.

## Limits

- Validation has one reference per question. Dev scores against several, so MBR's trade-off may differ there (plan §7.2).
- Decoding sees only the top ten candidates.
- Row 0 and row 1 also differ in epochs, windows and target unit, so compare settings within a reader.
- The rule's threshold is twice the seed std of the reference's EM + F1, measured on three seeds.
- A setting's interval covers question sampling; the seed spread is the ±.

"Questions scored differently" is the mean number of questions per run whose EM or F1 differs from the reference. Under the normalization-gap tables, "Elsewhere" means any span that is neither the gold nor a boundary variant of it, and counts are the mean per run.

## Results

### row1 (3 seeds)

| Setting | EM | F1 | EM + F1 | Against γ = 1 | Questions scored differently | Wins by the rule |
|---|---|---|---|---|---|---|
| map | 76.97 ± 0.07 | 89.46 ± 0.05 | 166.43 ± 0.12 | -0.18 [-0.31, -0.04] | 69.3 | no |
| mbr-g0 | 76.68 ± 0.02 | 89.67 ± 0.10 | 166.35 ± 0.12 | -0.26 [-0.50, -0.02] | 201.0 | no |
| mbr-g0.5 | 76.96 ± 0.01 | 89.59 ± 0.02 | 166.55 ± 0.03 | -0.06 [-0.16, +0.04] | 36.7 | no |
| mbr-g1 | 77.04 ± 0.03 | 89.57 ± 0.06 | 166.61 ± 0.09 | +0.00 [+0.00, +0.00] | 0.0 | no |
| mbr-g2 | 76.98 ± 0.03 | 89.52 ± 0.04 | 166.51 ± 0.08 | -0.10 [-0.18, -0.02] | 30.3 | no |
| map-t1 | 76.95 ± 0.13 | 89.45 ± 0.06 | 166.41 ± 0.19 | -0.20 [-0.40, +0.00] | 116.0 | no |
| mbr-g1-t1 | 77.00 ± 0.08 | 89.54 ± 0.06 | 166.53 ± 0.12 | -0.07 [-0.24, +0.10] | 77.0 | no |
| map-positional | 76.65 ± 0.07 | 89.30 ± 0.04 | 165.95 ± 0.05 | -0.65 [-0.97, -0.34] | 248.0 | no |
| mbr-g1-positional | 76.88 ± 0.06 | 89.55 ± 0.08 | 166.43 ± 0.14 | -0.18 [-0.40, +0.05] | 144.3 | no |
| map-string | 76.98 ± 0.06 | 89.45 ± 0.03 | 166.42 ± 0.09 | -0.18 [-0.34, -0.03] | 82.3 | no |
| mbr-g1-string | 77.05 ± 0.03 | 89.57 ± 0.04 | 166.62 ± 0.07 | +0.02 [-0.06, +0.10] | 24.3 | no |

The selection rule needs a gain above 0.18 (twice the seed std of the reference's EM + F1).

| Answer against the gold span | MAP, official EM 1 | MAP, official EM 0 | MBR, official EM 1 | MBR, official EM 0 |
|---|---|---|---|---|
| Identical span | 5124.0 | 0.0 | 5122.7 | 0.0 |
| Boundary variant | 759.3 | 1.3 | 765.7 | 1.3 |
| Elsewhere | 48.0 | 1755.3 | 48.0 | 1750.3 |
| Gold is not a word span | 1.7 | 18.3 | 1.7 | 18.3 |

### row0 (3 seeds)

| Setting | EM | F1 | EM + F1 | Against γ = 1 | Questions scored differently | Wins by the rule |
|---|---|---|---|---|---|---|
| map | 76.89 ± 0.12 | 89.62 ± 0.11 | 166.50 ± 0.21 | +0.18 [+0.02, +0.36] | 98.0 | no |
| mbr-g0 | 76.33 ± 0.11 | 89.70 ± 0.08 | 166.03 ± 0.19 | -0.29 [-0.57, -0.02] | 254.3 | no |
| mbr-g0.5 | 76.60 ± 0.05 | 89.67 ± 0.06 | 166.27 ± 0.10 | -0.05 [-0.16, +0.06] | 53.0 | no |
| mbr-g1 | 76.67 ± 0.08 | 89.65 ± 0.06 | 166.32 ± 0.13 | +0.00 [+0.00, +0.00] | 0.0 | no |
| mbr-g2 | 76.74 ± 0.10 | 89.64 ± 0.08 | 166.38 ± 0.16 | +0.05 [-0.04, +0.16] | 37.7 | no |
| map-t1 | 76.84 ± 0.17 | 89.59 ± 0.10 | 166.43 ± 0.26 | +0.11 [-0.09, +0.31] | 128.0 | no |
| mbr-g1-t1 | 76.74 ± 0.12 | 89.64 ± 0.07 | 166.39 ± 0.19 | +0.06 [-0.08, +0.21] | 66.0 | no |
| map-positional | 76.67 ± 0.22 | 89.47 ± 0.06 | 166.15 ± 0.27 | -0.17 [-0.46, +0.12] | 218.0 | no |
| mbr-g1-positional | 76.51 ± 0.15 | 89.58 ± 0.06 | 166.09 ± 0.21 | -0.23 [-0.43, -0.03] | 110.7 | no |
| map-string | 76.89 ± 0.09 | 89.63 ± 0.07 | 166.52 ± 0.14 | +0.19 [+0.02, +0.38] | 103.7 | no |
| mbr-g1-string | 76.77 ± 0.03 | 89.73 ± 0.04 | 166.49 ± 0.05 | +0.17 [+0.09, +0.26] | 25.0 | no |

The selection rule needs a gain above 0.26 (twice the seed std of the reference's EM + F1).

| Answer against the gold span | MAP, official EM 1 | MAP, official EM 0 | MBR, official EM 1 | MBR, official EM 0 |
|---|---|---|---|---|
| Identical span | 5313.3 | 0.0 | 5299.0 | 0.0 |
| Boundary variant | 566.7 | 1.3 | 565.7 | 1.3 |
| Elsewhere | 45.0 | 1761.7 | 43.7 | 1778.3 |
| Gold is not a word span | 1.3 | 18.7 | 1.3 | 18.7 |
