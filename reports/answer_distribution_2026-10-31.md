# Full-span reader diagnostics: issues #95 and #96

Collected 31 October 2026. Numbers are mean ± sample standard deviation across three seeds, not confidence intervals or causal estimates.

## Execution and validation

- Analysis commit: `aaba686bc8ac7980a1052415928aaeb6063fdc13`, main base `fd98971`. No training, temperature fitting, final test scoring or #97 experiments.
- CPU setup/tests **1048451**: COMPLETED, 6m56s; **389 passed, 13 skipped**. No reference tests changed. Some optional integration tests lack data/tokenizer caches in this environment.
- GPU smoke **1048453**: COMPLETED, 1m37s. #95 **1048460**: COMPLETED, 1m56s. #96 **1048461**: COMPLETED, 11m15s. All exit 0; A100-PCIE-40GB GPUs.
- Nine summaries and nine per-question artifacts validated: counts, exact summary recomputation, config/calibration hashes, clean code and job provenance. Six validation runs each contain 7,708 questions; three NQ runs each contain 1,250, including 250 held out.
- New artifacts only; original metrics, predictions, calibration, training metadata and checkpoints preserved.
- Rebuild/audit: `python -m squadqa.report_answer_distribution`. Reads committed artifacts; no GPU/checkpoint/data download needed.

## #95: multiple annotated answers in Natural Questions

**1,247 of 1,250 questions have at least two disjoint annotated regions**; all 250 held-out questions are eligible. Three questions merge into one region and are excluded from primary statistics, but retained in the per-question file. All NQ questions are unseen by these SQuAD-trained readers.

Region probability sums groups whose content core lies wholly inside that merged annotation region. Crossing/partial/other spans enter the rest bucket. Collapse is the top region mass divided by total annotated-region mass, not its share of the whole distribution.

| Measurement | All eligible, T=1 | All eligible, calibrated | Held-out 250, calibrated |
|---|---:|---:|---:|
| Probability on all annotated regions (%) | 28.13 ± 2.13 | 23.77 ± 1.57 | 24.66 ± 1.73 |
| Top region conditional share (%) | 78.95 ± 0.78 | 74.19 ± 0.83 | 74.15 ± 1.08 |
| Top-region share under constructed q (%) | 36.98 ± 0.00 | 36.98 ± 0.00 | 37.11 ± 0.00 |
| Questions with conditional share >90% (%) | 39.94 ± 2.39 | 27.80 ± 1.44 | 28.27 ± 1.62 |
| Questions with conditional share >99% (%) | 13.55 ± 0.87 | 5.96 ± 0.53 | 8.00 ± 0.00 |
| Second/first region mass | 0.271 ± 0.011 | 0.327 ± 0.012 | 0.339 ± 0.013 |
| Second region has a group ranked top 5 (%) | 25.07 ± 3.74 | 24.73 ± 3.74 | 22.27 ± 3.21 |
| Second region has a group ranked top 10 (%) | 44.00 ± 4.19 | 44.05 ± 3.81 | 44.27 ± 3.78 |
| KL(q || p), nats | 3.899 ± 0.189 | 3.327 ± 0.131 | 3.338 ± 0.147 |
| Jensen–Shannon distance | 0.687 ± 0.009 | 0.693 ± 0.008 | 0.692 ± 0.009 |

**Reading:** after calibration, about 74% of annotated-region probability lies on one region, versus 37% under constructed q. However, only 24% of total probability lies inside any annotated region. The reader favors one annotated alternative while allocating substantial mass outside this containment definition; it does not collapse to one valid answer on every question.

Temperature scaling reduces conditional concentration and KL, but also reduces total region mass and slightly increases JS distance. Better KL alone is not evidence of better answer decisions. No #97 decoder-quality measurements were performed.

The second region is second by model mass, and its best group is ranked against all boundary groups. This is **not actual top-5/top-10 span-list recall**: boundary variants can consume separate verifier slots. q is constructed uniformly over distinct annotated groups, not measured human frequencies. NQ differs from SQuAD, many answers are list items, and partial spans/incomplete annotations affect the rest bucket. The result does not isolate a causal effect of CE or establish a mismatch with natural annotator probabilities.

Per-question results and run summaries also include train/held-out splits, first-annotated status, region lengths and gaps, and corresponding descriptive breakdowns. The calibrated top region is the first annotated region on 70.17 ± 1.89% of eligible questions; this association does not establish an annotation-order bias.

## #96: full validation distributions

| Measurement | Row 0, T=1 | Row 1, T=1 | Row 0, calibrated | Row 1, calibrated |
|---|---:|---:|---:|---:|
| Top span probability (%) | 73.17 ± 0.16 | 73.27 ± 0.81 | 67.23 ± 0.04 | 65.61 ± 0.80 |
| Top group probability (%) | 80.24 ± 0.06 | 83.73 ± 0.20 | 75.12 ± 0.12 | 77.56 ± 0.17 |
| Span entropy (nats) | 0.845 ± 0.006 | 0.819 ± 0.025 | 1.127 ± 0.006 | 1.208 ± 0.016 |
| Group entropy (nats) | 0.620 ± 0.001 | 0.499 ± 0.005 | 0.853 ± 0.007 | 0.818 ± 0.007 |
| Effective groups exp(H) | 2.97 ± 0.08 | 2.27 ± 0.01 | 4.19 ± 0.15 | 3.92 ± 0.14 |
| Groups for 50% mass | 1.27 ± 0.02 | 1.14 ± 0.00 | 1.44 ± 0.03 | 1.32 ± 0.03 |
| Groups for 90% mass | 3.16 ± 0.11 | 2.48 ± 0.11 | 4.65 ± 0.13 | 4.97 ± 0.36 |
| Groups for 99% mass | 9.63 ± 0.19 | 8.50 ± 0.48 | 19.90 ± 0.69 | 47.29 ± 5.60 |
| Top-group mass on best span (%) | 90.57 ± 0.14 | 86.68 ± 0.89 | 88.63 ± 0.10 | 83.52 ± 1.07 |
| Questions with best span <50% of top group (%) | 1.68 ± 0.15 | 4.92 ± 0.52 | 2.72 ± 0.03 | 7.62 ± 0.64 |
| Questions with >=2 greedy 5%-mass modes (%) | 4.37 ± 0.17 | 9.32 ± 0.10 | 5.26 ± 0.17 | 11.60 ± 0.14 |
| Second disjoint group mass (%) | 0.76 ± 0.03 | 1.81 ± 0.08 | 0.89 ± 0.03 | 2.21 ± 0.07 |
| Mass overlapping gold content positions (%) | 95.67 ± 0.03 | 94.41 ± 0.16 | 95.40 ± 0.03 | 92.89 ± 0.18 |

**Reading:** at T=1, row 1 has lower group entropy but spreads more mass across equivalent boundary variants. Calibration broadens both distributions. At calibrated T, row 1 needs about 47 groups for 99% mass versus 20 for row 0, despite similar overall group entropy. The top and the low-probability tail can change differently.

Row 1 also has more substantial disjoint alternatives: 11.6% of questions have at least two greedy >=5%-mass modes, versus 5.3% for row 0. This is not proof of better accuracy or lower cross-span error; E3 measures the latter separately.

**Comparison limit:** these are the historical systems specified by #96, not a controlled pair-term ablation. They differ in head, target unit, epochs and window layout. Do not attribute all differences to the bilinear interaction. Modes use a deterministic probability-ordered greedy selection of disjoint content cores, not a unique density-mode decomposition.

### Shape by MAP outcome (row 1, calibrated)

Outcome labels use the existing top-10 MAP decoder and official string EM/F1; shape uses the full distribution. “Disjoint” means official F1=0 here, not necessarily positional disjointness.

| Outcome | Mean questions | Top-group probability | Group entropy | Groups for 99% mass |
|---|---:|---:|---:|---:|
| correct | 5932.3 ± 4.0 | 0.839 ± 0.001 | 0.618 ± 0.007 | 31.76 ± 4.54 |
| partial | 1532.7 ± 5.0 | 0.571 ± 0.004 | 1.419 ± 0.010 | 76.83 ± 7.40 |
| disjoint | 243.0 ± 3.0 | 0.526 ± 0.008 | 1.907 ± 0.064 | 240.39 ± 27.92 |

Incorrect predictions have broader distributions on average. This motivates decoder analysis but does not measure whether MBR fixes them.

### MAP reproduction check

Recomputed scores differ slightly from original saved results: maximum absolute differences are 0.052 EM and 0.038 F1 points. Different inference batching/hardware/numerics can change near-ties. Originals are preserved; these diagnostics do not replace published accuracy results.

| Reader | EM difference (points) | F1 difference (points) |
|---|---:|---:|
| 2026-10-23-row1-joint-s0-j998373 | -0.0259 | -0.0190 |
| 2026-10-23-row1-joint-s1-j998384 | -0.0259 | -0.0064 |
| 2026-10-23-row1-joint-s2-j998397 | +0.0259 | +0.0162 |
| 2026-10-05-row0-baseline-s0-j910132 | -0.0519 | -0.0378 |
| 2026-10-05-row0-baseline-s1-j910133 | +0.0000 | +0.0090 |
| 2026-10-05-row0-baseline-s2-j910134 | -0.0130 | -0.0024 |

## Artifacts and review

Row-1 folders contain `multi_answer{,_questions}.json` and `distribution_validation{,_questions}.json`; row-0 folders contain distribution files. Numerical reports retain every seed/temperature. No candidate dumps or checkpoints are committed. PR #107 contains code and results for teammate review. Issues #95/#96 await review/merge; assistants do not merge. #97 remains untouched.
