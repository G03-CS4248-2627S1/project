# Error analysis of the final systems (rows 1, 3 and 4)

Issue #117, plan §10. Validation (7,708 questions, one reference each), DeBERTa-v3-large, 3 seeds per system; row 4's seeds share reader seed 0. Generated with `sandbox/bruhmiuz/error_analysis.py`, for example:

```
python sandbox/bruhmiuz/error_analysis.py <out_dir> --rule combined_mbr --candidates 2026-10-23-row1-joint-s0-j998373 \
  2026-10-29-verifier-separate-s0-j1032972 2026-10-30-verifier-separate-s1-j1041306 2026-10-30-verifier-separate-s2-j1041319
```

The candidates aren't committed (workflow.md §2.2), so a run is analysed where they are. Shares are means over seeds; each seed's run is the first in its command.

**Correction.** Row 4 was regenerated after a decoding fix (`decode.answer_groups`, #115): on a verifier list, each answer had taken its group's earliest-starting variant, not its most probable one. The first version of finding 2 said the verifier raised the shifted share from 4.4% to 6.2%. That came from the bug, and the share actually falls. The same bug produced row 4's "the, extra on the left" count of 103 in the boundary-word table, now 9. The EM and F1 totals barely moved.

## Findings

1. **The ladder barely changes what goes wrong.** EM errors fall only from about 1,782 (row 0) to 1,760 (row 4, MBR), and boundary errors stay the large majority: 84% on row 0 and 82% on rows 1, 3 and 4. Too-short answers alone are over half.
2. **The verifier moves errors rather than removing them.** Against row 3 under MBR, row 4's too-short and no-overlap shares rise (52.3% to 55.6%, and 16.4% to 17.2%), and its too-long and shifted shares fall (25.9% to 23.4%, and 4.4% to 2.8%). This fits #116: it lengthens some answers and shortens others.
3. **Most of the remaining gap is reachable but not reached.** The scorer-correct answer is in the reader's top 10 on about 84% of errors, and second on 34–38% of them on rows 1 to 4. An oracle picking from the top 5 would score 93.7 EM, against row 4's 77.2, so better candidates aren't what's missing. About 16% of errors have no correct answer in the top 10.
4. **Long answers and open questions are where EM is lost.** On row 4, one-word answers score 88.8 EM and answers of ten or more words 31.5. "What year" scores 93.0, and "how" and "why" about 52–53. "What" questions are 57% of the errors, but they are also over half of the questions.
5. **Errors are shared across seeds.** On row 3, 1,460 questions are missed by all three seeds, about 82% of one seed's errors.
6. **Many no-overlap errors aren't model errors.** Of 25 sampled from 213, 7 were clear model mistakes, 8 had a wrong or poor gold answer, and 10 had two acceptable answers. The row 0 sample of 5 Oct gave a similar picture (about 7, 5 and 12). A single reference therefore makes this class look worse than it is. So 77 EM understates the reader, which fits the gap to dev's several references (AGENTS.md).
7. **Annotators' boundary conventions show up directly.** The most common missing words are prepositions and qualifiers ("in", "more than", "about", "around"). This fits #86's preposition count, which isn't committed yet (#120).

## Limits

- Validation has one reference per question, so these shares don't carry over to dev directly. Dev's predictions aren't read this way (spec rule 1).
- The hand-read verdicts are one reader's judgment, made by an AI assistant with the passages open, on 25 questions from one seed. A teammate should spot-check them before they go in the paper.
- An "oracle" means choosing whichever candidate the scorer accepts. No system can do that; it only bounds what a re-ranker could reach.
- Row 4's ranks are among the reader's candidates, so its rank 1 share (6.7%) is errors where the reader's top candidate was correct and the final answer wasn't. That share is 3.8% on row 3.

## Results

### Error classes, share of EM errors (%)

| System | EM | F1 | EM errors | Too short | Too long | Shifted | No overlap, F1 > 0 | No overlap, F1 = 0 | Gold not a word span |
|---|---|---|---|---|---|---|---|---|---|
| Row 0, MAP | 76.89 ± 0.12 | 89.62 ± 0.11 | 1782 | 54.9 | 26.1 | 3.1 | 4.3 | 10.5 | 1.0 |
| Row 1, MAP | 76.97 ± 0.07 | 89.46 ± 0.05 | 1775 | 53.2 | 25.0 | 4.3 | 4.6 | 12.0 | 1.0 |
| Row 3 (row 1, MBR) | 77.04 ± 0.03 | 89.57 ± 0.06 | 1770 | 52.3 | 25.9 | 4.4 | 4.7 | 11.7 | 1.0 |
| Row 4, MAP | 77.20 ± 0.07 | 89.37 ± 0.01 | 1758 | 56.2 | 22.6 | 2.8 | 4.8 | 12.6 | 1.1 |
| Row 4, MBR | 77.17 ± 0.04 | 89.46 ± 0.02 | 1760 | 55.6 | 23.4 | 2.8 | 4.9 | 12.3 | 1.1 |

### Where the scorer-correct answer sits among the reader's top 10 (share of EM errors, %), and the F1 lost

| System | Rank 1 | Rank 2 | Rank 3 | Rank 4 | Rank 5-10 | Not in top 10 | Oracle top 2 | Top 5 | Top 10 | F1 lost: boundary | No overlap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Row 0, MAP | 2.4 | 41.6 | 16.7 | 7.2 | 15.6 | 16.4 | 87.1 | 93.6 | 96.2 | 7.10 | 3.07 |
| Row 1, MAP | 3.3 | 38.2 | 16.7 | 7.9 | 18.1 | 15.8 | 86.5 | 93.7 | 96.4 | 6.92 | 3.41 |
| Row 3 (row 1, MBR) | 3.8 | 37.6 | 16.7 | 7.9 | 18.1 | 15.9 | 86.6 | 93.7 | 96.4 | 6.88 | 3.35 |
| Row 4, MAP | 6.6 | 34.7 | 15.7 | 8.5 | 18.6 | 15.9 | 86.6 | 93.7 | 96.4 | 6.86 | 3.54 |
| Row 4, MBR | 6.7 | 34.5 | 15.8 | 8.5 | 18.6 | 15.9 | 86.6 | 93.7 | 96.4 | 6.82 | 3.49 |

### Row 4, MBR, seed 0, by question type

| Type | Questions | EM | F1 | Share of EM errors |
|---|---|---|---|---|
| what | 4010 | 74.9 | 88.6 | 57.1% |
| who | 900 | 87.4 | 93.5 | 6.4% |
| when | 682 | 87.7 | 94.4 | 4.8% |
| which | 542 | 81.4 | 92.0 | 5.7% |
| how many | 407 | 74.9 | 89.0 | 5.8% |
| where | 331 | 71.3 | 87.4 | 5.4% |
| what year | 270 | 93.0 | 95.7 | 1.1% |
| how | 185 | 52.4 | 76.6 | 5.0% |
| how much | 114 | 64.9 | 82.9 | 2.3% |
| why | 81 | 53.1 | 82.5 | 2.2% |
| how long | 77 | 72.7 | 88.2 | 1.2% |
| other | 73 | 45.2 | 66.5 | 2.3% |
| name | 19 | 73.7 | 80.2 | 0.3% |
| how far | 10 | 30.0 | 71.5 | 0.4% |
| how old | 7 | 71.4 | 96.4 | 0.1% |

### Row 4, MBR, seed 0, by gold answer length (whitespace words)

| Words | Questions | EM | F1 |
|---|---|---|---|
| 1 | 2526 | 88.8 | 93.1 |
| 2 | 1871 | 81.8 | 92.0 |
| 3 | 1201 | 80.4 | 91.4 |
| 4-5 | 988 | 68.4 | 87.6 |
| 6-9 | 646 | 59.1 | 83.8 |
| 10+ | 476 | 31.5 | 66.9 |

### Row 4, MBR, seed 0, by the reader's top-1 probability

| Top-1 probability | Questions | EM |
|---|---|---|
| 0.0-0.3 | 837 | 40.9 |
| 0.3-0.5 | 1511 | 64.0 |
| 0.5-0.7 | 1673 | 75.6 |
| 0.7-0.9 | 2019 | 87.7 |
| 0.9-1.0 | 1668 | 96.2 |

### Most common boundary words (row 4, MBR, seed 0)

| Word or words | Side | Errors |
|---|---|---|
| the | missing on the left | 47 |
| in | missing on the left | 23 |
| . | missing on the right | 18 |
| in the | missing on the left | 13 |
| more than | missing on the left | 12 |
| about | missing on the left | 12 |
| around | missing on the left | 11 |
| over | extra on the left | 9 |
| the | extra on the left | 9 |
| . | extra on the right | 9 |
| " | missing on the right | 9 |
| century | missing on the right | 8 |

### The hand-read sample

25 random no-overlap errors with F1 = 0 (row 4, MBR, seed 0, of 213), each read with the whole passage open.

| Verdict | Question | Gold | Answer | Why |
|---|---|---|---|---|
| Model mistake | How many plants can only be found on Norfolk Island? | 51 | 174 | Took all native plants, not the endemic ones. |
| Model mistake | What did an established church, even at the state level, have? | opponents | state sanctioned religion | An awkward question, but the gold fits it and the answer doesn't. |
| Gold wrong or poor | What was railway area developed into | were developed following modernist ideas of the 1960s, in a brutalist style | shopping mall | The gold isn't what the area became; the mall is. |
| Both acceptable | The ownership of what islands are disputed with Japan? | the southernmost Kurils | Kuril Islands | Close: broader than the gold, and "Kuril" doesn't match "Kurils" as a token. |
| Both acceptable | What positive effects does a slower-evaporating filament enable? | more efficient light source | run hotter | The passage gives both effects in one sentence. |
| Gold wrong or poor | Israel's Supreme court has uses what internal method to review its own decisions? | a panel of three or more justices | further hearing | "Further hearing" is the method; the gold is who sits on the original panel. |
| Gold wrong or poor | What is the world in which the digimon live? | a manifestation of Earth's communication network | DigiWorld | The answer names the world; the gold describes it. |
| Model mistake | Where is the League of Nations located? | Switzerland | the Palace of Nations in Geneva | Took the United Nations' seat from the same sentence. |
| Both acceptable | How popular is the branch of Lutheranism? | third | approximately 80 million adherents | A rank and an adherent count both answer "how popular". |
| Gold wrong or poor | Which film did Madonna record the single Beautiful Stranger for? | Beautiful Stranger | Austin Powers: The Spy Who Shagged Me | The gold is the single; the answer is the film the question asks for. |
| Both acceptable | What happens to a digimon after it dissolves? | goes through its life cycle again | recomposes itself as a Digi-Egg | Consecutive steps in the same passage. |
| Gold wrong or poor | How many buses does the London public bus network operate? | more than 700 | 8,500 | The gold counts routes; the answer counts buses, as asked. |
| Gold wrong or poor | Whose ideas did Sextus build on? | Aetius | stoics | The passage has Sextus continuing the Stoics' idea; Aetius only summarizes it. |
| Both acceptable | Menander is one of the few remaining pieces of what time of liteary work? | New Comedy | Hellenistic period | "What time" is probably a typo for "type"; the answer fits the question as written. |
| Model mistake | To travel from Biscayne Bay to Miami, in what direction would one move? | west | east | Needs an inference from the bay's position, and went the wrong way. |
| Model mistake | What is the importance pertaining to the age of the hominid that was found in Eritrea? | it is the oldest skeletal find of its kind | provides a link between hominids and the earliest anatomically modern humans | Took the next clause, which isn't about age. |
| Both acceptable | What was the name of the album she released in 1983? | Madonna | self-titled | "Self-titled" gives the name indirectly. |
| Both acceptable | Stewart felt the court was not responsibly aided by the uncritical invocation of what? | metaphors | "wall of separation" metaphor | Same meaning; "metaphor" doesn't match "metaphors" as a token. |
| Gold wrong or poor | What is a popular tabloid magazine in Namibia? | Windhoek Observer | Informanté | The passage calls the answer the tabloid; the gold is a weekly. |
| Both acceptable | Repetition pitch is caused by what phenomenon? | the addition of a true repetition of the original sound | reflecting off a sound-reflecting surface | Both clauses describe the cause. |
| Gold wrong or poor | In what year did Mount Rainer receive a record amount of snowfall? | 1998-99 | 1971–72 | The gold's 1998-99 record is Mt. Baker's; the answer is Rainier's. |
| Both acceptable | What type of gas fills a tungsten filament bulb? | argon or nitrogen | inert | "An inert" gas is right in type, but the span is cut short. |
| Model mistake | What causes humans to underestimate time intervals? | depressants | Stimulants | Took the opposite: stimulants overestimate. |
| Both acceptable | Who did Windows 8 mock in their television campaign? | the iPad | Apple | The ad compares with the iPad and parodies Apple's. |
| Model mistake | What are pseudoproblems? | most metaphysical, ethical, aesthetic and other traditional philosophical problems | Any sentence that is not purely logical, or is unverifiable is devoid of meaning | Took the principle, not the problems it rules out. |
