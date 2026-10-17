# implementation_details.md — CS4248 Extractive QA (SQuAD v1.1)

Implementation spec for coding agents and team members.

- **What and why:** `Extractive_QA_Design_Plan.docx` (plan v1.8). **How:** this file. **When:** [timeline.md](timeline.md). **How we work:** [workflow.md](workflow.md).
- Week labels such as W2 or W5+ are plan weeks: W1–W7 are course weeks 7–13.
- If this file and the plan disagree, stop and ask a human.

---

## 1. Hard rules (MUST)

1. `dev-v1.1.json` is the **test set**. Never train on it, tune on it, or inspect its predictions except in final evaluation runs. All tuning uses the validation split (§3.1).
2. Never generate training data (questions, distractors) from `dev-v1.1.json` passages.
3. No closed-source LLM or API calls in training or inference code. (Use for data generation is pending a ruling; see plan §2.)
4. Locate gold answers with `answer_start`, never by string search. 11% of answers appear more than once in their paragraph. The same text at another position is **never** a correct answer, even though the official scorer would accept it. Only answers the dataset annotates, each at its own `answer_start`, are gold. Normalization changes only how a span at the annotated position is compared (§4.3–4.4). The one exception is the `string` decoding row of E1 (`mbr_utility: string`, §4.4), which groups equal texts at any position to measure what this rule costs; training targets never use it.
5. Every reported number comes from the official `evaluate-v2.0.py`. Predictions file: `{question_id: answer_text}` for **every** question, never an empty string (an empty string counts as "no answer").
6. Never edit a reference implementation or a test to make new code pass. If a test looks wrong, stop and report.
7. No silent fallbacks. If a fast path (FlexAttention, KV cache) is unavailable, raise an error or log it loudly. Never switch paths quietly, especially in benchmarks.
8. Every training job checkpoints and can resume. Default cluster GPU jobs are capped at 3 hours.
9. Do not start a hand-written Triton/CUDA kernel. Out of scope until after the deadline.

---

## 2. Config defaults

```yaml
model: microsoft/deberta-v3-large      # development: microsoft/deberta-v3-base
precision: bf16                        # bf16 autocast compute, fp32 master weights and optimizer state (§9); fp16 can be unstable with DeBERTa-v3
max_seq_len: 384
doc_stride: 128                        # window_layout even: the most passage tokens between consecutive window starts; left: the overlap (§3.3)
window_layout: even                    # even = the fewest full windows whose starts are at most doc_stride apart, spread evenly over the passage | left = runs before 17 Oct 2026: from the left, overlapping by doc_stride, the last window holding the rest (§3.3)
padding: dynamic                       # pad to longest in batch; batch by similar length
max_answer_words: 30                   # L; every dev question has a reference of ≤ 28 words
span_head: compound                    # Fajcik et al. (2021) objectives: compound (row 1+; I+J) = joint + auxiliary independent loss | joint (J; P2 ablation) | conditional (JC; 1b test) | standard (I; row 0)
aux_independent_weight: 1.0            # compound only (Fajcik et al., 2021)
head_dim: 256
W_init: zeros
bilinear_form: full                    # joint and compound pair term (§4.1): full = h_iᵀ W h_j (biaffine) | dot = (A h_i)ᵀ(B h_j) of rank bilinear_rank (Fajcik et al.'s dot product; 1b test)
bilinear_rank: 64                      # dot form only
length_bias: true                      # joint and compound: learned bonus b[j − i] per span length, zero at init (§4.1)
bilinear_pos: none                     # joint and compound, pair term only (§4.1; 1b tests): none | additive = sinusoidal encodings added to h_i, h_j | rotary = RoPE on the pair term's inputs (h_i, h_j for full; A h_i, B h_j for dot)
no_answer: false                       # true = a null option, the span (CLS, CLS), in every softmax; never predicted (rule 5); 1b tests set it (§4.1)
decode_topk: 10
decode_rules: [map, mbr]               # always report both
mbr_em_weight: 1.0                     # γ, the F1:EM decoding ratio: MBR utility F1 + γ·EM; 0 = expected F1
mbr_utility: normalized                # normalized = content-word F1/EM, boundary variants grouped | positional | string = the scorer's normalization at any position (both E1 rows; §4.4)
f1_target: normalized                  # normalized = content-word F1, boundary variants exact | positional (ablation; §4.3)
f1_target_weight: 0.0                  # α, the F1:EM loss ratio (§4.3); 0 = one-hot target (rows 0–1), tuned in E1
f1_target_temperature: 0.1             # τ: how evenly the F1 credit is spread
f1_target_floor: 0.5                   # φ: spans with F1 < φ get no F1 credit; 0 = every overlapping span
target_unit: span                      # span = targets per span (t_EM on the annotated span) | group = per group of boundary variants, so both parts count the gold answer once (§4.3; needs f1_target normalized)
risk_weight: 0.0                       # β: weight of the expected-F1 (risk) loss added to the target loss (§4.3); 0 = none
risk_em_weight: 0.0                    # γ_r: weight of the EM loss inside the risk term, ℒ_risk = Σ p · (ℓ_F1 + γ_r · ℓ_EM) (§4.3); 0 = F1 only. Separate from the decoding γ
calibration: temperature               # one T, fit on validation split
extra_train_data: []                   # row 5: [adversarialqa]; augmentation (P2) adds augmented
train:                                 # reader fine-tuning: the plan's sandbox recipe (plan §15.1); lr and epochs tuned in row 0 (plan §15.3)
  lr: 2.0e-5                           # write floats with a dot: YAML reads 2e-5 as a string
  epochs: 1
  warmup_frac: 0.1                     # linear schedule, warmup over this fraction of the steps
  weight_decay: 0.01                   # AdamW; none on biases and LayerNorm weights
  effective_batch: 32                  # questions per optimizer step (all their windows), via gradient accumulation
  grad_clip: 1.0
  head_dropout: 0.1
  checkpoint_every: 500                # optimizer steps
verifier:
  variant: shared_lora                 # shared_lora = frozen reader + LoRA | separate = ModernBERT-large (E2 compares both)
  separate_model: answerdotai/ModernBERT-large   # pinned in squadqa/download_model.py; base for development
  type: in_place                       # in_place | levitated | levitated_shared (both P3)
  shared_option: B                     # levitated_shared only (P3): A = fully shared multi-task, B = frozen reader + marker-only LoRA
  lora: {rank: 16, alpha: 32, targets: [query_proj, key_proj, value_proj]}   # shared_lora and option B; check module names in the installed HF version
  k_infer: 5
  k_sweep: [5, 10, 20, 30]             # levitated variants only (P3)
  train_candidates: sampled            # sampled = drawn from the final reader's own span distribution (plan §4.3) | crossfit (E2 ablation)
  crossfit_folds: 3                    # crossfit ablation only, on base
  sibling_negatives: false             # P2 ablation (§3.5)
  unanswerable_negatives: false        # row 5: SQuAD 2.0 unanswerable questions with a null candidate (§5.1)
seeds: [0, 1, 2]
```

---

## 3. Data

### 3.1 Splits
- **Validation:** hold out about 10% of *articles* from `train-v1.1.json` (44 articles, 7,708 questions). Split by article title, never by question: questions on the same article share paragraphs. Fixed seed; save the list to `data/splits/val_articles.txt`.
- **Cross-fitting folds (K = 3), E2 ablation only:** also by article, over the non-validation training articles.
- The final reader trains on all non-validation training articles.

### 3.2 Words (the unit of every span)
- A **word** is either a maximal run of characters that are neither whitespace nor punctuation, or a single punctuation character (Unicode category `P*`, plus ``$+<=>^`|~``). Same rule as BERT's BasicTokenizer.
- Measured on the official files: with this rule, answers start and end on word boundaries for **99.70% of train** and **99.59% of dev** (first reference). With whitespace-only words, only about 53% do. **Do not use whitespace-only words.**
- Training: skip the 265 questions whose answer is not word-aligned and the 78 whose answer is longer than 30 words (343 in total; the two sets do not overlap). Log the skipped IDs to `data/splits/skipped_train_ids.txt`.
- Dev: predict every question; skip nothing.

```python
import unicodedata

EXTRA_PUNCT = set("$+<=>^`|~")

def is_punct(ch: str) -> bool:
    return unicodedata.category(ch).startswith("P") or ch in EXTRA_PUNCT

def word_spans(text: str) -> list[tuple[int, int]]:
    """(start_char, end_char) for each word, end exclusive."""
    out, i, n = [], 0, len(text)
    while i < n:
        if text[i].isspace():
            i += 1
            continue
        if is_punct(text[i]):
            out.append((i, i + 1))
            i += 1
            continue
        j = i
        while j < n and not text[j].isspace() and not is_punct(text[j]):
            j += 1
        out.append((i, j))
        i = j
    return out
```

### 3.3 Words ↔ subword tokens
- Tokenize each passage once on its own with the fast tokenizer (`return_offsets_mapping=True`), and each question on its own. A window is `[CLS] question [SEP] passage tokens [SEP]`, at most 384 tokens, so it has room for B = 384 − 3 − (question tokens) passage tokens. A question's windows together cover the whole passage (`squadqa.data.make_windows`, `window_ranges`). `window_layout` sets how:
  - **`even`** (the default since 17 Oct 2026): a passage of n ≤ B tokens is one window. A longer one gets the fewest windows whose consecutive starts are at most `doc_stride` (128) tokens apart, k = 1 + ⌈(n − B) / 128⌉. Each holds B passage tokens, and their starts are spread evenly from 0 to n − B: the last window ends at the passage's end, and consecutive windows overlap by at least B − 128 tokens (within 1 token of each other).
  - **`left`** (every run before; `predict.ADDED_KEYS` fills it in for them): windows from the left, consecutive ones overlapping by exactly `doc_stride` tokens. The last window holds the rest of the passage, so it can be as short as 129 tokens (median 168 in train).
- **Do not use `return_overflowing_tokens`.** tokenizers 0.23 (every release through 0.23.2; 0.22.2 is correct) returns at most two windows per (question, passage) pair and sometimes cuts the second short, so the end of the passage is lost for 528 train, 57 validation and 146 dev questions (checked 2026-10-04). With `window_layout: left`, wherever the library covers the whole passage, our windows are identical to its windows (its `stride` is the overlap; `tests/test_data.py`).
- For each passage word in a window:
  - `first_tok[w]` = first subword token whose character span lies inside the word;
  - `last_tok[w]` = last such token.
- Use only words entirely inside the window.
- **DeBERTa-v3 offsets include the leading space of a word** (checked 2026-09-24: in "Who won", the token " won" has span (3, 7), not (4, 7)). Strip leading whitespace from each token's span before comparing it with word spans. Otherwise no token "lies inside" a word that follows a space, and `first_tok` silently finds nothing.

### 3.4 Windows
- About 5% of paragraphs exceed 250 words, so most questions produce one window: 689 of the 87,256 kept training questions have more than one under either layout, and none has more than 5 under `even` (4 under `left`). `even`'s overlaps are larger, so more gold answers lie wholly inside two or more windows: 470 of these questions (134 under `left`); 50 of the 7,708 validation questions (15).
- Several windows: put the valid span scores of all windows of a question into **one** softmax. A span present in several windows uses the score from the window where it is most central (largest `min(tokens to its left, tokens to its right)` within the window).
- Training with several windows: start and end (and later span) scores of all the question's windows share one softmax too. The gold answer's copies in every window that wholly holds it all count as correct: the loss is minus the log of their summed probability (`squadqa.heads.question_nll`).

### 3.5 Sibling answers (verifier negatives)
- For question q: the gold answers of the other questions on the same paragraph, as word spans.
- Drop any sibling whose word-overlap F1 with q's gold span is above 0.
- Used only with `sibling_negatives: true` (P2 ablation).

### 3.6 Additional data (ladder row 5, plan §4.5)
- All files come from `python -m squadqa.download_external`, which pins them and removes duplicate questions and answers. Its docstring says what each file may be used for.
- It does **not** drop validation articles. Every training loader must:
  - drop each question whose article title is in the validation split (`squadqa.splits.load_val_titles()`);
  - raise an error if any passage matches a dev-v1.1 passage (whitespace-normalized).
  - Measured 2026-10-03 (plan §15.2): 4,899 SQuAD 2.0 and 2,596 AdversarialQA training questions sit on validation articles; none sit on dev passages.
- **AdversarialQA** (`data/external/adversarialqa/train.json`, the combined split): extra reader training data when `extra_train_data` contains `adversarialqa`. Same format as SQuAD, one answer per question. `adversarialqa/dev.json` is built on dev-v1.1 passages, so never train or tune on it (rule 1).
- **SQuAD 2.0 unanswerable questions** (`data/external/squad2/train-v2.0.json`): after duplicate removal this file holds only the unanswerable questions, all on train-v1.1 passages. They are verifier negatives when `unanswerable_negatives: true` (§5.1). Their candidates are the reader's top-k plus each `plausible_answers` span, located with its `answer_start` (rule 4) and converted to word spans (§3.2). Drop plausible answers that are not word-aligned.
- **Augmentation (P2):** each rule (letter case, date format, name substitution) edits the passage, question and answer together, and carries `answer_start` through the edit by offset arithmetic, never by search (rule 4). The offset round-trip test (§9) must pass on the augmented set. Never augment validation articles or anything from dev-v1.1 (rules 1–2).

### 3.7 Multi-answer data for E6 (plan §7.2, §8)
- **Purpose:** check whether cross-entropy makes p match a known answer distribution q. E6 only: never ladder training or tuning data.
- **Source:** Natural Questions dev (Kwiatkowski et al., 2019) in the MRQA 2019 format (Fisch et al., 2019), pinned by URL and SHA-256 in `squadqa/multi_answer.py`; CC BY-SA 3.0. `python -m squadqa.multi_answer` builds `data/external/multi_answer/` (gitignored). SQuAD train has one answer per question (66 questions are asked twice about one passage with different answers), and the MRQA NewsQA dev file has one answer per question too (checked 2026-10-16).
- **Build** (counts in `summary.json`; on 2026-10-16: 12,836 questions → 1,000 train and 250 held out, 3.4 answers each):
  - Strip the HTML markup MRQA keeps in NQ contexts and remap answer offsets; drop answers that start or end in markup. Drop answers that are not word-aligned or exceed `max_answer_words`.
  - Merge boundary variants with `decode.group_keys` and keep questions with 2+ distinct answers. MRQA keeps no annotator counts, so q is equal over a question's answers; training draws its answers from q, so q is the true distribution by construction.
  - Drop passages that share a word 10-gram with a dev-v1.1 passage (9 on 2026-10-16). Split by passage, 20% held out, seed 0.
- **Files:** `train.json` holds each question once per answer as a one-answer question (id `<qid>_a<k>`) with `"weight": q(answer)`; `heldout.json` holds each held-out question with all its answers; `q.json` holds q for every question.
- **Training weights each copy's loss by its `weight`,** so every question totals 1 and the objective is soft-label cross-entropy against q. Without it a question with k answers counts k times, and a SQuAD + E6 mix over-weights E6 about 3.4×. Sampling one answer per question each epoch optimizes the same expected objective (Gheibi & Ghazizadeh, 2026) but needs several epochs; the frozen recipe has one.
- **E6 scoring** (not built yet; conventions follow ChaosNLI's `evaluation/tools.py`, Nie et al., 2020, read 2026-10-16):
  - Sum p over each group of boundary variants first, because q is defined over groups.
  - KL(q ‖ p) with the natural log, the direction ChaosNLI uses (human ‖ model). It needs p only on q's answers: spans with q = 0 add nothing, so it is the full-space value.
  - Jensen-Shannon distance (scipy's `jensenshannon`, natural log; the square root of the divergence), over q's answers plus one "rest" bucket holding p's mass elsewhere. The bucket gives exactly the full-space value, because each span outside q's support adds ½ · p(y) · log 2 either way.
  - Floor p at 1e-15 and renormalize, as ChaosNLI does. Report at T = 1 (ChaosNLI's plain softmax) and at the temperature fitted on SQuAD validation, and average over held-out questions without weights.
  - Post-hoc smoothing alone can halve KL (Zhang, Gong & Choi, 2021), so the decision-level test is expected EM and F1 under q of the MAP and MBR answers.
- **Caveat:** many of NQ's multiple answers are items of a list answer ("who are the co-hosts"), each correct on its own, rather than competing phrasings. q spreads equally over them.

---

## 4. Joint span head

### 4.1 Definition
```
h_i = MLP_start(x[first_tok[i]])          MLP = Linear(d_model→256), GELU, Dropout(0.1)
h_j = MLP_end(x[last_tok[j]])
s(i, j) = h_iᵀ W h_j + uᵀh_i + vᵀh_j + b[j − i]
valid(i, j) ⇔ both are passage words, i ≤ j, j − i + 1 ≤ 30
p(i, j) = softmax of s over all valid spans (flattened)
```
- `W` and `b` start at zero, so the head starts as the standard start + end model.
- Compute the full n × n word-level matrix and mask it. No band tricks are needed at these sizes.
- **Head options** (joint and compound; milestone 1b tests them; the survey behind them is in `literature_review.md` §8):
  - **Pair term** (`bilinear_form`): `full` is `h_iᵀ W h_j`, the biaffine scorer of Yu et al. (2020). `dot` is `(A h_i)ᵀ(B h_j)` with A and B of rank `bilinear_rank` (64): the dot product of projections that Fajcik et al. found best for BERT-type models, as a low-rank W. B starts at zero and A at random, so the pair term starts at zero and both factors get gradients; two zero factors would never move.
  - `length_bias: false` drops `b`.
  - **Positions in the pair term** (`bilinear_pos`), never in `u` and `v`:
    - `additive`: sinusoidal encodings (Vaswani et al., 2017) of each word's index in the passage, added to `h_i` and `h_j` right before the pair term: `(h_i + PE(i))ᵀ W (h_j + PE(j))`. The product has terms in both the distance and the absolute positions.
    - `rotary`: RoPE (Su et al., RoFormer) on the pair term's inputs, rotating each by its word's position.
      - Dot form, as in GlobalPointer (Su et al., 2022): `(R_i A h_i)ᵀ(R_j B h_j) = (A h_i)ᵀ R_{j−i} (B h_j)` depends only on the distance, in a way that depends on content.
      - Full form (the 1b run `compound-rotary`): `(R_i h_i)ᵀ W (R_j h_j)`. A general W doesn't commute with the rotations, so the score depends on both positions, not only on their distance.
    - Without them the pair term sees position only as far as the encoder states carry it implicitly. The passage index, not the window's, gives a span the same encodings in every window. The pair term starts at zero, so they change nothing at init and the head still starts as the standard model.
- **No-answer option** (`no_answer: true`; every 1b test sets it): the span (CLS, CLS) joins every softmax as a null, as in Fajcik et al.'s and BERT's SQuAD 2.0 models. Joint score: the pair term of `(h_c, h'_c)` plus `uᵀh_c + vᵀh'_c`, with `h_c = MLP_start(x[CLS])`, `h'_c = MLP_end(x[CLS])` and no length or position terms.
  - Each window has its own null. A question's no-answer probability is the sum over its windows, which treat them like copies of the gold (§3.4).
  - On SQuAD v1.1 no question targets it, so training only pushes it down. Decoding never returns it (rule 5, §4.4).
  - It gets a positive target only if a reader trains on unanswerable questions. That is not built: row 5 gives them to the verifier only (§5.1).

### 4.2 Reference code
```python
import torch
import torch.nn as nn

class JointSpanHead(nn.Module):
    def __init__(self, d_model=1024, d_h=256, max_len=30, dropout=0.1):
        super().__init__()
        self.start = nn.Sequential(nn.Linear(d_model, d_h), nn.GELU(), nn.Dropout(dropout))
        self.end = nn.Sequential(nn.Linear(d_model, d_h), nn.GELU(), nn.Dropout(dropout))
        self.W = nn.Parameter(torch.zeros(d_h, d_h))           # zero init
        self.u = nn.Linear(d_h, 1)
        self.v = nn.Linear(d_h, 1)
        self.len_bias = nn.Parameter(torch.zeros(max_len))      # b[length - 1], zero init
        self.max_len = max_len

    def forward(self, x_first, x_last, word_mask):
        """x_first, x_last: (B, n, d_model) encoder states at each word's first / last subword.
        word_mask: (B, n) bool, True for passage words. Returns (B, n, n) scores, -inf if invalid."""
        hs, he = self.start(x_first), self.end(x_last)
        s = hs @ self.W @ he.transpose(1, 2)
        s = s + self.u(hs) + self.v(he).transpose(1, 2)
        n = s.size(1)
        idx = torch.arange(n, device=s.device)
        length = idx[None, :] - idx[:, None] + 1                  # j - i + 1
        valid = (length >= 1) & (length <= self.max_len)
        s = s + self.len_bias[(length - 1).clamp(0, self.max_len - 1)]
        mask = valid[None] & word_mask[:, :, None] & word_mask[:, None, :]
        return s.masked_fill(~mask, float("-inf"))
```

### 4.3 Loss
Word-level positional F1 between a span y = (i, j) and the gold span y* = (i*, j*) (notation as in plan §6):
```
overlap = max(0, min(j, j*) − max(i, i*) + 1)
P = overlap / (j − i + 1);  R = overlap / (j* − i* + 1);  F1 = 2PR / (P + R)   (0 if overlap = 0)
t(y) = (1 − α) · t_EM(y) + α · t_F1(y)              # α, τ, φ tuned on validation
t_EM: all mass on the annotated span y*, as in the one-hot loss of rows 0–1
t_F1(y) ∝ exp(−ℓ_F1(y, y*) / τ) where F1 > 0 and F1 ≥ φ, else 0     # ℓ_F1 = 1 − F1
ℒ_soft = − Σ_y t(y) · log p(y)                    # α = 0: ℒ_CE = − log p(y*); target_unit span (the default)
target_unit group:  ℒ_soft = (1 − α) · (− log p(G*)) + α · (− Σ_G t_F1(G) · log p(G))
  G = a group of boundary variants, G* the gold's; p(G) = Σ_{y ∈ G} p(y); t_F1 as above, one weight per group
ℒ_risk = Σ_y p(y) · [ℓ_F1(y, y*) + γ_r · ℓ_EM(y, y*)]   # every valid span; no floor; γ_r = 0: F1 only
ℒ = ℒ_soft + β · ℒ_risk                            # β = risk_weight, γ_r = risk_em_weight (plan §6.2)
```
- **One objective for every training variant** (plan §6.2):

  | | β = 0 | β > 0 |
  |---|---|---|
  | α = 0 | one-hot (rows 0–1) | one-hot + risk |
  | α > 0 | F1-weighted targets | F1-weighted targets + risk |

- **The risk term's EM weight γ_r** (`risk_em_weight`, default 0 = F1 only; decided 16 Oct 2026, #49). On questions whose answer is uncertain, the risk loss alone puts all mass on the answer MBR would choose under the same utility, so a reader trained with it and decoded with MAP aims at MBR with utility F1 + γ_r · EM. In E1, set γ_r to the γ (`mbr_em_weight`) of the MBR cell it is compared with, so both routes target the same answer; the two keys stay separate.
  - It does not count EM twice. The (1 − α) part of ℒ_soft, −log p(gold), is the log form of the expected EM loss 1 − p(gold); it makes p estimate the posterior and leaves the decision alone. The linear EM term inside ℒ_risk moves probability toward exact matches, as the risk loss does for F1.
  - The linear term's gradient is weak while p(gold) is small, but ℒ_soft carries training then.
  - ℓ_EM is the EM of `f1_target`: under `normalized` a boundary variant has ℓ_EM = 0. Per question, the optimum of ℒ becomes p(y) = t(y) / (c + β · (ℓ_F1(y, y*) + γ_r · ℓ_EM(y, y*))), so γ_r adds the same penalty to every span that is not an exact match.
- **The risk term needs no floor.** It weights spans by p, not by how many overlap the gold; spans with F1 = 0 count as full loss, which is right there.
- `α` is the proposal's F1:EM loss ratio: the exact-match (one-hot) and F1-weighted parts are mixed `(1 − α) : α` (`f1_target_weight`).
- `τ` (`f1_target_temperature`) sets how evenly the F1 credit is spread; `φ` (`f1_target_floor`) sets which spans get any. φ = 0 credits every overlapping span.
- **Why the floor:** the number of overlapping spans grows with answer and span length, and the many barely-overlapping ones outweigh the few good ones. Example: a one-word answer has 465 overlapping spans of at most 30 words. At τ = 0.2 and φ = 0, 78% of t_F1 goes to spans with F1 < 0.5, and a span drawn from t_F1 has expected F1 0.29. At τ = 0.1 it is 7% and 0.91.
  - The floor is the RAML target with the reward cut off below φ, so the forward/reverse KL relation of plan §6.2 still holds, with ℓ_F1 replaced by the truncated loss (∞ below φ).
  - Boundary smoothing (Zhu & Li, 2022) avoids the same problem by crediting only spans near the gold boundaries.
  - Log the expected F1 of t_F1 during training as a check.
**Note: this F1 differs from the official scorer's.** The scorer normalizes both strings first: lowercase, delete ASCII punctuation, remove *a*/*an*/*the*, collapse whitespace (`normalize_answer` in `qa/evaluate-v2.0.py`). Here, punctuation words and articles count, so a leading "the" or a trailing "." costs about 0.2 F1 (gold "Eiffel Tower" vs. "the Eiffel Tower": 0.8 here, 1.0 official). This is part of the F1-weighted-target work (timeline 2.2), selected by `f1_target`, with an ablation of `positional` vs. `normalized` on base:
- **`normalized`:** give each word a 0/1 flag (0 for punctuation words and articles; these are *content words* when 1) and count only content words in the overlap and lengths, with prefix sums, so it stays vectorized on the GPU. A span with no content words ("the", ".") matches only the identical span: the scorer would call any two such spans equal, but they sit at different positions (rule 4), and the scorer gives such a prediction 0 against any gold answer with content words (2 of 87,334 aligned train answers have none). It is still positional: only words at the same positions overlap.
- **Boundary variants are exact matches:** a span that covers exactly the annotated answer's content words, differing only by leading or trailing articles or punctuation ("the Eiffel Tower" vs. "Eiffel Tower"), gets the same F1 credit as the annotated span (F1 = 1, so the full weight in t_F1) and ℓ_EM = 0. They are found from the annotated span, so no string search is involved.
  - **Equal t_EM mass on the variants is not used** (decided 16 Oct 2026, #49). 74% of aligned training answers have a boundary variant (2.5 spans on average, the gold included, up to 8), so it would move most of the EM part off the annotated span.
  - **Per span or per group** (`target_unit`, 16 Oct 2026, #49):
    - `span` (the default): t_EM puts all its mass on the annotated span, so α = 0 is the one-hot loss that rows 0–1 were trained with. But each boundary variant gets its own t_F1 weight, so the gold's group gets one unit per variant: extra EM weight inside the F1 part that varies by question. The target's EM:F1 ratio is exactly (1 − α) : α only per group.
    - `group`: both parts work on groups of boundary variants, grouped as in decoding (`decode.group_keys`: the same content words at the same position; a span without content words is its own group). The EM part is −log p(G*), t_F1 gives each group one weight, and p(G) sums the probabilities of the group's spans. Each part then counts the gold answer once, as normalized EM does, and the model splits a group's mass among its variants as it likes. α = 0 is then −log p(G*), not the rows 0–1 loss. Built for the EM part (`heads.group_nll`, for α = 0); the F1 part comes with the F1-weighted target (#49). Group membership is worked out per window from the gold's content core (`core_start`, `core_end` in the batch), so a trimmed variant also counts in a window that cuts off the gold's leading article. Compound's auxiliary start and end term stays one-hot.
    - `group` needs `f1_target: normalized` (under `positional` every span is its own group) and a span softmax (span_head joint or compound). The risk loss is the same under both, since every member of a group has the same loss.
- **The same text elsewhere in the passage gets no credit** (rule 4), whatever the scorer does. This is where `normalized` deliberately differs from the scorer, besides punctuation inside a word ("U.S." → "us", "well-known" → "wellknown") and the scorer's bag-of-tokens comparison.
- The size of the gap is measured in week 8 (timeline 1.6); it bounds what `normalized` can gain.

Cross-entropy, NaN-safe (masked entries have log p = −inf, and 0 · −inf = NaN):
```python
logp = scores.flatten(1).log_softmax(-1)
loss = -torch.where(target > 0, target * logp, torch.zeros_like(logp)).sum(-1).mean()
# risk term: masked spans have p = 0, but give them a finite loss so 0 · ℓ stays 0
span_loss = f1_loss + risk_em_weight * em_loss          # ℓ_F1 + γ_r · ℓ_EM
risk = (logp.exp() * torch.where(valid, span_loss, torch.zeros_like(span_loss))).sum(-1).mean()
```

**Compound objective** (`span_head: compound`, ladder row 1; Fajcik et al. 2021): add the standard independent loss on the head's start and end scores.
- Start logits are `u(h_s)` over the passage words of all of a question's windows (one softmax, as in §3.4); end logits are `v(h_e)`.
- Cross-entropy against i* and j*, weighted by `aux_independent_weight` (ω in the plan): ℒ = ℒ_joint + ω · ℒ_ind.
- This term always uses one-hot targets. The F1-weighted target (row 2) and the risk term apply to the joint term only.
- Decoding uses the joint scores s(i, j).

**The four objectives of Fajcik et al. (2021)**, chosen by `span_head` (their names in brackets; 1b compares them on base):

| `span_head` | Loss | Decoding score s(i, j) |
|---|---|---|
| `standard` (I) | −log p_start(i*) − log p_end(j*) | uᵀh_i + vᵀh_j |
| `conditional` (JC) | −log p_start(i*) − log p(j* \| i*) | uᵀh_i + log p(j \| i) |
| `joint` (J) | −log p(y*) | the joint score (§4.1) |
| `compound` (I+J) | −log p(y*) + ω · (−log p_start(i*) − log p_end(j*)) | the joint score (§4.1) |

- **Conditional end:** `p(j | i)` is the softmax over the valid ends of start i of `wᵀ tanh(A x[last_tok[j]] + B x[first_tok[i]] + c)`. This is Fajcik et al.'s form (checked 2026-10-15 against arXiv 2008.12804v4), with `head_dim` hidden units instead of d_model.
  - Training conditions on the gold start (teacher forcing). With several windows, each window that holds the answer conditions on its own copy of i*, and their end logits share one softmax (§3.4).
  - Decoding scores every valid span exactly, so the question's softmax is p(i) · p(j | i). Fajcik et al. used a top-10 × top-10 beam instead.
- **With the no-answer option:** the start and end softmaxes of `standard`, and of compound's auxiliary term, get the null's start and end logits `uᵀh_c` and `vᵀh'_c`. The conditional start softmax gets `uᵀh_c`, and its only end is CLS. The joint softmax gets `s_null`.

### 4.4 Decoding
- `topk` over flattened scores (k = 10) → (i, j) → characters `(word_start[i], word_end[j])` → answer = `context[a:b]`.
- **Group boundary variants** (`mbr_utility: normalized`, the default): spans that cover the same content words at the same positions (§4.3; "the Eiffel Tower" and "Eiffel Tower") are one answer, so add up their probabilities. The same text elsewhere in the passage stays a separate answer (rule 4).
- **String grouping** (`mbr_utility: string`, an E1 row; decoding only): candidates whose texts are equal after the official scorer's `normalize_answer` are one answer, wherever they are in the passage. This is the grouping the scorer rewards, so it measures what rule 4 costs at decoding time; training targets never use it. A candidate whose text normalizes to nothing ("the", ".") is its own group and matches only itself, as under `normalized`.
- **MAP** = the most probable group, returning its most probable span (top-1 span when `mbr_utility: positional`). Groups and their probabilities are formed among the top-k candidates, under the full softmax.
- **MBR:** after calibration, among the top-k pick `argmax_y Σ_y' p(y') · [F1(y, y') + γ · EM(y, y')]`.
  - With `normalized`, y ranges over groups, and F1 and EM are the content-word versions from §4.3 (EM = 1 when two spans cover the same content words). Return the group's most probable span.
  - With `positional` (an E1 row), F1 is positional word F1 and EM(y, y') = 1 if the spans are identical.
  - With `string`, y ranges over string groups, and F1 and EM are the scorer's own `compute_f1` and `compute_exact` (bag of tokens after normalization). Return the group's most probable span.
  - γ = `mbr_em_weight` (the proposal's F1:EM ratio), tuned on the validation split. γ = 0 is expected F1; a large γ approaches MAP.
- **Calibration:** fit one temperature T on the validation split (minimize the NLL of the gold span; under `target_unit: group`, of the gold's group, the summed probability of its boundary variants, the unit the reader was trained on, so T doesn't punish how it splits a group among variants; `calibration.json` records the unit). Questions whose answer is not a valid span (not word-aligned, or over 30 words) are left out of the fit and counted in `calibration.json`.
- **No-answer option:** the null stays in the softmax, for calibration and for every span's probability (`squadqa.decode.span_logp`). MAP and MBR still choose among spans only, because the predictions file needs a non-empty answer for every question (rule 5).
  - MBR's choice doesn't change: every answer scores 0 against "no answer", so the null adds the same 0 to each candidate's expected utility.
  - `metrics.json` reports how much mass the null takes (`no_answer_<split>`).

### 4.5 Diagnostics
- **Cross-span mass** (plan E3; also the W2 early check on the joint head; `squadqa/diagnostics.py`):
  1. Take the most probable span and the most probable span that does not overlap it and has a content word (§4.3). By position they are a = (i1, j1) and b = (i2, j2), with i1 < i2. A span of only punctuation or articles is no answer, and joining one to a would give a boundary variant of a, not a cross span.
  2. The cross span is c = (i1, j2), if valid; without one the ratio below is 0.
  3. Report p(c) / (p(a) + p(b) + p(c)), with p calibrated (the run's temperature), bucketed by top-1 probability (< 0.5, 0.5–0.9, > 0.9), for the standard and joint heads.
  4. Also count how often each decision rule answers with the cross span (under `mbr_utility: normalized` or `string`, a boundary variant of it counts), with the official scores of those answers. This shows whether cross spans cause errors (plan §5: expected-F1 decoding favours them).
- **Equivalence test:** with W = 0 and b = 0, the joint head's argmax must equal the argmax over valid (i, j) of `u·h_i + v·h_j`, for every `bilinear_form` and `bilinear_pos` (`tests/test_heads.py`).

---

## 5. Verifier

### 5.1 Candidates and targets
- **Training:** spans sampled from the final reader's span distribution on its own training questions, at a temperature above 1 tuned on the validation split (plan §4.3). For the E2 ablation, use the cross-fitted readers' top-k on the held-out fold (§5.2) instead. In both cases add the gold span if missing, plus boundary variants (±1 word at either end, valid only), plus siblings (§3.5) if `sibling_negatives: true`. Deduplicate.
- **Inference:** the reader's top-k.
- **Loss:** listwise. Softmax over a question's candidate scores; F1-soft target over the candidates (same α, τ, φ form and the same `f1_target` as §4.3); cross-entropy.
- **Unanswerable negatives** (`unanswerable_negatives: true`, row 5): append a null candidate with a fixed score of 0 to every training question's list. Answerable questions give it target 0; the unanswerable questions of §3.6 put all target mass on it. Inference never adds the null candidate.
- Track **top-k recall** (fraction of questions whose gold span is among the candidates). It caps verifier accuracy.

### 5.2 Cross-fitting (K = 3, E2 ablation only)
Not the default: the default candidates are sampled from the final reader (§5.1). Run on base.

1. Split non-validation training articles into 3 folds.
2. For each fold: train a reader on the other 2 folds; generate top-k candidates on the held-out fold.
3. Train the verifier on these candidates.
4. At test time use the reader trained on all non-validation training articles.

### 5.3 In-place markers (both variants)
- Insert the marker strings `<a>` and `</a>` into the passage text around the candidate's characters, with a space on each side, then tokenize. Register them as special tokens in each variant's tokenizer and resize its embeddings. Inserting text before tokenizing works with both tokenizers.
- Input: the question and the passage. Most passages fit one window. Otherwise keep the window (§3.4, reader tokenizer) in which the candidate is most central, as a character range, and give both variants that same range, so E2 compares backbones on identical text.
- Head: concatenate hidden states at `<a>`, `</a>` and the first token (`[CLS]`) → MLP → one score.
- One forward pass per candidate.

### 5.4 Variants (E2)
- **`shared_lora`:** load the frozen final reader and add PEFT LoRA (`verifier.lora`) to its encoder.
  - Reader passes run with the adapter disabled (`with model.disable_adapter():`); verifier passes run with it enabled.
  - Train only the LoRA weights, the two marker embeddings and the verifier head. Keep the existing embedding rows frozen, so the reader is unchanged (test V4).
- **`separate`:** `verifier.separate_model` (ModernBERT-large), fully fine-tuned with its own tokenizer.
  - Pinned in `MODELS` in `squadqa/download_model.py` (base and large); load it with an explicit dtype (§9).
  - Smoke-test it like any new training script (§9).

### 5.5 Levitated markers (P3)
- **Layout:** `[text tokens][S_1 E_1 … S_k E_k][padding]`. Pad questions with fewer candidates using dummy marker pairs, excluded from attention and loss.
- **Position IDs:** text 0 … n−1; `S_c` = position of `first_tok` of the candidate's first word; `E_c` = position of `last_tok` of its last word.
- **Mask:** text queries see text keys only. Marker queries see text keys plus their own partner marker.
- **DeBERTa relative positions:** build `relative_pos` from the position IDs using Hugging Face's own `build_relative_position` / `make_log_bucket_position` (do not reimplement the bucketing). Pass it to the encoder. First verify that the installed Hugging Face version accepts `relative_pos` and a 3D attention mask; if not, wrap `DebertaV2Encoder`.
- **Head:** concatenate `[h(S_c); h(E_c); h_text(first_tok of start word); h_text(last_tok of end word); h(CLS)]` → MLP → one score.

### 5.6 Shared levitated variant (P3)
- **Option A (fully shared):** one pass; loss = reader loss + β × verifier loss; all weights train.
- **Option B (default):** train the reader, then freeze it. Marker tokens use LoRA-adapted weights (`W + ΔW`, applied only where `is_marker` is true); text tokens use frozen base weights. Compute text keys and values under `torch.no_grad()`. Train only the LoRA weights, marker embeddings, and marker head.
- **Inference (default, simple):** reader pass → top-k → one full levitated pass (≈1.05× a reader pass). The reader's span scores are unchanged by the markers.
- **KV cache (optional):** keep each layer's text keys and values from the reader pass and run only the marker tokens against them.

### 5.7 Acceptance tests (those for the variant in use must pass before any verifier training run)
| ID | Test | Pass condition |
|---|---|---|
| V1 | Text unchanged (levitated, P3) | Text-token outputs with markers appended equal a plain pass without markers (fp32: `atol=1e-5`) |
| V2 | Order independence (levitated, P3) | Shuffling marker order changes no candidate score |
| V3 | Cache equality (levitated, P3) | KV-cache scores equal single-pass scores |
| V4 | Reader frozen (`shared_lora`; levitated option B) | Reader predictions identical before and after verifier training |
| V5 | Marker round trip (both variants) | For each tokenizer: removing the markers from the marked text restores the passage, and the text between them equals the candidate |

---

## 6. Analysis notes

- **Fix-type breakdown (plan E2):** compare strings after the official `normalize_answer` (it drops articles and punctuation).
  - *No change:* reader and verifier answers are equal after normalization.
  - *Boundary fix:* the answers overlap but differ after normalization.
  - *Answer switch:* the answers do not overlap.
- Report per-seed results, mean ± std over 3 seeds, and paired bootstrap 95% confidence intervals for headline differences.
- Score every frozen system on dev, Adversarial SQuAD (AddSent, AddOneSent) and the four SQuAD-Shifts sets, and report each separately.
- **Adversarial SQuAD** (`data/external/adversarial_squad/`): each file holds 1,000 dev-v1.1 questions unchanged plus adversarial versions whose ids add a suffix (`-high-conf-turk0`, …).
  - Main number, Jia & Liang's adversarial score, with the grouping of the authors' `eval_squad.py` (CodaLab bundle `0xc2824269ba0a4179ac63c08fd3997d1c`, checked 2026-10-04; `squadqa.evaluate.adversarial_groups`). Group questions by the id prefix before the first `-`. In a group with adversarial versions (787 of the 1,000), the original is not used: take the version with the lowest F1 (the first in file order on ties) and that version's EM, so EM is not minimized separately. A group without versions keeps its original. Average over the 1,000 groups.
  - Get per-question EM and F1 from the official scorer's own `get_raw_scores` (load `qa/evaluate-v2.0.py` with `importlib`); only the grouping is ours (rule 5).
  - Also report the scorer's plain average over all questions in the file, which many papers quote instead, and the average over the 1,000 originals alone.
- **Unnormalized scoring** (a diagnostic; `python -m squadqa.evaluate <set> <predictions> --no-normalize`, or `normalize=False`): the official scorer's code with `normalize_answer` replaced by the identity (`squadqa.evaluate.raw_scorer`; `qa/evaluate-v2.0.py` is not changed). EM is exact string equality and F1 is bag-of-tokens over whitespace tokens, with case, punctuation and articles kept. It shows how much the scorer's normalization forgives, and is never a reported number (rule 5).
  - Measured 2026-10-16 on validation: the row-0 large seeds score 69.5 EM / 87.1–87.2 F1 unnormalized against 76.8–77.0 / 89.5–89.7 official (MAP), about 7.4 EM and 2.4 F1 forgiven. The base runs (row-0 base, 1b `joint-plain`) lose the same within 0.3, so normalization does not change the ranking.
  - About 70% of the exact matches lost unnormalized differ from the gold only by a leading "the"; case alone accounts for 1–2. No validation gold answer has leading or trailing whitespace.
  - Raw F1 can exceed the official F1: "the dog" against gold "the cat" shares "the" (0.5 raw, 0 official).

---

## 7. FlexAttention version of DeBERTa attention (P3, optional)

Only after P1 experiments are done. Goal: a fused kernel and a simpler levitated mask, without hand-written kernel code.

**Match Hugging Face DeBERTa-v2/v3 exactly. Check every item against the installed source and `config.json`:**
- All three terms (content→content, content→position, position→content) share one scale: `1 / sqrt(head_dim × 3)`.
- `relative_pos = log_bucket(pos_q − pos_k)`; bucketing controlled by `position_buckets` and `max_relative_positions` (−1 means use `max_position_embeddings`).
- Content→position index: `clamp(relative_pos + att_span, 0, 2·att_span − 1)`.
- Position→content index: `clamp(−relative_pos + att_span, 0, 2·att_span − 1)`.
- With `share_att_key: true`, position keys and queries use the content key and query projections applied to the layer-normed relative embeddings.

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# per layer: q_c, k_c, v: (B, H, N, d); k_r, q_r: (H, 2S, d) projected relative embeddings
c2p = torch.einsum("bhnd,hsd->bhns", q_c, k_r) * scale
p2c = torch.einsum("bhnd,hsd->bhns", k_c, q_r) * scale
# c2p_idx / p2c_idx: lookup from (pos_q - pos_k + OFFSET) to the clamped bucket index

def score_mod(score, b, h, qi, kj):
    d = pos[b, qi] - pos[b, kj]
    return score + c2p[b, h, qi, c2p_idx[d + OFFSET]] + p2c[b, h, kj, p2c_idx[d + OFFSET]]

def mask_mod(b, h, qi, kj):
    is_text_key = cand[b, kj] < 0                  # cand = -1 for text, c for candidate c's markers
    same_cand = (cand[b, qi] >= 0) & (cand[b, qi] == cand[b, kj])
    return (is_text_key | same_cand) & keep[b, kj] # keep = False for padding

out = flex_attention(q_c, k_c, v, score_mod=score_mod,
                     block_mask=create_block_mask(mask_mod, B, None, N, N), scale=scale)
```
- **Precondition:** gradients must flow through `c2p` and `p2c`, which are used inside `score_mod`. Test with a tiny gradient check first; support depends on the PyTorch version.
- **Recompilation:** dynamic padding changes shapes. Round sequence lengths to a few fixed buckets.
- Follow §8 for all testing.

---

## 8. Testing workflow for custom attention code (including AI-written code)

1. **Reference:** the Hugging Face DeBERTa attention module, run in fp32 and fp64.
2. **Independent tests:** written or reviewed by a human, not generated together with the implementation. Agents must not modify them (rule 6).
3. **Test matrix:**
   - lengths 1, 63, 64, 65, 127, 128, 129, 384, 1024;
   - batch size 1 and > 1;
   - padding on/off; levitated mask on/off;
   - fp32 and bf16.
4. **Check outputs and gradients** for q, k, v, and the relative embeddings. Starting tolerances: fp32 `atol=1e-5, rtol=1e-4`; bf16 `atol=2e-2, rtol=2e-2`. Run fp64 `torch.autograd.gradcheck` at tiny sizes.
5. **End-to-end:** SQuAD F1 with the new path must be within seed noise of the reference path.
6. **Benchmarks:** warm-up iterations, `torch.cuda.synchronize()`, forward + backward time and peak memory at several lengths. Compare against Hugging Face eager and `torch.compile` of Hugging Face. Flag any code that branches on specific shapes.
7. **Forward pass first:** use the new path for evaluation and inference only until the backward pass passes every test.

---

## 9. Practical notes

- **Precision** means mixed precision.
  - `precision: bf16` means the computation runs in bf16 under `torch.autocast("cuda", dtype=torch.bfloat16)` on A100/H100.
  - Master weights, gradients and optimizer state stay in **fp32**.
  - Never train with bf16 or fp16 weights and a plain AdamW:
    - bf16 keeps only 8 bits of precision, so a weight of about 0.05 can only change in steps of about 4e-4. An Adam update of about the learning rate (2e-5) rounds to zero, with no error or NaN.
    - fp16 can overflow and produce NaNs with DeBERTa-v3.
- **Load with an explicit dtype.**
  - Transformers 5 loads a checkpoint in the dtype it was stored in. `microsoft/deberta-v3-large`'s `pytorch_model.bin` is stored in fp16.
  - Training: `from_pretrained(name, revision=..., dtype=torch.float32)`.
  - Inference only: `dtype=torch.bfloat16` is fine. On an A100 its hidden states matched fp32 with cosine similarity 0.99995.
  - An earlier sandbox run that left out `dtype` trained with fp16 weights and fp16 Adam state, and nothing raised an error.
- **Check dtypes rather than assume them** (rule 7). After loading the model and when saving a checkpoint, raise an error if any trainable parameter or optimizer state is not fp32.
  - A deberta-v3-large checkpoint with Adam state is about 5.2 GB. A checkpoint of about 2.6 GB means half-precision state.
- **Pinned and offline.**
  - Load with `revision=MODELS[name]["revision"]` from `squadqa/download_model.py`.
  - Set `HF_HUB_OFFLINE=1` in jobs, so a missing file raises an error instead of downloading something unpinned.
  - Pinned (2026-10-04): `microsoft/deberta-v3-large`, `microsoft/deberta-v3-base` (development), `answerdotai/ModernBERT-base` (development) and `answerdotai/ModernBERT-large` (results). Pin any other model the same way before using it.
- **Smoke-test every new training script** before a full run.
  - Train a few dozen steps with a checkpoint, then run again with the same run id.
  - Check that parameters and optimizer state are fp32, the loss falls, the run resumes at the saved step, and the official scorer accepts the predictions.
- **Tokenizer:** DeBERTa-v3's fast tokenizer provides character offsets; the offset round-trip test (below) must pass before training.
- **Offset round-trip test:** for every training question, `context[answer_start : answer_start + len(text)] == text` (true for all rows in the official files), and the word span built from §3.2–3.3 maps back to identical text, except for the 265 logged misaligned answers.
- **Slurm:** checkpoint every N steps; resume automatically; default GPU jobs run at most 3 hours.
- **Runs:** save config, git commit hash, and seed with every run under `results/<run_id>/`.

---

## 10. Dataset facts (for sanity checks)

- Measured on the official files. Word counts use the §3.2 rule, and dev answer rows use the first reference.

| | Train | Dev |
|---|---|---|
| Articles / paragraphs / questions | 442 / 18,896 / 87,599 | 48 / 2,067 / 10,570 |
| Questions per paragraph | 4.64 | 5.11 |
| Paragraph words (mean / 99th pct / max) | 140 / 333 / 815 | 146 / 379 / 708 |
| Paragraphs over 250 words | 5.1% | 5.4% |
| Answer words (mean / median / max) | 3.62 / 2 / 50 | 3.42 / 2 / 37 |
| Answers ≤ 10 words | 94.2% | 95.5% |
| Answers over 30 words | 78 | 3 (every question still has a reference of ≤ 28 words) |
| Reference answers per question | 1 | 3.29 mean (1.72 in the local file after duplicate removal; see AGENTS.md) |
| Answer text repeated in paragraph | 10.8% | 11.0% |
| Word-aligned answers (§3.2 rule) | 99.70% | 99.59% |
| Offset errors in raw data | 0 | 0 |
