# implementation_details.md — CS4248 Extractive QA (SQuAD v1.1)

Implementation spec for coding agents and team members.

- **What and why:** `Extractive_QA_Design_Plan.docx` (plan v0.4). **How:** this file.
- If this file and the plan disagree, stop and ask a human.

---

## 1. Hard rules (MUST)

1. `dev-v1.1.json` is the **test set**. Never train on it, tune on it, or inspect its predictions except in final evaluation runs. All tuning uses the validation split (§3.1).
2. Never generate training data (questions, distractors) from `dev-v1.1.json` passages.
3. No closed-source LLM or API calls in training or inference code. (Use for data generation is pending a ruling; see plan §2.)
4. Locate gold answers with `answer_start`, never by string search. 11% of answers appear more than once in their paragraph.
5. Every reported number comes from the official `evaluate-v2.0.py`. Predictions file: `{question_id: answer_text}` for **every** question, never an empty string (an empty string counts as "no answer").
6. Never edit a reference implementation or a test to make new code pass. If a test looks wrong, stop and report.
7. No silent fallbacks. If a fast path (FlexAttention, KV cache) is unavailable, raise an error or log it loudly. Never switch paths quietly, especially in benchmarks.
8. Every training job checkpoints and can resume. Default cluster GPU jobs are capped at 3 hours.
9. Do not start a hand-written Triton/CUDA kernel. Out of scope until after the deadline (plan §12).

---

## 2. Config defaults

```yaml
model: microsoft/deberta-v3-large      # development: microsoft/deberta-v3-base
precision: bf16                        # bf16 autocast compute, fp32 master weights and optimizer state (§9); fp16 can be unstable with DeBERTa-v3
max_seq_len: 384
doc_stride: 128
padding: dynamic                       # pad to longest in batch; batch by similar length
max_answer_words: 30                   # L; every dev question has a reference of ≤ 28 words
span_head: joint                       # joint | standard (baseline)
head_dim: 256
W_init: zeros
decode_topk: 10
decode_rules: [map, mbr]               # always report both
calibration: temperature               # one T, fit on validation split
verifier:
  type: in_place                       # in_place | levitated | levitated_shared
  shared_option: B                     # A = fully shared multi-task, B = frozen reader + marker-only LoRA
  lora: {rank: 16, alpha: 32, targets: [query_proj, key_proj, value_proj]}   # check module names in the installed HF version
  k_infer: 5
  k_sweep: [5, 10, 20, 30]             # levitated variants only
  crossfit_folds: 3
seeds: [0, 1, 2]
```

---

## 3. Data

### 3.1 Splits
- **Validation:** hold out about 10% of *articles* from `train-v1.1.json` (≈44 articles, ≈8.7k questions). Split by article title, never by question: questions on the same article share paragraphs. Fixed seed; save the list to `data/splits/val_articles.txt`.
- **Cross-fitting folds (K = 3):** also by article, over the non-validation training articles.
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
- Tokenize `(question, context)` with the fast tokenizer: `truncation="only_second"`, `max_length=384`, `stride=128`, `return_overflowing_tokens=True`, `return_offsets_mapping=True`.
- For each passage word in a window:
  - `first_tok[w]` = first subword token whose character span lies inside the word;
  - `last_tok[w]` = last such token.
- Use only words entirely inside the window.
- **DeBERTa-v3 offsets include the leading space of a word** (checked 2026-09-24: in "Who won", the token " won" has span (3, 7), not (4, 7)). Strip leading whitespace from each token's span before comparing it with word spans. Otherwise no token "lies inside" a word that follows a space, and `first_tok` silently finds nothing.

### 3.4 Windows
- About 5% of paragraphs exceed 250 words, so most questions produce one window.
- Several windows: put the valid span scores of all windows of a question into **one** softmax. A span present in several windows uses the score from the window where it is most central (largest `min(tokens to its left, tokens to its right)` within the window).

### 3.5 Sibling answers (verifier negatives)
- For question q: the gold answers of the other questions on the same paragraph, as word spans.
- Drop any sibling whose word-overlap F1 with q's gold span is above 0.

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
Word-level positional F1 between span (i, j) and gold (i*, j*):
```
overlap = max(0, min(j, j*) − max(i, i*) + 1)
P = overlap / (j − i + 1);  R = overlap / (j* − i* + 1);  F1 = 2PR / (P + R)   (0 if overlap = 0)
target = (1 − α) · onehot(gold) + α · softmax over {F1 > 0} of (F1 / τ)     # α, τ tuned on validation
```
Cross-entropy, NaN-safe (masked entries have log p = −inf, and 0 · −inf = NaN):
```python
logp = scores.flatten(1).log_softmax(-1)
loss = -torch.where(target > 0, target * logp, torch.zeros_like(logp)).sum(-1).mean()
```

### 4.4 Decoding
- `topk` over flattened scores (k = 10) → (i, j) → characters `(word_start[i], word_end[j])` → answer = `context[a:b]`.
- **MAP** = top-1 span.
- **MBR:** after calibration, among the top-k pick `argmax_y Σ_y' p(y') · F1(y, y')` using positional word F1.
- **Calibration:** fit one temperature T on the validation split (minimize the NLL of the gold span).

### 4.5 Diagnostics
- **Cross-span mass** (plan E3; also the W2 early check on the joint head):
  1. Take the two highest-probability non-overlapping spans a = (i1, j1) and b = (i2, j2), with i1 < i2.
  2. The cross span is c = (i1, j2), if valid.
  3. Report p(c) / (p(a) + p(b) + p(c)), bucketed by top-1 probability (< 0.5, 0.5–0.9, > 0.9), for the standard and joint heads.
- **Equivalence test:** with W = 0 and b = 0, the joint head's argmax must equal the argmax over valid (i, j) of `u·h_i + v·h_j`.

---

## 5. Verifier

### 5.1 Candidates and targets
- **Training:** cross-fitted reader top-k on the held-out fold, plus the gold span if missing, plus boundary variants (±1 word at either end, valid only), plus siblings (§3.5). Deduplicate.
- **Inference:** the reader's top-k.
- **Loss:** listwise. Softmax over a question's candidate scores; F1-soft target over the candidates (same α, τ form as §4.3); cross-entropy.
- Track **top-k recall** (fraction of questions whose gold span is among the candidates). It caps verifier accuracy.

### 5.2 Cross-fitting (K = 3)
1. Split non-validation training articles into 3 folds.
2. For each fold: train a reader on the other 2 folds; generate top-k candidates on the held-out fold.
3. Train the verifier on these candidates.
4. At test time use the reader trained on all non-validation training articles.

### 5.3 In-place markers
- Insert `<a>` and `</a>` tokens around the candidate's subwords in the input.
- Head: concatenate hidden states at `<a>`, `</a>`, `[CLS]` → MLP → one score.
- One forward pass per candidate.

### 5.4 Levitated markers
- **Layout:** `[text tokens][S_1 E_1 … S_k E_k][padding]`. Pad questions with fewer candidates using dummy marker pairs, excluded from attention and loss.
- **Position IDs:** text 0 … n−1; `S_c` = position of `first_tok` of the candidate's first word; `E_c` = position of `last_tok` of its last word.
- **Mask:** text queries see text keys only. Marker queries see text keys plus their own partner marker.
- **DeBERTa relative positions:** build `relative_pos` from the position IDs using Hugging Face's own `build_relative_position` / `make_log_bucket_position` (do not reimplement the bucketing). Pass it to the encoder. First verify that the installed Hugging Face version accepts `relative_pos` and a 3D attention mask; if not, wrap `DebertaV2Encoder`.
- **Head:** concatenate `[h(S_c); h(E_c); h_text(first_tok of start word); h_text(last_tok of end word); h(CLS)]` → MLP → one score.

### 5.5 Shared variant
- **Option A (fully shared):** one pass; loss = reader loss + β × verifier loss; all weights train.
- **Option B (default):** train the reader, then freeze it. Marker tokens use LoRA-adapted weights (`W + ΔW`, applied only where `is_marker` is true); text tokens use frozen base weights. Compute text keys and values under `torch.no_grad()`. Train only the LoRA weights, marker embeddings, and marker head.
- **Inference (default, simple):** reader pass → top-k → one full levitated pass (≈1.05× a reader pass). The reader's span scores are unchanged by the markers.
- **KV cache (optional):** keep each layer's text keys and values from the reader pass and run only the marker tokens against them.

### 5.6 Acceptance tests (all must pass before any verifier training run)
| ID | Test | Pass condition |
|---|---|---|
| V1 | Text unchanged | Text-token outputs with markers appended equal a plain pass without markers (fp32: `atol=1e-5`) |
| V2 | Order independence | Shuffling marker order changes no candidate score |
| V3 | Cache equality | KV-cache scores equal single-pass scores |
| V4 | Reader frozen (option B) | Reader predictions identical before and after verifier training |

---

## 6. Analysis notes

- **Fix-type breakdown (plan E2):** compare strings after the official `normalize_answer` (it drops articles and punctuation).
  - *No change:* reader and verifier answers are equal after normalization.
  - *Boundary fix:* the answers overlap but differ after normalization.
  - *Answer switch:* the answers do not overlap.
- Report per-seed results, mean ± std over 3 seeds, and paired bootstrap 95% confidence intervals for headline differences.

---

## 7. FlexAttention version of DeBERTa attention (optional, W5+)

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
