"""The verifier: in-place candidate markers, shared by both variants (spec §5.3, §5.7 V5), and the
shared_lora variant (spec §5.4, §5.7 V4), with its listwise loss and the reader + λ·verifier combination.

A candidate is a word span (i, j) of a passage, inclusive, as in squadqa.data. The verifier reads
the question and the passage with the candidate marked in place:

    lo, hi = candidate_window(make_windows(...), ctx, i, j)       # reader tokenizer, spec §3.4
    m = mark(context, ctx.words, i, j, lo, hi)                     # "... in <a> 1889 </a> for ..."
    open_id, close_id = add_markers(tokenizer)                     # each variant's own tokenizer
    enc = encode(tokenizer, question, m, open_id, close_id)        # input_ids and the head's positions

Both variants (#67 shared_lora, #68 separate) get the same character range of the passage, chosen
with the reader's tokenizer, so E2 compares backbones on identical text.

Resizing the embeddings: DeBERTa-v3's matrix has 128,100 rows but its tokenizer only 128,001
entries, so after add_markers len(tokenizer) is 128,003, below the matrix. A plain
model.resize_token_embeddings(len(tokenizer)) would shrink the reader's embeddings; use
resize_for_markers, which only ever grows them.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

OPEN, CLOSE = "<a>", "</a>"
OPEN_INSERT, CLOSE_INSERT = f" {OPEN} ", f" {CLOSE} "      # a space on each side (spec §5.3)


def candidate_window(windows, ctx, i: int, j: int) -> tuple[int, int]:
    """[start, end) character range of the passage window, from squadqa.data.make_windows with the
    reader's tokenizer, in which the candidate (i, j) is most central (spec §3.4): the largest
    min(passage tokens to its left, passage tokens to its right), the earlier window on ties, as in
    squadqa.decode.combine_windows. The range runs from the window's first word to its last.
    Raises if no window holds the whole candidate."""
    best, best_cent = None, -1
    for w in windows:
        w0, w1 = w.first_word, w.first_word + w.n_words - 1
        if not (w0 <= i <= j <= w1):
            continue
        cent = min(int(w.first_tok[i - w0]) - w.ctx_start, w.ctx_end - int(w.last_tok[j - w0]))
        if cent > best_cent:
            best, best_cent = (w0, w1), cent
    if best is None:
        raise ValueError(f"no window holds the whole candidate ({i}, {j})")
    return ctx.words[best[0]][0], ctx.words[best[1]][1]


@dataclass
class Marked:
    text: str           # the window's passage text with the markers inserted
    lo: int             # the window's character range in the passage
    hi: int
    open_at: int        # character offset of OPEN in text
    close_at: int       # character offset of CLOSE in text


def mark(context: str, words: list[tuple[int, int]], i: int, j: int, lo: int, hi: int) -> Marked:
    """context[lo:hi] with OPEN and CLOSE around the characters of words i..j. Raises if the passage
    already holds a marker string, since unmark could not then restore it (none does in train or
    dev; checked 2026-10-23), or if the candidate is not inside [lo, hi)."""
    a, b = words[i][0], words[j][1]
    if not (lo <= a < b <= hi):
        raise ValueError(f"candidate characters {(a, b)} are not inside the window {(lo, hi)}")
    window = context[lo:hi]
    if OPEN in window or CLOSE in window:
        raise ValueError(f"the passage already contains {OPEN!r} or {CLOSE!r}")
    left = context[lo:a] + OPEN_INSERT
    text = left + context[a:b] + CLOSE_INSERT + context[b:hi]
    return Marked(text, lo, hi, len(left) - len(OPEN) - 1, len(left) + (b - a) + 1)


def unmark(text: str) -> str:
    """The inverse of mark: text with OPEN_INSERT and CLOSE_INSERT removed. Raises unless each
    appears exactly once, OPEN first."""
    if text.count(OPEN_INSERT) != 1 or text.count(CLOSE_INSERT) != 1:
        raise ValueError("expected exactly one of each marker")
    k = text.index(OPEN_INSERT)
    if text.index(CLOSE_INSERT) < k:
        raise ValueError(f"{CLOSE!r} comes before {OPEN!r}")
    return text.replace(OPEN_INSERT, "", 1).replace(CLOSE_INSERT, "", 1)


def marked_span(text: str) -> str:
    """The text between the markers."""
    a = text.index(OPEN_INSERT) + len(OPEN_INSERT)
    return text[a:text.index(CLOSE_INSERT, a)]


def add_markers(tokenizer) -> tuple[int, int]:
    """Register OPEN and CLOSE as special tokens (once; a second call adds nothing) and return their
    ids. Raises if either does not encode as one token of its own."""
    tokenizer.add_tokens([OPEN, CLOSE], special_tokens=True)
    ids = tokenizer.convert_tokens_to_ids([OPEN, CLOSE])
    for s, k in zip((OPEN, CLOSE), ids):
        if k == tokenizer.unk_token_id or tokenizer(s, add_special_tokens=False)["input_ids"] != [k]:
            raise RuntimeError(f"{s!r} does not encode as a single token")
    return ids[0], ids[1]


def resize_for_markers(model, tokenizer) -> int:
    """Grow the model's input embeddings to hold every tokenizer id, never shrinking them. Returns
    the number of rows added (0 for DeBERTa-v3, whose matrix already has spare rows)."""
    rows = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) <= rows:
        return 0
    model.resize_token_embeddings(len(tokenizer))
    return len(tokenizer) - rows


@dataclass
class Encoded:
    input_ids: np.ndarray           # (T,) int64: the tokenizer's pair layout, question then marked text
    open_pos: int                   # positions in input_ids of OPEN and CLOSE; the head also reads
    close_pos: int                  # position 0, the first token ([CLS] for DeBERTa-v3)


def encode(tokenizer, question: str, marked: Marked, open_id: int, close_id: int) -> Encoded:
    """Tokenize (question, marked text) with the tokenizer's own pair layout and find the markers.
    Never truncates (spec rule 7): raises unless each marker id appears exactly once, in the passage,
    at the character offsets mark put it."""
    enc = tokenizer(question, marked.text, return_offsets_mapping=True)
    ids = np.asarray(enc["input_ids"], dtype=np.int64)
    seq = enc.sequence_ids()
    pos = []
    for k, at, s in ((open_id, marked.open_at, OPEN), (close_id, marked.close_at, CLOSE)):
        hits = np.flatnonzero(ids == k)
        if len(hits) != 1 or seq[hits[0]] != 1 or tuple(enc["offset_mapping"][hits[0]]) != (at, at + len(s)):
            raise RuntimeError(f"{s!r} is not where mark put it")
        pos.append(int(hits[0]))
    return Encoded(ids, pos[0], pos[1])


def collate_marked(encs: list[Encoded], pad_id: int) -> dict:
    """A batch of marked candidates, padded on the right: input_ids and attention_mask (B, T),
    open_pos and close_pos (B,)."""
    t = max(len(e.input_ids) for e in encs)
    ids = torch.full((len(encs), t), pad_id, dtype=torch.long)
    mask = torch.zeros((len(encs), t), dtype=torch.long)
    for b, e in enumerate(encs):
        ids[b, :len(e.input_ids)] = torch.from_numpy(e.input_ids)
        mask[b, :len(e.input_ids)] = 1
    return {"input_ids": ids, "attention_mask": mask,
            "open_pos": torch.tensor([e.open_pos for e in encs]), "close_pos": torch.tensor([e.close_pos for e in encs])}


# The shared_lora verifier (spec §5.4; #67): the frozen final reader plus LoRA.

@contextlib.contextmanager
def adapters(model: nn.Module, enabled: bool):
    """Switch every LoRA layer in model on or off for the block, then back on. Off, the encoder
    computes exactly what it did before the adapters were injected (spec §5.4: reader passes)."""
    from peft.tuners.tuners_utils import BaseTunerLayer

    layers = [m for m in model.modules() if isinstance(m, BaseTunerLayer)]
    for m in layers:
        m.enable_adapters(enabled)
    try:
        yield
    finally:
        for m in layers:
            m.enable_adapters(True)


class SharedLoraVerifier(nn.Module):
    """The frozen reader (squadqa.heads.Reader) with LoRA on its encoder, two marker embeddings and a
    scoring head. Only those three train; every reader parameter is frozen (V4).

    The marker embeddings are a separate (2, d) parameter, written over the encoder's word
    embeddings wherever an input id is OPEN or CLOSE, so the reader's embedding table itself never
    trains. Training just two rows of the table instead, by masking its gradient, would move every
    row: AdamW's weight decay ignores a zero gradient (checked 2026-10-23). DeBERTa-v3's table already
    has rows for the new ids, so nothing is resized either (resize_for_markers is for a tokenizer
    that outgrows its table). The markers start at the table's mean row.

    forward scores marked candidates with the adapters on; reader_span_scores runs the reader
    with them off.
    """

    def __init__(self, reader: nn.Module, open_id: int, close_id: int, cfg: dict):
        super().__init__()
        from peft import LoraConfig, inject_adapter_in_model

        for p in reader.parameters():
            p.requires_grad_(False)
        lora = cfg["verifier"]["lora"]
        # peft raises if a target name matches no module (a renamed attention layer, spec §2's note)
        inject_adapter_in_model(LoraConfig(r=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=0.0,
                                           target_modules=list(lora["targets"]), bias="none"), reader.encoder)
        self.reader = reader
        emb = reader.encoder.get_input_embeddings()
        if max(open_id, close_id) >= emb.num_embeddings:
            raise ValueError("a marker id is outside the embedding table; grow it with resize_for_markers first")
        self.open_id, self.close_id = open_id, close_id
        self.markers = nn.Parameter(emb.weight.detach().float().mean(0).repeat(2, 1))
        emb.register_forward_hook(self._insert_markers)
        d = reader.encoder.config.hidden_size
        self.head = nn.Sequential(nn.Linear(3 * d, cfg["head_dim"]), nn.GELU(),
                                  nn.Dropout(cfg["train"]["head_dropout"]), nn.Linear(cfg["head_dim"], 1))

    def _insert_markers(self, module, inputs, output):
        ids = inputs[0]
        for k, token in enumerate((self.open_id, self.close_id)):
            output = torch.where((ids == token)[..., None], self.markers[k].to(output.dtype), output)
        return output

    def forward(self, batch: dict) -> torch.Tensor:
        """(B,) scores of a collate_marked batch: [CLS], OPEN and CLOSE states → MLP."""
        h = self.reader.encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
        rows = torch.arange(h.size(0), device=h.device)
        feats = torch.cat([h[:, 0], h[rows, batch["open_pos"]], h[rows, batch["close_pos"]]], dim=-1)
        return self.head(feats).squeeze(-1).float()

    def reader_span_scores(self, batch: dict):
        """The reader's span scores (Reader.span_scores) with the adapters off."""
        with adapters(self.reader.encoder, False):
            return self.reader.span_scores(batch)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]


def listwise_loss(scores: torch.Tensor, question: torch.Tensor, targets: torch.Tensor,
                  n_questions: int) -> torch.Tensor:
    """(n_questions,) cross-entropy between each question's target (squadqa.candidates.targets,
    summing to 1 over its candidates) and the softmax of its candidates' scores (spec §5.1).
    scores, question (each candidate's question index) and targets are (N,)."""
    from squadqa.heads import _segment_logsumexp

    logp = scores - _segment_logsumexp(scores, question, n_questions)[question]
    contrib = torch.where(targets > 0, targets * logp, torch.zeros_like(logp))     # 0 · −inf stays 0
    return -torch.zeros(n_questions, device=scores.device, dtype=scores.dtype).index_add(0, question, contrib)


def combine(reader_logp: np.ndarray, verifier_score: np.ndarray, lam: float,
            temperature: float = 1.0) -> np.ndarray:
    """Log-probabilities over one list from (reader log-probability + λ · verifier score) / T
    (timeline 4.2: combine, softmax, recalibrate). λ = 0 and T = 1 is the reader alone."""
    from squadqa.decode import log_softmax

    z = np.asarray(reader_logp, dtype=np.float64) + lam * np.asarray(verifier_score, dtype=np.float64)
    return log_softmax(z / temperature)


def fit_lambda(items: list[tuple[np.ndarray, np.ndarray, int | np.ndarray]],
               grid: tuple[float, ...] = tuple(np.round(np.arange(0, 5.01, 0.1), 2)),
               temperatures: tuple[float, ...] = tuple(np.round(np.arange(0.5, 5.01, 0.05), 2))
               ) -> tuple[float, float]:
    """The (λ, T) on the grids with the lowest mean NLL of the gold on the validation lists that
    hold it: items are (reader_logp, verifier_score, gold index or the indices of its group).
    λ = 0, T = 1 is the reader alone. T is fit with λ, not after it: the reader's log-probability
    and the verifier's score are each about calibrated, so their sum is overconfident, and at T = 1
    the NLL turned the verifier off (λ = 0) on all three base runs of 2026-10-28 although it moved
    more golds up than down."""
    if not items:
        raise ValueError("no list holds its gold")
    n = max(len(r) for r, _, _ in items)
    reader = np.full((len(items), n), -np.inf)                   # padding: no probability at any (λ, T)
    score, gold = np.zeros_like(reader), np.zeros(reader.shape, dtype=bool)
    for i, (r, v, g) in enumerate(items):
        reader[i, :len(r)], score[i, :len(v)] = r, v
        gold[i, np.atleast_1d(g)] = True

    def nll(lam: float, t: float) -> float:
        z = (reader + lam * score) / t
        logp = z - np.logaddexp.reduce(z, axis=1, keepdims=True)
        return float(-np.mean(np.logaddexp.reduce(np.where(gold, logp, -np.inf), axis=1)))

    table = np.array([[nll(lam, t) for t in temperatures] for lam in grid])
    i, j = np.unravel_index(int(np.argmin(table)), table.shape)
    return float(grid[i]), float(temperatures[j])
