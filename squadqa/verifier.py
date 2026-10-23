"""Verifier inputs: in-place candidate markers, shared by both variants (spec §5.3, §5.7 V5).

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

from dataclasses import dataclass

import numpy as np

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
