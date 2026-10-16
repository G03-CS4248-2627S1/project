"""Data pipeline (spec §3): SQuAD loading, words, windows, word <-> token maps, batching.

Every span in the project is a pair of word indices (i, j), inclusive, over the words of
spec §3.2. Gold answers are located only by answer_start (spec rule 4), never by search.

    tok = load_tokenizer(cfg["model"])
    feats = featurize(load_split("train", cfg["extra_train_data"]), tok, cfg, training=True)
    for ids in bucket_batches(feats.question_lengths(), batch_size, seed, epoch):
        batch = collate(feats.batch(ids), tok.pad_token_id)

The functions the offset reference test (tests/reference/, issue #19) is written against:
word_spans, token_spans, context_tokens, answer_word_span, pair_layout, make_windows, featurize.

    python -m squadqa.data        # writes data/splits/skipped_train_ids.txt and prints counts

Run from the repo root, like every entry point.
"""

import bisect
import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from squadqa.splits import load_val_titles

TRAIN = Path("qa/train-v1.1.json")
DEV = Path("qa/dev-v1.1.json")
SKIPPED_IDS = Path("data/splits/skipped_train_ids.txt")
EXTRA_TRAIN = {"adversarialqa": Path("data/external/adversarialqa/train.json")}


# Loading (spec §3.1, §3.6).

@dataclass(frozen=True)
class Question:
    id: str
    title: str
    context: str
    question: str
    answers: tuple[tuple[str, int], ...]        # (text, answer_start) as annotated, in file order


def load_squad(path: str | Path) -> list[Question]:
    """Every question in a SQuAD-format file, in file order."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)["data"]
    return [Question(qa["id"], article["title"], p["context"], qa["question"],
                     tuple((a["text"], a["answer_start"]) for a in qa["answers"]))
            for article in data for p in article["paragraphs"] for qa in p["qas"]]


def drop_validation(questions: list[Question]) -> list[Question]:
    """The questions whose article title is not in the validation split."""
    val = load_val_titles()
    return [q for q in questions if q.title not in val]


def check_no_dev_passages(questions: list[Question]) -> None:
    """Raise if any passage equals a dev-v1.1 passage after whitespace normalization."""
    dev = {" ".join(q.context.split()) for q in load_squad(DEV)}
    hits = sorted({q.id for q in questions if " ".join(q.context.split()) in dev})
    if hits:
        raise RuntimeError(f"{len(hits)} training questions are on dev-v1.1 passages, e.g. {hits[:3]}")


def load_split(name: str, extra: list[str] | tuple[str, ...] = ()) -> list[Question]:
    """"train": train-v1.1 minus the validation articles, plus each extra_train_data set (row 5)
    with the same filtering and the dev-passage check. "validation": the held-out articles.
    Dev is loaded only for the final scoring (squadqa.evaluate)."""
    if name == "validation":
        if extra:
            raise ValueError("extra training data never goes into the validation split")
        val = load_val_titles()
        return [q for q in load_squad(TRAIN) if q.title in val]
    if name != "train":
        raise KeyError(f"unknown split {name!r}; choose train or validation")
    questions = drop_validation(load_squad(TRAIN))
    for key in extra:
        if key not in EXTRA_TRAIN:
            raise KeyError(f"unknown extra_train_data {key!r}; choose from {sorted(EXTRA_TRAIN)}")
        if not EXTRA_TRAIN[key].exists():
            raise FileNotFoundError(f"{EXTRA_TRAIN[key]} is missing; run python -m squadqa.download_external")
        added = drop_validation(load_squad(EXTRA_TRAIN[key]))
        check_no_dev_passages(added)
        questions += added
    return questions


# Words (spec §3.2). is_punct and word_spans are the spec's reference code.

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


def answer_word_span(words: list[tuple[int, int]], answer_start: int, text: str) -> tuple[int, int] | None:
    """(first word, last word) of the answer at answer_start, or None if it does not start on a
    word start and end on a word end (not word-aligned)."""
    end = answer_start + len(text)
    starts = [s for s, _ in words]
    i = bisect.bisect_left(starts, answer_start)
    if i == len(words) or words[i][0] != answer_start:
        return None
    j = bisect.bisect_left(starts, end) - 1
    if j < i or words[j][1] != end:
        return None
    return i, j


# Words <-> subword tokens (spec §3.3).

def token_spans(text: str, offsets: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Token character spans with leading whitespace removed: DeBERTa-v3 includes a word's
    leading space in its first token (" won" is (3, 7) in "Who won"; the word is (4, 7))."""
    out = []
    for s, e in offsets:
        while s < e and text[s].isspace():
            s += 1
        out.append((s, e))
    return out


@dataclass
class ContextTokens:
    """A passage tokenized once, with its word <-> token maps (indices into ids)."""
    words: list[tuple[int, int]]
    ids: list[int]
    spans: list[tuple[int, int]]            # token_spans of each token
    word_first: np.ndarray                  # (n_words,) first token inside each word
    word_last: np.ndarray                   # (n_words,) last token inside each word


def context_tokens(context: str, tokenizer) -> ContextTokens:
    """Tokenize a passage on its own and map its words to tokens.

    word_first[w] / word_last[w] = the first / last token whose span lies inside word w.
    Raises if a token is not inside a single word or a word has no token, so a map can never
    be silently empty (neither happens on train or dev; checked 2026-10-04).
    """
    enc = tokenizer(context, add_special_tokens=False, return_offsets_mapping=True)
    return _context_tokens(context, enc["input_ids"], enc["offset_mapping"])


def _context_tokens(context: str, ids: list[int], offsets: list[tuple[int, int]]) -> ContextTokens:
    words = word_spans(context)
    spans = token_spans(context, offsets)
    starts = [s for s, _ in words]
    first = np.full(len(words), -1, dtype=np.int32)
    last = np.full(len(words), -1, dtype=np.int32)
    for k, (s, e) in enumerate(spans):
        if s >= e:
            continue                                    # a whitespace-only token belongs to no word
        w = bisect.bisect_right(starts, s) - 1
        if w < 0 or not (words[w][0] <= s and e <= words[w][1]):
            raise RuntimeError(f"token {context[s:e]!r} at {(s, e)} is not inside one word")
        if first[w] < 0:
            first[w] = k
        last[w] = k
    if (first < 0).any():
        w = int(np.argmin(first))
        raise RuntimeError(f"word {context[words[w][0]:words[w][1]]!r} at {words[w]} has no token")
    return ContextTokens(words, ids, spans, first, last)


# Windows (spec §3.4).

def window_ranges(n_tokens: int, budget: int, stride: int) -> list[tuple[int, int]]:
    """[start, end) passage-token ranges of at most `budget` tokens, consecutive ones overlapping
    by `stride` tokens, together covering the whole passage.

    Built here rather than by the tokenizer's return_overflowing_tokens: tokenizers 0.23
    (every release through 0.23.2; 0.22.2 is correct) returns at most two windows for a
    (question, passage) pair, so the end of a long passage is lost (528 train and 146 dev
    questions; checked 2026-10-04).
    """
    if budget <= stride:
        raise ValueError(f"window budget {budget} must exceed the stride {stride}")
    out, start = [], 0
    while True:
        end = min(start + budget, n_tokens)
        out.append((start, end))
        if end == n_tokens:
            return out
        start = end - stride


@dataclass
class Window:
    example: int                            # index into Features.examples
    input_ids: np.ndarray                   # (T,) int32: special tokens, question, passage tokens
    first_word: int                         # global index of the window's first passage word
    first_tok: np.ndarray                   # (n,) position in input_ids of each window word's first token
    last_tok: np.ndarray                    # (n,) position in input_ids of each window word's last token
    ctx_start: int                          # positions in input_ids of the first and last passage token
    ctx_end: int
    gold: tuple[int, int] | None = None     # gold word span local to this window, if wholly inside
    content: np.ndarray | None = None       # (n,) bool: which window words are content words (spec §4.3)
    core: tuple[int, int] | None = None     # the gold's first and last content word, local, if both are inside

    @property
    def n_words(self) -> int:
        return len(self.first_tok)


def pair_layout(tokenizer) -> tuple[list[int], list[int], list[int]]:
    """The special tokens the tokenizer puts before, between and after a (question, passage)
    pair, read off one encoded pair: DeBERTa-v3 gives [CLS], [SEP], [SEP]."""
    enc = tokenizer("a b", "c d")
    seq = enc.sequence_ids()
    ids = enc["input_ids"]
    q = [k for k, s in enumerate(seq) if s == 0]
    c = [k for k, s in enumerate(seq) if s == 1]
    if q != list(range(q[0], q[-1] + 1)) or c != list(range(c[0], c[-1] + 1)) or q[-1] > c[0]:
        raise RuntimeError("unexpected pair layout")
    return ids[:q[0]], ids[q[-1] + 1:c[0]], ids[c[-1] + 1:]


def make_windows(question_ids: list[int], ctx: ContextTokens, layout: tuple[list[int], list[int], list[int]],
                 max_seq_len: int, stride: int, example: int = 0) -> list[Window]:
    """The windows of one question: layout = pair_layout(tokenizer).

    A window keeps only the passage words whose tokens are all inside it (a contiguous run).
    gold is left as None; featurize fills it in.
    """
    before, between, after = layout
    head = before + list(question_ids) + between
    offset = len(head)                                  # position of the first passage token
    budget = max_seq_len - offset - len(after)
    out = []
    for a, b in window_ranges(len(ctx.ids), budget, stride):
        w0 = int(np.searchsorted(ctx.word_first, a))                # first word starting at or after a
        w1 = int(np.searchsorted(ctx.word_last, b)) - 1             # last word ending before b
        if w1 < w0:
            raise RuntimeError("a window holds no complete passage word")
        ids = head + ctx.ids[a:b] + after
        shift = offset - a
        out.append(Window(example, np.array(ids, dtype=np.int32), w0,
                          ctx.word_first[w0:w1 + 1] + shift, ctx.word_last[w0:w1 + 1] + shift,
                          offset, offset + (b - a) - 1))
    return out


# Features.

@dataclass
class Example:
    question: Question
    context: ContextTokens
    gold: tuple[int, int] | None            # word span of answers[0], None if not word-aligned
    windows: list[int] = field(default_factory=list)   # indices into Features.windows


@dataclass
class Features:
    examples: list[Example]
    windows: list[Window]
    skipped: dict[str, str]                 # question id -> reason (training only)

    def question_lengths(self) -> list[int]:
        """Longest window of each example, in tokens: the key for length bucketing."""
        return [max(len(self.windows[w].input_ids) for w in ex.windows) for ex in self.examples]

    def batch(self, example_ids: list[int]) -> list[tuple[int, Window]]:
        """(position in the batch, window) for every window of the given examples."""
        return [(b, self.windows[w]) for b, e in enumerate(example_ids) for w in self.examples[e].windows]


def featurize(questions: list[Question], tokenizer, cfg: dict, training: bool) -> Features:
    """Windows of cfg["max_seq_len"] tokens overlapping by cfg["doc_stride"] (spec §3.3-3.4).

    training=True skips the questions whose answers[0] is not word-aligned or is longer than
    cfg["max_answer_words"] words (spec §3.2; 343 on train-v1.1), and raises if a kept question
    has no window holding its whole answer. Otherwise every question is kept (validation, dev).
    """
    unique = list(dict.fromkeys(q.context for q in questions))
    enc = tokenizer(unique, add_special_tokens=False, return_offsets_mapping=True)
    contexts = {c: _context_tokens(c, ids, offs) for c, ids, offs in zip(unique, enc["input_ids"], enc["offset_mapping"])}
    examples: list[Example] = []
    windows: list[Window] = []
    skipped: dict[str, str] = {}
    for q in questions:
        ctx = contexts[q.context]
        gold = answer_word_span(ctx.words, q.answers[0][1], q.answers[0][0]) if q.answers else None
        if training and gold is None:
            skipped[q.id] = "not word-aligned"
        elif training and gold[1] - gold[0] + 1 > cfg["max_answer_words"]:
            skipped[q.id] = f"longer than {cfg['max_answer_words']} words"
        else:
            examples.append(Example(q, ctx, gold))

    from squadqa.decode import content_flags        # here: squadqa.decode imports this module

    flags = {c: content_flags(c, ctx.words) for c, ctx in contexts.items()}
    layout = pair_layout(tokenizer)
    question_ids = tokenizer([ex.question.question for ex in examples], add_special_tokens=False)["input_ids"]
    for e, (ex, q_ids) in enumerate(zip(examples, question_ids)):
        f = flags[ex.question.context]
        core = None                                     # the gold's content core: its first and last content word
        if ex.gold is not None and f[ex.gold[0]:ex.gold[1] + 1].any():
            c = np.flatnonzero(f[ex.gold[0]:ex.gold[1] + 1])
            core = (ex.gold[0] + int(c[0]), ex.gold[0] + int(c[-1]))
        for win in make_windows(q_ids, ex.context, layout, cfg["max_seq_len"], cfg["doc_stride"], e):
            if ex.gold is not None and win.first_word <= ex.gold[0] and ex.gold[1] < win.first_word + win.n_words:
                win.gold = (ex.gold[0] - win.first_word, ex.gold[1] - win.first_word)
            win.content = f[win.first_word:win.first_word + win.n_words]
            if core is not None and win.first_word <= core[0] and core[1] < win.first_word + win.n_words:
                win.core = (core[0] - win.first_word, core[1] - win.first_word)
            ex.windows.append(len(windows))
            windows.append(win)

    if training:
        lost = [ex.question.id for ex in examples if all(windows[w].gold is None for w in ex.windows)]
        if lost:
            raise RuntimeError(f"{len(lost)} training questions have no window holding the whole answer, e.g. {lost[:3]}")
    return Features(examples, windows, skipped)


def load_tokenizer(name: str):
    """The pinned fast tokenizer of a model in squadqa.download_model.MODELS."""
    from squadqa.config import env
    from squadqa.download_model import MODELS
    env("HF_HOME")                      # loads .env before huggingface_hub reads the cache path
    from transformers import AutoTokenizer

    if name not in MODELS:
        raise KeyError(f"{name} is not pinned; add it to squadqa/download_model.py first (spec §9)")
    tok = AutoTokenizer.from_pretrained(name, revision=MODELS[name]["revision"])
    if not tok.is_fast:
        raise RuntimeError(f"{name}: character offsets need a fast tokenizer")
    return tok


# Batching: dynamic padding and length bucketing.

def bucket_batches(lengths: list[int], batch_size: int, seed: int, epoch: int, pool: int = 50) -> list[list[int]]:
    """Shuffled batches of example indices of similar length.

    Shuffles, cuts into pools of `pool` batches, sorts each pool by length, slices it into
    batches and shuffles the batches. A batch holds every window of its examples, so the
    windows of a question always share one softmax (spec §3.4). Deterministic in (seed, epoch).
    """
    rng = np.random.default_rng([seed, epoch])
    idx = rng.permutation(len(lengths)).tolist()
    batches = []
    for s in range(0, len(idx), batch_size * pool):
        part = sorted(idx[s:s + batch_size * pool], key=lambda i: lengths[i])
        batches += [part[k:k + batch_size] for k in range(0, len(part), batch_size)]
    return [batches[i] for i in rng.permutation(len(batches))]


def collate(rows: list[tuple[int, Window]], pad_id: int) -> dict:
    """Pad a batch of windows into tensors (W windows, T tokens, n = most words in a window):

    input_ids, attention_mask       (W, T)
    first_tok, last_tok, word_mask  (W, n): token positions of each window word; padding is 0 / False
    example                         (W,): the window's question, as a position in the batch
    first_word                      (W,): global index of the window's first word
    ctx_start, ctx_end              (W,): first and last passage token, for "most central" (§3.4)
    gold_start, gold_end            (W,): gold word span local to the window, -1 if not wholly inside
    content                         (W, n): content-word flags (spec §4.3); padding is False
    core_start, core_end            (W,): the gold's first and last content word, local, -1 unless both
                                    are inside (target_unit group: a trimmed boundary variant can sit in a
                                    window that cuts off the gold's leading "the")
    """
    import torch

    T = max(len(w.input_ids) for _, w in rows)
    n = max(w.n_words for _, w in rows)
    input_ids = torch.full((len(rows), T), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), T), dtype=torch.long)
    first_tok = torch.zeros((len(rows), n), dtype=torch.long)
    last_tok = torch.zeros((len(rows), n), dtype=torch.long)
    word_mask = torch.zeros((len(rows), n), dtype=torch.bool)
    content = torch.zeros((len(rows), n), dtype=torch.bool)
    for r, (_, w) in enumerate(rows):
        if w.content is not None:
            content[r, :w.n_words] = torch.from_numpy(np.asarray(w.content, dtype=bool))
        input_ids[r, :len(w.input_ids)] = torch.from_numpy(w.input_ids.astype(np.int64))
        attention_mask[r, :len(w.input_ids)] = 1
        first_tok[r, :w.n_words] = torch.from_numpy(w.first_tok.astype(np.int64))
        last_tok[r, :w.n_words] = torch.from_numpy(w.last_tok.astype(np.int64))
        word_mask[r, :w.n_words] = True
    gold = [w.gold or (-1, -1) for _, w in rows]
    core = [w.core or (-1, -1) for _, w in rows]
    return {
        "input_ids": input_ids, "attention_mask": attention_mask,
        "first_tok": first_tok, "last_tok": last_tok, "word_mask": word_mask,
        "example": torch.tensor([b for b, _ in rows]),
        "first_word": torch.tensor([w.first_word for _, w in rows]),
        "ctx_start": torch.tensor([w.ctx_start for _, w in rows]),
        "ctx_end": torch.tensor([w.ctx_end for _, w in rows]),
        "gold_start": torch.tensor([g[0] for g in gold]),
        "gold_end": torch.tensor([g[1] for g in gold]),
        "content": content,
        "core_start": torch.tensor([c[0] for c in core]),
        "core_end": torch.tensor([c[1] for c in core]),
    }


def main() -> None:
    from squadqa.config import load_config

    cfg = load_config()
    feats = featurize(load_squad(TRAIN), load_tokenizer(cfg["model"]), cfg, training=True)
    reasons: dict[str, int] = {}
    for r in feats.skipped.values():
        reasons[r] = reasons.get(r, 0) + 1
    text = "".join(f"{qid}\n" for qid in sorted(feats.skipped))
    if SKIPPED_IDS.exists() and SKIPPED_IDS.read_text(encoding="utf-8") != text:
        raise RuntimeError(f"{SKIPPED_IDS} exists and differs from what this script builds; not overwriting")
    with open(SKIPPED_IDS, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    multi = sum(len(ex.windows) > 1 for ex in feats.examples)
    print(f"{len(feats.skipped)} skipped {reasons} -> {SKIPPED_IDS}; {len(feats.examples)} kept, "
          f"{len(feats.windows)} windows, {multi} questions with more than one window, "
          f"at most {max(len(ex.windows) for ex in feats.examples)} windows")


if __name__ == "__main__":
    main()
