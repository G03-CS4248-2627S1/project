"""Span decoding (spec §3.4, §4.3-4.4): one softmax per question, calibration, MAP and MBR.

A span is a pair of global word indices (i, j), inclusive (squadqa.data). The reader scores
every valid span of every window; combine_windows turns those into one score per span of the
question, and everything after works on that.

    spans = combine_windows(windows, cfg["max_answer_words"])         # per question
    T = fit_temperature([(s.scores, s.index(gold)) ...])              # validation split only
    cands = top_k(spans, cfg["decode_topk"], T)
    i, j = decode(cands, flags, "mbr", cfg["mbr_utility"], cfg["mbr_em_weight"])
    answer = answer_text(context, words, i, j)

Content words (spec §4.3): every word except punctuation words and the articles a/an/the.
Content-word F1 and EM compare spans by the content words they cover at the same positions,
so boundary variants ("the Eiffel Tower", "Eiffel Tower") are the same answer, and the same
text elsewhere in the passage is a different one (spec rule 4). A span with no content word
("the", ".") matches only itself. span_f1_em is the shared function the F1-weighted training
targets reuse.

String grouping (mbr_utility: string, an E1 row; decoding only): candidates whose texts are equal
after the official scorer's normalize_answer are one answer wherever they are, and MBR uses the
scorer's own F1 and EM. It is the scorer-optimal grouping, so it measures what spec rule 4 costs at
decoding time; training targets never use it.

No-answer option (no_answer: true): the null joins the question's softmax, so it takes part in
calibration and lowers every span's probability, but it is never an answer (spec rule 5). MAP
and MBR choose among spans as before: any answer scores 0 against "no answer", so the null adds
the same 0 to every span's expected utility and leaves the MBR choice unchanged.
"""

from dataclasses import dataclass

import numpy as np

from squadqa.data import is_punct

ARTICLES = {"a", "an", "the"}


def content_flags(context: str, words: list[tuple[int, int]]) -> np.ndarray:
    """(n_words,) bool: False for punctuation words and the articles a/an/the (any case)."""
    return np.array([not (e - s == 1 and is_punct(context[s])) and context[s:e].lower() not in ARTICLES
                     for s, e in words], dtype=bool)


def span_f1_em(flags: np.ndarray, a_start, a_end, b_start, b_end) -> tuple[np.ndarray, np.ndarray]:
    """Positional F1 and EM between spans a and b (inclusive word indices; arrays broadcast).

    Only words with flags True count: content_flags for the normalized versions, all True for
    plain positional word F1 (where EM then means identical spans). A span with no counted
    word matches only the identical span (F1 = EM = 1) and nothing else: the scorer would call
    any two such spans equal, but they sit at different positions (spec rule 4).
    """
    prefix = np.concatenate([[0], np.cumsum(flags, dtype=np.int64)])
    a_start, a_end, b_start, b_end = (np.asarray(x) for x in (a_start, a_end, b_start, b_end))
    la = prefix[a_end + 1] - prefix[a_start]
    lb = prefix[b_end + 1] - prefix[b_start]
    lo, hi = np.maximum(a_start, b_start), np.minimum(a_end, b_end)
    overlap = np.where(lo <= hi, prefix[np.maximum(hi, lo - 1) + 1] - prefix[lo], 0)
    both_empty = (la == 0) & (lb == 0) & (a_start == b_start) & (a_end == b_end)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(la + lb > 0, 2.0 * overlap / (la + lb), 0.0)
    f1 = np.where(both_empty, 1.0, f1)
    em = np.where(both_empty, 1.0, ((overlap == la) & (overlap == lb) & (la > 0)).astype(float))
    return f1, em


def group_keys(flags: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> list[tuple[int, ...]]:
    """Each span's first and last content word, so boundary variants share a key. A span with
    no content word is its own group, (-1, start, end), so unrelated ones never pool."""
    idx = np.arange(len(flags))
    next_c = np.minimum.accumulate(np.where(flags, idx, len(flags))[::-1])[::-1]    # first content word >= w
    prev_c = np.maximum.accumulate(np.where(flags, idx, -1))                         # last content word <= w
    out = []
    for i, j in zip(starts.tolist(), ends.tolist()):
        f, l = int(next_c[i]), int(prev_c[j])
        out.append((f, l) if f <= l else (-1, i, j))
    return out


# One softmax per question (spec §3.4).

@dataclass
class SpanScores:
    starts: np.ndarray          # (m,) global word index of each valid span's first word
    ends: np.ndarray            # (m,) last word
    scores: np.ndarray          # (m,) float32 reader score s(i, j)
    null: float | None = None   # the no-answer option's score (no_answer: true), else None

    def with_null(self) -> np.ndarray:
        """The scores the question's softmax runs over: every valid span, then the null if any.
        Span k keeps position k, so probabilities of spans are log_softmax(...)[:m]."""
        return self.scores if self.null is None else np.append(self.scores, np.float32(self.null))

    def index(self, span: tuple[int, int] | None) -> int | None:
        """Position of a span, or None if it is not among the valid spans."""
        if span is None:
            return None
        hit = np.nonzero((self.starts == span[0]) & (self.ends == span[1]))[0]
        return int(hit[0]) if len(hit) else None


def combine_windows(windows: list[tuple[np.ndarray, int, np.ndarray, np.ndarray, int, int]],
                    max_answer_words: int, nulls: list[float] | None = None) -> SpanScores:
    """All valid spans of a question's windows, each scored once.

    windows: (scores (n, n) with -inf where invalid, first_word, first_tok, last_tok, ctx_start,
    ctx_end) per window, as in squadqa.data.Window. A span in several windows takes its score
    from the window where it is most central: the largest min(passage tokens to its left,
    passage tokens to its right); the earlier window on ties.
    nulls: each window's null score (no_answer), combined by logsumexp: training puts every
    window's null in the question's softmax, so the no-answer probability is their sum.
    """
    keys, cents, parts = [], [], []
    for scores, first_word, first_tok, last_tok, ctx_start, ctx_end in windows:
        n = len(first_tok)
        s = np.asarray(scores, dtype=np.float32)[:n, :n]
        i, j = np.nonzero(np.isfinite(s) & (np.arange(n)[None, :] >= np.arange(n)[:, None])
                          & (np.arange(n)[None, :] - np.arange(n)[:, None] < max_answer_words))
        parts.append((i + first_word, j + first_word, s[i, j]))
        cents.append(np.minimum(first_tok[i] - ctx_start, ctx_end - last_tok[j]))
    starts = np.concatenate([p[0] for p in parts])
    ends = np.concatenate([p[1] for p in parts])
    scores = np.concatenate([p[2] for p in parts])
    if len(windows) > 1:
        cent = np.concatenate(cents)
        order = np.lexsort((-cent, ends, starts))           # by span, most central first; stable
        keep = order[np.unique(np.stack([starts[order], ends[order]]), axis=1, return_index=True)[1]]
        starts, ends, scores = starts[keep], ends[keep], scores[keep]
    if not len(scores):
        raise RuntimeError("a question has no valid span")
    null = None if nulls is None else float(np.logaddexp.reduce(np.asarray(nulls, dtype=np.float64)))
    return SpanScores(starts.astype(np.int32), ends.astype(np.int32), scores.astype(np.float32), null)


def log_softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(scores, dtype=np.float64) / temperature
    z = z - z.max()
    return z - np.log(np.exp(z).sum())


# Calibration (spec §4.4): one temperature, fit on the validation split.

def temperature_nll(items: list[tuple[np.ndarray, int | np.ndarray]], temperature: float) -> float:
    """Mean negative log-likelihood of the gold at this temperature: of the gold span (an index), or
    of the gold's group (an array of indices, target_unit group)."""
    return float(np.mean([-np.logaddexp.reduce(np.atleast_1d(log_softmax(s, temperature)[g])) for s, g in items]))


def group_members(flags: np.ndarray, starts: np.ndarray, ends: np.ndarray, gold: tuple[int, int]) -> np.ndarray:
    """Indices of the spans in the gold span's group: those with its first and last content word, as
    group_keys groups them (the gold alone if it has no content word)."""
    idx = np.arange(len(flags))
    next_c = np.minimum.accumulate(np.where(flags, idx, len(flags))[::-1])[::-1]
    prev_c = np.maximum.accumulate(np.where(flags, idx, -1))
    f, l = int(next_c[gold[0]]), int(prev_c[gold[1]])
    if f > l:
        return np.flatnonzero((starts == gold[0]) & (ends == gold[1]))
    return np.flatnonzero((next_c[starts] == f) & (prev_c[ends] == l))


def fit_temperature(items: list[tuple[np.ndarray, int]], lo: float = 0.05, hi: float = 20.0,
                    tol: float = 1e-4) -> float:
    """T minimizing temperature_nll, by golden-section search over log T in [lo, hi].

    items: (scores of all valid spans, index of the gold span) per question; questions whose
    gold span is not a valid span are left out by the caller. Raises if the optimum is at an
    end of the range, which means the range is wrong rather than the fit.
    """
    if not items:
        raise ValueError("no questions to fit the temperature on")
    a, b = np.log(lo), np.log(hi)
    g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = temperature_nll(items, np.exp(c)), temperature_nll(items, np.exp(d))
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = temperature_nll(items, np.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = temperature_nll(items, np.exp(d))
    t = float(np.exp((a + b) / 2))
    if not (lo * 1.01 < t < hi / 1.01):
        raise RuntimeError(f"fitted temperature {t:.3g} is at the edge of [{lo}, {hi}]")
    return t


# Candidates and decision rules (spec §4.4).

@dataclass
class Candidates:
    starts: np.ndarray          # (k,) top-k spans, most probable first
    ends: np.ndarray
    logp: np.ndarray            # (k,) log-probability under the full softmax at temperature T


def span_logp(spans: SpanScores, temperature: float = 1.0) -> tuple[np.ndarray, float | None]:
    """Log-probabilities of every valid span and of the null (None without one), under the
    question's one softmax at temperature T."""
    logp = log_softmax(spans.with_null(), temperature)
    m = len(spans.scores)
    return logp[:m], (float(logp[m]) if spans.null is not None else None)


def top_k(spans: SpanScores, k: int, temperature: float = 1.0) -> Candidates:
    """The k most probable spans. With a null option their probabilities sum to less than 1; the
    null itself is never a candidate, because every SQuAD v1.1 answer is a span (spec rule 5)."""
    logp = span_logp(spans, temperature)[0]
    order = np.argsort(-logp, kind="stable")[:k]
    return Candidates(spans.starts[order], spans.ends[order], logp[order])


def string_f1_em(texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(G, G) official F1 and EM between the texts of G string groups. A text that normalizes to
    nothing ("the", ".") matches only itself, as under content-word F1: the scorer would call any
    two such texts equal, but they are unrelated spans."""
    from squadqa.evaluate import scorer

    s = scorer()
    f1 = np.array([[s.compute_f1(a, b) for b in texts] for a in texts], dtype=float)
    em = np.array([[s.compute_exact(a, b) for b in texts] for a in texts], dtype=float)
    empty = np.array([not s.normalize_answer(t) for t in texts])
    apart = (empty[:, None] | empty[None, :]) & ~np.eye(len(texts), dtype=bool)
    return np.where(apart, 0.0, f1), np.where(apart, 0.0, em)


def decode(cands: Candidates, flags: np.ndarray, rule: str, utility: str, em_weight: float = 1.0,
           texts: list[str] | None = None) -> tuple[int, int]:
    """The chosen span (i, j) among the candidates.

    utility "normalized": candidates are grouped by the content words they cover (boundary
    variants), a group's probability is the sum of its members', F1 and EM are content-word
    versions, and the answer is the chosen group's most probable span. "positional": every
    span is its own group, with plain positional word F1 and EM = identical spans. "string":
    candidates are grouped by the official normalization of their texts (texts: each
    candidate's passage text, in candidate order) at any position, with the scorer's F1 and EM;
    a text that normalizes to nothing is its own group.
    rule "map": the most probable group. "mbr": argmax over groups y of
    sum over groups y' of p(y') * (F1(y, y') + em_weight * EM(y, y')).
    """
    if utility == "normalized":
        keys = group_keys(flags, cands.starts, cands.ends)
        used = flags
    elif utility == "positional":
        keys = list(zip(cands.starts.tolist(), cands.ends.tolist()))
        used = np.ones_like(flags)
    elif utility == "string":
        if texts is None or len(texts) != len(cands.starts):
            raise ValueError("mbr_utility 'string' needs the text of every candidate")
        from squadqa.evaluate import scorer

        norm = [scorer().normalize_answer(t) for t in texts]
        keys = [(n,) if n else (-1, int(i), int(j)) for n, i, j in zip(norm, cands.starts, cands.ends)]
    else:
        raise ValueError(f"unknown mbr_utility {utility!r}")
    first: dict[tuple[int, ...], int] = {}
    mass: dict[tuple[int, ...], float] = {}
    p = np.exp(cands.logp)
    for c, key in enumerate(keys):                          # candidates are most probable first
        first.setdefault(key, c)
        mass[key] = mass.get(key, 0.0) + float(p[c])
    groups = list(first)
    rep = np.array([first[g] for g in groups])
    prob = np.array([mass[g] for g in groups])
    if rule == "map":
        best = int(np.argmax(prob))
    elif rule == "mbr":
        if utility == "string":
            f1, em = string_f1_em([texts[r] for r in rep])
        else:
            s, e = cands.starts[rep], cands.ends[rep]
            f1, em = span_f1_em(used, s[:, None], e[:, None], s[None, :], e[None, :])
        best = int(np.argmax((f1 + em_weight * em) @ prob))
    else:
        raise ValueError(f"unknown decode rule {rule!r}")
    return int(cands.starts[rep[best]]), int(cands.ends[rep[best]])


def answer_text(context: str, words: list[tuple[int, int]], i: int, j: int) -> str:
    """The passage text from the start of word i to the end of word j (never empty)."""
    return context[words[i][0]:words[j][1]]
