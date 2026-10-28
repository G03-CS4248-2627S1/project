"""Span heads, the reader model and their losses (spec §4).

Every head scores word spans: it reads the encoder state at each word's first subword (for
starts) and last subword (for ends), using first_tok / last_tok from squadqa.data.collate.
span_head picks the training objective, one of the four of Fajcik et al. (2021, "Rethinking the
Objectives of Extractive Question Answering", MRQA; their names in brackets):

- standard (I, independent; ladder row 0): h_i = MLP_start(x[first_tok[i]]), h_j = MLP_end(x[last_tok[j]]),
  start logit u.h_i, end logit v.h_j, span score s(i, j) = u.h_i + v.h_j. Loss: start NLL + end NLL.
- joint (J; ladder row 1, frozen at the 1b freeze): s(i, j) = pair(h_i, h_j) + u.h_i + v.h_j + b[j - i] (spec §4.1), one softmax over
  all valid spans. Loss: span NLL. With the pair term and b at zero it is the standard head.
- compound (I+J; an ablation): the joint loss + aux_independent_weight x the standard loss on the
  same u.h_i and v.h_j. Decoding uses the joint scores.
- conditional (JC): the standard start logit, and an end distribution given the start i,
  p(j | i) = softmax_j of w.tanh(A x[last_tok[j]] + B x[first_tok[i]] + c) (Fajcik et al.'s form,
  with head_dim hidden units). Training conditions on the gold start (teacher forcing); decoding
  scores every valid span exactly, s(i, j) = start logit + log p(j | i), not with their top-k beam.

Options of the bilinear heads (joint, compound):
- bilinear_form: the pair term. full = h_i' W h_j' with W zero at init (the biaffine scorer of
  Yu et al., 2020); dot = (A h_i')(B h_j'), A and B of rank bilinear_rank (the dot product Fajcik et
  al. found best for BERT-type models), B zero at init so B gets the first gradient and A gets one
  after B moves; diag =
  w'(h_i' * h_j'), Fajcik et al.'s weighted dot product, w zero at init; mlp = w' LN(relu(A h_i' +
  B h_j' + c)), their f_MLP (from Lee et al., 2019) of hidden width bilinear_rank, w zero at init,
  computed only for spans of at most max_answer_words words. Their additive scorer w'[h_i; h_j] is
  already u.h_i + v.h_j, and additive plus weighted product reduces to diag, so neither is a form.
  rotary works with full and dot only.
- length_bias: the learned bonus b[j - i], zero at init.
- bilinear_pos: position information in the pair term only; u.h_i and v.h_j see h itself, so the
  head still starts as the standard model. additive: sinusoidal encodings of each word's position
  in the passage (Vaswani et al., 2017) added right before the pair term, h' = h + PE. rotary: the
  pair term's inputs rotated by position (RoPE): A h_i and B h_j in the dot form, which then depends
  on j - i only, as in GlobalPointer (Su et al., 2022); h_i and h_j before W in the full form, which
  then depends on both positions, because a general W does not commute with the rotations.

no_answer adds a null option, the span (CLS, CLS) as in Fajcik et al.'s and BERT's SQuAD 2.0
models, to every softmax: one null per window, and a question's no-answer probability is the sum
over its windows (question_nll treats the copies like gold copies). Training data never targets
it on SQuAD v1.1, and decoding never returns it (spec rule 5); it only takes probability mass.

The loss puts all of a question's windows into one softmax (spec §3.4): see question_nll.

target_unit (joint and compound): span, the default, scores the annotated span, −log p(y*). group
scores the gold's group of boundary variants, −log p(G*), the sum over every span that covers the
gold's content words (group_nll), so the loss counts the gold answer once, as normalized EM does.
Compound's auxiliary start and end term stays one-hot either way (spec §4.3).
"""

import torch
import torch.nn as nn

NEG_INF = float("-inf")
HEADS = ("standard", "joint", "compound", "conditional")
BILINEAR = ("joint", "compound")
PAIR_FORMS = ("full", "dot", "diag", "mlp")
ROTARY_FORMS = ("full", "dot")


def _mlp(d_model: int, d_h: int, dropout: float) -> nn.Module:
    return nn.Sequential(nn.Linear(d_model, d_h), nn.GELU(), nn.Dropout(dropout))


def sinusoidal(positions: torch.Tensor, dim: int) -> torch.Tensor:
    """Transformer position encodings of integer positions (...,) -> (..., dim) float32:
    sin and cos of position / 10000^(2k / dim), interleaved."""
    if dim % 2:
        raise ValueError(f"sinusoidal encodings need an even dimension, got {dim}")
    k = torch.arange(0, dim, 2, device=positions.device, dtype=torch.float32)
    angle = positions[..., None].float() / (10000.0 ** (k / dim))
    return torch.stack([angle.sin(), angle.cos()], dim=-1).flatten(-2)


def rotary(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Rotary position embedding (Su et al., RoFormer): rotate each pair of dimensions (2k, 2k+1)
    of x (..., n, r) by position * 10000^(-2k / r), positions (..., n). Then
    rotary(q, i) . rotary(k, j) = q . R(j - i) k depends only on j - i. Returns float32."""
    r = x.size(-1)
    if r % 2:
        raise ValueError(f"rotary encodings need an even dimension, got {r}")
    k = torch.arange(0, r, 2, device=x.device, dtype=torch.float32)
    angle = positions[..., None].float() / (10000.0 ** (k / r))
    cos, sin = angle.cos(), angle.sin()
    x = x.float()
    x0, x1 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1).flatten(-2)


class StandardSpanHead(nn.Module):
    def __init__(self, d_model: int, d_h: int = 256, max_len: int = 30, dropout: float = 0.1):
        super().__init__()
        self.start = _mlp(d_model, d_h, dropout)
        self.end = _mlp(d_model, d_h, dropout)
        self.u = nn.Linear(d_h, 1)
        self.v = nn.Linear(d_h, 1)
        self.max_len = max_len

    def forward(self, x_first: torch.Tensor, x_last: torch.Tensor, word_mask: torch.Tensor,
                x_cls: torch.Tensor | None = None, positions: torch.Tensor | None = None) -> dict:
        """x_first, x_last: (W, n, d_model); word_mask: (W, n) bool; x_cls: (W, d_model) or None.
        Returns start and end logits (W, n), -inf on padding, and with x_cls the null's start
        and end logits (W,)."""
        start = self.u(self.start(x_first)).squeeze(-1).float()
        end = self.v(self.end(x_last)).squeeze(-1).float()
        out = {"start": start.masked_fill(~word_mask, NEG_INF), "end": end.masked_fill(~word_mask, NEG_INF)}
        if x_cls is not None:
            out["start_null"] = self.u(self.start(x_cls)).squeeze(-1).float()
            out["end_null"] = self.v(self.end(x_cls)).squeeze(-1).float()
        return out

    def span_scores(self, out: dict, word_mask: torch.Tensor) -> torch.Tensor:
        """(W, n, n) scores s(i, j) = start_i + end_j, -inf unless both are words, i <= j and
        j - i + 1 <= max_len (spec §4.1 valid set)."""
        return valid_spans(out["start"][:, :, None] + out["end"][:, None, :], word_mask, self.max_len)

    def null_score(self, out: dict) -> torch.Tensor | None:
        return out["start_null"] + out["end_null"] if "start_null" in out else None


class JointSpanHead(nn.Module):
    """s(i, j) = pair(h_i, h_j) + u.h_i + v.h_j + b[j - i] (spec §4.1-4.2); the pair term is
    h_i' W h_j' (form full), (A h_i')(B h_j') (dot), w'(h_i' * h_j') (diag) or w' LN(relu(A h_i' +
    B h_j' + c)) (mlp), h' = h (+ PE), rotated for rotary (full and dot only)."""

    def __init__(self, d_model: int, d_h: int = 256, max_len: int = 30, dropout: float = 0.1,
                 length_bias: bool = True, pos: str = "none", form: str = "full", rank: int = 64):
        super().__init__()
        bad_rotary = pos == "rotary" and form not in ROTARY_FORMS
        if pos not in ("none", "additive", "rotary") or form not in PAIR_FORMS or bad_rotary:
            raise ValueError(f"unsupported pair term: bilinear_form {form!r} with bilinear_pos {pos!r}")
        self.start = _mlp(d_model, d_h, dropout)
        self.end = _mlp(d_model, d_h, dropout)
        if form == "full":
            self.W = nn.Parameter(torch.zeros(d_h, d_h))                     # zero init (W_init: zeros)
        elif form == "dot":
            self.A = nn.Linear(d_h, rank, bias=False)
            self.B = nn.Linear(d_h, rank, bias=False)
            nn.init.zeros_(self.B.weight)                                   # zero pair term, nonzero gradient
        elif form == "diag":
            self.w = nn.Parameter(torch.zeros(d_h))                          # w'(h_i * h_j), zero at init
        else:                                                                # mlp: w' LN(relu(A h_i + B h_j + c))
            self.A = nn.Linear(d_h, rank)                                   # its bias is c
            self.B = nn.Linear(d_h, rank, bias=False)
            self.norm = nn.LayerNorm(rank)
            self.w = nn.Parameter(torch.zeros(rank))                        # zero pair term at init
        self.u = nn.Linear(d_h, 1)
        self.v = nn.Linear(d_h, 1)
        self.len_bias = nn.Parameter(torch.zeros(max_len)) if length_bias else None   # b[length - 1]
        self.pos, self.form = pos, form
        self.max_len = max_len

    def pair(self, bs: torch.Tensor, be: torch.Tensor, positions: torch.Tensor | None) -> torch.Tensor:
        """(W, n, n) pair terms of start features bs and end features be (W, n, d_h). The mlp form
        is computed only for spans of at most max_len words, one diagonal at a time (0 elsewhere,
        where valid_spans masks the span anyway)."""
        if self.form == "full":
            if self.pos == "rotary":
                bs, be = rotary(bs, positions), rotary(be, positions)
            return (bs @ self.W) @ be.transpose(1, 2)
        if self.form == "diag":
            return (bs * self.w) @ be.transpose(1, 2)
        q, k = self.A(bs), self.B(be)
        if self.form == "dot":
            if self.pos == "rotary":
                q, k = rotary(q, positions), rotary(k, positions)
            return q @ k.transpose(1, 2)
        n = bs.size(1)
        out = torch.zeros(bs.size(0), n, n, dtype=torch.float32, device=bs.device)
        for d in range(min(self.max_len, n)):                               # span length d + 1: (i, i + d)
            out.diagonal(d, 1, 2).copy_(self.mlp_term(q[:, :n - d], k[:, d:]))
        return out

    def mlp_term(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """w' LN(relu(q + k)) over the last dimension, in fp32: Fajcik et al.'s f_MLP, whose
        sigma is layer normalization of a ReLU, without its constant b2 (a softmax ignores it)."""
        return self.norm(torch.relu(q.float() + k.float())) @ self.w.float()

    def pair_null(self, cs: torch.Tensor, ce: torch.Tensor) -> torch.Tensor:
        """(W,) pair term of the null (CLS, CLS), without position terms, summed in fp32."""
        if self.form == "full":
            return ((cs @ self.W).float() * ce.float()).sum(-1)
        if self.form == "diag":
            return (cs.float() * self.w.float() * ce.float()).sum(-1)
        if self.form == "dot":
            return (self.A(cs).float() * self.B(ce).float()).sum(-1)
        return self.mlp_term(self.A(cs), self.B(ce))

    def forward(self, x_first: torch.Tensor, x_last: torch.Tensor, word_mask: torch.Tensor,
                x_cls: torch.Tensor | None = None, positions: torch.Tensor | None = None) -> dict:
        """As StandardSpanHead.forward, plus "span": (W, n, n) joint scores, -inf where invalid, and
        with x_cls "null": (W,) the joint score of the span (CLS, CLS), without position terms.
        positions: (W, n) passage word index of each window word; needed with bilinear_pos."""
        hs, he = self.start(x_first), self.end(x_last)
        start = self.u(hs).squeeze(-1).float()
        end = self.v(he).squeeze(-1).float()
        if self.pos != "none" and positions is None:
            raise ValueError(f"bilinear_pos {self.pos!r} needs the word positions")
        bs, be = hs, he
        if self.pos == "additive":
            pe = sinusoidal(positions, hs.size(-1))
            bs, be = hs + pe, he + pe
        span = self.pair(bs, be, positions).float() + start[:, :, None] + end[:, None, :]
        if self.len_bias is not None:
            idx = torch.arange(span.size(-1), device=span.device)
            span = span + self.len_bias[(idx[None, :] - idx[:, None]).clamp(0, self.max_len - 1)]
        out = {"start": start.masked_fill(~word_mask, NEG_INF), "end": end.masked_fill(~word_mask, NEG_INF),
               "span": valid_spans(span, word_mask, self.max_len)}
        if x_cls is not None:
            cs, ce = self.start(x_cls), self.end(x_cls)
            out["start_null"] = self.u(cs).squeeze(-1).float()
            out["end_null"] = self.v(ce).squeeze(-1).float()
            out["null"] = self.pair_null(cs, ce) + out["start_null"] + out["end_null"]
        return out

    def span_scores(self, out: dict, word_mask: torch.Tensor) -> torch.Tensor:
        return out["span"]

    def null_score(self, out: dict) -> torch.Tensor | None:
        return out.get("null")


class ConditionalSpanHead(nn.Module):
    """Start logit u.h_i; end given start i: w.tanh(A x_last[j] + B x_first[i] + c) (Fajcik et al.'s
    conditional objective; the null option is the start CLS, whose only end is CLS)."""

    def __init__(self, d_model: int, d_h: int = 256, max_len: int = 30, dropout: float = 0.1):
        super().__init__()
        self.start = _mlp(d_model, d_h, dropout)
        self.u = nn.Linear(d_h, 1)
        self.end_proj = nn.Linear(d_model, d_h)                   # A x_last[j] + c
        self.start_proj = nn.Linear(d_model, d_h, bias=False)     # B x_first[i]
        self.drop = nn.Dropout(dropout)
        self.w = nn.Linear(d_h, 1)
        self.max_len = max_len

    def forward(self, x_first: torch.Tensor, x_last: torch.Tensor, word_mask: torch.Tensor,
                x_cls: torch.Tensor | None = None, positions: torch.Tensor | None = None) -> dict:
        """Start logits (W, n), -inf on padding, the end and start projections (W, n, d_h), and
        with x_cls the null's start logit (W,)."""
        start = self.u(self.start(x_first)).squeeze(-1).float()
        out = {"start": start.masked_fill(~word_mask, NEG_INF), "a": self.end_proj(x_last), "b": self.start_proj(x_first)}
        if x_cls is not None:
            out["start_null"] = self.u(self.start(x_cls)).squeeze(-1).float()
        return out

    def end_logits(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Unnormalized end logits w.tanh(a + b) for aligned (..., d_h) projections -> (...)."""
        return self.w(self.drop(torch.tanh(a + b))).squeeze(-1).float()

    def gold_end_logits(self, out: dict, gold_start: torch.Tensor, word_mask: torch.Tensor) -> torch.Tensor:
        """(R, n) end logits given each row's start (R,), -inf unless j is a word, j >= start and
        j - start + 1 <= max_len. out holds R windows."""
        rows = torch.arange(len(gold_start), device=gold_start.device)
        e = self.end_logits(out["a"], out["b"][rows, gold_start][:, None, :])
        j = torch.arange(e.size(-1), device=e.device)[None, :]
        ok = (j >= gold_start[:, None]) & (j - gold_start[:, None] < self.max_len) & word_mask
        return e.masked_fill(~ok, NEG_INF)

    def span_scores(self, out: dict, word_mask: torch.Tensor) -> torch.Tensor:
        """(W, n, n) scores s(i, j) = start_i + log p(j | i): the softmax over a question's spans
        (and null) is then p(i) p(j | i) exactly. Computed one diagonal (span length) at a time."""
        a, b = out["a"], out["b"]
        n = a.size(1)
        cond = torch.full((a.size(0), n, n), NEG_INF, device=a.device)
        for k in range(min(self.max_len, n)):
            cond.diagonal(k, 1, 2).copy_(self.end_logits(a[:, k:], b[:, :n - k]))
        cond = valid_spans(cond, word_mask, self.max_len)
        has_end = torch.isfinite(cond).any(-1, keepdim=True)          # False for padding starts
        logp_end = torch.where(has_end, cond.masked_fill(~has_end, 0.0).log_softmax(-1), cond)
        return valid_spans(out["start"][:, :, None] + logp_end, word_mask, self.max_len)

    def null_score(self, out: dict) -> torch.Tensor | None:
        return out.get("start_null")

    def loss(self, out: dict, batch: dict, n_examples: int) -> torch.Tensor:
        """-log p(start*) - log p(end* | start*), one softmax per question (spec §3.4); the end term
        uses the windows that hold the answer, each conditioned on its own copy of the gold start."""
        start = with_null(out["start"], out.get("start_null"))
        nll = question_nll(start, batch["gold_start"], batch["example"], n_examples)
        rows = torch.nonzero(batch["gold_start"] >= 0).squeeze(1)
        sub = {"a": out["a"][rows], "b": out["b"][rows]}
        end = self.gold_end_logits(sub, batch["gold_start"][rows], batch["word_mask"][rows])
        return nll + question_nll(end, batch["gold_end"][rows], batch["example"][rows], n_examples)


def valid_spans(scores: torch.Tensor, word_mask: torch.Tensor, max_len: int) -> torch.Tensor:
    n = scores.size(-1)
    idx = torch.arange(n, device=scores.device)
    length = idx[None, :] - idx[:, None] + 1
    valid = (length >= 1) & (length <= max_len)
    mask = valid[None] & word_mask[:, :, None] & word_mask[:, None, :]
    return scores.masked_fill(~mask, NEG_INF)


def with_null(logits: torch.Tensor, null: torch.Tensor | None) -> torch.Tensor:
    """(W, m) logits with the null logit (W,) appended as column m, or unchanged without one."""
    return logits if null is None else torch.cat([logits, null[:, None]], dim=1)


def _segment_logsumexp(values: torch.Tensor, segment: torch.Tensor, n_segments: int) -> torch.Tensor:
    """logsumexp of values (W,) within each segment id (W,) -> (n_segments,); -inf if all -inf."""
    m = torch.full((n_segments,), NEG_INF, device=values.device, dtype=values.dtype)
    m = m.scatter_reduce(0, segment, values, reduce="amax", include_self=True)
    safe_m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    total = torch.zeros(n_segments, device=values.device, dtype=values.dtype)
    total = total.index_add(0, segment, torch.exp(values - safe_m[segment]))
    return torch.log(total) + safe_m


def question_nll(logits: torch.Tensor, gold: torch.Tensor, example: torch.Tensor, n_examples: int) -> torch.Tensor:
    """Negative log-likelihood per question, with one softmax over all its windows' words.

    logits: (W, n) word logits, -inf on padding; gold: (W,) local gold word index in each
    window, -1 where the window does not wholly hold the answer; example: (W,) the window's
    question. Every window copy of the gold word counts as correct (spec §3.4).
    Returns (n_examples,).
    """
    all_lse = _segment_logsumexp(torch.logsumexp(logits, dim=-1), example, n_examples)
    gold_logit = logits.gather(1, gold.clamp(min=0)[:, None]).squeeze(1)
    gold_logit = torch.where(gold >= 0, gold_logit, torch.full_like(gold_logit, NEG_INF))
    gold_lse = _segment_logsumexp(gold_logit, example, n_examples)
    if not torch.isfinite(gold_lse).all():
        raise RuntimeError("a training question has no window holding its answer")
    return all_lse - gold_lse


def independent_nll(out: dict, batch: dict, n_examples: int) -> torch.Tensor:
    """Start NLL + end NLL (the standard objective, and the compound's auxiliary term), with the
    null's start and end logits in the softmaxes when the head has them."""
    start = with_null(out["start"], out.get("start_null"))
    end = with_null(out["end"], out.get("end_null"))
    return (question_nll(start, batch["gold_start"], batch["example"], n_examples)
            + question_nll(end, batch["gold_end"], batch["example"], n_examples))


def joint_nll(out: dict, batch: dict, n_examples: int) -> torch.Tensor:
    """-log p(gold span) under one softmax over every valid span of the question's windows (and
    the null): the (W, n, n) scores are flattened, and span (i, j) is column i * n + j."""
    span = out["span"]
    n = span.size(-1)
    gold = torch.where(batch["gold_start"] >= 0, batch["gold_start"] * n + batch["gold_end"], -1)
    return question_nll(with_null(span.flatten(1), out.get("null")), gold, batch["example"], n_examples)


def gold_group(span: torch.Tensor, batch: dict) -> torch.Tensor:
    """(W, n, n) bool: the valid spans of each window in its question's gold group (target_unit group,
    spec §4.3), the spans that cover the gold's content words and only those: the boundary variants,
    grouped as decode.group_keys does. Worked out per window from the gold's content core, so a trimmed
    variant also counts in a window that cuts off the rest of the gold. A gold without a content word
    is its own group."""
    content = batch["content"]
    n = content.size(1)
    idx = torch.arange(n, device=content.device)
    next_c = torch.where(content, idx, n).flip(-1).cummin(-1).values.flip(-1)     # first content word >= i
    prev_c = torch.where(content, idx, -1).cummax(-1).values                       # last content word <= j
    cs, ce = batch["core_start"][:, None, None], batch["core_end"][:, None, None]
    member = (next_c[:, :, None] == cs) & (prev_c[:, None, :] == ce) & (cs >= 0)
    gs, ge = batch["gold_start"][:, None, None], batch["gold_end"][:, None, None]
    alone = (idx[None, :, None] == gs) & (idx[None, None, :] == ge)
    no_core = ((batch["core_start"] < 0) & (batch["gold_start"] >= 0))[:, None, None]
    return torch.where(no_core, alone, member) & torch.isfinite(span)


def group_nll(out: dict, batch: dict, n_examples: int) -> torch.Tensor:
    """-log p(G*) (target_unit group): the gold group's summed probability under one softmax over every
    valid span of the question's windows (and the null). Copies in overlapping windows add up, as
    question_nll does for the gold span."""
    span = out["span"]
    member = gold_group(span, batch)
    all_lse = _segment_logsumexp(torch.logsumexp(with_null(span.flatten(1), out.get("null")), -1),
                                 batch["example"], n_examples)
    gold = torch.logsumexp(span.masked_fill(~member, NEG_INF).flatten(1), -1)
    gold_lse = _segment_logsumexp(gold, batch["example"], n_examples)
    if not torch.isfinite(gold_lse).all():
        raise RuntimeError("a training question has no valid span in its gold group")
    return all_lse - gold_lse


# F1-weighted targets and the risk loss (milestone 1a; spec §4.3).

def gold_f1_em(span: torch.Tensor, batch: dict, n_examples: int,
               counted: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(W, n, n) F1 and EM of every window span against its question's gold span, positional over the
    words flagged in counted (W, n): the content words for loss_utility normalized, every word for
    positional; decode.span_f1_em is the same function on one passage. Worked out in passage
    coordinates from the windows that hold the gold, so a span in a window that cuts the gold still
    gets its partial credit. A span with no counted word matches only the identical span."""
    W, n = counted.shape
    dev = counted.device
    ex, fw, gs, ge = batch["example"], batch["first_word"], batch["gold_start"], batch["gold_end"]
    prefix = torch.cat([torch.zeros(W, 1, dtype=torch.long, device=dev), counted.long().cumsum(-1)], -1)
    rows = torch.nonzero(gs >= 0).squeeze(1)
    per_q = lambda v: torch.full((n_examples,), -1, dtype=torch.long, device=dev).scatter_reduce(
        0, ex[rows], v, "amax")
    g_start, g_end = per_q(fw[rows] + gs[rows]), per_q(fw[rows] + ge[rows])
    g_len = per_q(prefix[rows, ge[rows] + 1] - prefix[rows, gs[rows]])
    has = (g_start[ex] >= 0)[:, None, None]
    s_l, e_l = (g_start[ex] - fw)[:, None, None], (g_end[ex] - fw)[:, None, None]   # may lie outside the window
    lg = g_len[ex][:, None, None]
    i = torch.arange(n, device=dev)[None, :, None]
    j = torch.arange(n, device=dev)[None, None, :]
    ls = prefix[:, None, 1:] - prefix[:, :n, None]                                   # counted words in (i, j)
    lo, hi = torch.maximum(i, s_l), torch.minimum(j, e_l)
    take = lambda k: prefix.gather(1, k.clamp(0, n).expand(W, n, n).reshape(W, -1)).view(W, n, n)
    overlap = torch.where(lo <= hi, take(hi + 1) - take(lo), 0)
    identical = (i == s_l) & (j == e_l) & has
    f1 = torch.where(has & (ls + lg > 0), 2.0 * overlap / (ls + lg).clamp(min=1), 0.0)
    em = (overlap == ls) & (overlap == lg) & (lg > 0) & has
    return torch.where(identical, 1.0, f1), (em | identical).float()


def _question_lse(out: dict, batch: dict, n_examples: int) -> torch.Tensor:
    """(n_examples,) log of each question's softmax normalizer: every valid span of its windows and the null."""
    return _segment_logsumexp(torch.logsumexp(with_null(out["span"].flatten(1), out.get("null")), -1),
                              batch["example"], n_examples)


def f1_target_nll(out: dict, batch: dict, n_examples: int, temperature: float, floor: float, unit: str,
                  counted: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The F1 part of the soft target, −Σ_u t_F1(u) · log p(u) per question, and the expected F1 under
    t_F1 (spec §4.3). u ranges over distinct passage spans (target_unit span) or groups of boundary
    variants (group); p(u) sums the probabilities of u's spans in every window of the question, and
    t_F1(u) ∝ exp(−(1 − F1)/τ) on units with F1 > 0 and F1 ≥ φ, each unit counted once."""
    span = out["span"]
    ex, fw = batch["example"], batch["first_word"]
    f1, _ = gold_f1_em(span, batch, n_examples, counted)
    r, i, j = torch.nonzero(torch.isfinite(span) & (f1 > 0) & (f1 >= floor), as_tuple=True)
    logp = span[r, i, j].float() - _question_lse(out, batch, n_examples)[ex[r]]
    if unit == "group":
        n = counted.size(1)
        idx = torch.arange(n, device=counted.device)
        next_c = torch.where(counted, idx, n).flip(-1).cummin(-1).values.flip(-1)
        prev_c = torch.where(counted, idx, -1).cummax(-1).values
        a, b = next_c[r, i], prev_c[r, j]
        own = a > b                                     # no content word: a group of its own
        a, b, kind = torch.where(own, i, a), torch.where(own, j, b), own.long()
    else:
        a, b, kind = i, j, torch.zeros_like(i)
    big = int(fw.max()) + counted.size(1) + 1
    key = ((ex[r] * big + fw[r] + a) * big + fw[r] + b) * 2 + kind
    units, inv = torch.unique(key, return_inverse=True)
    m = len(units)
    unit_max = lambda v: torch.zeros(m, dtype=v.dtype, device=v.device).scatter_reduce(
        0, inv, v, "amax", include_self=False)
    logp_u = _segment_logsumexp(logp, inv, m)
    f1_u = unit_max(f1[r, i, j].float())                     # the same for every span of a unit
    ex_u = unit_max(ex[r])
    w_u = torch.exp(-(1 - f1_u) / temperature)
    z = torch.zeros(n_examples, device=w_u.device).index_add(0, ex_u, w_u)
    if (z[torch.unique(ex)] <= 0).any():
        raise RuntimeError("a training question has no span with F1 credit")
    t = w_u / z[ex_u]
    loss = torch.zeros(n_examples, device=t.device).index_add(0, ex_u, -t * logp_u)
    expected_f1 = torch.zeros(n_examples, device=t.device).index_add(0, ex_u, t * f1_u)
    return loss, expected_f1


def risk_loss(out: dict, batch: dict, n_examples: int, em_weight: float, counted: torch.Tensor) -> torch.Tensor:
    """Σ_y p(y) · (ℓ_F1 + γ_r · ℓ_EM) per question over every valid span of its windows (spec §4.3). The
    null is never a correct answer, so it adds p(null) · (1 + γ_r). Copies of a span carry the same loss,
    so the sum over window entries is the sum over passage spans."""
    span = out["span"]
    ex = batch["example"]
    f1, em = gold_f1_em(span, batch, n_examples, counted)
    lse = _question_lse(out, batch, n_examples)
    valid = torch.isfinite(span)
    p = torch.exp(torch.where(valid, span.float() - lse[ex][:, None, None], NEG_INF))
    cost = torch.where(valid, (1 - f1) + em_weight * (1 - em), 0.0)
    risk = torch.zeros(n_examples, device=span.device).index_add(0, ex, (p * cost).flatten(1).sum(-1))
    if out.get("null") is not None:
        risk = risk.index_add(0, ex, torch.exp(out["null"].float() - lse[ex]) * (1 + em_weight))
    return risk


def check_head_config(cfg: dict) -> None:
    """Raise on a head setting that is not built or does not apply (spec rule 7: nothing ignored)."""
    if cfg["span_head"] not in HEADS:
        raise ValueError(f"unknown span_head {cfg['span_head']!r}; choose from {HEADS}")
    if cfg["target_unit"] not in ("span", "group"):
        raise ValueError(f"unknown target_unit {cfg['target_unit']!r}; choose span or group")
    if cfg["target_unit"] == "group" and (cfg["loss_utility"] != "normalized" or cfg["span_head"] not in BILINEAR):
        raise ValueError("target_unit group pools boundary variants: it needs loss_utility normalized and a "
                         f"span softmax, span_head in {BILINEAR}")
    if cfg["risk_em_weight"] < 0 or (cfg["risk_em_weight"] and not cfg["risk_share"]):
        raise ValueError("risk_em_weight is the EM weight inside the risk loss: it must be >= 0, "
                         "and above 0 only with risk_share > 0")
    alpha, tau, floor, beta = (cfg["f1_target_weight"], cfg["f1_target_temperature"], cfg["f1_target_floor"],
                               cfg["risk_share"])
    if not (0 <= alpha <= 1 and tau > 0 and 0 <= floor <= 1 and 0 <= beta <= 1):
        raise ValueError("need 0 <= f1_target_weight <= 1, f1_target_temperature > 0, 0 <= f1_target_floor <= 1 "
                         "and 0 <= risk_share <= 1")
    if alpha and beta == 1:
        raise ValueError("risk_share 1 is the risk loss alone, so f1_target_weight would be unused: set it to 0")
    if (alpha or beta) and cfg["span_head"] not in BILINEAR:
        raise ValueError(f"F1-weighted targets and the risk loss need a span softmax: span_head in {BILINEAR}")
    if cfg["span_head"] in BILINEAR and cfg["W_init"] != "zeros":
        raise NotImplementedError(f"W_init {cfg['W_init']!r} is not built; only zeros")
    if cfg["bilinear_form"] not in PAIR_FORMS or cfg["bilinear_pos"] not in ("none", "additive", "rotary"):
        raise ValueError(f"unknown bilinear_form {cfg['bilinear_form']!r} or bilinear_pos {cfg['bilinear_pos']!r}")
    if cfg["bilinear_pos"] == "rotary" and cfg["bilinear_form"] not in ROTARY_FORMS:
        raise ValueError(f"bilinear_pos rotary rotates the pair term's inputs: bilinear_form in {ROTARY_FORMS}, "
                         f"not {cfg['bilinear_form']!r}")
    if cfg["span_head"] not in BILINEAR and (cfg["bilinear_pos"] != "none" or cfg["bilinear_form"] != "full"):
        raise ValueError(f"bilinear_pos and bilinear_form need a bilinear head {BILINEAR}, not {cfg['span_head']!r}")


class Reader(nn.Module):
    """Encoder + span head. forward(batch) returns the head's outputs; loss(batch, n) the
    per-question training loss; span_scores(batch) the (W, n, n) span scores and the (W,) null
    scores (None without no_answer) that squadqa.predict decodes."""

    def __init__(self, encoder: nn.Module, cfg: dict):
        super().__init__()
        check_head_config(cfg)
        self.encoder = encoder
        self.kind = cfg["span_head"]
        self.no_answer = cfg["no_answer"]
        self.aux_weight = cfg["aux_independent_weight"]
        self.target_unit = cfg["target_unit"]
        self.alpha, self.tau, self.floor = cfg["f1_target_weight"], cfg["f1_target_temperature"], cfg["f1_target_floor"]
        self.beta, self.risk_em_weight, self.loss_utility = cfg["risk_share"], cfg["risk_em_weight"], cfg["loss_utility"]
        self.stats: dict[str, float] = {}               # the last batch's training checks, for the log
        args = (encoder.config.hidden_size, cfg["head_dim"], cfg["max_answer_words"], cfg["train"]["head_dropout"])
        if self.kind == "standard":
            self.head = StandardSpanHead(*args)
        elif self.kind == "conditional":
            self.head = ConditionalSpanHead(*args)
        else:
            self.head = JointSpanHead(*args, length_bias=cfg["length_bias"], pos=cfg["bilinear_pos"],
                                      form=cfg["bilinear_form"], rank=cfg["bilinear_rank"])

    def forward(self, batch: dict) -> dict:
        x = self.encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
        d = x.size(-1)
        x_first = x.gather(1, batch["first_tok"][:, :, None].expand(-1, -1, d))
        x_last = x.gather(1, batch["last_tok"][:, :, None].expand(-1, -1, d))
        x_cls = x[:, 0] if self.no_answer else None          # [CLS]: pair_layout puts it first for DeBERTa-v3
        n = batch["word_mask"].size(1)
        positions = batch["first_word"][:, None] + torch.arange(n, device=x.device)[None, :]
        return self.head(x_first, x_last, batch["word_mask"], x_cls, positions)

    def loss(self, batch: dict, n_examples: int) -> torch.Tensor:
        """Per-question loss (n_examples,) of the span_head's objective."""
        out = self(batch)
        if self.kind == "standard":
            return independent_nll(out, batch, n_examples)
        if self.kind == "conditional":
            return self.head.loss(out, batch, n_examples)
        # spec §4.3: ℒ = (1 − β) · ℒ_soft + β · ℒ_risk on the joint term; β = 1 is the risk loss alone
        counted = batch["content"] if self.loss_utility == "normalized" else batch["word_mask"]
        joint = 0.0
        if self.beta < 1:
            joint = (group_nll if self.target_unit == "group" else joint_nll)(out, batch, n_examples)
            if self.alpha:
                f1_part, expected_f1 = f1_target_nll(out, batch, n_examples, self.tau, self.floor,
                                                     self.target_unit, counted)
                joint = (1 - self.alpha) * joint + self.alpha * f1_part
                self.stats["t_f1_expected_f1"] = float(expected_f1.mean())
        if self.beta:
            joint = (1 - self.beta) * joint + self.beta * risk_loss(out, batch, n_examples, self.risk_em_weight,
                                                                    counted)
        return joint if self.kind == "joint" else joint + self.aux_weight * independent_nll(out, batch, n_examples)

    def span_scores(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor | None]:
        out = self(batch)
        return self.head.span_scores(out, batch["word_mask"]), self.head.null_score(out)
