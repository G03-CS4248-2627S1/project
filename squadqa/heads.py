"""Span heads, the reader model and their losses (spec §4).

Every head scores word spans: it reads the encoder state at each word's first subword (for
starts) and last subword (for ends), using first_tok / last_tok from squadqa.data.collate.
span_head picks the training objective, one of the four of Fajcik et al. (2021, "Rethinking the
Objectives of Extractive Question Answering", MRQA; their names in brackets):

- standard (I, independent; ladder row 0): h_i = MLP_start(x[first_tok[i]]), h_j = MLP_end(x[last_tok[j]]),
  start logit u.h_i, end logit v.h_j, span score s(i, j) = u.h_i + v.h_j. Loss: start NLL + end NLL.
- joint (J): s(i, j) = pair(h_i, h_j) + u.h_i + v.h_j + b[j - i] (spec §4.1), one softmax over
  all valid spans. Loss: span NLL. With the pair term and b at zero it is the standard head.
- compound (I+J; ladder row 1): the joint loss + aux_independent_weight x the standard loss on the
  same u.h_i and v.h_j. Decoding uses the joint scores.
- conditional (JC): the standard start logit, and an end distribution given the start i,
  p(j | i) = softmax_j of w.tanh(A x[last_tok[j]] + B x[first_tok[i]] + c) (Fajcik et al.'s form,
  with head_dim hidden units). Training conditions on the gold start (teacher forcing); decoding
  scores every valid span exactly, s(i, j) = start logit + log p(j | i), not with their top-k beam.

Options of the bilinear heads (joint, compound):
- bilinear_form: the pair term. full = h_i' W h_j' with W zero at init (the biaffine scorer of
  Yu et al., 2020); dot = (A h_i')(B h_j'), A and B of rank bilinear_rank (the dot product Fajcik et
  al. found best for BERT-type models), B zero at init so the gradient reaches both.
- length_bias: the learned bonus b[j - i], zero at init.
- bilinear_pos: position information in the pair term only; u.h_i and v.h_j see h itself, so the
  head still starts as the standard model. additive: sinusoidal encodings of each word's position
  in the passage (Vaswani et al., 2017) added right before the pair term, h' = h + PE. rotary: the
  dot form's A h_i and B h_j rotated by position (RoPE, as in GlobalPointer, Su et al., 2022), so the
  pair term depends on j - i only.

no_answer adds a null option, the span (CLS, CLS) as in Fajcik et al.'s and BERT's SQuAD 2.0
models, to every softmax: one null per window, and a question's no-answer probability is the sum
over its windows (question_nll treats the copies like gold copies). Training data never targets
it on SQuAD v1.1, and decoding never returns it (spec rule 5); it only takes probability mass.

The loss puts all of a question's windows into one softmax (spec §3.4): see question_nll.
"""

import torch
import torch.nn as nn

NEG_INF = float("-inf")
HEADS = ("standard", "joint", "compound", "conditional")
BILINEAR = ("joint", "compound")


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
    h_i' W h_j' (form full) or (A h_i')(B h_j') (form dot), h' = h (+ PE), rotated for rotary."""

    def __init__(self, d_model: int, d_h: int = 256, max_len: int = 30, dropout: float = 0.1,
                 length_bias: bool = True, pos: str = "none", form: str = "full", rank: int = 64):
        super().__init__()
        if pos not in ("none", "additive", "rotary") or form not in ("full", "dot") or (pos, form) == ("rotary", "full"):
            raise ValueError(f"unsupported pair term: bilinear_form {form!r} with bilinear_pos {pos!r}")
        self.start = _mlp(d_model, d_h, dropout)
        self.end = _mlp(d_model, d_h, dropout)
        if form == "full":
            self.W = nn.Parameter(torch.zeros(d_h, d_h))                     # zero init (W_init: zeros)
        else:
            self.A = nn.Linear(d_h, rank, bias=False)
            self.B = nn.Linear(d_h, rank, bias=False)
            nn.init.zeros_(self.B.weight)                                   # zero pair term, nonzero gradient
        self.u = nn.Linear(d_h, 1)
        self.v = nn.Linear(d_h, 1)
        self.len_bias = nn.Parameter(torch.zeros(max_len)) if length_bias else None   # b[length - 1]
        self.pos, self.form = pos, form
        self.max_len = max_len

    def pair(self, bs: torch.Tensor, be: torch.Tensor, positions: torch.Tensor | None) -> torch.Tensor:
        """(W, n, n) pair terms of start features bs and end features be (W, n, d_h)."""
        if self.form == "full":
            return (bs @ self.W) @ be.transpose(1, 2)
        q, k = self.A(bs), self.B(be)
        if self.pos == "rotary":
            q, k = rotary(q, positions), rotary(k, positions)
        return q @ k.transpose(1, 2)

    def pair_null(self, cs: torch.Tensor, ce: torch.Tensor) -> torch.Tensor:
        """(W,) pair term of the null (CLS, CLS), without position terms, summed in fp32."""
        if self.form == "full":
            return ((cs @ self.W).float() * ce.float()).sum(-1)
        return (self.A(cs).float() * self.B(ce).float()).sum(-1)

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


def check_head_config(cfg: dict) -> None:
    """Raise on a head setting that is not built or does not apply (spec rule 7: nothing ignored)."""
    if cfg["span_head"] not in HEADS:
        raise ValueError(f"unknown span_head {cfg['span_head']!r}; choose from {HEADS}")
    if cfg["f1_target_weight"] or cfg["risk_weight"]:
        raise NotImplementedError("F1-weighted targets and the risk loss (milestone 1a) are not built yet; "
                                  "set f1_target_weight=0 and risk_weight=0")
    if cfg["span_head"] in BILINEAR and cfg["W_init"] != "zeros":
        raise NotImplementedError(f"W_init {cfg['W_init']!r} is not built; only zeros")
    if cfg["bilinear_form"] not in ("full", "dot") or cfg["bilinear_pos"] not in ("none", "additive", "rotary"):
        raise ValueError(f"unknown bilinear_form {cfg['bilinear_form']!r} or bilinear_pos {cfg['bilinear_pos']!r}")
    if cfg["span_head"] not in BILINEAR and (cfg["bilinear_pos"] != "none" or cfg["bilinear_form"] != "full"):
        raise ValueError(f"bilinear_pos and bilinear_form need a bilinear head {BILINEAR}, not {cfg['span_head']!r}")
    if cfg["bilinear_pos"] == "rotary" and cfg["bilinear_form"] != "dot":
        raise ValueError("bilinear_pos rotary keeps the pair term relative only with bilinear_form dot")


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
        joint = joint_nll(out, batch, n_examples)
        return joint if self.kind == "joint" else joint + self.aux_weight * independent_nll(out, batch, n_examples)

    def span_scores(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor | None]:
        out = self(batch)
        return self.head.span_scores(out, batch["word_mask"]), self.head.null_score(out)
