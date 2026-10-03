"""Span heads, the reader model and their losses (spec §4).

Every head scores word spans: it reads the encoder state at each word's first subword (for
starts) and last subword (for ends), using first_tok / last_tok from squadqa.data.collate.

- standard (ladder row 0): h_i = MLP_start(x[first_tok[i]]), h_j = MLP_end(x[last_tok[j]]),
  start logit u.h_i, end logit v.h_j, span score s(i, j) = u.h_i + v.h_j. It is the joint
  head of spec §4.1 with W = 0 and b = 0, so row 0 -> row 1 changes only the head.
- joint and compound (rows 1+, issue 1b) are not built yet; asking for them raises.

The loss puts all of a question's windows into one softmax (spec §3.4): see question_nll.
"""

import torch
import torch.nn as nn

NEG_INF = float("-inf")


class StandardSpanHead(nn.Module):
    def __init__(self, d_model: int, d_h: int = 256, max_len: int = 30, dropout: float = 0.1):
        super().__init__()
        self.start = nn.Sequential(nn.Linear(d_model, d_h), nn.GELU(), nn.Dropout(dropout))
        self.end = nn.Sequential(nn.Linear(d_model, d_h), nn.GELU(), nn.Dropout(dropout))
        self.u = nn.Linear(d_h, 1)
        self.v = nn.Linear(d_h, 1)
        self.max_len = max_len

    def forward(self, x_first: torch.Tensor, x_last: torch.Tensor, word_mask: torch.Tensor) -> dict:
        """x_first, x_last: (W, n, d_model); word_mask: (W, n) bool.
        Returns start and end logits (W, n), -inf on padding."""
        start = self.u(self.start(x_first)).squeeze(-1).float()
        end = self.v(self.end(x_last)).squeeze(-1).float()
        return {"start": start.masked_fill(~word_mask, NEG_INF), "end": end.masked_fill(~word_mask, NEG_INF)}

    def span_scores(self, out: dict, word_mask: torch.Tensor) -> torch.Tensor:
        """(W, n, n) scores s(i, j) = start_i + end_j, -inf unless both are words, i <= j and
        j - i + 1 <= max_len (spec §4.1 valid set)."""
        return valid_spans(out["start"][:, :, None] + out["end"][:, None, :], word_mask, self.max_len)


def valid_spans(scores: torch.Tensor, word_mask: torch.Tensor, max_len: int) -> torch.Tensor:
    n = scores.size(-1)
    idx = torch.arange(n, device=scores.device)
    length = idx[None, :] - idx[:, None] + 1
    valid = (length >= 1) & (length <= max_len)
    mask = valid[None] & word_mask[:, :, None] & word_mask[:, None, :]
    return scores.masked_fill(~mask, NEG_INF)


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


class Reader(nn.Module):
    """Encoder + span head. forward(batch) returns the head's logits; span_scores(batch) the
    (W, n, n) span scores that squadqa.predict decodes."""

    def __init__(self, encoder: nn.Module, cfg: dict):
        super().__init__()
        if cfg["span_head"] != "standard":
            raise NotImplementedError(f"span_head {cfg['span_head']!r} is not built yet (issue 1b); "
                                      "use configs/experiments/row0-baseline.yaml or --set span_head=standard")
        self.encoder = encoder
        self.head = StandardSpanHead(encoder.config.hidden_size, cfg["head_dim"], cfg["max_answer_words"],
                                     cfg["train"]["head_dropout"])

    def forward(self, batch: dict) -> dict:
        x = self.encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
        d = x.size(-1)
        x_first = x.gather(1, batch["first_tok"][:, :, None].expand(-1, -1, d))
        x_last = x.gather(1, batch["last_tok"][:, :, None].expand(-1, -1, d))
        return self.head(x_first, x_last, batch["word_mask"])

    def loss(self, batch: dict, n_examples: int) -> torch.Tensor:
        """Per-question loss (n_examples,): start NLL + end NLL (the standard objective)."""
        out = self(batch)
        return (question_nll(out["start"], batch["gold_start"], batch["example"], n_examples)
                + question_nll(out["end"], batch["gold_end"], batch["example"], n_examples))

    def span_scores(self, batch: dict) -> torch.Tensor:
        return self.head.span_scores(self(batch), batch["word_mask"])
