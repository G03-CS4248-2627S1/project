"""Run a reader over a set of questions and write its predictions (spec §3.4, §4.4).

The reader is any score_fn(batch) -> (W, n, n) tensor of span scores, -inf where a span is
invalid, for a batch from squadqa.data.collate. Loading a checkpoint into one is the reader's
job (squadqa/train_reader.py, issue #21), which also adds the command-line entry point.

    spans = score_questions(score_fn, feats, tok.pad_token_id, cfg["max_answer_words"])
    cal = calibrate(feats, spans)                       # validation only; dev reuses its T
    preds, cands = decode_questions(feats, spans, cfg, cal["temperature"])
    write_outputs(run_dir, "validation", preds, cands, cal)

Files in results/<run_id>/, for split = validation (or a final-scoring set):
    predictions_<split>_<rule>.json     {question_id: answer}, one per rule in decode_rules
    candidates_<split>.json             {question_id: [[start_word, end_word, logp], ...]}, top-k
    calibration.json                    the fitted temperature and the NLL before and after
"""

from pathlib import Path

import numpy as np

from squadqa.config import write_json
from squadqa.data import Features, collate
from squadqa.decode import (SpanScores, answer_text, combine_windows, content_flags, decode,
                            fit_temperature, temperature_nll, top_k)


def score_questions(score_fn, feats: Features, pad_id: int, max_answer_words: int,
                    batch_size: int = 64, device: str = "cpu") -> list[SpanScores]:
    """SpanScores for every example: its windows' scores in one softmax (combine_windows).
    Batches go by length, so padding stays small; the order does not affect the result."""
    import torch

    order = np.argsort(feats.question_lengths(), kind="stable").tolist()
    out: list[SpanScores | None] = [None] * len(feats.examples)
    for b in range(0, len(order), batch_size):
        ids = order[b:b + batch_size]
        rows = feats.batch(ids)
        batch = {k: v.to(device) for k, v in collate(rows, pad_id).items()}
        with torch.no_grad():
            scores = score_fn(batch).float().cpu().numpy()
        per_example: dict[int, list] = {}
        for r, (pos, w) in enumerate(rows):
            per_example.setdefault(pos, []).append(
                (scores[r], w.first_word, w.first_tok, w.last_tok, w.ctx_start, w.ctx_end))
        for pos, windows in per_example.items():
            out[ids[pos]] = combine_windows(windows, max_answer_words)
    return out


def calibrate(feats: Features, spans: list[SpanScores]) -> dict:
    """Fit the temperature on these questions (the validation split) against each question's
    annotated answer. Questions whose answer is not a valid span (not word-aligned or too long)
    are left out and counted."""
    items = []
    for ex, s in zip(feats.examples, spans):
        g = s.index(ex.gold)
        if g is not None:
            items.append((s.scores, g))
    t = fit_temperature(items)
    return {"temperature": t, "nll_at_1": temperature_nll(items, 1.0), "nll": temperature_nll(items, t),
            "questions": len(items), "left_out": len(spans) - len(items)}


def decode_questions(feats: Features, spans: list[SpanScores], cfg: dict,
                     temperature: float) -> tuple[dict[str, dict[str, str]], dict[str, list]]:
    """Answers for every question under each rule in cfg["decode_rules"], and the top-k
    candidates (cfg["decode_topk"]) with their calibrated log-probabilities."""
    preds: dict[str, dict[str, str]] = {rule: {} for rule in cfg["decode_rules"]}
    cands_out: dict[str, list] = {}
    for ex, s in zip(feats.examples, spans):
        q = ex.question
        flags = content_flags(q.context, ex.context.words)
        cands = top_k(s, cfg["decode_topk"], temperature)
        for rule in cfg["decode_rules"]:
            i, j = decode(cands, flags, rule, cfg["mbr_utility"], cfg["mbr_em_weight"])
            preds[rule][q.id] = answer_text(q.context, ex.context.words, i, j)
        cands_out[q.id] = [[int(a), int(b), round(float(lp), 6)]
                           for a, b, lp in zip(cands.starts, cands.ends, cands.logp)]
    return preds, cands_out


def write_outputs(run_dir: Path, split: str, preds: dict[str, dict[str, str]], candidates: dict[str, list],
                  calibration: dict | None = None) -> None:
    for rule, answers in preds.items():
        if any(not a.strip() for a in answers.values()):
            raise RuntimeError(f"an empty {rule} answer; the scorer would read it as 'no answer'")
        write_json(run_dir / f"predictions_{split}_{rule}.json", answers)
    write_json(run_dir / f"candidates_{split}.json", candidates)
    if calibration is not None:
        write_json(run_dir / "calibration.json", calibration)
