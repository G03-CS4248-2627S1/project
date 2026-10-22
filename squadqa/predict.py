"""Run a reader over a set of questions and write its predictions (spec §3.4, §4.4).

The reader is any score_fn(batch) -> (W, n, n) tensor of span scores, -inf where a span is
invalid, for a batch from squadqa.data.collate, or a (span scores, (W,) null scores) pair for a
reader with a no-answer option (reader_score_fn wraps a squadqa.heads.Reader).

From a trained run's checkpoint (squadqa.train_reader writes the validation outputs itself):
    python -m squadqa.predict --run-id <run_id>                          # validation again
    python -m squadqa.predict --run-id <run_id> --eval-set addsent --final   # stage 5 only
A final-scoring set reuses the temperature fitted on validation (calibration.json).

    spans = score_questions(score_fn, feats, tok.pad_token_id, cfg["max_answer_words"])
    cal = calibrate(feats, spans)                       # validation only; dev reuses its T
    preds, cands = decode_questions(feats, spans, cfg, cal["temperature"])
    write_outputs(run_dir, "validation", preds, cands, cal)

Files in results/<run_id>/, for split = validation (or a final-scoring set):
    predictions_<split>_<rule>.json     {question_id: answer}, one per rule in decode_rules
    candidates_<split>.json             {question_id: [[start_word, end_word, logp], ...]}, top-k
    calibration.json                    the fitted temperature and the NLL before and after
    metrics.json                        {split: {rule: official scores}, ...}, updated per split;
                                        with a no-answer option also no_answer_<split> (no_answer_summary)
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from squadqa import evaluate as E
from squadqa.config import RESULTS, env, write_json
from squadqa.data import Features, collate, featurize, load_split, load_squad, load_tokenizer
from squadqa.decode import (SpanScores, answer_text, combine_windows, content_flags, decode,
                            fit_temperature, group_members, span_logp, temperature_nll, top_k)


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
            scored = score_fn(batch)
        scores, null = scored if isinstance(scored, tuple) else (scored, None)
        scores = scores.float().cpu().numpy()
        null = None if null is None else null.float().cpu().numpy()
        per_example: dict[int, list] = {}
        nulls: dict[int, list[float]] = {}
        for r, (pos, w) in enumerate(rows):
            per_example.setdefault(pos, []).append(
                (scores[r], w.first_word, w.first_tok, w.last_tok, w.ctx_start, w.ctx_end))
            if null is not None:
                nulls.setdefault(pos, []).append(float(null[r]))
        for pos, windows in per_example.items():
            out[ids[pos]] = combine_windows(windows, max_answer_words, nulls.get(pos))
    return out


def calibrate(feats: Features, spans: list[SpanScores], unit: str = "span") -> dict:
    """Fit the temperature on these questions (the validation split) against each question's
    annotated answer: its span, or with unit "group" (target_unit) the summed probability of its
    group of boundary variants, the unit the reader was trained on. Questions whose answer is not a
    valid span (not word-aligned or too long; under group, no valid span in its group) are left out
    and counted. A no-answer option stays in the softmax being fitted."""
    items = []
    for ex, s in zip(feats.examples, spans):
        if unit == "group" and ex.gold is not None:
            g = group_members(content_flags(ex.question.context, ex.context.words), s.starts, s.ends, ex.gold)
            g = g if len(g) else None
        else:
            g = s.index(ex.gold)
        if g is not None:
            items.append((s.with_null(), g))
    t = fit_temperature(items)
    return {"temperature": t, "nll_at_1": temperature_nll(items, 1.0), "nll": temperature_nll(items, t),
            "questions": len(items), "left_out": len(spans) - len(items), "unit": unit}


def no_answer_summary(spans: list[SpanScores], temperature: float) -> dict | None:
    """How much probability the no-answer option takes at temperature T: its mean, median and
    maximum, and the share of questions where it is over 0.5 or above the most probable span
    (where a reader allowed to abstain would). None for a reader without the option."""
    if not spans or spans[0].null is None:
        return None
    rows = []
    for s in spans:
        logp, null = span_logp(s, temperature)
        rows.append((np.exp(null), np.exp(logp.max())))
    p, top = np.array(rows).T
    return {"questions": len(p), "mean_p": float(p.mean()), "median_p": float(np.median(p)), "max_p": float(p.max()),
            "share_over_0.5": float((p > 0.5).mean()), "share_above_top_span": float((p > top).mean())}


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
        texts = [answer_text(q.context, ex.context.words, a, b) for a, b in zip(cands.starts, cands.ends)]
        for rule in cfg["decode_rules"]:
            i, j = decode(cands, flags, rule, cfg["mbr_utility"], cfg["mbr_em_weight"], texts)
            preds[rule][q.id] = answer_text(q.context, ex.context.words, i, j)
        cands_out[q.id] = [[int(a), int(b), round(float(lp), 6)]
                           for a, b, lp in zip(cands.starts, cands.ends, cands.logp)]
    return preds, cands_out


def write_outputs(run_dir: Path, split: str, preds: dict[str, dict[str, str]], candidates: dict[str, list] | None,
                  calibration: dict | None = None) -> None:
    for rule, answers in preds.items():
        if any(not a.strip() for a in answers.values()):
            raise RuntimeError(f"an empty {rule} answer; the scorer would read it as 'no answer'")
        write_json(run_dir / f"predictions_{split}_{rule}.json", answers)
    if candidates is not None:
        write_json(run_dir / f"candidates_{split}.json", candidates)
    if calibration is not None:
        write_json(run_dir / "calibration.json", calibration)


def reader_score_fn(model, device: str):
    """score_fn for a squadqa.heads.Reader: eval mode, bf16 autocast on the GPU."""
    import torch

    model.eval()
    use_amp = device.startswith("cuda")

    def score_fn(batch):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            return model.span_scores(batch)
    return score_fn


def predict_split(model, tok, cfg: dict, run_dir: Path, device: str, split: str, questions, dataset,
                  temperature: float | None = None, batch_size: int = 64, save_candidates: bool = True) -> dict:
    """Predict, decode and score one split; fit the temperature when temperature is None
    (validation only). Writes the outputs and returns {split: {rule: scores}, ...}."""
    feats = featurize(questions, tok, cfg, training=False)
    spans = score_questions(reader_score_fn(model, device), feats, tok.pad_token_id, cfg["max_answer_words"],
                            batch_size, device)
    cal = calibrate(feats, spans, cfg["target_unit"]) if temperature is None else None
    t = cal["temperature"] if cal else temperature
    preds, cands = decode_questions(feats, spans, cfg, t)
    write_outputs(run_dir, split, preds, cands if save_candidates else None, cal)
    out = {split: {rule: E.metrics(split, dataset, p) for rule, p in preds.items()}}
    if cal:
        out["calibration"] = cal
    no_answer = no_answer_summary(spans, t)
    if no_answer is not None:
        out[f"no_answer_{split}"] = no_answer
    return out


def predict_validation(model, tok, cfg: dict, run_dir: Path, device: str) -> dict:
    return predict_split(model, tok, cfg, run_dir, device, "validation", load_split("validation"),
                         E.load_eval_set("validation"))


def eval_questions(name: str):
    """The questions of an evaluation set. Load it with E.load_eval_set first, which refuses the
    final-scoring sets without final=True."""
    return load_split("validation") if name == "validation" else load_squad(E.FINAL_SETS[name])


# Config keys added after some runs were trained, with the values that reproduce how those runs
# were trained (the row-0 runs have none of them; at f1_target_weight 0, τ and φ are unused).
ADDED_KEYS = {"f1_target_weight": 0.0, "f1_target_temperature": 0.1, "f1_target_floor": 0.5,
              "risk_weight": 0.0, "risk_em_weight": 0.0, "target_unit": "span",
              "no_answer": False, "length_bias": True, "bilinear_form": "full", "bilinear_rank": 64, "bilinear_pos": "none",
              "window_layout": "left"}
# Config keys renamed after some runs were trained: old name -> current name.
RENAMED_KEYS = {"f1_target": "loss_utility"}


def run_config(run_id: str) -> dict:
    """A trained run's resolved config, with ADDED_KEYS filled in where the run predates them and
    RENAMED_KEYS under their current names."""
    import yaml

    with open(RESULTS / run_id / "config.yaml", encoding="utf-8") as f:
        cfg = {RENAMED_KEYS.get(k, k): v for k, v in yaml.safe_load(f).items()}
    return {**ADDED_KEYS, **cfg}


def load_reader(run_id: str, device: str):
    """A trained run's resolved config, tokenizer and reader with its checkpoint loaded, and the
    checkpoint's step."""
    from squadqa.heads import Reader
    from squadqa.train_reader import CHECKPOINT, load_checkpoint, load_encoder

    cfg = run_config(run_id)
    tok = load_tokenizer(cfg["model"])
    model = Reader(load_encoder(cfg["model"]), cfg).to(device)
    step = load_checkpoint(Path(env("SQUADQA_SCRATCH")) / "runs" / run_id / CHECKPOINT, model)
    return cfg, tok, model, step


def main(argv: list[str] | None = None) -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch

    parser = argparse.ArgumentParser(description="Predict with a trained reader's checkpoint.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--eval-set", default="validation", choices=E.EVAL_SETS)
    parser.add_argument("--final", action="store_true", help="allow dev and the robustness sets (stage 5 only)")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)

    run_dir = RESULTS / args.run_id
    dataset = E.load_eval_set(args.eval_set, args.final)            # raises for dev etc. without --final
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, tok, model, step = load_reader(args.run_id, device)
    if args.eval_set == "validation":
        out = predict_split(model, tok, cfg, run_dir, device, "validation", eval_questions("validation"), dataset,
                            batch_size=args.batch_size)
    else:
        temperature = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))["temperature"]
        out = predict_split(model, tok, cfg, run_dir, device, args.eval_set, eval_questions(args.eval_set),
                            dataset, temperature, args.batch_size)
    path = run_dir / "metrics.json"
    metrics = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    metrics.update(out)
    write_json(path, metrics)
    print(f"step {step}:", json.dumps(out[args.eval_set]), flush=True)


if __name__ == "__main__":
    main()
