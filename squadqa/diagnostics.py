"""Diagnostics of a reader's span distribution (spec §4.5).

Cross-span mass (plan E3; the early check on the joint head, plan §5). A head that scores a
span's start and end separately cannot tie them together: when it is split between two answers,
it also gives probability to the cross span that runs from the start of the first to the end of
the second. Per question, with p the calibrated probabilities over all valid spans:

    a = the most probable span
    b = the most probable span that does not overlap a and has a content word (spec §4.3)
    relabel the two by position: a = (i1, j1), b = (i2, j2), i1 < i2
    c = (i1, j2) if it is a valid span (at most max_answer_words words), else there is none
    ratio = p(c) / (p(a) + p(b) + p(c)), 0 without c

b needs a content word because a span of only punctuation or articles is no answer, and joining
one to a would give a boundary variant of a rather than a cross span. The summary buckets the
ratio by top-1 probability (< 0.5, 0.5-0.9, > 0.9) and counts how often each decision rule
answers with the cross span (a boundary variant of it counts under mbr_utility normalized),
with the official scores of those answers.

    python -m squadqa.diagnostics --run-id <run_id> [<run_id> ...]           # validation
    python -m squadqa.diagnostics --run-id <run_id> --eval-set addsent --final   # stage 5 only

The temperature comes from the run's calibration.json, fitted on validation. Files in
results/<run_id>/, for split = validation (or a final-scoring set):
    cross_span_<split>.json             the summary
    cross_span_<split>_questions.json   {question_id: [p(top 1), p(a), p(b), p(c)]}, 0 for no b or c
"""

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np

from squadqa import evaluate as E
from squadqa import predict as P
from squadqa.config import RESULTS, git_state, write_json
from squadqa.data import Features, featurize
from squadqa.decode import SpanScores, answer_text, content_flags, decode, log_softmax, span_f1_em, top_k

BUCKETS = ["<0.5", "0.5-0.9", ">0.9"]           # by top-1 probability (spec §4.5)


@dataclass
class CrossSpan:
    a: tuple[int, int]                  # the two spans, a before b in the passage
    b: tuple[int, int] | None           # None if every span with a content word overlaps the top one
    c: tuple[int, int] | None           # (start of a, end of b), None if it is not a valid span
    p_top1: float
    p_a: float
    p_b: float
    p_c: float

    @property
    def ratio(self) -> float:
        """p(c) / (p(a) + p(b) + p(c)); 0 without a cross span."""
        return self.p_c / (self.p_a + self.p_b + self.p_c) if self.c is not None else 0.0


def cross_span(spans: SpanScores, flags: np.ndarray, temperature: float = 1.0) -> CrossSpan:
    """The cross span of one question's span distribution. flags: content_flags of its words."""
    logp = log_softmax(spans.scores, temperature)
    p = np.exp(logp)
    top = int(np.argmax(logp))
    s, e = int(spans.starts[top]), int(spans.ends[top])
    prefix = np.concatenate([[0], np.cumsum(flags, dtype=np.int64)])
    has_content = prefix[spans.ends + 1] - prefix[spans.starts] > 0
    other = np.flatnonzero(((spans.ends < s) | (spans.starts > e)) & has_content)
    if not len(other):
        return CrossSpan((s, e), None, None, float(p[top]), float(p[top]), 0.0, 0.0)
    second = int(other[np.argmax(logp[other])])
    (i1, j1, pa), (i2, j2, pb) = sorted([(s, e, p[top]), (int(spans.starts[second]), int(spans.ends[second]), p[second])])
    k = spans.index((i1, j2))
    return CrossSpan((i1, j1), (i2, j2), None if k is None else (i1, j2), float(p[top]), float(pa), float(pb),
                     0.0 if k is None else float(p[k]))


def is_cross(x: CrossSpan, span: tuple[int, int], flags: np.ndarray, utility: str) -> bool:
    """Whether an answer span is the cross span: the same content words under mbr_utility
    normalized (so boundary variants count), the identical span under positional."""
    if x.c is None:
        return False
    used = flags if utility == "normalized" else np.ones_like(flags)
    return bool(span_f1_em(used, span[0], span[1], x.c[0], x.c[1])[1] == 1.0)


def bucket(p_top1: float) -> str:
    return BUCKETS[0] if p_top1 < 0.5 else BUCKETS[1] if p_top1 <= 0.9 else BUCKETS[2]


def cross_spans(feats: Features, spans: list[SpanScores], cfg: dict, temperature: float
                ) -> tuple[dict[str, CrossSpan], dict[str, dict[str, str]], dict[str, set[str]]]:
    """Per question id: its CrossSpan; per rule in cfg["decode_rules"]: the answers, decoded as
    squadqa.predict does, and the ids whose answer is the cross span."""
    rows: dict[str, CrossSpan] = {}
    preds: dict[str, dict[str, str]] = {rule: {} for rule in cfg["decode_rules"]}
    picks: dict[str, set[str]] = {rule: set() for rule in cfg["decode_rules"]}
    for ex, s in zip(feats.examples, spans):
        q = ex.question
        flags = content_flags(q.context, ex.context.words)
        x = rows[q.id] = cross_span(s, flags, temperature)
        cands = top_k(s, cfg["decode_topk"], temperature)
        for rule in cfg["decode_rules"]:
            i, j = decode(cands, flags, rule, cfg["mbr_utility"], cfg["mbr_em_weight"])
            preds[rule][q.id] = answer_text(q.context, ex.context.words, i, j)
            if is_cross(x, (i, j), flags, cfg["mbr_utility"]):
                picks[rule].add(q.id)
    return rows, preds, picks


def ratio_stats(xs: list[CrossSpan]) -> dict:
    if not xs:
        return {"questions": 0}
    r = np.array([x.ratio for x in xs])
    return {"questions": len(xs), "mean": float(r.mean()), "median": float(np.median(r)),
            "q90": float(np.quantile(r, 0.9)), "max": float(r.max()), "share_over_0.1": float((r > 0.1).mean()),
            "mean_p_cross": float(np.mean([x.p_c for x in xs]))}


def summarize(rows: dict[str, CrossSpan], picks: dict[str, set[str]],
              scores: dict[str, tuple[dict[str, float], dict[str, float]]]) -> dict:
    """The ratio overall and by top-1 bucket, and the decision rules' cross-span answers.

    scores: per rule, the official (EM, F1) per question of its answers (E.per_question). For
    each rule's cross-span answers it reports every rule's scores on those questions, so a
    rule that picks the cross span can be compared with one that does not.
    """
    xs = list(rows.values())
    out = {
        "questions": len(xs),
        "no_second_span": sum(x.b is None for x in xs),
        "no_cross_span": sum(x.c is None for x in xs),
        "ratio": {"all": ratio_stats(xs), **{k: ratio_stats([x for x in xs if bucket(x.p_top1) == k]) for k in BUCKETS}},
        "scores": {rule: E.summarize(*scores[rule]) for rule in picks},
        "cross_answers": {},
    }
    for rule, ids in picks.items():
        out["cross_answers"][rule] = {"questions": len(ids)}
        if ids:
            out["cross_answers"][rule]["scores"] = {
                r: E.summarize({q: scores[r][0][q] for q in ids}, {q: scores[r][1][q] for q in ids}) for r in picks}
    return out


def main(argv: list[str] | None = None) -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch

    parser = argparse.ArgumentParser(description="Cross-span diagnostic (plan E3) of trained readers.")
    parser.add_argument("--run-id", required=True, nargs="+")
    parser.add_argument("--eval-set", default="validation", choices=E.EVAL_SETS)
    parser.add_argument("--final", action="store_true", help="allow dev and the robustness sets (stage 5 only)")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)

    dataset = E.load_eval_set(args.eval_set, args.final)            # raises for dev etc. without --final
    questions = P.eval_questions(args.eval_set)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    commit, dirty = git_state()
    for run_id in args.run_id:
        run_dir = RESULTS / run_id
        temperature = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))["temperature"]
        cfg, tok, model, step = P.load_reader(run_id, device)
        feats = featurize(questions, tok, cfg, training=False)
        spans = P.score_questions(P.reader_score_fn(model, device), feats, tok.pad_token_id,
                                  cfg["max_answer_words"], args.batch_size, device)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        rows, preds, picks = cross_spans(feats, spans, cfg, temperature)
        scores = {rule: E.per_question(dataset, answers) for rule, answers in preds.items()}
        summary = {"split": args.eval_set, "temperature": temperature, "checkpoint_step": step,
                   "git_commit": commit, "git_dirty": dirty, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                   **summarize(rows, picks, scores)}
        write_json(run_dir / f"cross_span_{args.eval_set}.json", summary)
        write_json(run_dir / f"cross_span_{args.eval_set}_questions.json",
                   {q: [round(v, 6) for v in (x.p_top1, x.p_a, x.p_b, x.p_c)] for q, x in rows.items()})
        print(f"{run_id}:", json.dumps({"ratio": summary["ratio"], "scores": summary["scores"],
                                        "cross_answers": summary["cross_answers"]}), flush=True)


if __name__ == "__main__":
    main()
