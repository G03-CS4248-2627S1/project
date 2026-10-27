"""Verifier candidate lists (spec §5.1; issue #66), from a trained reader's span distribution.

Two kinds of list, one per question:
- training (the reader's own training questions): verifier.k_train distinct spans sampled
  without replacement from softmax(s / verifier.sample_temperature), where s are the reader's raw
  span scores (temperature 1 = the reader's own distribution), plus the gold span if the sample
  missed it, plus its boundary variants (±1 word at either end, valid spans only). A span is a
  position (rule 4), so deduplication is by (start, end), never by text.
- inference (validation; dev and the robustness sets only in stage-5 final scoring): the reader's
  top verifier.k_infer spans. Nothing is added: the gold is only used to measure recall.

Every candidate carries its F1 and EM against the gold (loss_utility, spec §4.3) and its reader
log-probability at the run's calibrated temperature, so the verifier's target (soft or one-hot;
targets below) and the reader + λ·verifier combination need no second pass over the reader.

    python -m squadqa.candidates --run-id <reader run> --split train          # sampled lists
    python -m squadqa.candidates --run-id <reader run> --split validation     # top-k lists

The lists go to $SQUADQA_SCRATCH/candidates/<run_id>/candidates_<split>.json (large; workflow.md
§2.2 keeps them out of the repo), with a summary next to them (summary below: top-k recall, how
often the sample held the gold, list sizes).
"""
from __future__ import annotations

import argparse
import json
import os
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from squadqa.decode import SpanScores, content_flags, log_softmax, span_f1_em, span_logp

SPLITS = ("train", "validation")
# Verifier keys that change the lists (spec §5.1) but are not built yet, with the only value allowed.
NOT_BUILT = {"train_candidates": "sampled", "sibling_negatives": False, "unanswerable_negatives": False}


@dataclass
class CandidateList:
    starts: np.ndarray          # (c,) global word index of each candidate's first word
    ends: np.ndarray            # (c,) last word, inclusive
    logp: np.ndarray            # (c,) reader log-probability at the calibrated temperature
    f1: np.ndarray              # (c,) F1 and EM against the gold (loss_utility), NaN without a gold
    em: np.ndarray
    gold: int | None            # position of the annotated span in the list, None if absent
    sampled_gold: bool          # the gold was among the sampled (training) or top-k (inference) spans
    sampled_em: bool = False    # some sampled / top-k span has EM 1 (the gold or a boundary variant)

    def to_json(self) -> dict:
        """JSON-ready, with null F1 and EM for a question without a word-aligned gold."""
        return {"spans": [[int(a), int(b)] for a, b in zip(self.starts, self.ends)],
                "logp": [round(float(x), 6) for x in self.logp],
                "f1": [None if np.isnan(x) else round(float(x), 6) for x in self.f1],
                "em": [None if np.isnan(x) else int(x) for x in self.em],
                "gold": self.gold, "sampled_gold": self.sampled_gold, "sampled_em": self.sampled_em}


def question_rng(seed: int, question_id: str) -> np.random.Generator:
    """One generator per question, so a list doesn't depend on which other questions ran with it."""
    return np.random.default_rng([seed, zlib.crc32(question_id.encode())])


def sample_spans(scores: np.ndarray, k: int, temperature: float, rng: np.random.Generator) -> np.ndarray:
    """Indices of min(k, len(scores)) distinct spans drawn without replacement from
    softmax(scores / temperature): the Gumbel-top-k trick (Kool et al., 2019), exact for sampling
    without replacement. -inf scores are never drawn."""
    if temperature <= 0:
        raise ValueError(f"the sampling temperature must be positive, not {temperature}")
    keys = np.asarray(scores, dtype=np.float64) / temperature + rng.gumbel(size=len(scores))
    k = min(k, int(np.isfinite(keys).sum()))
    return np.argsort(-keys, kind="stable")[:k]


def boundary_variants(spans: SpanScores, gold: tuple[int, int]) -> list[int]:
    """Indices of the valid spans one word longer or shorter than the gold at either end."""
    i, j = gold
    out = [spans.index(v) for v in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))]
    return [k for k in out if k is not None]


def _build(spans: SpanScores, picked: np.ndarray, sampled: np.ndarray, gold: tuple[int, int] | None,
           flags: np.ndarray, temperature: float) -> CandidateList:
    keep = np.unique(picked)
    keep = keep[np.lexsort((spans.ends[keep], spans.starts[keep]))]        # by position, so the order is fixed
    starts, ends = spans.starts[keep], spans.ends[keep]
    g = None if gold is None else spans.index(gold)      # None too for a gold over max_answer_words
    if gold is None:                                     # labels need only the gold's words, at any length
        f1, em = np.full(len(keep), np.nan), np.full(len(keep), np.nan)
    else:
        f1, em = span_f1_em(flags, starts, ends, gold[0], gold[1])
    pos = None if g is None else np.flatnonzero(keep == g)
    hit_em = gold is not None and bool((span_f1_em(flags, spans.starts[sampled], spans.ends[sampled], gold[0], gold[1])[1] == 1).any())
    return CandidateList(starts, ends, span_logp(spans, temperature)[0][keep], f1, em,
                         None if pos is None or not len(pos) else int(pos[0]),
                         g is not None and bool(np.isin(g, sampled)), hit_em)


def training_list(spans: SpanScores, gold: tuple[int, int], k: int, sample_temperature: float,
                  rng: np.random.Generator, flags: np.ndarray, temperature: float) -> CandidateList:
    """k sampled spans, plus the gold if missed, plus its boundary variants, deduplicated. Raises if
    the gold is not a valid span (featurize(training=True) has already skipped those questions)."""
    g = spans.index(gold)
    if g is None:
        raise ValueError(f"the gold {gold} is not a valid span")
    sampled = sample_spans(spans.scores, k, sample_temperature, rng)
    picked = np.concatenate([sampled, [g], boundary_variants(spans, gold)]).astype(np.int64)
    return _build(spans, picked, sampled, gold, flags, temperature)


def inference_list(spans: SpanScores, k: int, flags: np.ndarray, temperature: float,
                   gold: tuple[int, int] | None = None) -> CandidateList:
    """The k most probable spans and nothing else; the gold, if given, only fills f1, em and gold."""
    top = np.argsort(-span_logp(spans, temperature)[0], kind="stable")[:k]
    return _build(spans, top, top, gold, flags, temperature)


def targets(f1: np.ndarray, gold: int, alpha: float, tau: float, floor: float) -> np.ndarray:
    """The listwise target over one list, spec §4.3's per-span form: t = (1 − α) t_EM + α t_F1.

    t_EM puts all mass on the annotated span; t_F1 ∝ exp(−(1 − F1) / τ) over candidates with F1 > 0
    and F1 ≥ φ. α = 0 is the one-hot target. There is no group form here: §4.3's group objective,
    −log p(G*) with p(G*) summed over the gold's boundary variants, is not a per-candidate target
    vector, so it belongs in the verifier's loss (#67), as in heads.f1_target_nll(unit="group").
    """
    if not (0 <= alpha <= 1 and tau > 0 and 0 <= floor <= 1):
        raise ValueError("need 0 <= alpha <= 1, tau > 0 and 0 <= floor <= 1")
    t_em = (np.arange(len(f1)) == gold).astype(np.float64)
    w = np.where((f1 > 0) & (f1 >= floor), np.exp(-(1 - f1) / tau), 0.0)
    return (1 - alpha) * t_em + alpha * w / w.sum()


def summary(lists: list[CandidateList], split: str) -> dict:
    """What a split's lists look like.

    training: list sizes, and how often the sample held the gold before it was added: the annotated
    span (sampled_gold) or any EM 1 span (sampled_em). A check on sample_temperature: every
    training list gets the gold added, so this is about the sample alone; a sampled_em close to
    validation recall_em@k_infer means the sampled distractors are about as hard as the reader's
    top-k at inference.
    inference: top-k recall for each k up to the list size (the verifier's ceiling, spec §5.1),
    and what the reader's top-1 is: correct (EM 1), partial (F1 > 0) or disjoint (F1 0).
    """
    sizes = np.array([len(c.starts) for c in lists])
    out = {"questions": len(lists), "mean_size": float(sizes.mean()), "max_size": int(sizes.max())}
    if split == "train":
        out["sampled_gold"] = float(np.mean([c.sampled_gold for c in lists]))
        out["sampled_em"] = float(np.mean([c.sampled_em for c in lists]))
        return out
    scored = [c for c in lists if not np.isnan(c.f1).all()]       # questions with a word-aligned gold, of any length
    out["with_gold"] = len(scored)
    ranks = [int((c.logp > c.logp[c.gold]).sum()) if c.gold is not None else None for c in scored]
    # Two definitions of "found" (both over the questions with a word-aligned gold):
    # recall@k: the annotated span itself, by position, is in the top k (spec §5.1's wording);
    # recall_em@k: some top-k candidate has EM 1 under loss_utility, so a boundary variant counts,
    # the unit a reader trained with target_unit group aims at. squadqa.report's top-k recall uses
    # the official scorer instead, which also accepts the same text elsewhere (rule 4 doesn't).
    first_em = [int(np.flatnonzero(c.em[np.argsort(-c.logp, kind="stable")] == 1)[0])
                if (c.em == 1).any() else None for c in scored]
    for k in range(1, int(sizes.min()) + 1):
        out[f"recall@{k}"] = float(np.mean([r is not None and r < k for r in ranks]))
        out[f"recall_em@{k}"] = float(np.mean([r is not None and r < k for r in first_em]))
    top = [int(np.argmax(c.logp)) for c in scored]
    em = np.array([c.em[t] for c, t in zip(scored, top)])
    f1 = np.array([c.f1[t] for c, t in zip(scored, top)])
    out["top1"] = {"correct": float((em == 1).mean()), "partial": float(((em == 0) & (f1 > 0)).mean()),
                   "disjoint": float((f1 == 0).mean())}
    return out


def build_lists(examples, spans: list[SpanScores], split: str, cfg: dict, temperature: float,
                seed: int) -> dict[str, CandidateList]:
    """The lists of one batch of examples (squadqa.data.Example) with their SpanScores. cfg needs
    "verifier" (k_train, sample_temperature, k_infer) and "loss_utility"."""
    v = cfg["verifier"]
    for key, default in NOT_BUILT.items():
        if v.get(key, default) != default:
            raise NotImplementedError(f"verifier.{key}: {v[key]!r} is not built; only {default!r}")
    positional = cfg["loss_utility"] == "positional"
    out = {}
    for ex, s in zip(examples, spans):
        q = ex.question
        flags = np.ones(len(ex.context.words), dtype=bool) if positional else content_flags(q.context, ex.context.words)
        if split == "train":
            out[q.id] = training_list(s, ex.gold, v["k_train"], v["sample_temperature"],
                                      question_rng(seed, q.id), flags, temperature)
        else:
            out[q.id] = inference_list(s, v["k_infer"], flags, temperature, ex.gold)
    return out


def main(argv: list[str] | None = None) -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch

    from squadqa.config import RESULTS, env, load_config, write_json
    from squadqa.data import featurize, load_split
    from squadqa.predict import load_reader, reader_score_fn, score_questions

    parser = argparse.ArgumentParser(description="Verifier candidate lists from a trained reader.")
    parser.add_argument("--run-id", required=True, help="the reader run to sample from")
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument("--config", default=None, help="experiment file for the verifier keys (default: base)")
    parser.add_argument("--set", action="append", default=[], help="override, e.g. verifier.k_train=12")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed (training lists)")
    parser.add_argument("--chunk", type=int, default=4096, help="questions scored at a time, to bound memory")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)

    # The reader's own config sets how it windows and scores; the current config sets the lists
    # (the reader run predates the verifier keys) and the F1 / EM they are labelled with.
    current = load_config(args.config, args.set)
    list_cfg = {"verifier": current["verifier"], "loss_utility": current["loss_utility"]}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, tok, model, step = load_reader(args.run_id, device)
    temperature = json.loads((RESULTS / args.run_id / "calibration.json").read_text(encoding="utf-8"))["temperature"]
    questions = load_split(args.split)
    score_fn = reader_score_fn(model, device)
    lists: dict[str, CandidateList] = {}
    skipped = 0
    for c in range(0, len(questions), args.chunk):
        feats = featurize(questions[c:c + args.chunk], tok, cfg, training=args.split == "train")
        spans = score_questions(score_fn, feats, tok.pad_token_id, cfg["max_answer_words"], args.batch_size, device)
        lists.update(build_lists(feats.examples, spans, args.split, list_cfg, temperature, args.seed))
        skipped += len(feats.skipped)
        print(f"{min(c + args.chunk, len(questions))}/{len(questions)} questions", flush=True)

    out_dir = Path(env("SQUADQA_SCRATCH")) / "candidates" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / f"candidates_{args.split}.json", {q: c.to_json() for q, c in lists.items()})
    info = {"run_id": args.run_id, "checkpoint_step": step, "split": args.split, "seed": args.seed,
            "temperature": temperature, "skipped": skipped, **list_cfg, **summary(list(lists.values()), args.split)}
    write_json(out_dir / f"summary_{args.split}.json", info)
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
