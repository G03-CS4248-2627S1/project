"""Verifier candidate lists (spec §5.1; issue #66), from a trained reader's span distribution.

Two kinds of list, one per question:
- training (the reader's own training questions): verifier.k_train distinct spans sampled
  without replacement from softmax(s / verifier.sample_temperature), where s are the reader's raw
  span scores (temperature 1 = the reader's own distribution), plus the gold span if the sample
  missed it, plus its boundary variants (±1 word at either end, valid spans only). A span is a
  position (rule 4), so deduplication is by (start, end), never by text.
- inference (validation; dev and the robustness sets only in stage-5 final scoring): the reader's
  top verifier.k_infer spans. Nothing is added: the gold is only used to measure recall.

verifier.list_unit: group makes every candidate a group of boundary variants instead (spans with
the same content core, decode.group_cores; the unit of target_unit: group): its probability is the
sum over all its valid spans, its span is its most probable member, and training lists sample k
distinct groups from the tempered group distribution and add the gold's group, with no boundary
variants (they are in the gold's group already).

Sources (--source, training lists only besides squad): squad (train-v1.1, as the reader trained),
adversarialqa (AdversarialQA train, which the reader never saw) and squad2-unanswerable (SQuAD
2.0's unanswerable questions: there is no gold, every candidate is wrong and the verifier's null
candidate is the target; the list adds the plausible answer, a distractor, in place of the gold,
and the labels are measured against it).

Every candidate carries labels against the reference span (the gold, or the plausible answer),
all between content cores, so they are the same for every member of a group (span_labels):
f1, em, precision and recall (loss_utility, spec §4.3), relation, kind, start_shift, end_shift and
gap, official_f1 and official_em (the official scorer on the texts; analysis only, never a target:
it calls the same text elsewhere correct, rule 4 doesn't), sibling (the candidate's core is the
gold core of one of the reader's training questions on the same passage), and the reader's
log-probability at the run's calibrated temperature. Each list also records the reader's own top-1
(span or group) with the same labels.

    python -m squadqa.candidates --run-id <reader run> --split train          # sampled lists
    python -m squadqa.candidates --run-id <reader run> --split validation     # top-k lists
    python -m squadqa.candidates --run-id <reader run> --split train --source squad2-unanswerable \
        --set verifier.list_unit=group

The lists go to $SQUADQA_SCRATCH/candidates/<run_id>/candidates_<split>.json, or
candidates_<source>_<split>.json for the other sources (large; workflow.md §2.2 keeps them out of
the repo), with a summary next to them (summary below: top-k recall, how often the sample held the
gold, list sizes, what the reader's top-1 and the other candidates are).
"""
from __future__ import annotations

import argparse
import json
import os
import zlib
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from squadqa.data import answer_word_span
from squadqa.decode import (SpanScores, answer_text, content_flags, group_cores, log_softmax, span_f1_em,
                            span_logp, span_precision_recall, top_k)

SPLITS = ("train", "validation")
SOURCES = ("squad", "adversarialqa", "squad2-unanswerable")
LIST_UNITS = ("span", "group")
# Verifier keys that change the lists (spec §5.1) but are not built yet, with the only value allowed.
NOT_BUILT = {"train_candidates": "sampled", "sibling_negatives": False, "unanswerable_negatives": False}
# Labels against the reference, besides f1 and em (span_labels; official_* and sibling in build_lists).
LABELS = ("precision", "recall", "relation", "kind", "start_shift", "end_shift", "gap")


@dataclass
class CandidateList:
    starts: np.ndarray          # (c,) global word index of each candidate's first word
    ends: np.ndarray            # (c,) last word, inclusive
    logp: np.ndarray            # (c,) reader log-probability at the calibrated temperature
    f1: np.ndarray              # (c,) F1 and EM against the reference (loss_utility), NaN without one
    em: np.ndarray
    gold: int | None            # position of the reference in the list, None if absent
    sampled_gold: bool          # the reference was among the sampled (training) or top-k (inference) ones
    sampled_em: bool = False    # some sampled / top-k candidate has EM 1 (the gold or a boundary variant)
    labels: dict[str, list] = field(default_factory=dict)    # LABELS, official_*, sibling: one value per candidate
    sizes: np.ndarray | None = None    # group lists: the number of valid spans in each candidate's group
    unanswerable: bool = False  # SQuAD 2.0: the reference is the plausible answer, a distractor, and gold is None
    plausible: int | None = None       # unanswerable lists: the plausible answer's position, None if absent
    top1: dict | None = None    # the reader's most probable span (group): span, p, and the same labels

    def to_json(self) -> dict:
        """JSON-ready, with null labels for a question without a word-aligned reference."""
        out = {"spans": [[int(a), int(b)] for a, b in zip(self.starts, self.ends)],
               "logp": [round(float(x), 6) for x in self.logp],
               "f1": [None if np.isnan(x) else round(float(x), 6) for x in self.f1],
               "em": [None if np.isnan(x) else int(x) for x in self.em],
               **{k: [_plain(x) for x in v] for k, v in self.labels.items()}}
        if self.sizes is not None:
            out["size"] = [int(x) for x in self.sizes]
        if self.unanswerable:
            out.update(gold=None, unanswerable=True, plausible=self.plausible,
                       sampled_plausible=self.sampled_gold, sampled_em=self.sampled_em)
        else:
            out.update(gold=self.gold, sampled_gold=self.sampled_gold, sampled_em=self.sampled_em)
        if self.top1 is not None:
            out["top1"] = {k: _plain(v) for k, v in self.top1.items()}
        return out


def _plain(x):
    """A JSON value: NaN as null, floats rounded, numpy scalars as Python ones."""
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else round(float(x), 6)
    return None if x is None else str(x)


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


def span_labels(flags: np.ndarray, starts: np.ndarray, ends: np.ndarray,
                reference: tuple[int, int] | None) -> dict[str, list]:
    """LABELS of candidate spans against the reference span, all between content cores
    (decode.group_cores), so every member of a group gets the same values:
    - precision, recall: the halves of f1 (low precision = too long, low recall = too short);
    - relation: same (the reference's group, EM 1), inside, contains, crosses (overlap, neither
      holds the other) or disjoint;
    - kind: correct (EM 1), boundary (F1 > 0: the reader's kind of error) or wrong (F1 0: the
      verifier's kind);
    - start_shift, end_shift: candidate core minus reference core, in words (signed);
    - gap: words strictly between the two cores when disjoint, else 0.
    None for every candidate without a reference (a gold or plausible answer that isn't word-aligned).
    """
    n = len(starts)
    if reference is None:
        return {k: [None] * n for k in LABELS}
    i0, j0 = reference
    f1, em = span_f1_em(flags, starts, ends, i0, j0)
    precision, recall = span_precision_recall(flags, starts, ends, i0, j0)
    cs, ce = group_cores(flags, starts, ends)
    (rs,), (re_,) = group_cores(flags, [i0], [j0])
    disjoint = (ce < rs) | (cs > re_)
    relation = np.select([(cs == rs) & (ce == re_), disjoint, (cs >= rs) & (ce <= re_), (cs <= rs) & (ce >= re_)],
                         ["same", "disjoint", "inside", "contains"], "crosses")
    kind = np.select([em == 1, f1 > 0], ["correct", "boundary"], "wrong")
    gap = np.where(disjoint, np.maximum(cs - re_, rs - ce) - 1, 0)
    return {"precision": precision.tolist(), "recall": recall.tolist(), "relation": relation.tolist(),
            "kind": kind.tolist(), "start_shift": (cs - rs).tolist(), "end_shift": (ce - re_).tolist(),
            "gap": gap.tolist()}


def _top1(i: int, j: int, logp: float, flags: np.ndarray, reference: tuple[int, int] | None) -> dict:
    out = {"span": [int(i), int(j)], "p": float(np.exp(logp))}
    if reference is None:
        out.update(f1=None, em=None)
    else:
        f1, em = span_f1_em(flags, i, j, reference[0], reference[1])
        out.update(f1=float(f1), em=float(em))
    out.update({k: v[0] for k, v in span_labels(flags, np.array([i]), np.array([j]), reference).items()})
    return out


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
    logp = span_logp(spans, temperature)[0]
    best = int(np.argmax(logp))
    return CandidateList(starts, ends, logp[keep], f1, em,
                         None if pos is None or not len(pos) else int(pos[0]),
                         g is not None and bool(np.isin(g, sampled)), hit_em,
                         labels=span_labels(flags, starts, ends, gold),
                         top1=_top1(spans.starts[best], spans.ends[best], logp[best], flags, gold))


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


def unanswerable_list(spans: SpanScores, plausible: tuple[int, int] | None, k: int, sample_temperature: float,
                      rng: np.random.Generator, flags: np.ndarray, temperature: float) -> CandidateList:
    """k sampled spans plus the plausible answer when it is a valid span (no variants: it is a
    distractor, not an answer). Labels are measured against the plausible answer; gold is None."""
    sampled = sample_spans(spans.scores, k, sample_temperature, rng)
    p = spans.index(plausible)
    picked = sampled if p is None else np.append(sampled, p).astype(np.int64)
    return _unanswerable(_build(spans, picked, sampled, plausible, flags, temperature))


def _unanswerable(c: CandidateList) -> CandidateList:
    return replace(c, gold=None, plausible=c.gold, unanswerable=True)


def inference_list(spans: SpanScores, k: int, flags: np.ndarray, temperature: float,
                   gold: tuple[int, int] | None = None) -> CandidateList:
    """The k most probable spans and nothing else; the gold, if given, only fills f1, em and gold."""
    best = top_k(spans, k, temperature)                  # the reader's own top-k, so the lists stay in step with it
    top = np.array([spans.index((int(i), int(j))) for i, j in zip(best.starts, best.ends)], dtype=np.int64)
    return _build(spans, top, top, gold, flags, temperature)


def span_groups(spans: SpanScores, flags: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Every valid span's group id (equal content cores), and per group: its most probable member
    (an index into spans), its size, and its core."""
    cs, ce = group_cores(flags, spans.starts, spans.ends)
    cores, gid = np.unique(np.stack([cs, ce], axis=1), axis=0, return_inverse=True)
    gid = gid.reshape(-1)
    order = np.argsort(-spans.scores, kind="stable")
    _, first = np.unique(gid[order], return_index=True)              # groups 0..G-1, best member first
    return gid, order[first], np.bincount(gid, minlength=len(cores)), cores[:, 0], cores[:, 1]


def group_logsumexp(values: np.ndarray, gid: np.ndarray, n_groups: int) -> np.ndarray:
    """log Σ exp(values) over each group's members."""
    out = np.full(n_groups, -np.inf)
    np.logaddexp.at(out, gid, np.asarray(values, dtype=np.float64))
    return out


def group_list(spans: SpanScores, k: int, flags: np.ndarray, temperature: float,
               reference: tuple[int, int] | None, rng: np.random.Generator | None = None,
               sample_temperature: float | None = None) -> CandidateList:
    """A list of groups. With rng (training): k distinct groups sampled from the groups' tempered
    probabilities Σ softmax(s / sample_temperature) over members, plus the reference's group if
    missed (when some valid span has its core). Without (inference): the k most probable groups,
    nothing added. Each candidate is its group's most probable span, with the group's probability."""
    gid, rep, sizes, core_s, core_e = span_groups(spans, flags)
    n = len(rep)
    logp = group_logsumexp(span_logp(spans, temperature)[0], gid, n)
    ref = None
    if reference is not None:
        (rs,), (re_,) = group_cores(flags, [reference[0]], [reference[1]])
        hit = np.flatnonzero((core_s == rs) & (core_e == re_))
        ref = int(hit[0]) if len(hit) else None
    if rng is None:
        sampled = np.argsort(-logp, kind="stable")[:k]
        picked = sampled
    else:
        sampled = sample_spans(group_logsumexp(spans.scores / sample_temperature, gid, n), k, 1.0, rng)
        picked = sampled if ref is None else np.append(sampled, ref)
    keep = np.unique(picked)
    keep = keep[np.lexsort((spans.ends[rep[keep]], spans.starts[rep[keep]]))]
    starts, ends = spans.starts[rep[keep]], spans.ends[rep[keep]]
    if reference is None:
        f1, em, hit_em = np.full(len(keep), np.nan), np.full(len(keep), np.nan), False
    else:
        f1, em = span_f1_em(flags, starts, ends, reference[0], reference[1])
        r = rep[sampled]
        hit_em = bool((span_f1_em(flags, spans.starts[r], spans.ends[r], reference[0], reference[1])[1] == 1).any())
    pos = np.flatnonzero(keep == ref) if ref is not None else []
    best = int(np.argmax(logp))
    return CandidateList(starts, ends, logp[keep], f1, em, int(pos[0]) if len(pos) else None,
                         ref is not None and bool(np.isin(ref, sampled)), hit_em,
                         labels=span_labels(flags, starts, ends, reference), sizes=sizes[keep],
                         top1=_top1(spans.starts[rep[best]], spans.ends[rep[best]], logp[best], flags, reference))


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
    top-k at inference. Unanswerable lists report sampled_plausible instead of sampled_gold. Both
    add what the reader's top-1 is (reader_top1) and what the other candidates are (others).
    inference: top-k recall for each k up to the list size (the verifier's ceiling, spec §5.1),
    and what the reader's top-1 is: correct (EM 1), partial (F1 > 0) or disjoint (F1 0).
    """
    if not lists:
        raise ValueError(f"no {split} lists to summarize")
    sizes = np.array([len(c.starts) for c in lists])
    out = {"questions": len(lists), "mean_size": float(sizes.mean()), "max_size": int(sizes.max())}
    if lists[0].sizes is not None:
        out["mean_group_size"] = float(np.mean(np.concatenate([c.sizes for c in lists])))
    if split == "train":
        out["sampled_plausible" if lists[0].unanswerable else "sampled_gold"] = float(np.mean([c.sampled_gold for c in lists]))
        out["sampled_em"] = float(np.mean([c.sampled_em for c in lists]))
        out.update(_label_summary(lists))
        return out
    scored = [c for c in lists if not np.isnan(c.f1).all()]       # questions with a word-aligned gold, of any length
    if not scored:
        raise ValueError(f"no {split} question has a word-aligned gold, so recall is undefined")
    out["with_gold"] = len(scored)
    ranks = [int((c.logp > c.logp[c.gold]).sum()) if c.gold is not None else None for c in scored]
    # Two definitions of "found" (both over the questions with a word-aligned gold):
    # recall@k: the annotated span itself, by position, is in the top k (spec §5.1's wording);
    # recall_em@k: some top-k candidate has EM 1 under loss_utility, so a boundary variant counts,
    # the unit a reader trained with target_unit group aims at. squadqa.report's top-k recall uses
    # the official scorer instead, which also accepts the same text elsewhere (rule 4 doesn't).
    first_em = [int(np.flatnonzero(c.em[np.argsort(-c.logp, kind="stable")] == 1)[0])
                if (c.em == 1).any() else None for c in scored]
    # Ties: recall@k ranks by strictly higher logp, so a gold tied with another span counts as the
    # better one; recall_em@k takes the stable sort, which puts tied spans in list (position) order.
    # Exact float ties are rare in real scores, but the two recalls can differ on them.
    for k in range(1, int(sizes.max()) + 1):           # a shorter list counts as a miss at the ranks it lacks
        out[f"recall@{k}"] = float(np.mean([r is not None and r < k for r in ranks]))
        out[f"recall_em@{k}"] = float(np.mean([r is not None and r < k for r in first_em]))
    top = [int(np.argmax(c.logp)) for c in scored]
    em = np.array([c.em[t] for c, t in zip(scored, top)])
    f1 = np.array([c.f1[t] for c, t in zip(scored, top)])
    out["top1"] = {"correct": float((em == 1).mean()), "partial": float(((em == 0) & (f1 > 0)).mean()),
                   "disjoint": float((f1 == 0).mean())}
    return out


def _shares(values: list) -> dict:
    keys, counts = np.unique(np.array(values, dtype=object).astype(str), return_counts=True)
    return {str(k): float(c / len(values)) for k, c in zip(keys, counts)}


def _label_summary(lists: list[CandidateList]) -> dict:
    """The reader's top-1 against the reference (for unanswerable lists, kind correct = it picked
    the plausible distractor's group) and the candidates other than the reference: their kind,
    relation and, when labelled, sibling shares. Lists without a reference or labels are left out."""
    labelled = [c for c in lists if c.top1 is not None and c.top1.get("kind") is not None]
    if not labelled:
        return {}
    out = {"with_reference": len(labelled) / len(lists),
           "reader_top1": {"kind": _shares([c.top1["kind"] for c in labelled]),
                           "relation": _shares([c.top1["relation"] for c in labelled]),
                           "mean_p": float(np.mean([c.top1["p"] for c in labelled]))}}
    if all("sibling" in c.top1 for c in labelled):
        out["reader_top1"]["sibling"] = float(np.mean([c.top1["sibling"] for c in labelled]))
    ref = [c.plausible if c.unanswerable else c.gold for c in labelled]
    if all("sibling" in c.labels for c in labelled) and any(r is not None for r in ref):
        # for unanswerable lists: how often the distractor is another question's answer on the passage
        out["reference_sibling"] = float(np.mean([c.labels["sibling"][r] for c, r in zip(labelled, ref) if r is not None]))
    others = [(c, k) for c, r in zip(labelled, ref) for k in range(len(c.starts)) if k != r]
    if others:
        out["others"] = {"candidates": len(others),
                         "kind": _shares([c.labels["kind"][k] for c, k in others]),
                         "relation": _shares([c.labels["relation"][k] for c, k in others])}
        if all("sibling" in c.labels for c in labelled):
            out["others"]["sibling"] = float(np.mean([c.labels["sibling"][k] for c, k in others]))
    return out


def sibling_index(questions) -> dict[str, list[tuple[str, int, str]]]:
    """passage → (question id, answer_start, text) of the gold of each question on it: the reader's
    training questions, for the sibling label."""
    out: dict[str, list[tuple[str, int, str]]] = {}
    for q in questions:
        if q.answers:
            out.setdefault(q.context, []).append((q.id, q.answers[0][1], q.answers[0][0]))
    return out


def _text_labels(c: CandidateList, ex, flags: np.ndarray, siblings: dict | None) -> None:
    """Add official_f1 / official_em (the official scorer, the candidate's text against the
    reference text, at any position) and, with a sibling index, sibling, to c and its top1."""
    from squadqa.evaluate import scorer

    q, words = ex.question, ex.context.words
    ref = q.answers[0][0] if q.answers else None
    spans = [(int(i), int(j)) for i, j in zip(c.starts, c.ends)] + [tuple(c.top1["span"])]
    texts = [answer_text(q.context, words, i, j) for i, j in spans]
    s = scorer()
    f1 = [None if ref is None else float(s.compute_f1(ref, t)) for t in texts]
    em = [None if ref is None else int(s.compute_exact(ref, t)) for t in texts]
    c.labels["official_f1"], c.labels["official_em"] = f1[:-1], em[:-1]
    c.top1.update(official_f1=f1[-1], official_em=em[-1])
    if siblings is None:
        return
    golds = [answer_word_span(words, start, text) for qid, start, text in siblings.get(q.context, ()) if qid != q.id]
    golds = [g for g in golds if g is not None]
    cores = set()
    if golds:
        gs, ge = group_cores(flags, [g[0] for g in golds], [g[1] for g in golds])
        cores = set(zip(gs.tolist(), ge.tolist()))
    cs, ce = group_cores(flags, [i for i, _ in spans], [j for _, j in spans])
    hit = [(a, b) in cores for a, b in zip(cs.tolist(), ce.tolist())]
    c.labels["sibling"] = hit[:-1]
    c.top1["sibling"] = hit[-1]


def build_lists(examples, spans: list[SpanScores], split: str, cfg: dict, temperature: float,
                seed: int, unanswerable: bool = False, siblings: dict | None = None) -> dict[str, CandidateList]:
    """The lists of one batch of examples (squadqa.data.Example) with their SpanScores. cfg needs
    "verifier" (k_train, sample_temperature, k_infer, list_unit) and "loss_utility". unanswerable:
    the examples' gold is the plausible answer of an unanswerable question (training lists only).
    siblings: sibling_index of the reader's training questions, for the sibling label."""
    v = cfg["verifier"]
    for key, default in NOT_BUILT.items():
        if v.get(key, default) != default:
            raise NotImplementedError(f"verifier.{key}: {v[key]!r} is not built; only {default!r}")
    if v["list_unit"] not in LIST_UNITS:
        raise ValueError(f"verifier.list_unit must be one of {LIST_UNITS}, not {v['list_unit']!r}")
    if unanswerable and split != "train":
        raise ValueError("unanswerable questions give training lists only")
    positional = cfg["loss_utility"] == "positional"
    out = {}
    for ex, s in zip(examples, spans):
        q = ex.question
        flags = np.ones(len(ex.context.words), dtype=bool) if positional else content_flags(q.context, ex.context.words)
        if split == "train":
            rng = question_rng(seed, q.id)
            if v["list_unit"] == "group":
                c = group_list(s, v["k_train"], flags, temperature, ex.gold, rng, v["sample_temperature"])
                if not unanswerable and c.gold is None:
                    raise ValueError(f"{q.id}: no valid span has the gold's core")
            elif unanswerable:
                c = unanswerable_list(s, ex.gold, v["k_train"], v["sample_temperature"], rng, flags, temperature)
            else:
                c = training_list(s, ex.gold, v["k_train"], v["sample_temperature"], rng, flags, temperature)
            if unanswerable and not c.unanswerable:
                c = _unanswerable(c)
        elif v["list_unit"] == "group":
            c = group_list(s, v["k_infer"], flags, temperature, ex.gold)
        else:
            c = inference_list(s, v["k_infer"], flags, temperature, ex.gold)
        _text_labels(c, ex, flags, siblings)
        out[q.id] = c
    return out


def load_source(source: str, split: str):
    """The questions a source's lists are built on."""
    from squadqa.data import load_extra, load_split, load_unanswerable

    if source == "squad":
        return load_split(split)
    if split != "train":
        raise ValueError(f"--source {source} gives training lists only")
    return load_unanswerable() if source == "squad2-unanswerable" else load_extra(source)


def main(argv: list[str] | None = None) -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch

    from squadqa.config import RESULTS, env, git_state, load_config, write_json
    from squadqa.data import featurize, load_split
    from squadqa.predict import load_reader, reader_score_fn, score_questions

    parser = argparse.ArgumentParser(description="Verifier candidate lists from a trained reader.")
    parser.add_argument("--run-id", required=True, help="the reader run to sample from")
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument("--source", default="squad", choices=SOURCES,
                        help="the questions: train-v1.1 (default), or for training lists AdversarialQA "
                             "or SQuAD 2.0's unanswerable questions")
    parser.add_argument("--config", default=None, help="experiment file for the verifier keys (default: base)")
    parser.add_argument("--set", action="append", default=[], help="override, e.g. verifier.k_train=12")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed (training lists)")
    parser.add_argument("--chunk", type=int, default=4096, help="questions scored at a time, to bound memory")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)
    if args.source != "squad" and args.split != "train":
        parser.error(f"--source {args.source} gives training lists only")

    # The reader's own config sets how it windows and scores; the current config sets the lists
    # (the reader run predates the verifier keys) and the F1 / EM they are labelled with.
    current = load_config(args.config, args.set)
    list_cfg = {"verifier": current["verifier"], "loss_utility": current["loss_utility"]}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, tok, model, step = load_reader(args.run_id, device)
    if cfg.get("extra_train_data"):
        # A reader trained on extra data (row 5: AdversarialQA) needs training lists over those
        # questions too, and row 5 needs the unanswerable_negatives lists. Nothing uses such a
        # reader yet, so this waits for row 5 instead of writing lists that skip half its data.
        raise NotImplementedError(f"lists for a reader trained with extra_train_data {cfg['extra_train_data']} are not built")
    temperature = json.loads((RESULTS / args.run_id / "calibration.json").read_text(encoding="utf-8"))["temperature"]
    questions = load_source(args.source, args.split)
    siblings = sibling_index(load_split("train"))         # the reader's training questions
    unanswerable = args.source == "squad2-unanswerable"
    score_fn = reader_score_fn(model, device)
    lists: dict[str, CandidateList] = {}
    skipped = 0
    for c in range(0, len(questions), args.chunk):
        # Unanswerable questions keep a plausible answer that isn't word-aligned: the list then has no reference.
        feats = featurize(questions[c:c + args.chunk], tok, cfg, training=args.split == "train" and not unanswerable)
        spans = score_questions(score_fn, feats, tok.pad_token_id, cfg["max_answer_words"], args.batch_size, device)
        lists.update(build_lists(feats.examples, spans, args.split, list_cfg, temperature, args.seed,
                                 unanswerable, siblings))
        skipped += len(feats.skipped)
        print(f"{min(c + args.chunk, len(questions))}/{len(questions)} questions", flush=True)

    name = args.split if args.source == "squad" else f"{args.source}_{args.split}"
    out_dir = Path(env("SQUADQA_SCRATCH")) / "candidates" / args.run_id      # write_json creates it
    write_json(out_dir / f"candidates_{name}.json", {q: c.to_json() for q, c in lists.items()},
               compact=args.split == "train")                                # ~830k candidates
    commit, dirty = git_state()
    info = {"run_id": args.run_id, "checkpoint_step": step, "split": args.split, "source": args.source,
            "seed": args.seed, "git_commit": commit, "git_dirty": dirty, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "temperature": temperature, "skipped": skipped, **list_cfg, **summary(list(lists.values()), args.split)}
    write_json(out_dir / f"summary_{name}.json", info)
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
