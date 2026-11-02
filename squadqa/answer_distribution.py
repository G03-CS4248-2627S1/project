"""Full-span diagnostics for #95 (NQ regions) and #96 (validation shape).

No training, temperature fitting, final-test prediction, or #97 decoder sweep.
Score small chunks and release full-span matrices rather than retain a dataset in RAM.
Modes are defined reproducibly as probability-ordered, greedily nonoverlapping content
cores with group mass >= .05 (not an inferred density-mixture component count).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np

from squadqa.data import Question, answer_word_span, load_split, load_squad
from squadqa.decode import (SpanScores, answer_text, content_flags, decode, group_cores,
                            span_logp, top_k)

ROW1 = [f"2026-10-23-row1-joint-s{s}-j{j}" for s, j in enumerate((998373, 998384, 998397))]
ROW0 = [f"2026-10-05-row0-baseline-s{s}-j{j}" for s, j in enumerate((910132, 910133, 910134))]
NQ = Path("data/external/multi_answer")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grouped(spans: SpanScores, flags: np.ndarray, temperature: float) -> dict:
    """All distinct content-core groups, ranked by mass; never truncate before pooling."""
    logp, null_logp = span_logp(spans, temperature)
    p = np.exp(logp)
    cs, ce = group_cores(flags, spans.starts, spans.ends)
    keys, ids = np.unique(np.stack([cs, ce], axis=1), axis=0, return_inverse=True)
    mass = np.bincount(ids, weights=p, minlength=len(keys))
    maximum = np.zeros(len(keys))
    np.maximum.at(maximum, ids, p)
    order = np.argsort(-mass, kind="stable")
    return {"p": p, "keys": keys[order], "mass": mass[order], "maximum": maximum[order],
            "null": 0.0 if null_logp is None else float(np.exp(null_logp))}


def entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def disjoint_modes(keys: np.ndarray, mass: np.ndarray, flags: np.ndarray) -> list[int]:
    """Greedy disjoint high-mass cores, including neither empty cores nor sub-.05 groups."""
    selected = []
    for k in np.flatnonzero(mass >= 0.05):
        a, b = keys[k]
        if not flags[a:b + 1].any():
            continue
        if all(b < keys[q, 0] or a > keys[q, 1] for q in selected):
            selected.append(int(k))
    return selected


def shape(spans: SpanScores, flags: np.ndarray, gold: tuple[int, int] | None,
          temperature: float) -> dict:
    g = grouped(spans, flags, temperature)
    keys, mass, p = g["keys"], g["mass"], g["p"]
    if not np.isclose(p.sum() + g["null"], 1.0, atol=1e-9):
        raise ValueError("span plus null probabilities do not sum to one")
    hp = entropy(np.append(p, g["null"]))
    hg = entropy(np.append(mass, g["null"]))
    modes = disjoint_modes(keys, mass, flags)
    # Second region = highest-probability nonempty core disjoint from the top group.
    disjoint = (keys[:, 1] < keys[0, 0]) | (keys[:, 0] > keys[0, 1])
    nonempty = flags[keys[:, 0]]
    others = np.flatnonzero(disjoint & nonempty)
    cumul = np.cumsum(mass)
    out = {"top1_span_probability": float(p.max()), "top1_group_probability": float(mass[0]),
           "span_entropy": hp, "group_entropy": hg, "effective_groups": float(np.exp(hg)),
           "top_group_representative_share": float(g["maximum"][0] / mass[0]),
           "modes_ge_0.05": len(modes),
           "second_disjoint_group_mass": float(mass[others[0]]) if len(others) else 0.0,
           "null_mass": g["null"], "valid_spans": len(p), "groups": len(mass)}
    for threshold in (0.5, 0.9, 0.99):
        # Null is a separate outcome: reaching a target may be impossible using answer groups.
        out[f"groups_for_{threshold:g}_mass"] = (int(np.searchsorted(cumul, threshold) + 1)
                                                    if cumul[-1] >= threshold else None)
    if gold is not None:
        prefix = np.concatenate([[0], np.cumsum(flags)])
        lo, hi = np.maximum(spans.starts, gold[0]), np.minimum(spans.ends, gold[1])
        overlap = (lo <= hi) & (prefix[np.maximum(lo, hi + 1)] - prefix[lo] > 0)
        out["mass_overlapping_gold"] = float(p[overlap].sum())
        out["mass_disjoint_gold"] = float(p[~overlap].sum())
    else:
        out["mass_overlapping_gold"] = out["mass_disjoint_gold"] = None
    return out


def regions(answers: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Merge overlapping (not merely adjacent) gold cores; q uniform over input groups."""
    merged: list[tuple[int, int]] = []
    for a, b in sorted(set(answers)):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    q = np.array([sum(a <= s and e <= b for s, e in answers) / len(answers) for a, b in merged])
    return merged, q


def divergences(q: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Natural-log KL(q||p), sqrt(JS), including a rest bucket; floor model p then normalize."""
    p = np.maximum(p, 1e-15)
    p = p / p.sum()
    m = (p + q) / 2
    nz = q > 0
    kl = float((q[nz] * np.log(q[nz] / p[nz])).sum())
    js = .5 * float((q[nz] * np.log(q[nz] / m[nz])).sum()) + .5 * float((p * np.log(p / m)).sum())
    return kl, float(np.sqrt(max(0.0, js)))


def multi_answer(spans: SpanScores, flags: np.ndarray, golds: list[tuple[int, int]],
                 temperature: float) -> dict:
    g = grouped(spans, flags, temperature)
    cs, ce = group_cores(flags, [x[0] for x in golds], [x[1] for x in golds])
    cores = list(dict.fromkeys(zip(cs.tolist(), ce.tolist())))
    rs, q = regions(cores)
    inside = np.array([(g["keys"][:, 0] >= a) & (g["keys"][:, 1] <= b) for a, b in rs])
    nonempty = flags[g["keys"][:, 0]]
    inside &= nonempty[None, :]
    mass = inside @ g["mass"]
    order = np.argsort(-mass, kind="stable")
    total = float(mass.sum())
    outside = float(max(0.0, 1.0 - total))
    kl, js = divergences(np.append(q, 0.0), np.append(mass, outside))
    second = int(order[1]) if len(order) > 1 else None
    rank = np.flatnonzero(inside[second]) if second is not None else np.array([], dtype=int)
    top = int(order[0])
    first_region = next(k for k, (a, b) in enumerate(rs) if a <= cores[0][0] and cores[0][1] <= b)
    return {"regions": [list(x) for x in rs], "q_regions": q.tolist(), "p_regions": mass.tolist(),
            "eligible_2_regions": len(rs) >= 2, "answer_region_mass": total, "outside_mass": outside,
            "collapse": float(mass[top] / total) if total > 0 else None,
            "zero_region_mass": total == 0, "q_top_region_share": float(q.max()),
            "second_to_first_mass": float(mass[second] / mass[top]) if second is not None and mass[top] > 0 else None,
            "second_region_best_group_rank": int(rank[0] + 1) if len(rank) else None,
            "second_region_in_top5": bool(len(rank) and rank[0] < 5),
            "second_region_in_top10": bool(len(rank) and rank[0] < 10),
            "kl_q_p": kl, "js_distance": js, "top_region_is_first_annotated": top == first_region,
            "region_lengths": [b - a + 1 for a, b in rs],
            "minimum_region_gap": min((rs[k][0] - rs[k-1][1] - 1 for k in range(1, len(rs))), default=None)}


def stats(values: list[float]) -> dict:
    x = np.asarray(values, dtype=float)
    if not len(x):
        return {"n": 0}
    return {"n": len(x), "mean": float(x.mean()), "std": float(x.std()), "min": float(x.min()),
            "max": float(x.max()), "quantiles": {str(v): float(np.quantile(x, v)) for v in (0,.1,.25,.5,.75,.9,.99,1)}}


def aggregate(rows: list[dict]) -> dict:
    out = {"questions": len(rows)}
    numeric = sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float))})
    for key in numeric:
        out[key] = stats([r[key] for r in rows if isinstance(r.get(key), (int, float))])
    if rows and "top1_span_probability" in rows[0]:
        p = np.array([r["top1_span_probability"] for r in rows])
        out["top1_probability_shares"] = {"below_0.5": float((p < .5).mean()), "at_least_0.9": float((p >= .9).mean())}
    if rows and "collapse" in rows[0]:
        x = [r["collapse"] for r in rows if r["collapse"] is not None]
        out["collapse_shares"] = {str(v): float(np.mean(np.array(x) > v)) if x else None for v in (.9,.99)}
    return out


def summary(rows: dict[str, dict], mode: str) -> dict:
    vals = list(rows.values())
    if mode == "shape":
        return {"all": aggregate(vals), "by_map_outcome": {k: aggregate([r for r in vals if r["map_outcome"] == k])
                                                               for k in ("correct", "partial", "disjoint")}}
    out = {}
    for split in ("all", "heldout", "train"):
        selected = [r for r in vals if split == "all" or r["split"] == split]
        eligible = [r for r in selected if r["eligible_2_regions"]]
        out[split] = {"total": len(selected), "eligible": len(eligible), "measurements": aggregate(eligible),
                      "by_first_annotated": {str(v): aggregate([r for r in eligible if r["top_region_is_first_annotated"] == v])
                                             for v in (True, False)},
                      "by_minimum_gap": {label: aggregate([r for r in eligible if (r["minimum_region_gap"] <= 10 if label == "0-10" else r["minimum_region_gap"] > 10)])
                                         for label in ("0-10", ">10")},
                      "by_max_region_length": {label: aggregate([r for r in eligible if (max(r["region_lengths"]) <= 5 if label == "<=5" else max(r["region_lengths"]) > 5)])
                                               for label in ("<=5", ">5")}}
    return out


def nq_questions(root: Path) -> tuple[list[Question], dict]:
    """Collapse E6 training answer-copies back to one question with q.json's gold answers."""
    q = json.loads((root / "q.json").read_text())
    found = {}
    for split in ("train", "heldout"):
        for ex in load_squad(root / f"{split}.json"):
            qid = ex.id if split == "heldout" else ex.id.rsplit("_a", 1)[0]
            if qid not in q or q[qid]["split"] != split:
                raise ValueError(f"E6 split/id mismatch: {ex.id}")
            rebuilt = Question(qid, ex.title, ex.context, ex.question,
                               tuple((text, start) for start, text, weight in q[qid]["answers"]))
            if qid in found and found[qid] != rebuilt:
                raise ValueError(f"conflicting E6 answer copies: {qid}")
            found[qid] = rebuilt
    if set(found) != set(q):
        raise ValueError("E6 files and q.json contain different questions")
    return list(found.values()), q


def load_checkpoint_reader(run_id: str, device: str):
    """Build the pinned architecture, then strictly load the COMPLETE trained reader.

    No need to download a second copy of pretrained weights which the full checkpoint
    immediately replaces. Only the pinned config/tokenizer are needed in the HF cache.
    """
    import torch
    from transformers import AutoConfig, AutoModel
    from squadqa import predict as P
    from squadqa.config import env
    from squadqa.data import load_tokenizer
    from squadqa.download_model import MODELS
    from squadqa.heads import Reader
    from squadqa.train_reader import load_checkpoint

    cfg = P.run_config(run_id)
    env("HF_HOME")
    config = AutoConfig.from_pretrained(cfg["model"], revision=MODELS[cfg["model"]]["revision"])
    # Build on CPU; load_checkpoint uses strict state_dict loading and map_location=cpu.
    encoder = AutoModel.from_config(config, dtype=torch.float32)
    model = Reader(encoder, cfg)
    step = load_checkpoint(Path(env("SQUADQA_SCRATCH")) / "runs" / run_id / "checkpoint.pt", model)
    return cfg, load_tokenizer(cfg["model"]), model.to(device), step


def main(argv=None):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    from squadqa import evaluate as E, predict as P
    from squadqa.config import RESULTS, env, git_state, write_json
    from squadqa.data import featurize

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("shape", "multi-answer"))
    ap.add_argument("--run-id", nargs="+")
    ap.add_argument("--include-baseline", action="store_true")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--nq-dir", type=Path, default=NQ)
    ap.add_argument("--output-root", type=Path, default=RESULTS)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    if args.limit is not None and args.output_root == RESULTS:
        ap.error("smoke --limit requires a separate --output-root")
    if args.batch_size < 1 or args.chunk_size < 1:
        ap.error("batch and chunk sizes must be positive")
    if args.include_baseline and args.mode != "shape":
        ap.error("baselines apply only to #96 shape")
    run_ids = args.run_id or ROW1 + (ROW0 if args.include_baseline else [])
    qs, q = (nq_questions(args.nq_dir) if args.mode == "multi-answer" else (load_split("validation"), None))
    if args.limit:
        qs = qs[:args.limit]
    datafiles = ([args.nq_dir / n for n in ("q.json", "train.json", "heldout.json", "summary.json")]
                 if q is not None else [Path("qa/train-v1.1.json"), Path("data/splits/val_articles.txt")])
    commit, dirty = git_state()
    if dirty:
        raise RuntimeError("diagnostics must run from committed, clean tracked code")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = E.scorer()
    for run_id in run_ids:
        prefix = "distribution_validation" if args.mode == "shape" else "multi_answer"
        dest = args.output_root / run_id
        paths = [dest / f"{prefix}{suffix}.json" for suffix in ("", "_questions")]
        if not args.overwrite and any(p.exists() for p in paths):
            raise FileExistsError(f"refusing to overwrite {paths}; use --overwrite deliberately")
        cal_path = RESULTS / run_id / "calibration.json"
        temperature = json.loads(cal_path.read_text())["temperature"]
        cfg, tok, model, step = load_checkpoint_reader(run_id, device)
        checkpoint = Path(env("SQUADQA_SCRATCH")) / "runs" / run_id / "checkpoint.pt"
        rows = {"at_1": {}, "calibrated": {}}
        print(f"{run_id}: {len(qs)} questions, T={temperature}, {device}", flush=True)
        for start in range(0, len(qs), args.chunk_size):
            feats = featurize(qs[start:start + args.chunk_size], tok, cfg, training=False)
            scored = P.score_questions(P.reader_score_fn(model, device), feats, tok.pad_token_id,
                                       cfg["max_answer_words"], args.batch_size, device)
            for ex, spans in zip(feats.examples, scored):
                flags = content_flags(ex.question.context, ex.context.words)
                for label, t in (("at_1", 1.0), ("calibrated", temperature)):
                    if q is None:
                        r = shape(spans, flags, ex.gold, t)
                        candidates = top_k(spans, cfg["decode_topk"], t)
                        texts = [answer_text(ex.question.context, ex.context.words, int(a), int(b))
                                 for a, b in zip(candidates.starts, candidates.ends)]
                        a, b = decode(candidates, flags, "map", cfg["mbr_utility"], cfg["mbr_em_weight"], texts)
                        answer = answer_text(ex.question.context, ex.context.words, a, b)
                        em = max(scorer.compute_exact(text, answer) for text, _ in ex.question.answers)
                        f1 = max(scorer.compute_f1(text, answer) for text, _ in ex.question.answers)
                        r.update(map_answer=answer, official_em=em, official_f1=f1,
                                 map_outcome="correct" if em else "partial" if f1 > 0 else "disjoint")
                    else:
                        golds = [answer_word_span(ex.context.words, off, text) for text, off in ex.question.answers]
                        if any(gold is None for gold in golds):
                            raise ValueError(f"unaligned E6 answer: {ex.question.id}")
                        r = multi_answer(spans, flags, golds, t)
                        r["split"] = q[ex.question.id]["split"]
                    rows[label][ex.question.id] = r
            del scored, feats
            print(f"  {min(start + args.chunk_size, len(qs))}/{len(qs)}", flush=True)
        source_run = json.loads((RESULTS / run_id / "run.json").read_text())
        meta = {"schema_version": 1, "mode": args.mode, "reader_run_id": run_id, "checkpoint_step": step,
                "checkpoint": str(checkpoint.resolve()), "checkpoint_bytes": checkpoint.stat().st_size,
                "training_provenance": source_run, "analysis_commit": commit, "git_dirty": dirty,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "host": platform.node(),
                "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                "temperature": temperature, "questions": len(qs), "limit": args.limit,
                "batch_size": args.batch_size, "chunk_size": args.chunk_size,
                "data_sha256": {str(p): digest(p) for p in datafiles},
                "calibration_sha256": digest(cal_path), "reader_config_sha256": digest(RESULTS / run_id / "config.yaml"),
                "definitions": {"modes": "greedy probability-ordered disjoint content cores, group p >= 0.05",
                                "region_mass": "all groups whose nonempty content core lies wholly within merged gold region",
                                "second_region": "second-highest model region mass, rank among ALL content-core groups",
                                "q": "uniform over distinct annotated groups, aggregated into regions; not human frequencies",
                                "divergences": "natural log; rest bucket; p floored 1e-15 and normalized; JS square root",
                                "gold_overlap": "overlap at counted content positions; invalid gold recorded as null",
                                "map": "existing top-k normalized decoder; shape statistics use the FULL span/group distribution"}}
        summary_data = {label: summary(v, args.mode) for label, v in rows.items()}
        if q is None:
            expected = json.loads((RESULTS / run_id / "metrics.json").read_text())["validation"]["map"]
            vals = list(rows["calibrated"].values())
            measured = {"exact": 100 * float(np.mean([r["official_em"] for r in vals])),
                        "f1": 100 * float(np.mean([r["official_f1"] for r in vals]))}
            meta["map_validation_check"] = {"measured": measured, "expected": expected}
            if args.limit is None and any(abs(measured[k] - expected[k]) > .1 for k in measured):
                raise RuntimeError(f"MAP validation differs by >0.1 points: {meta['map_validation_check']}")
        # Summaries written last, so a summary implies the per-question artifact finished.
        write_json(paths[1], {"metadata": meta, **rows})
        write_json(paths[0], {"metadata": meta, **summary_data})
        print(f"Wrote {paths[0]}", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
