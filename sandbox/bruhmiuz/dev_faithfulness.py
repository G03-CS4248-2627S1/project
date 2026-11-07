"""Is the reader's p faithful to the annotators? (#119; plan §7.2; spec §3.7's conventions for E6.)

Dev's original file (commit 8a3a52b) keeps every annotator's answer, so q, the share of a question's annotators who
gave each answer, can be measured. This compares q with the reader's p on dev, from each run's saved top-10 candidates
at its validation temperature, for row 0 and row 1 (3 seeds each). Nothing is fitted or chosen here: every system was
frozen and scored before this ran (#115), and qa/ is not changed.

- q is over groups of boundary variants, as decoding groups them (decode.group_keys). An annotation that is not
  word-aligned has no group and is left out of q (its question keeps the others); a question with none is skipped.
- p of a group is the sum of its members' probabilities among the top 10; the rest of p lies outside them.
- KL(q || p), natural log, p floored at 1e-15, only on questions where every group of q is in the top 10 ("covered"),
  because elsewhere p of a missing group is unknown.
- Jensen-Shannon distance (scipy, natural log) over q's groups in the top 10 plus one bucket for everything else. It
  is exact on covered questions and a lower bound on the others, where q's missing groups share the bucket.
- Reliability: every (question, top-10 group) pair binned by p, with the mean q in each bin, and the same for the most
  probable group alone. The gap is the bin-weighted |mean p - mean q|.
- Decisions: the MAP and MBR answers (decoded again from the candidates, and checked against the committed
  predictions) scored by their expected EM and F1 over the annotators, with the official scorer, so q also counts
  annotations that are not word-aligned. "Best in top 10" is the candidate with the highest expected EM + F1 under q.

    python sandbox/bruhmiuz/dev_faithfulness.py <out.json>

CPU only. Needs results/<run>/candidates_dev.json, which is not committed (workflow.md §2.2): copy it from the cluster
clone, where final scoring wrote it. Nothing in squadqa/ uses this script.
"""
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.distance import jensenshannon

sys.path.insert(0, ".")
from squadqa import data as D
from squadqa import decode as dec
from squadqa import evaluate as E
from squadqa import predict as P

ROWS = {
    "row0": ["2026-10-05-row0-baseline-s0-j910132", "2026-10-05-row0-baseline-s1-j910133", "2026-10-05-row0-baseline-s2-j910134"],
    "row1": ["2026-10-23-row1-joint-s0-j998373", "2026-10-23-row1-joint-s1-j998384", "2026-10-23-row1-joint-s2-j998397"],
}
ORIGINAL = "8a3a52b:qa/dev-v1.1.json"
FLOOR = 1e-15
BINS = 10
RESAMPLES = 10_000


def questions():
    """Per question id: context, words, content flags, q over groups, the annotators' texts, unaligned count."""
    original = json.loads(subprocess.run(["git", "show", ORIGINAL], capture_output=True, check=True).stdout)
    current = {qa["id"]: p["context"] for a in E.load_eval_set("dev", final=True) for p in a["paragraphs"] for qa in p["qas"]}
    out = {}
    for a in original["data"]:
        for p in a["paragraphs"]:
            context = p["context"]
            words = D.word_spans(context)
            flags = dec.content_flags(context, words)
            for qa in p["qas"]:
                if current[qa["id"]] != context:
                    raise RuntimeError(f"{qa['id']}: the original passage differs from qa/dev-v1.1.json's")
                keys, unaligned = [], 0
                for ans in qa["answers"]:
                    span = D.answer_word_span(words, ans["answer_start"], ans["text"])
                    if span is None:
                        unaligned += 1
                        continue
                    keys.append(dec.group_keys(flags, np.array([span[0]]), np.array([span[1]]))[0])
                counts = Counter(keys)
                q = {k: c / len(keys) for k, c in counts.items()}
                out[qa["id"]] = {"context": context, "words": words, "flags": flags, "q": q,
                                 "texts": [ans["text"] for ans in qa["answers"]], "unaligned": unaligned}
    if set(out) != set(current):
        raise RuntimeError("the original dev file and qa/dev-v1.1.json hold different questions")
    return out


def expected(sc, pred, texts):
    """Expected EM and F1 of pred over the annotators (each annotation once)."""
    return (float(np.mean([sc.compute_exact(t, pred) for t in texts])),
            float(np.mean([sc.compute_f1(t, pred) for t in texts])))


def run_stats(run, qs, sc):
    """Per question values for one run."""
    cfg = P.run_config(run)
    if cfg["mbr_utility"] != "normalized":
        raise RuntimeError(f"{run}: this script groups as mbr_utility normalized does")
    cands = json.loads((Path("results") / run / "candidates_dev.json").read_text(encoding="utf-8"))
    saved = {r: E.load_predictions(Path("results") / run / f"predictions_dev_{r}.json") for r in ("map", "mbr")}
    per, pairs, top_pairs, mismatched = {}, [], [], 0
    for qid, x in qs.items():
        if not x["q"]:
            continue
        a = np.asarray(cands[qid], dtype=np.float64)
        starts, ends, logp = a[:, 0].astype(np.int64), a[:, 1].astype(np.int64), a[:, 2]
        texts = [dec.answer_text(x["context"], x["words"], int(i), int(j)) for i, j in zip(starts, ends)]
        pg = defaultdict(float)
        for k, lp in zip(dec.group_keys(x["flags"], starts, ends), logp):
            pg[k] += math.exp(lp)
        q = x["q"]
        covered = all(k in pg for k in q)
        kl = sum(v * math.log(v / max(pg.get(k, 0.0), FLOOR)) for k, v in q.items()) if covered else None
        inside = [k for k in q if k in pg]
        pv = [pg[k] for k in inside] + [max(1.0 - sum(pg[k] for k in inside), 0.0)]
        qv = [q[k] for k in inside] + [sum(v for k, v in q.items() if k not in pg)]
        pv = np.maximum(np.array(pv), FLOOR)
        js = float(jensenshannon(pv / pv.sum(), np.array(qv)))
        for k, v in pg.items():
            pairs.append((v, q.get(k, 0.0)))
        top = max(pg, key=pg.get)
        top_pairs.append((pg[top], q.get(top, 0.0)))
        row = {"covered": covered, "kl": kl, "js": js, "disagree": len(q) > 1,
               "entropy_q": -sum(v * math.log(v) for v in q.values())}
        cand = dec.Candidates(starts, ends, logp)
        for rule in ("map", "mbr"):
            i, j = dec.decode(cand, x["flags"], rule, cfg["mbr_utility"], cfg["mbr_em_weight"], texts)
            ans = dec.answer_text(x["context"], x["words"], i, j)
            mismatched += ans != saved[rule][qid]
            row[rule] = expected(sc, ans, x["texts"])
        best = max((expected(sc, t, x["texts"]) for t in set(texts)), key=lambda s: s[0] + s[1])
        row["best"] = best
        per[qid] = row
    return per, pairs, top_pairs, mismatched


def reliability(pairs):
    p = np.array([a for a, _ in pairs])
    q = np.array([b for _, b in pairs])
    b = np.minimum((p * BINS).astype(int), BINS - 1)
    bins = [{"bin": f"{k / BINS:.1f}-{(k + 1) / BINS:.1f}", "n": int((b == k).sum()),
             "mean_p": float(p[b == k].mean()), "mean_q": float(q[b == k].mean())} for k in range(BINS) if (b == k).any()]
    gap = float(sum(x["n"] * abs(x["mean_p"] - x["mean_q"]) for x in bins) / len(p))
    return {"bins": bins, "gap": gap, "pairs": len(p)}


def bootstrap(a, b, ids, seed=0):
    """Mean of b - a over ids, with a 95% percentile interval over resampled questions."""
    d = np.array([b[i] - a[i] for i in ids])
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(RESAMPLES, len(d)))].mean(axis=1)
    return {"diff": float(d.mean()), "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))], "n": len(d)}


def mean_std(xs):
    return {"mean": float(np.mean(xs)), "std": float(np.std(xs, ddof=1))}


def main(out_path):
    sc = E.scorer()
    qs = questions()
    n_annot = sum(len(x["texts"]) for x in qs.values())
    out = {"questions": len(qs), "annotations": n_annot, "unaligned_annotations": sum(x["unaligned"] for x in qs.values()),
           "questions_without_q": sum(not x["q"] for x in qs.values()),
           "questions_disagree": sum(len(x["q"]) > 1 for x in qs.values()),
           "annotators": dict(sorted(Counter(min(len(x["texts"]), 5) for x in qs.values()).items())), "rows": {}}
    seed_means = {}
    for row, runs in ROWS.items():
        seeds, per_run = [], []
        all_pairs, all_top = [], []
        for run in runs:
            per, pairs, top_pairs, mismatched = run_stats(run, qs, sc)
            if mismatched:
                raise RuntimeError(f"{run}: {mismatched} answers decoded from the candidates differ from the committed predictions")
            per_run.append(per)
            all_pairs += pairs
            all_top += top_pairs
            s = {}
            for subset, keep in (("all", lambda r: True), ("disagree", lambda r: r["disagree"])):
                rows = [r for r in per.values() if keep(r)]
                cov = [r for r in rows if r["covered"]]
                s[subset] = {"questions": len(rows), "covered": len(cov) / len(rows),
                             "kl_covered": float(np.mean([r["kl"] for r in cov])),
                             "entropy_q_covered": float(np.mean([r["entropy_q"] for r in cov])),
                             "js": float(np.mean([r["js"] for r in rows]))}
                for rule in ("map", "mbr", "best"):
                    s[subset][rule] = {"exact": 100 * float(np.mean([r[rule][0] for r in rows])),
                                       "f1": 100 * float(np.mean([r[rule][1] for r in rows]))}
            seeds.append(s)
        ids = sorted(per_run[0])
        avg = {qid: {k: np.mean([p[qid][k] for p in per_run], axis=0) for k in ("map", "mbr")} for qid in ids}
        seed_means[row] = {qid: {"js": float(np.mean([p[qid]["js"] for p in per_run]))} for qid in ids}
        summary = {}
        for subset in ("all", "disagree"):
            summary[subset] = {key: mean_std([s[subset][key] for s in seeds]) for key in ("covered", "kl_covered", "entropy_q_covered", "js")}
            for rule in ("map", "mbr", "best"):
                summary[subset][rule] = {m: mean_std([s[subset][rule][m] for s in seeds]) for m in ("exact", "f1")}
            keep = [qid for qid in ids if subset == "all" or per_run[0][qid]["disagree"]]
            summary[subset]["mbr_minus_map"] = {
                "exact": bootstrap({i: 100 * avg[i]["map"][0] for i in keep}, {i: 100 * avg[i]["mbr"][0] for i in keep}, keep),
                "f1": bootstrap({i: 100 * avg[i]["map"][1] for i in keep}, {i: 100 * avg[i]["mbr"][1] for i in keep}, keep),
                "exact_plus_f1": bootstrap({i: 100 * avg[i]["map"].sum() for i in keep}, {i: 100 * avg[i]["mbr"].sum() for i in keep}, keep)}
        out["rows"][row] = {"runs": runs, "seeds": seeds, "summary": summary,
                            "reliability_all_groups": reliability(all_pairs), "reliability_top_group": reliability(all_top)}
    ids = sorted(seed_means["row0"])
    out["row1_minus_row0_js"] = bootstrap({i: seed_means["row0"][i]["js"] for i in ids}, {i: seed_means["row1"][i]["js"] for i in ids}, ids)
    Path(out_path).write_text(json.dumps(out, indent=1) + "\n", encoding="ascii", newline="\n")
    print(report(out))


def report(out):
    ms = lambda d, f="{:.3f}": f"{f.format(d['mean'])} ± {f.format(d['std'])}"
    ci = lambda c: f"{c['diff']:+.2f} [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}]"
    lines = [f"{out['questions']} questions, {out['annotations']} annotations ({out['unaligned_annotations']} not word-aligned), "
             f"{out['questions_disagree']} with 2+ groups, {out['questions_without_q']} without q", ""]
    for subset in ("all", "disagree"):
        lines += [f"### {subset}", "", "| Row | Covered | KL(q‖p), covered | H(q), covered | JS distance |", "|---|---|---|---|---|"]
        for row, r in out["rows"].items():
            s = r["summary"][subset]
            lines.append(f"| {row} | {ms(s['covered'])} | {ms(s['kl_covered'])} | {ms(s['entropy_q_covered'])} | {ms(s['js'])} |")
        lines += ["", "| Row | MAP EM / F1 under q | MBR EM / F1 under q | Best in top 10 | MBR - MAP, EM | F1 | EM + F1 |", "|---|---|---|---|---|---|---|"]
        for row, r in out["rows"].items():
            s = r["summary"][subset]
            cell = lambda d: f"{d['exact']['mean']:.2f} / {d['f1']['mean']:.2f}"
            c = s["mbr_minus_map"]
            lines.append(f"| {row} | {cell(s['map'])} | {cell(s['mbr'])} | {cell(s['best'])} | {ci(c['exact'])} | {ci(c['f1'])} | {ci(c['exact_plus_f1'])} |")
        lines.append("")
    for kind in ("reliability_all_groups", "reliability_top_group"):
        lines += [f"### {kind}", "", "| p | " + " | ".join(f"{row} n, mean p, mean q" for row in out["rows"]) + " |", "|---|" + "---|" * len(out["rows"])]
        by = {row: {b["bin"]: b for b in r[kind]["bins"]} for row, r in out["rows"].items()}
        for name in sorted({b for d in by.values() for b in d}):
            cells = [f"{d[name]['n']}, {d[name]['mean_p']:.3f}, {d[name]['mean_q']:.3f}" if name in d else "" for d in by.values()]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("| gap | " + " | ".join(f"{r[kind]['gap']:.3f}" for r in out["rows"].values()) + " |")
        lines.append("")
    lines.append(f"row 1 - row 0, JS distance: {out['row1_minus_row0_js']['diff']:+.4f} [{out['row1_minus_row0_js']['ci95'][0]:+.4f}, {out['row1_minus_row0_js']['ci95'][1]:+.4f}]")
    return "\n".join(lines)


if __name__ == "__main__":
    main(sys.argv[1])
