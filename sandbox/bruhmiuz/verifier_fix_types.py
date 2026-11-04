"""What the separate verifier (ladder row 4) changes against the reader on the same lists (#116; plan E2; spec §6).

Rebuilds each answer's span from the committed scores, the lists and metrics.json's λ and temperature,
and stops unless the rebuilt answers equal the committed predictions. Fix types are spec §6's, between
the reader's answer and the combined one:
- no change: equal after the official normalize_answer;
- boundary fix: the spans overlap but the answers differ (split here into longer, shorter and crossing);
- answer switch: the spans do not overlap.
It also compares errors between readers and verifiers: when one is wrong on EM, how often is the other
right? That tests why E2 chose a separate encoder (plan §4.3, "less correlated errors").

    python sandbox/bruhmiuz/verifier_fix_types.py <out.json>

CPU only, on validation. It needs data/verifier/<reader>/candidates_validation.json, the lists the
runs scored (gitignored; python -m squadqa.candidates rebuilds them), and the reader's tokenizer.
Nothing in squadqa/ uses this script.
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from squadqa import data as D
from squadqa import evaluate as E
from squadqa import predict as P
from squadqa import train_verifier as TV
from squadqa import verifier as V
from squadqa.config import RESULTS, write_json
from squadqa.decode import Candidates, answer_text, content_flags, decode, log_softmax

READER = "2026-10-23-row1-joint-s0-j998373"
READER_SEEDS = [READER, "2026-10-23-row1-joint-s1-j998384", "2026-10-23-row1-joint-s2-j998397"]
RUNS = ["2026-10-29-verifier-separate-s0-j1032972", "2026-10-30-verifier-separate-s1-j1041306",
        "2026-10-30-verifier-separate-s2-j1041319"]
SHARED = "2026-10-28-verifier-shared-lora-s0-j1030695"
RULES = ("map", "mbr")
KINDS = ("longer", "shorter", "crossing", "switch")
BUCKETS = (("< 0.5", 0.0, 0.5), ("0.5-0.9", 0.5, 0.9), ("> 0.9", 0.9, 1.01))
sc = E.scorer()


def kind_of(reader_text, combined_text, r, c):
    """Spec §6's fix type of one change; r and c are the (first word, last word) of each answer."""
    if sc.normalize_answer(reader_text) == sc.normalize_answer(combined_text):
        return "none"
    if r[1] < c[0] or c[1] < r[0]:
        return "switch"
    if c[0] <= r[0] and c[1] >= r[1]:
        return "longer"
    if c[0] >= r[0] and c[1] <= r[1]:
        return "shorter"
    return "crossing"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rebuild(run, examples, entries, rcfg, rule):
    """(question id -> reader span and text, combined span and text) for one run and rule."""
    metrics = load_json(RESULTS / run / "metrics.json")
    lam, temperature = metrics["lambda"], metrics["temperature"]["combined"]
    saved = load_json(RESULTS / run / TV.SCORES)
    out = {}
    for ex, e in zip(examples, entries):
        q = ex.question
        flags = content_flags(q.context, ex.context.words)
        starts, ends = np.array([a for a, _ in e["spans"]]), np.array([b for _, b in e["spans"]])
        texts = [answer_text(q.context, ex.context.words, a, b) for a, b in e["spans"]]
        s = np.array(saved[q.id], dtype=np.float64)
        pick = {}
        for who, logp in (("reader", np.array(e["logp"], dtype=np.float64)),
                          ("combined", log_softmax(V.combine(e["logp"], s, lam), temperature))):
            i, j = decode(Candidates(starts, ends, logp), flags, rule, rcfg["mbr_utility"], rcfg["mbr_em_weight"], texts)
            pick[who] = ((i, j), answer_text(q.context, ex.context.words, i, j))
        out[q.id] = pick
    for who in ("reader", "combined"):
        committed = E.load_predictions(RESULTS / run / f"predictions_validation_{who}_{rule}.json")
        differ = sum(out[k][who][1] != committed[k] for k in out)
        if differ:
            raise RuntimeError(f"{run} {who}_{rule}: {differ} rebuilt answers differ from the committed predictions")
    return out


def tally(rows):
    """Counts over one run's changed questions."""
    out = Counter()
    for r in rows:
        out["changed"] += 1
        out["em_gained"] += r["dem"] > 0
        out["em_lost"] += r["dem"] < 0
        out["f1_better"] += r["df1"] > 1e-9
        out["f1_worse"] += r["df1"] < -1e-9
    return out


def mean_std(values):
    v = np.array(values, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}


def contribution(per_seed_rows, ids, keep):
    """A kind's share of combined - reader, in points, over all questions, with bootstrap intervals."""
    em, f1 = {k: 0.0 for k in ids}, {k: 0.0 for k in ids}
    for rows in per_seed_rows:
        for r in rows:
            if keep(r):
                em[r["id"]] += r["dem"] / len(per_seed_rows)
                f1[r["id"]] += r["df1"] / len(per_seed_rows)
    zero = {k: 0.0 for k in ids}
    total = {k: em[k] + f1[k] for k in ids}
    return {"exact": E.paired_bootstrap(zero, em), "f1": E.paired_bootstrap(zero, f1), "exact_plus_f1": E.paired_bootstrap(zero, total)}


def pair_stats(a_em, b_em):
    """How two systems' EM errors overlap, over the same questions."""
    keys = sorted(a_em)
    a, b = np.array([a_em[k] for k in keys]), np.array([b_em[k] for k in keys])
    return {"both_right": float(((a == 1) & (b == 1)).mean() * 100), "a_only": float(((a == 1) & (b == 0)).mean() * 100),
            "b_only": float(((a == 0) & (b == 1)).mean() * 100), "both_wrong": float(((a == 0) & (b == 0)).mean() * 100),
            "b_right_when_a_wrong": float((b[a == 0] == 1).mean() * 100), "a_right_when_b_wrong": float((a[b == 0] == 1).mean() * 100),
            "phi": float(np.corrcoef(a, b)[0, 1])}


def main(out_path):
    rcfg = P.run_config(READER)
    tok = D.load_tokenizer(rcfg["model"])
    feats = D.featurize(D.load_split("validation"), tok, rcfg, training=False)
    examples, entries = TV.by_id(feats, TV.load_lists(TV.lists_dir(READER, None), "validation"))
    dataset = E.load_eval_set("validation")
    ids = E.question_ids(dataset)
    top1 = {ex.question.id: float(np.exp(max(e["logp"]))) for ex, e in zip(examples, entries)}
    by_id = {ex.question.id: ex for ex in examples}
    out = {"reader": READER, "runs": RUNS, "questions": len(ids), "rules": {}}
    for rule in RULES:
        per_seed = []
        for run in RUNS:
            picks = rebuild(run, examples, entries, rcfg, rule)
            er, fr = E.per_question(dataset, {k: v["reader"][1] for k, v in picks.items()})
            ec, fc = E.per_question(dataset, {k: v["combined"][1] for k, v in picks.items()})
            rows = []
            for k, v in picks.items():
                kind = kind_of(v["reader"][1], v["combined"][1], v["reader"][0], v["combined"][0])
                if kind != "none":
                    rows.append({"id": k, "kind": kind, "dem": ec[k] - er[k], "df1": fc[k] - fr[k], "top1": top1[k],
                                 "reader": v["reader"][1], "combined": v["combined"][1], "run": run})
            per_seed.append(rows)
        table = {}
        groups = {**{k: (lambda r, k=k: r["kind"] == k) for k in KINDS},
                  "boundary": lambda r: r["kind"] != "switch", "all": lambda r: True}
        for name, keep in groups.items():
            counts = [tally([r for r in rows if keep(r)]) for rows in per_seed]
            table[name] = {f: mean_std([c[f] for c in counts]) for f in ("changed", "em_gained", "em_lost", "f1_better", "f1_worse")}
            table[name]["contribution"] = contribution(per_seed, ids, keep)
        buckets = {}
        for label, lo, hi in BUCKETS:
            keep = lambda r, lo=lo, hi=hi: lo <= r["top1"] < hi
            counts = [tally([r for r in rows if keep(r)]) for rows in per_seed]
            buckets[label] = {"questions": sum(lo <= p < hi for p in top1.values()),
                              **{f: mean_std([c[f] for c in counts]) for f in ("changed", "em_gained", "em_lost")},
                              "contribution": contribution(per_seed, ids, keep)}
        out["rules"][rule] = {"kinds": table, "by_reader_top1": buckets}
        if rule == "map":
            first = per_seed[0]
            chosen = []
            for kind in KINDS:
                for sign, name in ((1, "gain"), (-1, "loss")):
                    hit = next((r for r in first if r["kind"] == kind and r["dem"] * sign > 0), None)
                    if hit:
                        q = by_id[hit["id"]].question
                        at = q.answers[0][1]
                        chosen.append({"kind": kind, "effect": name, "id": hit["id"], "question": q.question, "gold": q.answers[0][0],
                                       "reader": hit["reader"], "combined": hit["combined"], "top1": round(hit["top1"], 3),
                                       "passage": q.context[max(0, at - 220):at + len(q.answers[0][0]) + 220]})
            out["examples"] = chosen

    def em_of(run, rule):
        return E.per_question(dataset, E.load_predictions(RESULTS / run / f"predictions_validation_{rule}.json"))[0]

    pairs = {}
    separate = [pair_stats(em_of(run, "reader_map"), em_of(run, "verifier_map")) for run in RUNS]
    pairs["reader vs separate verifier alone (3 seeds)"] = {k: float(np.mean([s[k] for s in separate])) for k in separate[0]}
    pairs["reader vs shared verifier alone (seed 0, its own lists)"] = pair_stats(em_of(SHARED, "reader_map"), em_of(SHARED, "verifier_map"))
    readers = [pair_stats(em_of(READER_SEEDS[0], "map"), em_of(READER_SEEDS[i], "map")) for i in (1, 2)]
    pairs["reader seed 0 vs reader seeds 1, 2"] = {k: float(np.mean([s[k] for s in readers])) for k in readers[0]}
    verifiers = [pair_stats(em_of(RUNS[0], "verifier_map"), em_of(RUNS[i], "verifier_map")) for i in (1, 2)]
    pairs["separate verifier seed 0 vs seeds 1, 2"] = {k: float(np.mean([s[k] for s in verifiers])) for k in verifiers[0]}
    out["error_overlap_map"] = pairs
    write_json(Path(out_path), out)
    print(report(out))


def ci(d):
    return f"{d['diff']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]"


def report(out):
    lines = []
    for rule in RULES:
        r = out["rules"][rule]
        lines += [f"### {rule.upper()}: by fix type", "",
                  "| Fix type | Changed | EM gained | EM lost | F1 better | F1 worse | Share of EM | Share of F1 | Share of EM + F1 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for name, t in r["kinds"].items():
            c = t["contribution"]
            lines.append(f"| {name} | {t['changed']['mean']:.1f} ± {t['changed']['std']:.1f} | {t['em_gained']['mean']:.1f} | {t['em_lost']['mean']:.1f} "
                         f"| {t['f1_better']['mean']:.1f} | {t['f1_worse']['mean']:.1f} | {ci(c['exact'])} | {ci(c['f1'])} | {ci(c['exact_plus_f1'])} |")
        lines += ["", f"### {rule.upper()}: by the reader's top-1 probability", "",
                  "| Top-1 probability | Questions | Changed | EM gained | EM lost | Share of EM + F1 |", "|---|---|---|---|---|---|"]
        for label, b in r["by_reader_top1"].items():
            lines.append(f"| {label} | {b['questions']} | {b['changed']['mean']:.1f} | {b['em_gained']['mean']:.1f} | {b['em_lost']['mean']:.1f} "
                         f"| {ci(b['contribution']['exact_plus_f1'])} |")
        lines.append("")
    lines += ["### Error overlap (MAP, EM)", "",
              "| Pair (a vs b) | Both right | Only a | Only b | Both wrong | b right when a wrong | a right when b wrong | φ |",
              "|---|---|---|---|---|---|---|---|"]
    for name, p in out["error_overlap_map"].items():
        lines.append(f"| {name} | {p['both_right']:.2f} | {p['a_only']:.2f} | {p['b_only']:.2f} | {p['both_wrong']:.2f} "
                     f"| {p['b_right_when_a_wrong']:.1f} | {p['a_right_when_b_wrong']:.1f} | {p['phi']:.3f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    main(sys.argv[1])
