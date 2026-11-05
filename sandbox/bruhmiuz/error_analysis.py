"""Error analysis of a reader on the validation split (exploratory; nothing in squadqa/ uses it).

Classifies each exact-match miss of the first run's predictions by where the predicted span
sits relative to the gold span (too short, too long, shifted, elsewhere), counts the words added
or dropped at the edges, breaks EM and F1 down by question type and answer length, counts how
many of the runs miss each question, finds the rank of the first acceptable candidate in
candidates_validation.json, and bins EM by the top candidate's probability. It also writes up
to 25 random misses per class, with passage context, to <out_dir>/error_samples.txt, and the main
counts to <out_dir>/summary.json.

    python sandbox/bruhmiuz/error_analysis.py <out_dir> [--rule RULE] [--candidates RUN] [run_id ...]

The runs default to the three row-0 seeds, and RULE to map: combined_mbr, for instance, reads a verifier
run's combined answers. Each run needs results/<run_id>/predictions_validation_<rule>.json. The candidates
come from the first run's candidates_validation.json, or from --candidates RUN (a verifier run's answers
lie in its reader's candidates). They are not committed: regenerate them with
python -m squadqa.predict --run-id <run_id> (workflow.md §2.2).
"""
import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from squadqa import evaluate as E
from squadqa.data import answer_word_span, load_split, word_spans

parser = argparse.ArgumentParser()
parser.add_argument("out_dir")
parser.add_argument("--rule", default="map")
parser.add_argument("--candidates", help="the run whose candidates_validation.json to use (default: the first run)")
parser.add_argument("runs", nargs="*", default=["2026-10-05-row0-baseline-s0-j910132", "2026-10-05-row0-baseline-s1-j910133",
                                                "2026-10-05-row0-baseline-s2-j910134"])
args = parser.parse_intermixed_args()
RUNS = args.runs
OUT = Path(args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)
sc = E.scorer()
qs = load_split("validation")
preds = [json.loads(Path(f"results/{r}/predictions_validation_{args.rule}.json").read_text(encoding="utf-8")) for r in RUNS]
cands = json.loads(Path(f"results/{args.candidates or RUNS[0]}/candidates_validation.json").read_text(encoding="utf-8"))
# Matched in this order, so "how many" wins over "how" and "what year" over "what".
QTYPES = ["how many", "how much", "how long", "how old", "how far", "what year", "which year", "when", "who",
          "whom", "whose", "where", "why", "how", "which", "what", "name"]


def em_f1(gold, pred):
    return sc.compute_exact(gold, pred), sc.compute_f1(gold, pred)


def qtype(q):
    t = q.lower()
    for k in QTYPES:
        if re.search(r"\b" + k + r"\b", t):
            return {"whom": "who", "whose": "who", "which year": "what year"}.get(k, k)
    return "other"


rows = []
for q in qs:
    gold_text, gold_start = q.answers[0]
    words = word_spans(q.context)
    g = answer_word_span(words, gold_start, gold_text)
    p = [pr[q.id] for pr in preds]
    em, f1 = em_f1(gold_text, p[0])
    runs_wrong = sum(em_f1(gold_text, x)[0] == 0 for x in p)
    cl = cands.get(q.id, [])
    # position of the prediction: the most probable candidate whose text equals it
    pos = next(((i, j, lp) for i, j, lp in cl if q.context[words[i][0]:words[j][1]] == p[0]), None)
    # rank of the first candidate the scorer counts as EM
    em_rank = next((r for r, (i, j, _) in enumerate(cl)
                    if sc.compute_exact(gold_text, q.context[words[i][0]:words[j][1]])), None)
    top_p = float(np.exp(cl[0][2])) if cl else float("nan")
    rows.append(dict(q=q, words=words, g=g, p=p[0], pos=pos, em=em, f1=f1, runs_wrong=runs_wrong,
                     em_rank=em_rank, top_p=top_p, type=qtype(q.question), glen=len(gold_text.split())))

n = len(rows)
wrong = [r for r in rows if r["em"] == 0]
lines = []
P = lines.append
P(f"questions {n}; EM {100*np.mean([r['em'] for r in rows]):.2f} F1 {100*np.mean([r['f1'] for r in rows]):.2f}; "
  f"EM errors {len(wrong)}")

# 1. error classes
BOUNDARY = ("too", "overlap")
cls = Counter()
edge = Counter()
for r in wrong:
    g, pos = r["g"], r["pos"]
    if g is None or pos is None:
        c = "unlocated"
    else:
        (gi, gj), (pi, pj) = g, pos[:2]
        if pj < gi or pi > gj:
            c = "disjoint, F1>0" if r["f1"] > 0 else "disjoint, F1=0"
        elif pi <= gi and pj >= gj:
            c = "too long (contains gold)"
        elif pi >= gi and pj <= gj:
            c = "too short (inside gold)"
        else:
            c = "overlapping, shifted"
        if c.startswith(BOUNDARY):
            w, ctx = r["words"], r["q"].context
            side = lambda a, b: ctx[w[a][0]:w[b][1]] if a <= b else ""
            if pi < gi: edge["extra left: " + side(pi, gi - 1).lower()] += 1
            if pj > gj: edge["extra right: " + side(gj + 1, pj).lower()] += 1
            if pi > gi: edge["missing left: " + side(gi, pi - 1).lower()] += 1
            if pj < gj: edge["missing right: " + side(pj + 1, gj).lower()] += 1
    r["cls"] = c
    cls[c] += 1
P("\n## Error classes (share of EM errors; mean F1 within class)")
for c, k in cls.most_common():
    P(f"  {c:28s} {k:5d}  {100*k/len(wrong):5.1f}%  meanF1 {100*np.mean([r['f1'] for r in wrong if r['cls']==c]):5.1f}")
P(f"  F1 lost to boundary errors (too long/short/shifted): "
  f"{100*sum(1-r['f1'] for r in wrong if r['cls'].startswith(BOUNDARY))/n:.2f} F1 points; "
  f"to disjoint: {100*sum(1-r['f1'] for r in wrong if r['cls'].startswith('disjoint'))/n:.2f}")

edge_len = Counter()
for key, k in edge.items():
    d, txt = key.split(": ", 1)
    edge_len[(d, len(txt.split()) if len(txt.split()) < 4 else "4+")] += k
P("\n## Boundary edits: side and size (words)")
for key, k in sorted(edge_len.items(), key=lambda x: -x[1])[:12]:
    P(f"  {key}: {k}")
P("\n## Most common boundary edits")
for key, k in edge.most_common(30):
    P(f"  {k:4d}  {key!r}")

# 2. by question type
P("\n## By question type: n, EM, F1, share of all EM errors")
bt = defaultdict(list)
for r in rows:
    bt[r["type"]].append(r)
for t, rs in sorted(bt.items(), key=lambda x: -len(x[1])):
    P(f"  {t:10s} {len(rs):5d}  EM {100*np.mean([r['em'] for r in rs]):5.1f}  F1 {100*np.mean([r['f1'] for r in rs]):5.1f}"
      f"  errors {100*sum(r['em']==0 for r in rs)/len(wrong):5.1f}%")

# 3. by gold answer length
P("\n## By gold answer length (whitespace words): n, EM, F1")
for lo, hi in [(1, 1), (2, 2), (3, 3), (4, 5), (6, 9), (10, 999)]:
    rs = [r for r in rows if lo <= r["glen"] <= hi]
    P(f"  {lo}-{hi if hi<999 else '+'}: {len(rs):5d}  EM {100*np.mean([r['em'] for r in rs]):5.1f}  F1 {100*np.mean([r['f1'] for r in rs]):5.1f}")

# 4. agreement between runs
P(f"\n## Runs that miss each question (EM), of {len(RUNS)}")
for k, v in sorted(Counter(r["runs_wrong"] for r in rows).items()):
    P(f"  {k}: {v} ({100*v/n:.1f}%)")

# 5. rank of the first acceptable candidate
k_max = max(len(c) for c in cands.values())
P(f"\n## Rank of the first candidate the scorer counts as EM (errors only; k = {k_max})")
rk = Counter()
for r in wrong:
    e = r["em_rank"]
    rk["not in top-k" if e is None else (str(e + 1) if e < 4 else "5-10" if e < 10 else "11+")] += 1
for key in ["1", "2", "3", "4", "5-10", "11+", "not in top-k"]:
    if rk[key]:
        P(f"  {key}: {rk[key]} ({100*rk[key]/len(wrong):.1f}%)")
correct = n - len(wrong)
for top in (2, k_max):
    hits = sum(1 for r in wrong if r["em_rank"] is not None and r["em_rank"] < top)
    P(f"  oracle EM choosing among top {top}: {100*(correct+hits)/n:.1f}")

# 6. confidence
P("\n## Top-candidate probability (temperature-scaled)")
for name, rs in [("correct", [r for r in rows if r["em"]]),
                 ("boundary err", [r for r in wrong if r["cls"].startswith(BOUNDARY)]),
                 ("disjoint err", [r for r in wrong if r["cls"].startswith("disjoint")])]:
    ps = np.array([r["top_p"] for r in rs])
    P(f"  {name:13s} n {len(rs):5d}  mean {ps.mean():.2f}  share p>0.9 {np.mean(ps>0.9):.2f}")
for lo, hi in [(0, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)]:
    rs = [r for r in rows if lo <= r["top_p"] < hi]
    P(f"  p in [{lo:.1f},{min(hi,1):.1f}): n {len(rs):5d}  EM {100*np.mean([r['em'] for r in rs]):5.1f}")

print("\n".join(lines))
(OUT / "summary.json").write_text(json.dumps({
    "runs": RUNS, "rule": args.rule, "questions": n, "em": 100 * float(np.mean([r["em"] for r in rows])),
    "f1": 100 * float(np.mean([r["f1"] for r in rows])), "errors": len(wrong), "classes": dict(cls),
    "f1_lost_boundary": 100 * sum(1 - r["f1"] for r in wrong if r["cls"].startswith(BOUNDARY)) / n,
    "f1_lost_disjoint": 100 * sum(1 - r["f1"] for r in wrong if r["cls"].startswith("disjoint")) / n,
    "first_em_rank": dict(rk), "oracle_em": {str(top): 100 * (correct + sum(1 for r in wrong if r["em_rank"] is not None and r["em_rank"] < top)) / n
                                              for top in (2, 5, k_max)},
    "runs_wrong": {str(k): v for k, v in Counter(r["runs_wrong"] for r in rows).items()}}, indent=1), encoding="utf-8")

# samples for reading
random.seed(0)
with open(OUT / "error_samples.txt", "w", encoding="utf-8") as f:
    for c in cls:
        rs = [r for r in wrong if r["cls"] == c]
        f.write(f"\n######## {c} ({len(rs)})\n")
        for r in random.sample(rs, min(25, len(rs))):
            q = r["q"]
            gs = q.answers[0][1]
            snip = q.context.replace("\n", " ") if c == "disjoint, F1=0" else q.context[max(0, gs - 150):gs + len(q.answers[0][0]) + 150].replace("\n", " ")
            f.write(f"\n[{q.id}] {q.question}\n  gold: {q.answers[0][0]!r}\n  pred: {r['p']!r}  (F1 {r['f1']:.2f}, "
                    f"p {r['top_p']:.2f}, runs wrong {r['runs_wrong']})\n  ctx: ...{snip}...\n")
