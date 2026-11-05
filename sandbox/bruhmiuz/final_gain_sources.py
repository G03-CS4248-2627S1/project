"""Where row 4's gain comes from, per set, after the fix: combined against the reader on the same lists, split by whether
the two answers differ only by non-ASCII punctuation at an edge (which the scorer doesn't strip) or otherwise. Also how
far the reader on the lists now is from predict's reader (k = 5 against 10). Reporting only: nothing here is fitted, and
nothing feeds back into a system. Writes reports/final_scoring/gain_sources.json.

    python sandbox/bruhmiuz/final_gain_sources.py

Nothing in squadqa/ uses this script.
"""
import json
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, ".")
from squadqa import evaluate as E

READER = Path("results/2026-10-23-row1-joint-s0-j998373")
RUNS = ["2026-10-29-verifier-separate-s0-j1032972", "2026-10-30-verifier-separate-s1-j1041306", "2026-10-30-verifier-separate-s2-j1041319"]
SETS = ["validation", *E.FINAL_SETS]
norm = E.scorer().normalize_answer


def core(s):
    s = s.strip()
    while s and ord(s[0]) > 127 and unicodedata.category(s[0]).startswith("P"):
        s = s[1:].strip()
    while s and ord(s[-1]) > 127 and unicodedata.category(s[-1]).startswith("P"):
        s = s[:-1].strip()
    return norm(s)


def mean(xs):
    return sum(xs) / len(xs)


out = {}
for s in SETS:
    data = E.load_eval_set(s, final=s != "validation")
    ids = set(E.question_ids(data))
    predict = {q: a for q, a in E.load_predictions(READER / f"predictions_{s}_mbr.json").items() if q in ids}
    pe, pf = E.per_question(data, predict)
    row = {"questions": len(ids), "by_seed": []}
    for run in RUNS:
        lists = E.load_predictions(Path("results") / run / f"predictions_{s}_reader_mbr.json")
        comb = E.load_predictions(Path("results") / run / f"predictions_{s}_combined_mbr.json")
        le, lf = E.per_question(data, lists)
        ce, cf = E.per_question(data, comb)
        edge, other = [], []
        for q in ids:
            if norm(lists[q]) == norm(comb[q]):
                continue
            (edge if core(lists[q]) == core(comb[q]) else other).append(q)
        n = len(ids)
        row["by_seed"].append({
            "lists_minus_predict_em": 100 * sum(le[q] - pe[q] for q in ids) / n,
            "lists_differ_from_predict": 100 * sum(norm(lists[q]) != norm(predict[q]) for q in ids) / n,
            "gain_em": 100 * sum(ce[q] - le[q] for q in ids) / n,
            "gain_f1": 100 * sum(cf[q] - lf[q] for q in ids) / n,
            "edge_share": 100 * len(edge) / n,
            "edge_gain_em": 100 * sum(ce[q] - le[q] for q in edge) / n,
            "edge_gain_f1": 100 * sum(cf[q] - lf[q] for q in edge) / n,
            "other_share": 100 * len(other) / n,
            "other_gain_em": 100 * sum(ce[q] - le[q] for q in other) / n,
            "other_gain_f1": 100 * sum(cf[q] - lf[q] for q in other) / n,
        })
    row["mean"] = {k: mean([b[k] for b in row["by_seed"]]) for k in row["by_seed"][0]}
    out[s] = row
    m = row["mean"]
    print(f"{s:28s} lists-predict EM {m['lists_minus_predict_em']:+.2f} (differ {m['lists_differ_from_predict']:.2f}%) | gain EM {m['gain_em']:+.2f} F1 {m['gain_f1']:+.2f} | "
          f"edge-only {m['edge_share']:.2f}%: EM {m['edge_gain_em']:+.2f} F1 {m['edge_gain_f1']:+.2f} | other {m['other_share']:.2f}%: EM {m['other_gain_em']:+.2f} F1 {m['other_gain_f1']:+.2f}", flush=True)
Path("reports/final_scoring/gain_sources.json").write_text(json.dumps(out, indent=1) + "\n", encoding="ascii", newline="\n")
print("written")
