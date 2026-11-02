"""Decoding-side analysis from committed files only: no GPU, checkpoints or candidate files.

1. Where MBR changes MAP's answer (the decoder-side question of #97, bucketed by top-1
   probability): how often, what kind of change, and whether the changes gain or lose EM / F1.
2. Selective answering (plan E5): EM and F1 when answering only the most confident questions,
   for MAP and MBR, with two confidence signals.

Inputs, per run: predictions_validation_{map,mbr}.json, and cross_span_validation_questions.json
(E3, squadqa.diagnostics), which holds {qid: [p_top1, p_a, p_b, p_c]} at the run's calibrated
temperature. a and b are the top span and the best non-overlapping span with a content word (the
"rival"), in passage order; c joins them. Scores come from the official scorer (squadqa.evaluate).

    python sandbox/ychai/decoding_analysis.py        # from the repo root, under a minute on a laptop

writes reports/decoding_analysis_2026-11-01.json and .md.
"""

import json
import math
import re
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repo root, for squadqa
from squadqa import evaluate as E  # noqa: E402
from squadqa.config import RESULTS, write_json  # noqa: E402

OUT = Path("reports/decoding_analysis_2026-11-01")
MODELS = {   # name: (what it is, PRs that committed its predictions and E3 files, runs for seeds 0-2)
    "row 0": ("baseline: standard head, 1 epoch", "#38, #43",
              ["2026-10-05-row0-baseline-s0-j910132", "2026-10-05-row0-baseline-s1-j910133",
               "2026-10-05-row0-baseline-s2-j910134"]),
    "row 1": ("joint head, one-hot group targets, 2 epochs", "#73, #80",
              ["2026-10-23-row1-joint-s0-j998373", "2026-10-23-row1-joint-s1-j998384",
               "2026-10-23-row1-joint-s2-j998397"]),
    "soft target": ("row 1 trained on F1 soft targets (τ 0.3, φ 0)", "#81, #82",
                    ["2026-10-25-row1-joint-soft-t0.3-f0.0-s0-j1012180",
                     "2026-10-25-row1-joint-soft-t0.3-f0.0-s1-j1012202",
                     "2026-10-25-row1-joint-soft-t0.3-f0.0-s2-j1012210"]),
    "risk share 0.25": ("row 1 trained on 75% CE + 25% risk", "#105",
                        ["2026-10-30-row1-joint-risk-share0.25-s0-j1044574",
                         "2026-10-30-row1-joint-risk-share0.25-s1-j1044582",
                         "2026-10-30-row1-joint-risk-share0.25-s2-j1044589"]),
    "risk share 0.5": ("row 1 trained on 50% CE + 50% risk (risk_weight 1)", "#81, #82",
                       ["2026-10-25-row1-joint-risk-b1.0-s0-j1012186", "2026-10-25-row1-joint-risk-b1.0-s1-j1012206",
                        "2026-10-25-row1-joint-risk-b1.0-s2-j1012213"]),
    "row 5": ("row 1 + AdversarialQA training data (no E3 files)", "#106",
              ["2026-10-30-row1-joint-aqa-s0-j1045742", "2026-10-30-row1-joint-aqa-s1-j1045747",
               "2026-10-30-row1-joint-aqa-s2-j1045751"]),
}
KINDS = ["longer", "shorter", "overlapping", "separate"]
KIND_TEXT = {"longer": "longer: contains MAP's answer", "shorter": "shorter: inside MAP's answer",
             "overlapping": "overlapping, neither inside the other", "separate": "separate: another part of the passage"}
UNCERTAIN = 0.5             # MBR (γ ≥ 1) can change the answer only below this top-1 probability (appendix)
RIVAL = 0.1                 # a rival this likely makes the question "torn between two answers"
COVERAGES = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
TARGETS = [90, 95]          # coverage reachable while the answered questions keep this EM
EXAMPLE_RUN = MODELS["row 1"][2][0]


def bucket(p_top1: float) -> str:
    """squadqa.diagnostics.bucket, copied because that module imports torch."""
    return "<0.5" if p_top1 < 0.5 else "0.5-0.9" if p_top1 <= 0.9 else ">0.9"


def occurrences(context: str, text: str) -> list[tuple[int, int]]:
    """Character spans where text occurs in context as whole words: an answer starts and ends at
    word boundaries, so "Roman" does not occur inside "Romans"."""
    head = r"(?<!\w)" if re.match(r"\w", text) else ""
    tail = r"(?!\w)" if re.search(r"\w$", text) else ""
    found = [(m.start(), m.end()) for m in re.finditer(head + re.escape(text) + tail, context)]
    if not found:
        raise ValueError(f"answer {text!r} is not in its passage")
    return found


def change_kind(context: str, map_text: str, mbr_text: str) -> str:
    """Where MBR's answer lies relative to MAP's in the passage. A text that occurs more than
    once takes the occurrence that relates closest (containment first, then overlap)."""
    found = set()
    for a0, a1 in occurrences(context, map_text):
        for b0, b1 in occurrences(context, mbr_text):
            if b0 <= a0 and a1 <= b1:
                found.add("longer")
            elif a0 <= b0 and b1 <= a1:
                found.add("shorter")
            elif a0 < b1 and b0 < a1:
                found.add("overlapping")
            else:
                found.add("separate")
    return next(k for k in KINDS if k in found)


def coverage_curve(signal: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Mean score of the k most confident questions, for k = 1..n. Ties keep file order; at most
    4 questions share a value in these files."""
    order = np.argsort(-signal, kind="stable")
    return np.cumsum(score[order]) / np.arange(1, len(score) + 1)


def selective(signal: np.ndarray, em: np.ndarray, f1: np.ndarray) -> dict:
    """EM and F1 at each coverage, their averages over all coverages (the areas under the
    curves), and the largest coverage that keeps each EM target."""
    n = len(em)
    cum_em, cum_f1 = coverage_curve(signal, em), coverage_curve(signal, f1)
    at = {c: max(1, round(c / 100 * n)) - 1 for c in COVERAGES}
    out = {"em_at": {c: 100 * cum_em[i] for c, i in at.items()},
           "f1_at": {c: 100 * cum_f1[i] for c, i in at.items()},
           "area_em": 100 * cum_em.mean(), "area_f1": 100 * cum_f1.mean()}
    for t in TARGETS:
        ok = np.nonzero(cum_em >= t / 100)[0]
        out[f"coverage_em{t}"] = 100 * (ok.max() + 1) / n if len(ok) else 0.0
    return out


def analyse_run(run: str, dataset: list[dict], qids: list[str], contexts: dict[str, str]) -> dict:
    d = RESULTS / run
    preds = {rule: E.load_predictions(d / f"predictions_validation_{rule}.json") for rule in ("map", "mbr")}
    scores = {rule: E.per_question(dataset, preds[rule]) for rule in preds}
    em = {rule: np.array([scores[rule][0][q] for q in qids]) for rule in preds}
    f1 = {rule: np.array([scores[rule][1][q] for q in qids]) for rule in preds}
    changed = [q for q in qids if preds["map"][q] != preds["mbr"][q]]
    scorer = E.scorer()
    if any(scorer.compute_exact(preds["map"][q], preds["mbr"][q]) for q in changed):
        raise AssertionError(f"{run}: a changed answer equals MAP's after normalization, so it can't score differently")
    changed_set = set(changed)
    kind = np.array([change_kind(contexts[q], preds["map"][q], preds["mbr"][q]) if q in changed_set else ""
                     for q in qids])
    out = {"run": run, "preds": preds, "em": em, "f1": f1, "kind": kind,
           "summary": {rule: E.summarize(*scores[rule]) for rule in preds}}
    path = d / "cross_span_validation_questions.json"
    if path.exists():
        e3 = json.loads(path.read_text(encoding="utf-8"))
        p = np.array([e3[q] for q in qids])
        top1 = p[:, 0]
        if not np.array_equal(np.maximum(p[:, 1], p[:, 2]), top1):
            raise AssertionError(f"{run}: the top span is neither a nor b")
        rival = p[:, 1] + p[:, 2] - top1                    # 0 when there is no rival span
        joined = json.loads((d / "cross_span_validation.json").read_text(encoding="utf-8"))["cross_answers"]["mbr"]
        out.update(top1=top1, rival=rival, joined=joined,
                   selective={rule: {"top-1 probability": selective(top1, em[rule], f1[rule]),
                                     "margin": selective(top1 - rival, em[rule], f1[rule])}
                              for rule in preds},
                   oracle_area_em={rule: 100 * coverage_curve(em[rule], em[rule]).mean() for rule in preds})
    return out


def stats(values) -> dict:
    values = [float(v) for v in values]
    return {"mean": statistics.mean(values), "std": statistics.stdev(values), "seeds": values}


def contribution(delta: np.ndarray, mask: np.ndarray, qids: list[str]) -> dict:
    """MBR - MAP from the questions in mask, in points of the whole validation set: paired
    bootstrap over questions of each question's mean over seeds (as squadqa.report compare)."""
    c = (delta * mask).mean(axis=0)
    return E.paired_bootstrap(dict.fromkeys(qids, 0.0), dict(zip(qids, c)))


def count(mask: np.ndarray) -> dict:
    return stats(mask.sum(axis=1))


def analyse_model(name: str, runs: list[dict], qids: list[str]) -> dict:
    d_em = np.stack([r["em"]["mbr"] - r["em"]["map"] for r in runs])
    d_f1 = np.stack([r["f1"]["mbr"] - r["f1"]["map"] for r in runs])
    delta = d_em + d_f1
    changed = np.stack([r["kind"] != "" for r in runs])
    out: dict = {
        "runs": [r["run"] for r in runs],
        "scores": {rule: {m: stats([r["summary"][rule][m] for r in runs]) for m in ("exact", "f1")}
                   for rule in ("map", "mbr")},
        "changed": count(changed), "em_gained": count(d_em > 0), "em_lost": count(d_em < 0),
        "f1_better": count(d_f1 > 0), "f1_worse": count(d_f1 < 0),
        "mbr_minus_map": {"exact": contribution(d_em, changed, qids), "f1": contribution(d_f1, changed, qids),
                          "exact_plus_f1": contribution(delta, changed, qids)},
        "kinds": {},
    }
    for k in KINDS:
        mask = np.stack([r["kind"] == k for r in runs])
        out["kinds"][k] = {"changed": count(mask), "em_gained": count(mask & (d_em > 0)),
                           "em_lost": count(mask & (d_em < 0)), "exact_plus_f1": contribution(delta, mask, qids)}
    if "top1" not in runs[0]:
        return out
    top1, rival = np.stack([r["top1"] for r in runs]), np.stack([r["rival"] for r in runs])
    buckets = np.vectorize(bucket)(top1)
    out["buckets"] = {b: {"questions": count(buckets == b), "changed": count(changed & (buckets == b))}
                      for b in ("<0.5", "0.5-0.9", ">0.9")}
    if np.any(changed & (top1 >= UNCERTAIN)):
        raise AssertionError(f"{name}: MBR changed an answer whose top-1 probability is at least {UNCERTAIN}")
    uncertain = top1 < UNCERTAIN
    out["uncertain"] = {}
    kinds = np.stack([r["kind"] for r in runs])
    for label, mask in [("two answers", uncertain & (rival >= RIVAL)), ("boundaries", uncertain & (rival < RIVAL))]:
        out["uncertain"][label] = {"questions": count(mask), "changed": count(mask & changed),
                                   "em_gained": count(mask & (d_em > 0)), "em_lost": count(mask & (d_em < 0)),
                                   "exact_plus_f1": contribution(delta, mask, qids), "kinds": {}}
        for k in KINDS:
            km = mask & (kinds == k)
            out["uncertain"][label]["kinds"][k] = {"changed": count(km), "em_gained": count(km & (d_em > 0)),
                                                   "em_lost": count(km & (d_em < 0))}
    out["joined_span"] = {   # E3: questions whose MBR answer is the span joining the top span and its rival
        "questions": stats([r["joined"]["questions"] for r in runs]),
        **{f"{m}_{rule}": stats([r["joined"]["scores"][rule][m] for r in runs])
           for m in ("exact", "f1") for rule in ("map", "mbr")}}
    out["selective"] = {
        rule: {sig: {key: ({c: stats([r["selective"][rule][sig][key][c] for r in runs]) for c in COVERAGES}
                           if key in ("em_at", "f1_at") else stats([r["selective"][rule][sig][key] for r in runs]))
                     for key in runs[0]["selective"][rule][sig]}
               for sig in ("top-1 probability", "margin")}
        for rule in ("map", "mbr")}
    out["oracle_area_em"] = {rule: stats([r["oracle_area_em"][rule] for r in runs]) for rule in ("map", "mbr")}
    out["signal_difference"] = {   # top-1 probability minus margin, per seed
        m: [r["selective"]["map"]["top-1 probability"][m] - r["selective"]["map"]["margin"][m] for r in runs]
        for m in ("area_em", "area_f1")}
    reports = [RESULTS / run / "report_validation.json" for run in out["runs"]]
    if all(p.exists() for p in reports):
        rep = [json.loads(p.read_text(encoding="utf-8")) for p in reports]
        out["calibration"] = {k: stats([x[k] for x in rep]) for k in ("temperature", "ece_map")}
        out["calibration"]["unit"] = rep[0].get("calibration_unit")
    return out


def examples(run: dict, qids: list[str], info: dict) -> list[dict]:
    """The first EM gain and the first EM loss of each kind of change, in file order."""
    first: dict[tuple[str, bool], dict] = {}
    for i, q in enumerate(qids):
        k = run["kind"][i]
        d = run["em"]["mbr"][i] - run["em"]["map"][i]
        if not k or d == 0 or (k, d > 0) in first:
            continue
        question, golds, _ = info[q]
        first[k, d > 0] = {"id": q, "kind": k, "question": question, "map": run["preds"]["map"][q],
                           "mbr": run["preds"]["mbr"][q], "gold": golds,
                           "top1": float(run["top1"][i]), "rival": float(run["rival"][i]),
                           "em": [float(run["em"]["map"][i]), float(run["em"]["mbr"][i])],
                           "f1": [float(run["f1"]["map"][i]), float(run["f1"]["mbr"][i])]}
    return [first[k, gain] for k in KINDS for gain in (True, False) if (k, gain) in first]


def half_rule_check(cases: int = 20_000, seed: int = 0) -> dict:
    """Random candidate sets through squadqa.decode: how often MBR differs from MAP when MAP's
    answer holds more than half of the candidates' probability (the appendix proves never, for
    γ ≥ 1). Counts may differ under another NumPy version; the zeros shouldn't."""
    from squadqa.decode import Candidates, answer_groups, decode

    rng = np.random.default_rng(seed)
    gammas = [0.0, 0.5, 1.0, 2.0]
    out = {"cases": cases, "seed": seed, "above_half": 0, "mbr_differs": {str(g): 0 for g in gammas}}
    for _ in range(cases):
        n_words = int(rng.integers(3, 12))
        flags = rng.random(n_words) > 0.2                     # about one word in five isn't a content word
        spans = [(i, j) for i in range(n_words) for j in range(i, min(n_words, i + 6))]
        k = int(rng.integers(2, min(10, len(spans)) + 1))
        pick = [spans[i] for i in rng.choice(len(spans), size=k, replace=False)]
        p = rng.dirichlet(np.ones(k) * rng.choice([0.3, 1.0, 3.0])) * rng.uniform(0.5, 1.0)    # total m <= 1
        order = np.argsort(-p, kind="stable")
        cands = Candidates(np.array([pick[o][0] for o in order]), np.array([pick[o][1] for o in order]),
                           np.log(p[order]))
        prob = answer_groups(cands, flags, "normalized")[1]
        if prob.max() <= prob.sum() / 2 + 1e-12:
            continue
        out["above_half"] += 1
        for g in gammas:
            if decode(cands, flags, "mbr", "normalized", g) != decode(cands, flags, "map", "normalized", g):
                out["mbr_differs"][str(g)] += 1
    return out


def pm(s: dict, digits: int = 2) -> str:
    return f"{s['mean']:.{digits}f} ± {s['std']:.{digits}f}"


def ci(c: dict) -> str:
    return f"{c['diff']:+.2f} [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}]"


def share(x: dict, n: int) -> float:
    """A per-seed count's mean as a percentage of n."""
    return x["mean"] / n * 100


def in_all(values: list[float], positive: bool) -> str:
    hits = sum((v > 0) == positive for v in values)
    return f"all {len(values)}" if hits == len(values) else f"{hits} of {len(values)}"


def markdown(res: dict) -> str:
    m, n, hr = res["models"], res["questions"], res["half_rule_check"]
    e3 = [k for k in m if "selective" in m[k]]
    r0, r1, r5, soft = m["row 0"], m["row 1"], m["row 5"], m["soft target"]
    sel = {k: m[k]["selective"]["map"]["top-1 probability"] for k in e3}
    unc1 = r1["buckets"]["<0.5"]["questions"]
    same_upto = math.floor(min(100 - v / n * 100 for v in unc1["seeds"]))     # in every seed
    d_em = [v for k in e3 for v in m[k]["signal_difference"]["area_em"]]
    d_f1 = [v for k in e3 for v in m[k]["signal_difference"]["area_f1"]]
    others = ("risk share 0.25", "risk share 0.5", "row 5")
    longer_other = [m[k]["kinds"]["longer"]["exact_plus_f1"]["diff"] for k in others]
    two0, two1, bnd1 = r0["uncertain"]["two answers"], r1["uncertain"]["two answers"], r1["uncertain"]["boundaries"]
    j0, j1 = r0["joined_span"], r1["joined_span"]
    L = ["# Decoding-side analysis: where MBR changes MAP's answer, and selective answering", "",
         f"Validation split ({n:,} questions), 3 seeds per model, from committed predictions and E3 files only. "
         "Generated by `python sandbox/ychai/decoding_analysis.py`; every number is in the JSON next to this file. "
         "± is the sample standard deviation over seeds; brackets are 95% paired bootstrap intervals over questions "
         "(10,000 resamples) of each question's mean over seeds. Counts are questions per seed.", "",
         "MBR is the project's decoder: expected F1 + γ·EM with γ = 1 over the top 10 candidates, grouped by "
         "content words (`mbr_utility: normalized`). The *top-1 probability* is the most likely span's probability "
         "at the calibrated temperature. The *rival* is the most likely span that doesn't overlap it and has a "
         "content word. Both come from E3.", "",
         "## Findings", "",
         f"1. **MBR changes few answers, and only uncertain ones.** It changes "
         f"{min(share(m[k]['changed'], n) for k in m):.2f}–{max(share(m[k]['changed'], n) for k in m):.2f}% of the "
         f"answers (row 1: {r1['changed']['mean']:.0f} of {n:,}). With γ = 1 it cannot change an answer that holds "
         "more than half of the candidates' probability (appendix), and in every run every change has top-1 "
         f"probability below 0.5. On row 1 that leaves {share(unc1, n):.0f}% of the questions where any MBR with "
         "γ ≥ 1 can act.",
         f"2. **On row 1, MBR gains by lengthening answers whose boundaries are unclear.** Of its "
         f"{ci(r1['mbr_minus_map']['exact_plus_f1'])} EM + F1, longer answers (containing MAP's) give "
         f"{ci(r1['kinds']['longer']['exact_plus_f1'])}: EM gained on {r1['kinds']['longer']['em_gained']['mean']:.1f} "
         f"questions and lost on {r1['kinds']['longer']['em_lost']['mean']:.1f}. The gain is on questions torn "
         f"between boundaries ({ci(bnd1['exact_plus_f1'])}); on questions torn between two separate answers it is "
         f"{ci(two1['exact_plus_f1'])}.",
         f"3. **On row 0, MBR hedges between two answers and loses.** Row 0's MBR − MAP is "
         f"{ci(r0['mbr_minus_map']['exact_plus_f1'])}, and the questions torn between two answers alone give "
         f"{ci(two0['exact_plus_f1'])}: EM gained on {two0['em_gained']['mean']:.1f} and lost on "
         f"{two0['em_lost']['mean']:.1f}, and {two0['kinds']['longer']['changed']['mean']:.0f} of the "
         f"{two0['changed']['mean']:.0f} changes there are longer answers. E3 points the same way: on the "
         f"{j0['questions']['mean']:.0f} questions where row 0's MBR answers with the span joining the top span and its "
         f"rival, its EM is {j0['exact_mbr']['mean']:.1f} against MAP's {j0['exact_map']['mean']:.1f}; on row 1 such "
         f"answers gain ({j1['questions']['mean']:.0f} questions, {j1['exact_map']['mean']:.1f} → "
         f"{j1['exact_mbr']['mean']:.1f}). The joint head more than halved the cross-span probability (#80), and on "
         "row 1 MBR's hedges no longer lose.",
         f"4. **The other readers leave MBR little to gain.** The soft target is sure of more answers (only "
         f"{share(soft['buckets']['<0.5']['questions'], n):.0f}% below 0.5), so MBR changes "
         f"{share(soft['changed'], n):.2f}% of the answers ({ci(soft['mbr_minus_map']['exact_plus_f1'])}). On the risk-loss "
         f"readers and row 5, longer answers gain only {min(longer_other):+.2f} to {max(longer_other):+.2f}, losing "
         "EM about as often as they gain it. Row 5 also switches to another part of the passage twice as often as "
         f"row 1 ({r5['kinds']['separate']['changed']['mean']:.1f} against "
         f"{r1['kinds']['separate']['changed']['mean']:.1f}), losing EM on {r5['kinds']['separate']['em_lost']['mean']:.1f} "
         f"and gaining on {r5['kinds']['separate']['em_gained']['mean']:.1f}. MBR − MAP: "
         + ", ".join(f"{k} {ci(m[k]['mbr_minus_map']['exact_plus_f1'])}" for k in others) + ".",
         f"5. **Abstaining moves accuracy far more than decoding.** Answering only its most confident half, row 1 "
         f"scores {sel['row 1']['em_at'][50]['mean']:.1f} EM / {sel['row 1']['f1_at'][50]['mean']:.1f} F1, and its "
         f"most confident 30% {sel['row 1']['em_at'][30]['mean']:.1f} EM, against "
         f"{sel['row 1']['em_at'][100]['mean']:.1f} / {sel['row 1']['f1_at'][100]['mean']:.1f} on everything. MBR "
         "adds nothing once abstention starts: ranked by top-1 probability, MAP and MBR give the same answers at "
         f"every coverage up to {same_upto:.0f}%, because the questions MBR could change are the first ones "
         "abstained on.",
         f"6. **Top-1 probability is the better signal for EM; the margin, slightly, for F1.** Top-1 probability has "
         f"the larger area under the EM curve in {in_all(d_em, True)} runs (by {min(d_em):+.2f} to {max(d_em):+.2f}), "
         f"and the margin the larger area under the F1 curve in {in_all(d_f1, False)} (by {-max(d_f1):.2f} to "
         f"{-min(d_f1):.2f}). The margin ignores doubt about boundaries, which costs EM but little F1.",
         f"7. **Calibrated is not the same as well ranked.** Row 0 ranks its answers better than row 1: area under "
         f"the EM curve {pm(sel['row 0']['area_em'])} against {pm(sel['row 1']['area_em'])}, and "
         f"{sel['row 0']['coverage_em95']['mean']:.0f}% of questions answerable at EM ≥ 95 against "
         f"{sel['row 1']['coverage_em95']['mean']:.0f}%, though row 1 has the lower ECE "
         f"({r1['calibration']['ece_map']['mean']:.3f} per {r1['calibration']['unit']} against "
         f"{r0['calibration']['ece_map']['mean']:.3f} per {r0['calibration']['unit']}). ECE asks "
         "whether the probabilities are right on average; selective answering needs the right answers to be the "
         "confident ones. The risk-loss readers rank slightly below row 1 and the soft target lowest "
         f"({sel['soft target']['coverage_em95']['mean']:.0f}% at EM ≥ 95). Row 0 and row 1 also differ in epochs and "
         "windows (plan 9.1), so this doesn't single out the head.", "",
         "## Inputs", "", "| Model | What it is | Committed in |", "|---|---|---|"]
    for name, (what, prs, _) in MODELS.items():
        L.append(f"| {name} | {what} | {prs} |")

    L += ["", "## 1. Where MBR changes MAP's answer", "",
          "*EM gained / lost*: questions where MBR's answer is an exact match and MAP's isn't, or the reverse. "
          "MBR − MAP is over the whole validation set, in points; all of it comes from the changed questions.", "",
          "| Model | Changed | EM gained | EM lost | F1 better | F1 worse | MBR − MAP, EM | MBR − MAP, F1 | "
          "MBR − MAP, EM + F1 |", "|---|---|---|---|---|---|---|---|---|"]
    for name, x in m.items():
        L.append(f"| {name} | {pm(x['changed'], 1)} ({share(x['changed'], n):.2f}%) "
                 f"| {pm(x['em_gained'], 1)} | {pm(x['em_lost'], 1)} | {pm(x['f1_better'], 1)} | {pm(x['f1_worse'], 1)} "
                 f"| {ci(x['mbr_minus_map']['exact'])} | {ci(x['mbr_minus_map']['f1'])} "
                 f"| {ci(x['mbr_minus_map']['exact_plus_f1'])} |")

    L += ["", "### Only uncertain questions can change", "",
          "With γ ≥ 1, MBR returns MAP's answer whenever that answer holds more than half of the candidates' "
          "probability (appendix). A top-1 span with probability ≥ 0.5 makes its answer that one, so no such "
          "question can change. In the data, every change in every run is in the < 0.5 bucket.", "",
          "| Model | Questions < 0.5 | Changed | Questions 0.5–0.9 | Changed | Questions > 0.9 | Changed |",
          "|---|---|---|---|---|---|---|"]
    for name in e3:
        b = m[name]["buckets"]
        L.append(f"| {name} | {pm(b['<0.5']['questions'], 0)} | {pm(b['<0.5']['changed'], 1)} "
                 f"| {pm(b['0.5-0.9']['questions'], 0)} | {b['0.5-0.9']['changed']['mean']:.0f} "
                 f"| {pm(b['>0.9']['questions'], 0)} | {b['>0.9']['changed']['mean']:.0f} |")

    L += ["", "### What kind of change", "",
          "Where MBR's answer lies in the passage relative to MAP's. The last column is that kind's share of "
          "MBR − MAP (EM + F1, whole-set points).", "",
          "| Model | Kind | Changed | EM gained | EM lost | Contribution to EM + F1 |", "|---|---|---|---|---|---|"]
    for name, x in m.items():
        for k in KINDS:
            y = x["kinds"][k]
            L.append(f"| {name} | {KIND_TEXT[k]} | {pm(y['changed'], 1)} | {pm(y['em_gained'], 1)} "
                     f"| {pm(y['em_lost'], 1)} | {ci(y['exact_plus_f1'])} |")

    L += ["", "### Torn between two answers, or between boundaries?", "",
          f"The uncertain questions (top-1 probability < {UNCERTAIN}) split by their rival: *two answers* when the "
          f"rival, in another part of the passage, has probability ≥ {RIVAL}; *boundaries* otherwise, when the "
          "missing probability sits on spans overlapping the top one or is spread thin. The *longer* and "
          "*separate* columns give those kinds' changes, with EM gained / lost.", "",
          "| Model | Torn between | Questions | Changed | EM gained | EM lost | Longer (EM + / −) | Separate (EM + / −) "
          "| Contribution to EM + F1 |", "|---|---|---|---|---|---|---|---|---|"]
    for name in e3:
        for label, y in m[name]["uncertain"].items():
            kc = {k: f"{y['kinds'][k]['changed']['mean']:.1f} (+{y['kinds'][k]['em_gained']['mean']:.1f} / "
                     f"−{y['kinds'][k]['em_lost']['mean']:.1f})" for k in ("longer", "separate")}
            L.append(f"| {name} | {label} | {pm(y['questions'], 0)} | {pm(y['changed'], 1)} | {pm(y['em_gained'], 1)} "
                     f"| {pm(y['em_lost'], 1)} | {kc['longer']} | {kc['separate']} | {ci(y['exact_plus_f1'])} |")
    L += ["", "E3's own count (`cross_answers` in each run's `cross_span_validation.json`): the questions whose MBR "
          "answer is the span joining the top span and its rival, and both decoders' scores on them.", "",
          "| Model | MBR answers with the joined span | EM: MAP → MBR | F1: MAP → MBR |", "|---|---|---|---|"]
    for name in e3:
        j = m[name]["joined_span"]
        L.append(f"| {name} | {pm(j['questions'], 0)} | {j['exact_map']['mean']:.1f} → {j['exact_mbr']['mean']:.1f} "
                 f"| {j['f1_map']['mean']:.1f} → {j['f1_mbr']['mean']:.1f} |")

    shown = {(x["kind"], x["em"][1] > x["em"][0]) for x in res["examples"]}
    missing = [f"{k} change that {'gains' if gain else 'loses'} EM" for k in KINDS for gain in (True, False)
               if (k, gain) not in shown]
    L += ["", f"### Examples (row 1, seed 0, `{EXAMPLE_RUN}`)", "",
          "The first EM gain and the first EM loss of each kind of change, in file order. Seed 0 has no "
          + ", no ".join(missing) + ". Validation has one gold answer per question.", "",
          "| Kind | Question | MAP | MBR | Gold | Top-1 / rival | EM | F1 |", "|---|---|---|---|---|---|---|---|"]
    for x in res["examples"]:
        cells = [x["kind"], x["question"], x["map"], x["mbr"], x["gold"][0]]
        L.append("| " + " | ".join(c.replace("|", "\\|") for c in cells)
                 + f" | {x['top1']:.2f} / {x['rival']:.2f} | {x['em'][0]:.0f} → {x['em'][1]:.0f} "
                 f"| {x['f1'][0]:.2f} → {x['f1'][1]:.2f} |")

    worst = max(r1["selective"][rule][sig][key][c]["std"] for rule in ("map", "mbr")
                for sig in ("top-1 probability", "margin") for key in ("em_at", "f1_at") for c in COVERAGES)
    L += ["", "## 2. Selective answering (E5)", "",
          "Answer only the most confident questions and abstain on the rest; *coverage* is the share answered. Two "
          "confidence signals: the **top-1 probability**, and the **margin**, top-1 probability minus the rival's, "
          "which ignores probability on boundary variants of the top answer.", "",
          "### Row 1 (joint head): EM and F1 of the answered questions", "",
          "| Coverage | " + " | ".join(f"{c}%" for c in COVERAGES) + " |", "|---|" + "---|" * len(COVERAGES)]
    for rule in ("map", "mbr"):
        for sig in ("top-1 probability", "margin"):
            s = r1["selective"][rule][sig]
            for met, key in (("EM", "em_at"), ("F1", "f1_at")):
                L.append(f"| {rule.upper()}, {sig}, {met} | "
                         + " | ".join(f"{s[key][c]['mean']:.2f}" for c in COVERAGES) + " |")
    L += ["", f"The seeds' std is at most {worst:.2f} in this table. Ranked by top-1 probability, MAP and MBR give "
          f"the same answers at every coverage up to {same_upto:.0f}% in every seed.", "",
          "### Every model: the whole curve in one number", "",
          "*Area*: EM (or F1) of the answered questions averaged over all coverages, the area under the curve; an "
          "oracle that answers its correct questions first reaches the last column. *Coverage at EM ≥ 90 / 95*: the "
          "most questions one can answer while the answered ones keep that EM. MAP answers. ECE (of the MAP answer, "
          "in the unit the run was calibrated in) and the temperature T come from each run's "
          "`report_validation.json`.", "",
          "| Model | Signal | Area, EM | Area, F1 | Coverage at EM ≥ 90 | Coverage at EM ≥ 95 | ECE (unit) | T "
          "| Oracle area, EM |", "|---|---|---|---|---|---|---|---|---|"]
    for name in e3:
        x = m[name]
        cal = x["calibration"]
        for sig in ("top-1 probability", "margin"):
            s = x["selective"]["map"][sig]
            L.append(f"| {name} | {sig} | {pm(s['area_em'])} | {pm(s['area_f1'])} | {pm(s['coverage_em90'], 1)}% "
                     f"| {pm(s['coverage_em95'], 1)}% | {cal['ece_map']['mean']:.3f} ({cal['unit']}) "
                     f"| {cal['temperature']['mean']:.2f} | {x['oracle_area_em']['map']['mean']:.2f} |")
    L += ["", "Top-1 probability minus margin, per seed (MAP). Area under the EM curve: "
          + "; ".join(f"{k} " + ", ".join(f"{v:+.2f}" for v in m[k]["signal_difference"]["area_em"]) for k in e3)
          + ". Area under the F1 curve: "
          + "; ".join(f"{k} " + ", ".join(f"{v:+.2f}" for v in m[k]["signal_difference"]["area_f1"]) for k in e3) + "."]

    L += ["", "## Appendix: why MBR keeps an answer with more than half the probability", "",
          "Let g be MAP's answer (the most likely group of boundary variants) with probability p, m ≤ 1 the "
          "candidates' total probability, and y any other answer, with probability q ≤ m − p and f = F1(g, y). "
          "Different answers have EM 0 with each other, and F1 ≤ 1. Then", "",
          "- U(g) ≥ (1 + γ)·p + f·q (g against itself and against y);",
          "- U(y) ≤ (1 + γ)·q + f·p + (m − p − q) (y against itself, against g, and at most 1 against the rest).", "",
          "So U(g) − U(y) ≥ (2 + γ − f)·p − (γ − f)·q − m. For γ ≥ 1 ≥ f the q term is never positive, and q ≤ m − p "
          "gives U(g) − U(y) ≥ (1 + γ − f)·(2p − m), which is positive once p > m / 2: MBR returns g. A top-1 span with "
          "probability ≥ 0.5 belongs to a group with at least that much, which no other group can beat, so that "
          "group is g and p ≥ 0.5 ≥ m / 2.", "",
          "With γ < 1 the guarantee weakens to p > m / (1 + γ), and with γ = 0 (expected F1 alone) it is gone. For "
          "example, \"A B\" with probability 0.55 and \"B C\" with 0.45 give the candidate \"A B C\", even with "
          "almost no probability of its own, an expected F1 of 0.80 against 0.775 for \"A B\". With γ = 1, \"A B\" "
          "keeps the lead (1.325 against 0.80).", "",
          f"A random check through `squadqa.decode` agrees (`half_rule_check`, seed {hr['seed']}). Of "
          f"{hr['cases']:,} random candidate sets, {hr['above_half']:,} have MAP's answer above half the "
          "candidates' probability. MBR changes that answer in "
          f"{hr['mbr_differs']['1.0']} of them with γ = 1, {hr['mbr_differs']['2.0']} with γ = 2 and "
          f"{hr['mbr_differs']['0.5']} with γ = 0.5, against {hr['mbr_differs']['0.0']} with γ = 0."]
    return "\n".join(L) + "\n"


def main() -> None:
    dataset = E.load_eval_set("validation")
    qids = E.question_ids(dataset)
    info = {qa["id"]: (qa["question"], [a["text"] for a in qa["answers"]], p["context"])
            for a in dataset for p in a["paragraphs"] for qa in p["qas"]}
    contexts = {q: v[2] for q, v in info.items()}
    res: dict = {"split": "validation", "questions": len(qids), "uncertain_below": UNCERTAIN, "rival_at_least": RIVAL,
                 "models": {}}
    for name, (_, _, runs) in MODELS.items():
        analysed = [analyse_run(run, dataset, qids, contexts) for run in runs]
        res["models"][name] = analyse_model(name, analysed, qids)
        if runs[0] == EXAMPLE_RUN:
            res["examples"] = examples(analysed[0], qids, info)
        print(f"{name}: done", file=sys.stderr)
    res["half_rule_check"] = half_rule_check()
    write_json(OUT.with_suffix(".json"), res)
    OUT.with_suffix(".md").write_text(markdown(res), encoding="utf-8")
    print(f"wrote {OUT}.json and .md", file=sys.stderr)


if __name__ == "__main__":
    main()
