"""Reports over finished runs, decode-only on the CPU (spec §6): calibration error, top-k recall,
and comparisons of systems over seeds.

Per run, from results/<run_id>/: its config, calibration.json, candidates_<split>.json and MAP
predictions. Candidates are not committed (workflow.md §2.2), so a run is reported where its
candidates are: the machine that fetched it. The report records:
- T, and the NLL of the gold at T and at 1, from calibration.json.
- ECE of the MAP answer (Guo et al., 2017): the answer's probability, the summed probability of
  the chosen group among the saved top-k candidates at T (decode.answer_groups under the run's
  mbr_utility), in 10 equal-width bins, against its official EM. A group can have members below
  the top k, so this probability is a lower bound.
- Top-k recall for k in 1, 5, 10 (up to the saved k): the share of questions where some top-k
  candidate has official EM 1 against a reference. It caps a verifier that picks among them.

A system is one decoding rule over the runs (seeds) of one config, written name=rule:run,run,...
compare gives each system's EM, F1 and EM + F1 as mean ± sample std over its runs, the mean of
its runs' reports where they exist, and paired bootstrap 95% intervals (evaluate.paired_bootstrap)
for contrasts: signed sums of systems, such as B-A, or D+A-B-C for an interaction. A contrast
uses per-question scores averaged over each system's runs and resamples questions, so its
interval covers question sampling; the seed spread is the std. A rule names a predictions file: map
or mbr for a reader, or reader_, verifier_ or combined_ before them for a verifier run. --split scores
another evaluation set, which needs --final (stage 5 only, workflow.md §6.1); on Adversarial SQuAD
every score is over the worst-case groups, as in evaluate.metrics. --first-reference scores against
each question's first answer alone, the score to set beside validation's one reference.

heldout checks a verifier run's combination on articles it wasn't fitted on: λ and the temperature
are fitted on half the validation articles and scored on the other half, over random splits, both
ways round. It needs the run's verifier_scores_validation.json and the lists they belong to.

    python -m squadqa.report run <run_id>...        # writes results/<run_id>/report_validation.json
    python -m squadqa.report compare A=map:<run>,<run> B=mbr:<run>,<run> --contrast B-A [-o out.json]
    python -m squadqa.report compare A=map:<run> B=combined_mbr:<run> --contrast B-A --split dev --final
    python -m squadqa.report heldout <verifier run_id>... [--lists DIR] [--splits N]   # writes heldout_validation.json

Run from the repo root, like every entry point.
"""

import argparse
import json
import re
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from squadqa import evaluate as E
from squadqa.config import RESULTS, write_json
from squadqa.data import word_spans
from squadqa.decode import Candidates, answer_groups, answer_text, content_flags, decode, log_softmax

BINS = 10
RECALL_KS = (1, 5, 10)
NAME = r"[A-Za-z0-9_]+"
RULE = r"(?:(?:reader|verifier|combined)_)?(?:map|mbr)"


def ece(confidence: Sequence[float], correct: Sequence[float], bins: int = BINS) -> float:
    """Expected calibration error: |accuracy - mean confidence| per equal-width bin of confidence,
    weighted by the bin's share of questions. A confidence of exactly 1 falls in the last bin."""
    conf = np.asarray(confidence, dtype=np.float64)
    ok = np.asarray(correct, dtype=np.float64)
    if conf.ndim != 1 or conf.shape != ok.shape or not len(conf):
        raise ValueError("confidence and correct must be non-empty and of the same length")
    if ((conf < 0) | (conf > 1)).any():
        raise ValueError("confidences must lie in [0, 1]")
    b = np.minimum((conf * bins).astype(int), bins - 1)
    return float(sum((b == k).mean() * abs(ok[b == k].mean() - conf[b == k].mean())
                     for k in range(bins) if (b == k).any()))


def question_diagnostics(context: str, golds: Sequence[str], cands: Sequence[Sequence[float]], utility: str,
                         ks: Sequence[int] = RECALL_KS) -> tuple[str, float, list[bool]]:
    """One question's MAP answer text, its probability, and for each k whether some top-k candidate
    has official EM 1 against a gold answer. cands: [[start_word, end_word, logp], ...] as saved by
    squadqa.predict, most probable first."""
    words = word_spans(context)
    arr = np.asarray(cands, dtype=np.float64)
    c = Candidates(arr[:, 0].astype(np.int64), arr[:, 1].astype(np.int64), arr[:, 2])
    texts = [answer_text(context, words, int(i), int(j)) for i, j in zip(c.starts, c.ends)]
    rep, prob = answer_groups(c, content_flags(context, words), utility, texts)
    best = int(np.argmax(prob))
    hit = [max(E.scorer().compute_exact(g, t) for g in golds) for t in texts]
    return texts[rep[best]], float(min(prob[best], 1.0)), [any(hit[:k]) for k in ks]


def run_report(run_id: str, split: str = "validation") -> dict:
    """Calibration and top-k recall of one run (see the module docstring)."""
    from squadqa.predict import run_config

    run_dir = RESULTS / run_id
    cfg = run_config(run_id)
    cal = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))
    cands_path = run_dir / f"candidates_{split}.json"
    if not cands_path.exists():
        raise FileNotFoundError(f"{cands_path} is missing; candidates stay on the machine that fetched the run")
    cands = json.loads(cands_path.read_text(encoding="utf-8"))
    preds = E.load_predictions(run_dir / f"predictions_{split}_map.json")
    dataset = E.load_eval_set(split)
    exact, _ = E.per_question(dataset, preds)
    ks = [k for k in RECALL_KS if k <= cfg["decode_topk"]]
    conf, correct, recall, differ = [], [], [], 0
    for article in dataset:
        for p in article["paragraphs"]:
            for qa in p["qas"]:
                text, prob, rec = question_diagnostics(p["context"], [a["text"] for a in qa["answers"]],
                                                       cands[qa["id"]], cfg["mbr_utility"], ks)
                differ += text != preds[qa["id"]]          # near-ties: the saved logp are rounded
                conf.append(prob)
                correct.append(exact[qa["id"]])
                recall.append(rec)
    if differ > 0.001 * len(conf):
        raise RuntimeError(f"{run_id}: the MAP answer rebuilt from the candidates differs from the saved one "
                           f"on {differ} of {len(conf)} questions; wrong mbr_utility or candidates?")
    rec = np.array(recall)
    return {"split": split, "questions": len(conf), "temperature": cal["temperature"],
            "nll": cal["nll"], "nll_at_1": cal["nll_at_1"], "calibration_unit": cal.get("unit", "span"),
            "ece_map": ece(conf, correct), "ece_bins": BINS,
            "mean_confidence_map": float(np.mean(conf)), "accuracy_map": float(np.mean(correct)),
            "topk_recall": {str(k): float(rec[:, n].mean()) for n, k in enumerate(ks)},
            "map_rebuilt_differs": differ}


def parse_system(spec: str) -> tuple[str, str, list[str]]:
    """'B=mbr:run1,run2' -> ('B', 'mbr', ['run1', 'run2'])."""
    m = re.fullmatch(rf"({NAME})=({RULE}):([^,]+(?:,[^,]+)*)", spec)
    if not m:
        raise ValueError(f"a system is name=rule:run,run,... (name of letters, digits, _; rule map, mbr, or "
                         f"reader_, verifier_ or combined_ before them), got {spec!r}")
    return m[1], m[2], m[3].split(",")


def parse_contrast(expr: str, names: Sequence[str]) -> dict[str, int]:
    """'D+A-B-C' -> {'D': 1, 'A': 1, 'B': -1, 'C': -1}; a repeated name adds up."""
    if not re.fullmatch(rf"[+-]?{NAME}(?:[+-]{NAME})*", expr):
        raise ValueError(f"a contrast is a signed sum of system names, such as B-A or D+A-B-C; got {expr!r}")
    signs: dict[str, int] = {}
    for sign, name in re.findall(rf"([+-]?)({NAME})", expr):
        if name not in names:
            raise ValueError(f"contrast {expr!r} names {name!r}, which is not a system")
        signs[name] = signs.get(name, 0) + (-1 if sign == "-" else 1)
    return signs


def signed_sums(per: Mapping[str, Mapping[str, float]], signs: Mapping[str, int]) -> tuple[dict, dict]:
    """Per-question sums of the negative and of the positive terms of a contrast, so that
    paired_bootstrap(neg, pos) estimates the contrast."""
    keys = next(iter(per.values())).keys()
    neg = {k: sum(-s * per[n][k] for n, s in signs.items() if s < 0) for k in keys}
    pos = {k: sum(s * per[n][k] for n, s in signs.items() if s > 0) for k in keys}
    return neg, pos


def _stats(values: Sequence[float]) -> dict:
    return {"mean": statistics.mean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "n": len(values)}


def compare(systems: Sequence[tuple[str, str, list[str]]], contrasts: Sequence[str], split: str = "validation",
            resamples: int = 10_000, final: bool = False, first_reference: bool = False) -> dict:
    """Each system's scores over its runs and the contrasts' bootstrap intervals (module docstring)."""
    if len({name for name, _, _ in systems}) != len(systems):
        raise ValueError("system names must differ")
    dataset = E.load_eval_set(split, final)
    if first_reference:
        dataset = E.first_reference(dataset)
    out: dict = {"split": split, "first_reference": first_reference, "systems": {}, "contrasts": {}}
    per: dict[str, dict[str, dict[str, float]]] = {"exact": {}, "f1": {}, "exact_plus_f1": {}}
    for name, rule, runs in systems:
        ems, f1s, rows, reports = [], [], [], []
        for run in runs:
            preds = E.load_predictions(RESULTS / run / f"predictions_{split}_{rule}.json")
            em, f1 = E.per_question(dataset, preds)
            row: dict = {"run": run}
            if split in E.ADVERSARIAL:                       # spec §6 reports these beside the worst case
                m = E.metrics(split, dataset, preds)
                row["all_questions"] = {"exact": m["exact"], "f1": m["f1"]}
                row["original"] = {"exact": m["original"]["exact"], "f1": m["original"]["f1"]}
                em, f1 = E.adversarial_groups(dataset, em, f1)
            s = E.summarize(em, f1)
            ems.append(em)
            f1s.append(f1)
            rows.append({**row, "exact": s["exact"], "f1": s["f1"]})
            path = RESULTS / run / f"report_{split}.json"
            if path.exists():
                reports.append(json.loads(path.read_text(encoding="utf-8")))
        sys_out = {"rule": rule, "runs": rows,
                   "exact": _stats([r["exact"] for r in rows]), "f1": _stats([r["f1"] for r in rows]),
                   "exact_plus_f1": _stats([r["exact"] + r["f1"] for r in rows])}
        for key in ("all_questions", "original"):
            if key in rows[0]:
                sys_out[key] = {m: _stats([r[key][m] for r in rows]) for m in ("exact", "f1")}
        if reports and len(reports) == len(runs):
            sys_out["report"] = {key: _stats([r[key] for r in reports]) for key in ("temperature", "nll", "ece_map")}
            sys_out["report"]["topk_recall"] = {k: _stats([r["topk_recall"][k] for r in reports])
                                                for k in reports[0]["topk_recall"]}
        out["systems"][name] = sys_out
        em_mean, f1_mean = E.mean_over_seeds(ems), E.mean_over_seeds(f1s)
        per["exact"][name], per["f1"][name] = em_mean, f1_mean
        per["exact_plus_f1"][name] = {k: em_mean[k] + f1_mean[k] for k in em_mean}
    for expr in contrasts:
        signs = parse_contrast(expr, list(out["systems"]))
        out["contrasts"][expr] = {metric: E.paired_bootstrap(*signed_sums(scores, signs), resamples)
                                  for metric, scores in per.items()}
    return out


def markdown(result: dict) -> str:
    """The systems and contrasts as two Markdown tables, and on Adversarial SQuAD a third with the
    averages over all questions and over the originals."""
    def ms(s: dict) -> str:
        return f"{s['mean']:.2f} ± {s['std']:.2f}"

    lines = ["| System | Rule | Runs | EM | F1 | EM + F1 |", "|---|---|---|---|---|---|"]
    for name, s in result["systems"].items():
        lines.append(f"| {name} | {s['rule'].upper()} | {len(s['runs'])} | {ms(s['exact'])} | {ms(s['f1'])} "
                     f"| {ms(s['exact_plus_f1'])} |")
    lines += ["", "| Contrast | EM | F1 | EM + F1 |", "|---|---|---|---|"]
    for expr, c in result["contrasts"].items():
        cells = [f"{c[m]['diff']:+.2f} [{c[m]['ci95'][0]:+.2f}, {c[m]['ci95'][1]:+.2f}]"
                 for m in ("exact", "f1", "exact_plus_f1")]
        lines.append(f"| {expr} | " + " | ".join(cells) + " |")
    if any("all_questions" in s for s in result["systems"].values()):
        lines += ["", "| System | EM, all questions | F1, all questions | EM, originals | F1, originals |",
                  "|---|---|---|---|---|"]
        for name, s in result["systems"].items():
            a, o = s["all_questions"], s["original"]
            lines.append(f"| {name} | {ms(a['exact'])} | {ms(a['f1'])} | {ms(o['exact'])} | {ms(o['f1'])} |")
    return "\n".join(lines)


def article_splits(titles: Sequence[str], splits: int) -> list[tuple[set[str], set[str]]]:
    """(fit titles, test titles): `splits` random halves of the articles, each used both ways round."""
    titles = sorted(titles)
    out = []
    for seed in range(splits):
        half = set(np.random.default_rng(seed).permutation(titles)[: len(titles) // 2].tolist())
        rest = set(titles) - half
        out += [(half, rest), (rest, half)]
    return out


def heldout(examples: Sequence, entries: Sequence[dict], scores: Sequence[np.ndarray], rcfg: Mapping,
            splits: int = 5) -> dict:
    """Per rule, the gain in points of the combination over the reader on the same lists, on articles
    left out of fitting λ and the temperature, which are fitted as train_verifier.evaluate fits them.
    Answers are scored with the official scorer against each question's references."""
    from squadqa import train_verifier as TV
    from squadqa import verifier as V

    sc = E.scorer()
    questions = []
    for ex, e in zip(examples, entries):
        q = ex.question
        golds = [text for text, _ in q.answers]
        texts = [answer_text(q.context, ex.context.words, a, b) for a, b in e["spans"]]
        questions.append({
            "flags": content_flags(q.context, ex.context.words),
            "starts": np.array([a for a, _ in e["spans"]]), "ends": np.array([b for _, b in e["spans"]]),
            "at": {tuple(span): k for k, span in enumerate(e["spans"])},
            "exact": [max(sc.compute_exact(g, t) for g in golds) for t in texts],
            "f1": [max(sc.compute_f1(g, t) for g in golds) for t in texts]})

    def answer(k: int, logp: np.ndarray, rule: str) -> tuple[float, float]:
        q = questions[k]
        span = decode(Candidates(q["starts"], q["ends"], logp), q["flags"], rule, rcfg["mbr_utility"], rcfg["mbr_em_weight"])
        c = q["at"][span]
        return q["exact"][c], q["f1"][c]

    reader = {rule: [answer(k, np.array(e["logp"], dtype=np.float64), rule) for k, e in enumerate(entries)]
              for rule in rcfg["decode_rules"]}
    titles = [ex.question.title for ex in examples]
    gains: dict[str, dict[str, list[float]]] = {rule: {"exact": [], "f1": []} for rule in rcfg["decode_rules"]}
    fits = []
    for fit_titles, test_titles in article_splits(set(titles), splits):
        items = []
        for k, title in enumerate(titles):
            em = np.array([0 if x is None else x for x in entries[k]["em"]])
            if title in fit_titles and (em == 1).any():
                items.append((np.array(entries[k]["logp"], dtype=np.float64), scores[k], np.flatnonzero(em == 1)))
        lam = V.fit_lambda(items)[0]
        temperature, _ = TV.fit_temperature_or_edge([(V.combine(r, s, lam), g) for r, s, g in items])
        fits.append({"lambda": float(lam), "temperature": float(temperature)})
        test = [k for k, title in enumerate(titles) if title in test_titles]
        for rule in rcfg["decode_rules"]:
            combined = [answer(k, log_softmax(V.combine(entries[k]["logp"], scores[k], lam), temperature), rule) for k in test]
            base = [reader[rule][k] for k in test]
            for m, metric in enumerate(("exact", "f1")):
                gains[rule][metric].append(100.0 * float(np.mean([c[m] - b[m] for c, b in zip(combined, base)])))
    out: dict = {"splits": splits, "halves": 2 * splits, "fits": fits, "rules": {}}
    for rule, by_metric in gains.items():
        out["rules"][rule] = {metric: {"gains": g, "mean": float(np.mean(g)), "positive": int(sum(x > 0 for x in g))}
                              for metric, g in by_metric.items()}
    return out


def heldout_run(run_id: str, lists: str | None = None, splits: int = 5) -> dict:
    """heldout for a finished verifier run; writes results/<run_id>/heldout_validation.json."""
    from squadqa import train_verifier as TV

    rcfg, _, examples, entries, scores = TV.load_finished(run_id, lists)
    result = heldout(examples, entries, scores, rcfg, splits)
    write_json(RESULTS / run_id / "heldout_validation.json", result)
    return result


def heldout_markdown(results: Mapping[str, dict]) -> str:
    """One row per run and rule."""
    lines = ["| Run | Rule | EM gain | Halves positive | F1 gain | Halves positive |", "|---|---|---|---|---|---|"]
    for run, r in results.items():
        for rule, m in r["rules"].items():
            lines.append(f"| {run} | {rule.upper()} | {m['exact']['mean']:+.2f} | {m['exact']['positive']}/{r['halves']} "
                         f"| {m['f1']['mean']:+.2f} | {m['f1']['positive']}/{r['halves']} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Reports over finished runs.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="calibration and top-k recall per run")
    run.add_argument("runs", nargs="+", metavar="RUN_ID")
    run.add_argument("--split", default="validation")
    cmp_ = sub.add_parser("compare", help="systems over seeds, and bootstrap intervals for contrasts")
    cmp_.add_argument("systems", nargs="+", metavar="NAME=RULE:RUN,RUN")
    cmp_.add_argument("--contrast", action="append", default=[], help="a signed sum of systems, e.g. B-A")
    cmp_.add_argument("--resamples", type=int, default=10_000)
    cmp_.add_argument("-o", "--out", help="also write the result as JSON here")
    cmp_.add_argument("--split", default="validation", choices=E.EVAL_SETS)
    cmp_.add_argument("--final", action="store_true", help="allow dev and the robustness sets (stage 5 only)")
    cmp_.add_argument("--first-reference", action="store_true", help="score against each question's first answer alone")
    held = sub.add_parser("heldout", help="a verifier's gain on articles left out of fitting λ and T")
    held.add_argument("runs", nargs="+", metavar="RUN_ID")
    held.add_argument("--lists", help="folder of the candidate lists (default: data/verifier/<reader>)")
    held.add_argument("--splits", type=int, default=5, help="random halves, each used both ways round")
    args = parser.parse_args(argv)
    if args.command == "run":
        for run_id in args.runs:
            rep = run_report(run_id, args.split)
            write_json(RESULTS / run_id / f"report_{args.split}.json", rep)
            recall = ", ".join(f"@{k} {v:.3f}" for k, v in rep["topk_recall"].items())
            print(f"{run_id}: T {rep['temperature']:.3f}, NLL {rep['nll']:.3f}, ECE {rep['ece_map']:.4f}, "
                  f"recall {recall}")
    elif args.command == "heldout":
        print(heldout_markdown({run_id: heldout_run(run_id, args.lists, args.splits) for run_id in args.runs}))
    else:
        result = compare([parse_system(s) for s in args.systems], args.contrast, args.split, args.resamples, args.final,
                         args.first_reference)
        if args.out:
            write_json(Path(args.out), result)
        print(markdown(result))


if __name__ == "__main__":
    main()
