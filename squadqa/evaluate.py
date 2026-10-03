"""Scoring with the official scorer (spec §1 rule 5, §6; workflow.md §6.1).

Per-question EM and F1 always come from get_raw_scores in qa/evaluate-v2.0.py, which is
loaded with importlib and never modified, and summaries from its make_eval_dict. This module
adds only what the scorer lacks: checks on the predictions, Adversarial SQuAD's worst-case
grouping, and paired bootstrap intervals.

Evaluation sets: the validation split during development. Dev, Adversarial SQuAD (addsent,
addonesent) and the four SQuAD-Shifts sets are scored once, in the final scoring, so loading
one of them needs final=True (--final on the command line).

    python -m squadqa.evaluate validation results/<run_id>/predictions.json
    python -m squadqa.evaluate validation a.json --compare b.json     # paired bootstrap of b - a
    python -m squadqa.evaluate addsent predictions.json --final -o metrics_addsent.json

Run from the repo root, like every entry point.
"""

import argparse
import importlib.util
import json
from collections.abc import Mapping
from functools import cache
from pathlib import Path

import numpy as np

from squadqa.config import write_json
from squadqa.splits import load_val_titles

SCORER = Path("qa/evaluate-v2.0.py")
TRAIN = Path("qa/train-v1.1.json")
EXTERNAL = Path("data/external")
FINAL_SETS = {
    "dev": Path("qa/dev-v1.1.json"),
    "addsent": EXTERNAL / "adversarial_squad/addsent.json",
    "addonesent": EXTERNAL / "adversarial_squad/addonesent.json",
    "squadshifts-new_wiki": EXTERNAL / "squadshifts/new_wiki.json",
    "squadshifts-nyt": EXTERNAL / "squadshifts/nyt.json",
    "squadshifts-reddit": EXTERNAL / "squadshifts/reddit.json",
    "squadshifts-amazon_reviews": EXTERNAL / "squadshifts/amazon_reviews.json",
}
ADVERSARIAL = {"addsent", "addonesent"}
EVAL_SETS = ["validation", *FINAL_SETS]


@cache
def scorer():
    """qa/evaluate-v2.0.py as a module."""
    spec = importlib.util.spec_from_file_location("squad_official_eval", SCORER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_data(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["data"]


def load_eval_set(name: str, final: bool = False) -> list[dict]:
    """The articles of an evaluation set. Anything but validation needs final=True."""
    if name == "validation":
        titles = load_val_titles()
        data = [a for a in _load_data(TRAIN) if a["title"] in titles]
        if {a["title"] for a in data} != titles:
            raise RuntimeError(f"some validation titles are not in {TRAIN}")
        return data
    if name not in FINAL_SETS:
        raise KeyError(f"unknown evaluation set {name!r}; choose from {EVAL_SETS}")
    if not final:
        raise RuntimeError(f"{name} is scored only in the final scoring (workflow.md §6.1); "
                           "pass final=True (--final) there")
    path = FINAL_SETS[name]
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run python -m squadqa.download_external")
    return _load_data(path)


def question_ids(dataset: list[dict]) -> list[str]:
    """Question ids in file order."""
    return [qa["id"] for article in dataset for p in article["paragraphs"] for qa in p["qas"]]


def load_predictions(path: str | Path) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        preds = json.load(f)
    if not isinstance(preds, dict):
        raise TypeError(f"{path} must hold one JSON object {{question_id: answer_text}}")
    return preds


def check_predictions(dataset: list[dict], preds: Mapping[str, str]) -> None:
    """Raise unless preds holds a non-empty answer string for exactly the dataset's questions.

    The scorer skips a missing id with a print and scores an empty answer as "no answer"
    (spec rule 5), and it ignores extra ids, which usually means the wrong dataset.
    """
    ids = question_ids(dataset)
    if len(ids) != len(set(ids)):
        raise ValueError("the dataset repeats question ids")
    missing = [i for i in ids if i not in preds]
    extra = sorted(set(preds) - set(ids))
    empty = [i for i in ids if i in preds and not (isinstance(preds[i], str) and preds[i].strip())]
    problems = [f"{len(found)} {what}, e.g. {found[:3]}"
                for what, found in [("missing", missing), ("not in the dataset", extra),
                                    ("empty or not a string", empty)] if found]
    if problems:
        raise ValueError("bad predictions: " + "; ".join(problems))


def per_question(dataset: list[dict], preds: Mapping[str, str]) -> tuple[dict[str, float], dict[str, float]]:
    """(EM, F1) per question id, each 0-1, from the official scorer (max over gold answers)."""
    check_predictions(dataset, preds)
    exact, f1 = scorer().get_raw_scores(dataset, preds)
    return {k: float(v) for k, v in exact.items()}, {k: float(v) for k, v in f1.items()}


def summarize(exact: Mapping[str, float], f1: Mapping[str, float]) -> dict:
    """{exact, f1, total}: the scorer's own averages, in points (0-100)."""
    return dict(scorer().make_eval_dict(exact, f1))


def adversarial_groups(dataset: list[dict], exact: Mapping[str, float],
                       f1: Mapping[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    """Jia & Liang's worst-case (EM, F1) per original question id.

    Same grouping as the authors' eval_squad.py (CodaLab bundle 0xc2824269ba0a4179ac63c08fd3997d1c,
    sha256 1d756bb6...; checked 2026-10-04). An id's group is its prefix before the first "-".
    When the group has adversarial versions, the original is not used: the version with the lowest
    F1 is chosen, the first in file order on ties, and its EM comes with it (EM is not minimized
    separately). A group with no versions keeps its original question.
    """
    chosen: dict[str, str] = {}
    for qid in question_ids(dataset):
        orig = qid.split("-")[0]
        if qid == orig:
            chosen.setdefault(orig, qid)
        elif orig not in chosen or chosen[orig] == orig or f1[qid] < f1[chosen[orig]]:
            chosen[orig] = qid
    return {o: exact[q] for o, q in chosen.items()}, {o: f1[q] for o, q in chosen.items()}


def metrics(name: str, dataset: list[dict], preds: Mapping[str, str]) -> dict:
    """Summary for one evaluation set. Adversarial sets add the worst-case score over the
    original ids ("adversarial", the main number) and the originals alone ("original")."""
    exact, f1 = per_question(dataset, preds)
    out = summarize(exact, f1)
    if name in ADVERSARIAL:
        adv_exact, adv_f1 = adversarial_groups(dataset, exact, f1)
        missing = [o for o in adv_exact if o not in exact]
        if missing:
            raise ValueError(f"{len(missing)} adversarial groups lack their original question, e.g. {missing[:3]}")
        out["adversarial"] = summarize(adv_exact, adv_f1)
        out["original"] = summarize({o: exact[o] for o in adv_exact}, {o: f1[o] for o in adv_exact})
    return out


def mean_over_seeds(runs: list[Mapping[str, float]]) -> dict[str, float]:
    """Per-question scores averaged over runs (seeds) of one system."""
    if not runs or any(r.keys() != runs[0].keys() for r in runs):
        raise ValueError("every run must score the same questions")
    return {k: sum(r[k] for r in runs) / len(runs) for k in runs[0]}


def paired_bootstrap(a: Mapping[str, float], b: Mapping[str, float],
                     resamples: int = 10_000, seed: int = 0) -> dict:
    """Mean of b - a in points, with a 95% percentile interval.

    Resamples the units (questions, or adversarial groups) with replacement; both systems
    share every resample. a and b are per-unit scores in 0-1 over the same units.
    """
    if a.keys() != b.keys():
        raise ValueError("a and b must score the same questions")
    keys = sorted(a)
    diff = 100.0 * (np.array([b[k] for k in keys]) - np.array([a[k] for k in keys]))
    n = len(diff)
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    chunk = max(1, 5_000_000 // n)                      # about 40 MB of indices at a time
    for start in range(0, resamples, chunk):
        m = min(chunk, resamples - start)
        means[start:start + m] = diff[rng.integers(0, n, size=(m, n))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"diff": float(diff.mean()), "ci95": [float(lo), float(hi)],
            "n": n, "resamples": resamples, "seed": seed}


def compare(name: str, dataset: list[dict], preds_a: Mapping[str, str], preds_b: Mapping[str, str],
            resamples: int = 10_000, seed: int = 0) -> dict:
    """Paired bootstrap of b - a for EM and F1 (over adversarial groups on Adversarial SQuAD)."""
    (ea, fa), (eb, fb) = per_question(dataset, preds_a), per_question(dataset, preds_b)
    if name in ADVERSARIAL:
        (ea, fa), (eb, fb) = adversarial_groups(dataset, ea, fa), adversarial_groups(dataset, eb, fb)
    return {"exact": paired_bootstrap(ea, eb, resamples, seed),
            "f1": paired_bootstrap(fa, fb, resamples, seed)}


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Score predictions with the official SQuAD scorer.")
    parser.add_argument("eval_set", choices=EVAL_SETS)
    parser.add_argument("predictions", help="JSON object {question_id: answer_text}")
    parser.add_argument("--compare", metavar="PREDICTIONS_B",
                        help="paired bootstrap of PREDICTIONS_B minus predictions")
    parser.add_argument("--final", action="store_true", help="allow dev and the robustness sets (stage 5 only)")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("-o", "--out", help="write the result as JSON here (default: print it)")
    args = parser.parse_args(argv)

    dataset = load_eval_set(args.eval_set, args.final)
    preds = load_predictions(args.predictions)
    if args.compare:
        result = compare(args.eval_set, dataset, preds, load_predictions(args.compare), args.resamples)
    else:
        result = metrics(args.eval_set, dataset, preds)
    if args.out:
        write_json(Path(args.out), result)
    else:
        print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
