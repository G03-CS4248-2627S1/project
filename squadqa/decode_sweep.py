"""The decoding sweep (#25, plan §9.2): γ, calibration and grouping, decode-only on the CPU.

Every setting re-decodes a finished reader's saved top-k candidates on validation and scores the
answers with the official scorer. Candidates are not committed (workflow.md §2.2), so a run is swept
where its candidates are: the machine that fetched it, or after python -m squadqa.predict --run-id.

    python -m squadqa.decode_sweep row1=<run>,<run>,<run> row0=<run>,<run>,<run> [--resamples N] [-o out.json]

A system is one reader's seeds, written name=run,run,... For each it reports:
- Each setting's EM, F1 and EM + F1, as mean ± sample std over the runs, and a paired bootstrap of
  EM + F1 against the reference (MBR at γ = 1: row 3), over questions.
- Plan §15.3's selection rule: a setting replaces the reference only if it wins by more than twice the
  seed std of the reference's EM + F1.
- The normalization gap: where each decoder's answer sits against the gold span, next to the official EM.

A run's saved logp are at its fitted temperature T. Multiplying them by T gives the probabilities at
T = 1 up to a constant, and decoding only compares them, so the "-t1" settings need nothing else. A
temperature keeps the order of the spans, so the top k are the same; only their masses move.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from squadqa import evaluate as E
from squadqa.config import RESULTS, write_json
from squadqa.data import answer_word_span, word_spans
from squadqa.decode import Candidates, answer_text, content_flags, decode, span_f1_em
from squadqa.report import _stats

# name: rule, γ, mbr_utility, at the fitted temperature. MAP is γ = ∞ (spec §4.4).
SETTINGS = {
    "map": ("map", 1.0, "normalized", True),
    "mbr-g0": ("mbr", 0.0, "normalized", True),
    "mbr-g0.5": ("mbr", 0.5, "normalized", True),
    "mbr-g1": ("mbr", 1.0, "normalized", True),
    "mbr-g2": ("mbr", 2.0, "normalized", True),
    "map-t1": ("map", 1.0, "normalized", False),
    "mbr-g1-t1": ("mbr", 1.0, "normalized", False),
    "map-positional": ("map", 1.0, "positional", True),
    "mbr-g1-positional": ("mbr", 1.0, "positional", True),
    "map-string": ("map", 1.0, "string", True),
    "mbr-g1-string": ("mbr", 1.0, "string", True),
}
REFERENCE = "mbr-g1"
SAVED = {"map": "map", "mbr-g1": "mbr"}         # settings that must rebuild a run's committed predictions
GAP_SETTINGS = ("map", "mbr-g1")
POSITIONS = ("identical", "boundary_variant", "elsewhere", "no_gold_span")


def decode_question(cands: Candidates, flags: np.ndarray, texts: list[str], temperature: float,
                    setting: str) -> tuple[int, int]:
    """One setting's answer among one question's candidates."""
    rule, gamma, utility, calibrated = SETTINGS[setting]
    logp = cands.logp if calibrated else cands.logp * temperature
    return decode(Candidates(cands.starts, cands.ends, logp), flags, rule, utility, gamma, texts)


def position(flags: np.ndarray, gold: tuple[int, int] | None, span: tuple[int, int]) -> str:
    """Where an answer sits against the gold span: the same span, a boundary variant of it (the same
    content words at the same positions, spec §4.3), or anywhere else."""
    if gold is None:
        return "no_gold_span"
    if span == gold:
        return "identical"
    em = span_f1_em(flags, span[0], span[1], gold[0], gold[1])[1]
    return "boundary_variant" if float(em) == 1.0 else "elsewhere"


def decode_run(run_id: str, dataset: list[dict]) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    """A run's answers under every setting, and for GAP_SETTINGS each answer's position against the
    gold. Raises if the default settings don't rebuild the run's committed predictions."""
    run_dir = RESULTS / run_id
    path = run_dir / "candidates_validation.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; candidates stay on the machine that fetched the run")
    saved = json.loads(path.read_text(encoding="utf-8"))
    temperature = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))["temperature"]
    answers = {s: {} for s in SETTINGS}
    places = {s: {} for s in GAP_SETTINGS}
    for article in dataset:
        for p in article["paragraphs"]:
            words = word_spans(p["context"])
            flags = content_flags(p["context"], words)
            for qa in p["qas"]:
                arr = np.asarray(saved[qa["id"]], dtype=np.float64)
                c = Candidates(arr[:, 0].astype(np.int64), arr[:, 1].astype(np.int64), arr[:, 2])
                texts = [answer_text(p["context"], words, int(i), int(j)) for i, j in zip(c.starts, c.ends)]
                gold = answer_word_span(words, qa["answers"][0]["answer_start"], qa["answers"][0]["text"])
                for s in SETTINGS:
                    i, j = decode_question(c, flags, texts, temperature, s)
                    answers[s][qa["id"]] = answer_text(p["context"], words, i, j)
                    if s in places:
                        places[s][qa["id"]] = position(flags, gold, (i, j))
    for s, rule in SAVED.items():
        file = run_dir / f"predictions_validation_{rule}.json"
        preds = E.load_predictions(file)
        differ = sum(answers[s][q] != preds[q] for q in preds)
        if differ > 0.001 * len(preds):         # near-ties: the saved logp are rounded
            raise RuntimeError(f"{run_id}: {s} differs from {file.name} on {differ} of {len(preds)} questions")
    return answers, places


def gap_counts(exact: Mapping[str, float], places: Mapping[str, str]) -> dict[str, int]:
    """Questions by the answer's position and its official EM."""
    c = Counter((places[q], int(exact[q])) for q in exact)
    return {f"{where}_em{em}": c[(where, em)] for where in POSITIONS for em in (1, 0)}


def sweep(systems: Sequence[tuple[str, list[str]]], resamples: int = 10_000, seed: int = 0) -> dict:
    """Every setting over each system's runs (module docstring)."""
    dataset = E.load_eval_set("validation")
    out: dict = {"split": "validation", "reference": REFERENCE,
                 "settings": {k: dict(zip(("rule", "gamma", "utility", "calibrated"), v)) for k, v in SETTINGS.items()},
                 "systems": {}}
    for name, runs in systems:
        ems: dict[str, list] = {s: [] for s in SETTINGS}
        f1s: dict[str, list] = {s: [] for s in SETTINGS}
        gaps: dict[str, list] = {s: [] for s in GAP_SETTINGS}
        for run in runs:
            answers, places = decode_run(run, dataset)
            for s in SETTINGS:
                em, f1 = E.per_question(dataset, answers[s])
                ems[s].append(em)
                f1s[s].append(f1)
                if s in places:
                    gaps[s].append(gap_counts(em, places[s]))
        rows = {}
        sums = {}
        for s in SETTINGS:
            em_mean, f1_mean = E.mean_over_seeds(ems[s]), E.mean_over_seeds(f1s[s])
            sums[s] = {q: em_mean[q] + f1_mean[q] for q in em_mean}
        ref_sum = sums[REFERENCE]
        per_run = {s: [E.summarize(a, b) for a, b in zip(ems[s], f1s[s])] for s in SETTINGS}
        ref_std = _stats([r["exact"] + r["f1"] for r in per_run[REFERENCE]])["std"]
        for s in SETTINGS:
            total = _stats([r["exact"] + r["f1"] for r in per_run[s]])
            boot = E.paired_bootstrap(ref_sum, sums[s], resamples, seed)
            rows[s] = {"exact": _stats([r["exact"] for r in per_run[s]]), "f1": _stats([r["f1"] for r in per_run[s]]),
                       "exact_plus_f1": total, "vs_reference": boot,
                       "changed_from_reference": _mean_changed(ems, f1s, s),
                       "wins_by_selection_rule": bool(boot["diff"] > 2 * ref_std)}
        out["systems"][name] = {"runs": list(runs), "reference_std": ref_std, "settings": rows,
                                "gap": {s: {k: float(np.mean([g[k] for g in gaps[s]])) for k in gaps[s][0]}
                                        for s in GAP_SETTINGS}}
    return out


def _mean_changed(ems: Mapping[str, list], f1s: Mapping[str, list], setting: str) -> float:
    """Mean number of questions per run whose score differs from the reference's."""
    counts = [sum(e[q] != re[q] or f[q] != rf[q] for q in e)
              for e, f, re, rf in zip(ems[setting], f1s[setting], ems[REFERENCE], f1s[REFERENCE])]
    return float(np.mean(counts))


def markdown(result: dict) -> str:
    """One table of settings and one of the normalization gap per system."""
    def ms(s: dict) -> str:
        return f"{s['mean']:.2f} ± {s['std']:.2f}"

    lines = []
    for name, system in result["systems"].items():
        lines += [f"### {name} ({len(system['runs'])} seeds)", "",
                  "| Setting | EM | F1 | EM + F1 | Against γ = 1 | Questions scored differently | Wins by the rule |",
                  "|---|---|---|---|---|---|---|"]
        for s, r in system["settings"].items():
            b = r["vs_reference"]
            lines.append(f"| {s} | {ms(r['exact'])} | {ms(r['f1'])} | {ms(r['exact_plus_f1'])} "
                         f"| {b['diff']:+.2f} [{b['ci95'][0]:+.2f}, {b['ci95'][1]:+.2f}] "
                         f"| {r['changed_from_reference']:.1f} | {'yes' if r['wins_by_selection_rule'] else 'no'} |")
        n_rule = 2 * system["reference_std"]
        lines += ["", f"The selection rule needs a gain above {n_rule:.2f} (twice the seed std of the reference's EM + F1).", "",
                  "| Answer against the gold span | MAP, official EM 1 | MAP, official EM 0 | MBR, official EM 1 | MBR, official EM 0 |",
                  "|---|---|---|---|---|"]
        labels = {"identical": "Identical span", "boundary_variant": "Boundary variant",
                  "elsewhere": "Elsewhere", "no_gold_span": "Gold is not a word span"}
        for where in POSITIONS:
            g, h = system["gap"]["map"], system["gap"]["mbr-g1"]
            lines.append(f"| {labels[where]} | {g[where + '_em1']:.1f} | {g[where + '_em0']:.1f} "
                         f"| {h[where + '_em1']:.1f} | {h[where + '_em0']:.1f} |")
        lines.append("")
    return "\n".join(lines)


def parse_system(spec: str) -> tuple[str, list[str]]:
    """'row1=r1,r2' -> ('row1', ['r1', 'r2'])."""
    name, _, runs = spec.partition("=")
    if not name or not runs or not all(runs.split(",")):
        raise ValueError(f"a system is name=run,run,..., got {spec!r}")
    return name, runs.split(",")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Re-decode finished readers under γ, calibration and grouping settings.")
    parser.add_argument("systems", nargs="+", metavar="NAME=RUN,RUN")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("-o", "--out", help="also write the result as JSON here")
    args = parser.parse_args(argv)
    result = sweep([parse_system(s) for s in args.systems], args.resamples)
    if args.out:
        write_json(Path(args.out), result)
    print(markdown(result))


if __name__ == "__main__":
    main()
