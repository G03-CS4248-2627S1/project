"""Turn fetched MAP loss search summaries into a comparison table.

python -m squadqa.loss_search_report results/<e-search>/summary.json \
    results/<c-search>/summary.json -o reports/map-loss-search.md
"""

import argparse
import json
from pathlib import Path


def render(summaries: list[dict]) -> str:
    if not summaries:
        raise ValueError("at least one summary is required")
    if len({s["commit"] for s in summaries}) != 1:
        raise ValueError("search phases must use the same code commit")
    e = next((s for s in summaries if s["phase"] == "e"), None)
    baseline = next((t for t in e["trials"] if t["parameters"] == {"beta": 0.0}), None) if e else None
    reference = baseline["scores"]["mbr_em"] + baseline["scores"]["mbr_f1"] if baseline else None
    lines = ["# MAP loss search on training-derived validation", "",
             f"Code commit: `{summaries[0]['commit']}`. Objective: official validation MAP EM + F1. "
             "Scores use the validation split, not the held-out test set.", "",
             "| Phase | Trial | Choice method | α | τ | β | MAP EM | MAP F1 | MAP sum | MBR sum | MAP − B | Run ID |",
             "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for summary in summaries:
        for trial in summary["trials"]:
            p, s = trial["parameters"], trial["scores"]
            mbr_sum = s["mbr_em"] + s["mbr_f1"]
            delta = f"{trial['objective'] - reference:+.3f}" if reference is not None else "—"
            method = "Initial anchor" if trial["selection"] == "anchor" else "GP expected improvement"
            lines.append(f"| {summary['phase'].upper()} | {trial['trial']} | {method} | "
                         f"{p.get('alpha', '—')} | {p.get('tau', '—')} | {p.get('beta', '—')} | "
                         f"{s['map_em']:.3f} | {s['map_f1']:.3f} | {trial['objective']:.3f} | "
                         f"{mbr_sum:.3f} | {delta} | `{trial['run_id']}` |")
    lines += ["", "B is MBR from the matched β=0 one-hot run in phase E. "
              "MAP − B compares each MAP score to that run's MBR score. "
              "The first three trials in each phase cover fixed anchors; later trials maximize "
              "Gaussian-process expected improvement from the completed trials. "
              "Phase E searches β ∈ [0, 3]. Phase C searches α ∈ [0.1, 0.9] and "
              "log τ ∈ [log 0.05, log 1], with φ=0. Both use seed 0 and one epoch. "
              "A winner needs confirmation on seeds 1–3 before a performance claim.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", nargs="+", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in args.summary]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(summaries), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
