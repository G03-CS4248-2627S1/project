"""The tables of reports/final_scoring_2026-11-06.md, as Markdown, from reports/final_scoring/*.json (and
before_decode_fix/ for the before and after). Nothing in squadqa/ uses this script.

    python sandbox/bruhmiuz/final_tables.py
"""
import json
from pathlib import Path

NEW = Path("reports/final_scoring")
OLD = NEW / "before_decode_fix"
SETS = [("validation", "Validation"), ("dev", "Dev"), ("dev-first-reference", "Dev, first reference"), ("addsent", "AddSent"),
        ("addonesent", "AddOneSent"), ("squadshifts-new_wiki", "Shifts: new_wiki"), ("squadshifts-nyt", "Shifts: nyt"),
        ("squadshifts-reddit", "Shifts: reddit"), ("squadshifts-amazon_reviews", "Shifts: amazon_reviews")]


def load(folder, name):
    return json.loads((folder / f"{name}.json").read_text(encoding="utf-8"))


def cell(s):
    return f"{s['exact']['mean']:.2f} / {s['f1']['mean']:.2f}"


def ci(c):
    return f"{c['diff']:+.2f} [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}]"


print("### Table 1\n")
cols = [("R0", "Row 0, MAP"), ("R1", "Row 1, MAP"), ("R3", "Row 3, MBR"), ("R4map", "Row 4, MAP"), ("R4", "Row 4, MBR"), ("R5map", "Row 5, MAP"), ("R5", "Row 5, MBR")]
print("| Set | " + " | ".join(c for _, c in cols) + " |")
print("|---|" + "---|" * len(cols))
for name, label in SETS:
    d = load(NEW, name)["systems"]
    print(f"| {label} | " + " | ".join(cell(d[k]) for k, _ in cols) + " |")

print("\n### Spread\n")
print("| Set | " + " | ".join(c for _, c in cols) + " |")
print("|---|" + "---|" * len(cols))
for name, label in SETS:
    d = load(NEW, name)["systems"]
    print(f"| {label} | " + " | ".join(f"{d[k]['exact']['std']:.2f} / {d[k]['f1']['std']:.2f}" for k, _ in cols) + " |")

print("\n### Contrasts\n")
for expr, title in (("R1-R0", "Row 1 - row 0"), ("R3-R1", "Row 3 - row 1"), ("R4-R3s0", "Row 4 - row 3, reader seed 0"), ("R5-R3", "Row 5 - row 3")):
    print(f"**{title}**\n\n| Set | EM | F1 | EM + F1 |\n|---|---|---|---|")
    for name, label in SETS:
        c = load(NEW, name)["contrasts"][expr]
        print(f"| {label} | {ci(c['exact'])} | {ci(c['f1'])} | {ci(c['exact_plus_f1'])} |")
    print()

print("### Before and after\n")
print("| Set | Row 4 MBR before | after | Row 4 MAP before | after | Row 4 - row 3, EM before | after | Row 4 - row 3, F1 before | after |")
print("|---|---|---|---|---|---|---|---|---|")
for name, label in SETS:
    if name == "validation":
        continue
    o, n = load(OLD, name), load(NEW, name)
    a, b = o["contrasts"]["R4-R3s0"], n["contrasts"]["R4-R3s0"]
    print(f"| {label} | {cell(o['systems']['R4'])} | {cell(n['systems']['R4'])} | {cell(o['systems']['R4map'])} | {cell(n['systems']['R4map'])} | "
          f"{ci(a['exact'])} | {ci(b['exact'])} | {ci(a['f1'])} | {ci(b['f1'])} |")

print("\n### Gain sources\n")
g = json.loads((NEW / "gain_sources.json").read_text(encoding="utf-8"))
print("| Set | Reader on lists - predict's, EM | Row 4 MBR - reader on lists, EM | F1 | Edge-only changes, % of questions | Their EM | Their F1 | Other changes, % of questions | Their EM | Their F1 |")
print("|---|---|---|---|---|---|---|---|---|---|")
for name, label in SETS:
    if name == "dev-first-reference":
        continue
    m = g[name]["mean"]
    print(f"| {label} | {m['lists_minus_predict_em']:+.2f} | {m['gain_em']:+.2f} | {m['gain_f1']:+.2f} | {m['edge_share']:.2f} | {m['edge_gain_em']:+.2f} | {m['edge_gain_f1']:+.2f} | "
          f"{m['other_share']:.2f} | {m['other_gain_em']:+.2f} | {m['other_gain_f1']:+.2f} |")
