"""Build the validation split (spec §3.1).

Run from the repo root:  python -m squadqa.splits

Holds out 10% of train-v1.1 articles (44 of 442), chosen by title with a fixed
seed, and writes their titles to data/splits/val_articles.txt, one per line,
sorted. The split is by article, never by question, because questions on the
same article share paragraphs. The file is committed; re-running reproduces it
exactly, so the script refuses to overwrite a different existing file.
"""

import json
import random
from pathlib import Path

TRAIN = Path("qa/train-v1.1.json")
VAL_ARTICLES = Path("data/splits/val_articles.txt")
VAL_FRACTION = 0.10
SEED = 0


def load_val_titles() -> set[str]:
    """The validation article titles. Raises if the split has not been built."""
    if not VAL_ARTICLES.exists():
        raise FileNotFoundError(f"{VAL_ARTICLES} is missing; run python -m squadqa.splits")
    return set(VAL_ARTICLES.read_text(encoding="utf-8").split("\n")) - {""}


def main() -> None:
    with open(TRAIN, encoding="utf-8") as f:
        data = json.load(f)["data"]
    titles = sorted(a["title"] for a in data)
    if len(titles) != len(set(titles)):
        raise RuntimeError("duplicate article titles in train-v1.1.json")
    val = sorted(random.Random(SEED).sample(titles, round(VAL_FRACTION * len(titles))))
    text = "\n".join(val) + "\n"
    if VAL_ARTICLES.exists() and VAL_ARTICLES.read_text(encoding="utf-8") != text:
        raise RuntimeError(f"{VAL_ARTICLES} exists and differs from what this script builds; not overwriting")
    VAL_ARTICLES.parent.mkdir(parents=True, exist_ok=True)
    with open(VAL_ARTICLES, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)

    val_set = set(val)
    n_q = {True: 0, False: 0}
    for a in data:
        n_q[a["title"] in val_set] += sum(len(p["qas"]) for p in a["paragraphs"])
    print(f"{len(val)} of {len(titles)} articles -> {VAL_ARTICLES}: "
          f"{n_q[True]} validation questions, {n_q[False]} training questions")


if __name__ == "__main__":
    main()
