"""Find and remove duplicate questions and answers in SQuAD-format data.

Two kinds of duplicate, and the first copy is always the one kept:
- Answer: same `text` and `answer_start` as an earlier answer of the same
  question (also checked in SQuAD 2.0's `plausible_answers`). qa/dev-v1.1.json
  was cleaned with the same rule.
- Question: same `id`, or same passage and question text, as an earlier
  question in the same file or in an earlier file. Passage and question are
  compared with whitespace collapsed, and the question also ignoring case.

Detector (changes nothing; earlier files win):
    python -m squadqa.dedup qa/train-v1.1.json qa/dev-v1.1.json data/external/squad2/train-v2.0.json
"""

import json
import sys
from collections import Counter


def _norm(s: str) -> str:
    return " ".join(s.split())


def question_key(context: str, question: str) -> tuple[str, str]:
    return _norm(context), _norm(question).lower()


def iter_qas(data: dict):
    for article in data["data"]:
        for para in article["paragraphs"]:
            for qa in para["qas"]:
                yield article, para, qa


class Seen:
    """Questions seen so far: id and question key -> (file, id) of the first copy."""

    def __init__(self) -> None:
        self.ids: dict[str, tuple[str, str]] = {}
        self.keys: dict[tuple[str, str], tuple[str, str]] = {}

    def match(self, para: dict, qa: dict) -> tuple[str, str] | None:
        return self.ids.get(qa["id"]) or self.keys.get(question_key(para["context"], qa["question"]))

    def add(self, name: str, para: dict, qa: dict) -> None:
        self.ids.setdefault(qa["id"], (name, qa["id"]))
        self.keys.setdefault(question_key(para["context"], qa["question"]), (name, qa["id"]))


def dedupe_answers(data: dict) -> int:
    """Remove repeated answers in place. Returns how many were removed."""
    removed = 0
    for _, _, qa in iter_qas(data):
        for field in ("answers", "plausible_answers"):
            if field not in qa:
                continue
            seen, keep = set(), []
            for ans in qa[field]:
                k = (ans["text"], ans["answer_start"])
                if k not in seen:
                    seen.add(k)
                    keep.append(ans)
            removed += len(qa[field]) - len(keep)
            qa[field] = keep
    return removed


def find_duplicate_questions(name: str, data: dict, seen: Seen) -> list[dict]:
    """Questions in `data` that repeat one in `seen` or an earlier one in `data`.

    Every question that is not a duplicate is added to `seen` under `name`.
    `index` is the question's position in iter_qas order. A duplicate can share
    its id with the copy that is kept, so removal goes by index, not id.
    """
    found = []
    for index, (article, para, qa) in enumerate(iter_qas(data)):
        first = seen.match(para, qa)
        if first:
            found.append({"index": index, "id": qa["id"], "title": article.get("title"),
                          "duplicate_of": {"file": first[0], "id": first[1]}})
        else:
            seen.add(name, para, qa)
    return found


def remove_questions(data: dict, indices: set[int]) -> None:
    """Remove questions by iter_qas index in place, then any paragraph or article left empty."""
    index = 0
    for article in data["data"]:
        for para in article["paragraphs"]:
            keep = []
            for qa in para["qas"]:
                if index not in indices:
                    keep.append(qa)
                index += 1
            para["qas"] = keep
        article["paragraphs"] = [p for p in article["paragraphs"] if p["qas"]]
    data["data"] = [a for a in data["data"] if a["paragraphs"]]


def main(paths: list[str]) -> None:
    seen = Seen()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        n = sum(1 for _ in iter_qas(data))
        answers = dedupe_answers(data)
        dups = find_duplicate_questions(path, data, seen)
        where = Counter("same file" if d["duplicate_of"]["file"] == path else d["duplicate_of"]["file"]
                        for d in dups)
        print(f"{path}: {n} questions, {len(dups)} duplicate questions {dict(where)}, "
              f"{answers} duplicate answers")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
