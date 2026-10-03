"""Download the external datasets into data/external/ (gitignored).

Run from the repo root:  python -m squadqa.download_external

Every source is a fixed URL (pinned to a git commit where possible) with a
SHA-256 of the downloaded bytes. A mismatch raises an error. If every output
is already in place the script does nothing; delete data/external/ to rebuild.
JSON files are written in the same layout as qa/*.json: ASCII-only (json.dump
default, because qa/evaluate-v2.0.py opens data files without an encoding,
which is cp1252 on Windows), 2-space indent, LF line endings and a trailing
newline.

Duplicates are removed before writing (rules in squadqa/dedup.py; the first
copy is kept):
- Repeated answers within a question, in every file.
- Questions that repeat a question in qa/train-v1.1.json, qa/dev-v1.1.json,
  an earlier file of CLEANED below, or earlier in the same file. The qa/ files
  are only read, never changed. SQuAD 2.0 train shrinks to the questions that
  v1.1 train does not already have, which are the unanswerable ones.
- Adversarial SQuAD is checked only within each file: it repeats dev-v1.1 on
  purpose (see below).
- Paraphrased SQuAD is checked only within each file. Its *_orig files are the
  dev-v1.1 questions lowercased, kept on purpose as the like-for-like baseline
  for *_para. Para and orig are paired by article title (one question each);
  the ids are offset by one (orig 106201 pairs with para 106200). A question
  removed from one file of a pair is removed from the other as well.
Removed ids, with the question each one repeats, go to data/external/duplicates.json.

What each dataset may be used for. Passage overlap was checked against the
course files on 2026-09-24:

| Path                     | Passages from | Use                                      |
|--------------------------|---------------|------------------------------------------|
| squad2/train-v2.0.json   | train-v1.1    | training only (plausible_answers = hard  |
|                          |               | verifier negatives); drop val articles   |
| adversarialqa/train.json | train-v1.1    | training; drop val articles              |
| adversarialqa/dev.json   | dev-v1.1      | evaluation ONLY, never tune on it        |
|                          |               | (spec §1 rules 1-2)                      |
| squadshifts/             | new text      | evaluation only                          |
| paraphrased_squad/       | dev-v1.1      | evaluation only                          |
| adversarial_squad/       | dev-v1.1      | evaluation only                          |

AdversarialQA is kept as its combined/ split (the union of the dbidaf, dbert
and droberta subsets). Its test.json files have no answers and its
squad_splits/ is dev-v1.1 re-split, so neither is kept. SQuAD 2.0 dev is not
downloaded: it comes from the dev-v1.1 articles.

Adversarial SQuAD (Jia & Liang, 2017) has AddSent and AddOneSent. Each file
holds 1,000 dev-v1.1 questions unchanged (same id and passage) plus adversarial
versions of them, whose ids add a suffix such as "-high-conf-turk0" and whose
passages have one distracting sentence added. Score as in implementation_details.md §6.
"""

import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

from squadqa.dedup import Seen, dedupe_answers, find_duplicate_questions, iter_qas, remove_questions

OUT = Path("data/external")
REFERENCE = ["qa/train-v1.1.json", "qa/dev-v1.1.json"]

SHIFTS = "https://raw.githubusercontent.com/modestyachts/squadshifts-website/077771091a3993f5310cabc98f24a41d067c36cf/datasets"
PARA = "https://raw.githubusercontent.com/nusnlp/paraphrasing-squad/860c4924d7746e17294eae548776e882f9bf52dd/datasets"
CODALAB = "https://worksheets.codalab.org/rest/bundles"   # bundles are immutable

# (destination under OUT, url, sha256 of the downloaded bytes)
JSON_FILES = [
    # SQuAD 2.0 (Rajpurkar et al., 2018)
    ("squad2/train-v2.0.json",
     "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json",
     "68dcfbb971bd3e96d5b46c7177b16c1a4e7d4bdef19fb204502738552dede002"),
    # SQuAD-Shifts (Miller et al., 2020)
    ("squadshifts/new_wiki.json", f"{SHIFTS}/new_wiki_v1.0.json",
     "1d50a0ca4cdab88d038b52ea8a2d0510af6038d7f41a0a256b9659237769d0ee"),
    ("squadshifts/nyt.json", f"{SHIFTS}/nyt_v1.0.json",
     "fd55ae252cc735a695fab4473d4cd605b154183ac3b3eb63068f53cfc86f36d6"),
    ("squadshifts/reddit.json", f"{SHIFTS}/reddit_v1.0.json",
     "b4ac346dbf1816fdec80fdcf91d06a4b754fa51e47960fa8614a94f37ea42a4b"),
    ("squadshifts/amazon_reviews.json", f"{SHIFTS}/amazon_reviews_v1.0.json",
     "9db7bdee84a91f0fb80b9e74f365042fbbb217cb8e0a7e73bb071245472589b6"),
    # Paraphrased SQuAD (Gan & Ng, 2019)
    ("paraphrased_squad/dev_para.json", f"{PARA}/dev_para.json",
     "aa55be585ece44d7ea489f3b96dddc474c48faa43caba65490e7f0291b6cca1a"),
    ("paraphrased_squad/dev_orig.json", f"{PARA}/dev_orig.json",
     "a07aa501ab98f9eed859249398d25296a7277774c9d6392eb2482438a99c7349"),
    ("paraphrased_squad/adv_para.json", f"{PARA}/adv_para.json",
     "eefc9f44a67d55546a9a2f320467c67e271b85885608eabe8944b5d209560652"),
    ("paraphrased_squad/adv_orig.json", f"{PARA}/adv_orig.json",
     "dc2d3a5e01fe8424d6204bf401c5aad39f60f269f04599d75679f53251e75c40"),
    # Adversarial SQuAD (Jia & Liang, 2017), the authors' CodaLab release
    ("adversarial_squad/addsent.json", f"{CODALAB}/0xb765680b60c64d088f5daccac08b3905/contents/blob/",
     "40e3602aa5195cdacd03904a9c301ceb17ccf730cc32bd3ab998b66b4401e660"),
    ("adversarial_squad/addonesent.json", f"{CODALAB}/0x3ac9349d16ba4e7bb9b5920e3b1af393/contents/blob/",
     "50420ac8d8b7547cd3715347c9a276802bc9466328ba0814adce4c20495e2889"),
]

# AdversarialQA (Bartolo et al., 2020): one zip, keep the combined split.
AQA_URL = "https://adversarialqa.github.io/data/aqa_v1.0.zip"
AQA_SHA256 = "f4f3c23224a5060b28c35e35581bd5cf46256dda3665418fb83d036d0e0c93cf"
AQA_MEMBERS = {"combined/train.json": "adversarialqa/train.json",
               "combined/dev.json": "adversarialqa/dev.json"}

# Question deduplication order after the qa/ reference files: earlier files win.
CLEANED = ["squad2/train-v2.0.json", "adversarialqa/train.json", "adversarialqa/dev.json",
           "squadshifts/new_wiki.json", "squadshifts/nyt.json",
           "squadshifts/reddit.json", "squadshifts/amazon_reviews.json"]
# Checked only within each file, because they repeat dev-v1.1 on purpose.
WITHIN_FILE = ["adversarial_squad/addsent.json", "adversarial_squad/addonesent.json"]
PARA_PAIRS = [("paraphrased_squad/dev_para.json", "paraphrased_squad/dev_orig.json"),
              ("paraphrased_squad/adv_para.json", "paraphrased_squad/adv_orig.json")]


def fetch(url: str, sha256: str) -> bytes:
    print(f"GET {url}")
    with urllib.request.urlopen(url) as r:
        data = r.read()
    got = hashlib.sha256(data).hexdigest()
    if got != sha256:
        raise RuntimeError(f"SHA-256 mismatch for {url}: expected {sha256}, got {got}")
    return data


def write_json(data: dict, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    # Same layout as qa/*.json: 2-space indent, LF line endings, trailing newline.
    with open(tmp, "w", encoding="ascii", newline="\n") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    tmp.replace(dest)


def remove_duplicates(files: dict[str, dict]) -> dict:
    """Remove duplicate answers and questions in place. Returns the report."""
    report = {}
    seen = Seen()
    for path in REFERENCE:
        with open(path, encoding="utf-8") as f:
            ref = json.load(f)
        for _, para, qa in iter_qas(ref):
            seen.add(path, para, qa)

    def record(rel: str, dups: list[dict], answers: int, before: int) -> None:
        report[rel] = {"questions_before": before,
                       "questions_after": sum(1 for _ in iter_qas(files[rel])),
                       "answers_removed": answers, "questions_removed": dups}

    for rel in CLEANED:
        data = files[rel]
        before = sum(1 for _ in iter_qas(data))
        answers = dedupe_answers(data)
        dups = find_duplicate_questions(rel, data, seen)
        remove_questions(data, {d["index"] for d in dups})
        record(rel, dups, answers, before)

    for rel in WITHIN_FILE:
        data = files[rel]
        before = sum(1 for _ in iter_qas(data))
        answers = dedupe_answers(data)
        dups = find_duplicate_questions(rel, data, Seen())
        remove_questions(data, {d["index"] for d in dups})
        record(rel, dups, answers, before)

    for pair in PARA_PAIRS:
        found = {}
        for rel in pair:
            answers = dedupe_answers(files[rel])
            found[rel] = (find_duplicate_questions(rel, files[rel], Seen()), answers)
        titles = {d["title"] for dups, _ in found.values() for d in dups}
        for rel in pair:
            data = files[rel]
            before = sum(1 for _ in iter_qas(data))
            dups, answers = found[rel]
            # The partner of a removed question is removed too, recorded without duplicate_of.
            listed = {d["index"] for d in dups}
            partners = [{"index": i, "id": qa["id"], "title": a["title"], "duplicate_of": None}
                        for i, (a, _, qa) in enumerate(iter_qas(data))
                        if a["title"] in titles and i not in listed]
            remove_questions(data, listed | {d["index"] for d in partners})
            record(rel, dups + partners, answers, before)
    return report


def main() -> None:
    outputs = [rel for rel, _, _ in JSON_FILES] + list(AQA_MEMBERS.values())
    if all((OUT / rel).exists() for rel in outputs):
        print(f"All files are in {OUT}; delete it to rebuild.")
        return

    files = {rel: json.loads(fetch(url, sha).decode("utf-8")) for rel, url, sha in JSON_FILES}
    with zipfile.ZipFile(io.BytesIO(fetch(AQA_URL, AQA_SHA256))) as z:
        for member, rel in AQA_MEMBERS.items():
            files[rel] = json.loads(z.read(member).decode("utf-8"))

    report = remove_duplicates(files)
    for rel, r in report.items():
        print(f"{rel}: {r['questions_before']} -> {r['questions_after']} questions, "
              f"{r['answers_removed']} duplicate answers removed")
    for rel, data in files.items():
        write_json(data, OUT / rel)
    write_json(report, OUT / "duplicates.json")
    print(f"Wrote {len(files)} files and duplicates.json to {OUT}")


if __name__ == "__main__":
    main()
