"""E6 data (plan §7.2, §8): questions with several annotated answers and a known answer distribution q.

E6 checks whether cross-entropy makes the reader's probabilities match q (plan §7.2, "Checking
faithfulness directly"). SQuAD train has one answer per question and dev is off limits, so the
questions come from another dataset: the dev split of Natural Questions (Kwiatkowski et al., 2019)
in the MRQA 2019 shared-task format (Fisch et al., 2019), pinned by URL and SHA-256. Its license is
CC BY-SA 3.0, so the files stay in data/external/ (gitignored) and are rebuilt from the pin.

    python -m squadqa.multi_answer          # download once, then build data/external/multi_answer/

Build steps (the counts at each step go to summary.json):
1. Strip the HTML markup that MRQA keeps in NQ contexts (<P>, <Td>, <Li>, ...) and remap each
   answer's offsets. An answer that starts or ends inside markup is dropped.
2. Map answers to word spans (squadqa.data); drop those that are not word-aligned or are longer
   than max_answer_words.
3. Group a question's answers by the content words they cover (squadqa.decode.group_keys), so
   boundary variants are one answer, represented by its first annotated span. Keep questions with
   two or more groups. MRQA keeps no annotator counts, so each group gets probability 1/k. By
   construction that is the true q: training draws its answers from it.
4. Drop passages that share a word 10-gram with a dev-v1.1 passage (spec §1, rules 1-2).
5. Split by passage: HELDOUT_FRAC held out, seed 0.

Outputs in data/external/multi_answer/, SQuAD format, ASCII (json.dump's default):
- train.json: each question once per answer, as an ordinary one-answer question (id <qid>_a<k>), so
  the answer frequencies equal q. Training stays plain cross-entropy.
- heldout.json: each held-out question once, with all its answers (the scorer's multi-reference format).
- q.json: {question id: {"split": ..., "answers": [[answer_start, text, probability], ...]}}, the known q.
- summary.json: the counts.
E6 only: never ladder training or tuning data.
"""

import gzip
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np

from squadqa.data import answer_word_span, word_spans
from squadqa.decode import content_flags, group_keys
from squadqa.download_external import fetch, write_json

OUT = Path("data/external/multi_answer")
RAW = OUT / "raw" / "NaturalQuestionsShort-dev.jsonl.gz"
NQ_URL = "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NaturalQuestionsShort.jsonl.gz"
NQ_SHA256 = "2ba8b2181b520f81b49d62c0e4a23819f33d5dec0e8cf4a623edcda0feb73530"   # MD5 matches MRQA's README
DEV = Path("qa/dev-v1.1.json")
HELDOUT_FRAC = 0.2
NGRAM = 10
TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9]*(?:\s[^<>]*)?>")


def strip_markup(text: str) -> tuple[str, list[int]]:
    """text without markup tags, each tag and whitespace run becoming one space; and for each
    original character, its position in the new text (-1 if removed)."""
    out: list[str] = []
    pos = [-1] * len(text)

    def space():
        if out and out[-1] != " ":
            out.append(" ")

    k = 0
    for m in [*TAG.finditer(text), None]:
        end = m.start() if m else len(text)
        for i in range(k, end):
            if text[i].isspace():
                space()
            else:
                pos[i] = len(out)
                out.append(text[i])
        if m:
            space()
            k = m.end()
    while out and out[-1] == " ":
        out.pop()
    return "".join(out), pos


def ngrams(text: str, n: int = NGRAM) -> set[tuple[str, ...]]:
    words = [text[s:e].lower() for s, e in word_spans(text)]
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def question_groups(context: str, words, flags, pos: list[int], qa: dict, max_answer_words: int,
                    counts: Counter) -> list[tuple[int, str]]:
    """The question's distinct answers as (answer_start, text) in the stripped context, one per
    group of boundary variants, in annotation order."""
    groups: dict[tuple, tuple[int, str]] = {}
    for d in qa["detected_answers"]:
        for s, e in d["char_spans"]:                    # inclusive character offsets
            counts["answers"] += 1
            ns, ne = pos[s], pos[e]
            if ns < 0 or ne < 0:
                counts["answers_in_markup"] += 1
                continue
            text = context[ns:ne + 1]
            span = answer_word_span(words, ns, text)
            if span is None:
                counts["answers_not_word_aligned"] += 1
                continue
            if span[1] - span[0] + 1 > max_answer_words:
                counts["answers_too_long"] += 1
                continue
            key = group_keys(flags, np.array([span[0]]), np.array([span[1]]))[0]
            groups.setdefault(key, (ns, text))
    return list(groups.values())


def build(records, dev_contexts: list[str], max_answer_words: int, heldout_frac: float = HELDOUT_FRAC,
          seed: int = 0) -> dict[str, dict]:
    """The E6 files from MRQA context records: {file name: JSON content}."""
    dev_grams = set().union(*(ngrams(c) for c in dev_contexts)) if dev_contexts else set()
    counts: Counter = Counter()
    passages = []
    for rec in records:
        counts["passages"] += 1
        context, pos = strip_markup(rec["context"])
        words = word_spans(context)
        flags = content_flags(context, words)
        qas = []
        for qa in rec["qas"]:
            counts["questions"] += 1
            answers = question_groups(context, words, flags, pos, qa, max_answer_words, counts)
            if len(answers) < 2:
                counts["questions_with_fewer_than_2_answers"] += 1
                continue
            qas.append((qa["qid"], qa["question"], answers))
        if not qas:
            continue
        if ngrams(context) & dev_grams:
            counts["passages_overlapping_dev"] += 1
            counts["questions_overlapping_dev"] += len(qas)
            continue
        passages.append((context, qas))

    order = list(range(len(passages)))
    random.Random(seed).shuffle(order)
    held = set(order[:round(heldout_frac * len(passages))])
    files = {"train.json": {"version": "e6-nq-mrqa-dev-train", "data": []},
             "heldout.json": {"version": "e6-nq-mrqa-dev-heldout", "data": []}}
    q: dict[str, dict] = {}
    for k, (context, qas) in enumerate(passages):
        split = "heldout" if k in held else "train"
        out_qas = []
        for qid, question, answers in qas:
            if qid in q:
                raise ValueError(f"question id {qid} appears twice")
            q[qid] = {"split": split, "answers": [[s, t, 1 / len(answers)] for s, t in answers]}
            if split == "train":
                out_qas += [{"id": f"{qid}_a{a}", "question": question, "answers": [{"text": t, "answer_start": s}]}
                            for a, (s, t) in enumerate(answers)]
            else:
                out_qas.append({"id": qid, "question": question,
                                "answers": [{"text": t, "answer_start": s} for s, t in answers]})
            counts[f"{split}_questions"] += 1
            counts[f"{split}_answers"] += len(answers)
        files[f"{split}.json"]["data"].append({"title": f"nq-{k:05d}", "paragraphs": [{"context": context, "qas": out_qas}]})
        counts[f"{split}_passages"] += 1
    files["q.json"] = q
    files["summary.json"] = {"source": NQ_URL, "sha256": NQ_SHA256, "max_answer_words": max_answer_words,
                             "heldout_frac": heldout_frac, "seed": seed, "ngram": NGRAM, **dict(sorted(counts.items()))}
    return files


def read_mrqa(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        lines = f.read().splitlines()
    if "header" not in json.loads(lines[0]):
        raise ValueError(f"{path} has no MRQA header line")
    return [json.loads(line) for line in lines[1:] if line.strip()]


def main() -> None:
    from squadqa.config import load_config

    if RAW.exists():
        got = hashlib.sha256(RAW.read_bytes()).hexdigest()
        if got != NQ_SHA256:
            raise RuntimeError(f"{RAW} has SHA-256 {got}, expected {NQ_SHA256}; delete it to download again")
    else:
        RAW.parent.mkdir(parents=True, exist_ok=True)
        RAW.write_bytes(fetch(NQ_URL, NQ_SHA256))
    with open(DEV, encoding="utf-8") as f:
        dev = [p["context"] for a in json.load(f)["data"] for p in a["paragraphs"]]
    files = build(read_mrqa(RAW), dev, load_config()["max_answer_words"])
    for name, data in files.items():
        write_json(data, OUT / name)
    s = files["summary.json"]
    print(f"{s['questions']} questions -> {s.get('train_questions', 0)} train ({s.get('train_answers', 0)} one-answer copies), "
          f"{s.get('heldout_questions', 0)} held out; {s.get('passages_overlapping_dev', 0)} passages dropped for "
          f"overlapping dev; wrote {OUT}")


if __name__ == "__main__":
    main()
