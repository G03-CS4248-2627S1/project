"""E6 data (squadqa.multi_answer): markup stripping, answer groups, one-answer copies, the dev filter."""

from collections import Counter

import pytest

from squadqa import multi_answer as M


def span(context, sub, nth=0):
    """Inclusive MRQA character span of the nth occurrence of sub."""
    s = -1
    for _ in range(nth + 1):
        s = context.index(sub, s + 1)
    return [s, s + len(sub) - 1]


def record(context, questions):
    """An MRQA context record; questions: {qid: [answer substring or (substring, nth), ...]}."""
    qas = []
    for qid, answers in questions.items():
        found = [span(context, *(a if isinstance(a, tuple) else (a,))) for a in answers]
        qas.append({"qid": qid, "question": f"question {qid}?",
                    "detected_answers": [{"text": context[s:e + 1], "char_spans": [[s, e]]} for s, e in found]})
    return {"context": context, "qas": qas}


def test_strip_markup_maps_every_kept_character():
    text = "<P> The <Td colspan=\"2\"> cat </Td> sat . </P>"
    clean, pos = M.strip_markup(text)
    assert clean == "The cat sat ."
    for i, ch in enumerate(text):
        if pos[i] >= 0:
            assert clean[pos[i]] == ch
    assert pos[text.index("cat")] == clean.index("cat") and pos[text.index("<Td")] == -1


def test_boundary_variants_are_one_answer_and_single_answer_questions_drop():
    context = "<P> Visit the Eiffel Tower or Big Ben . </P>"
    rec = record(context, {"two": ["the Eiffel Tower", "Eiffel Tower", "Big Ben"], "one": ["the Eiffel Tower", "Eiffel Tower"]})
    files = M.build([rec], [], 30, heldout_frac=0.0)
    q = files["q.json"]
    assert set(q) == {"two"}
    assert [a[1] for a in q["two"]["answers"]] == ["the Eiffel Tower", "Big Ben"]      # first span of each group
    assert sum(a[2] for a in q["two"]["answers"]) == pytest.approx(1.0)
    assert files["summary.json"]["questions_with_fewer_than_2_answers"] == 1


def test_train_has_one_copy_per_answer_and_heldout_keeps_them_all():
    recs = [record(f"<P> Paris or Lyon , item {k} , also Nice . </P>", {f"q{k}": ["Paris", "Lyon", "Nice"]}) for k in range(10)]
    files = M.build(recs, [], 30, heldout_frac=0.2, seed=0)
    train, held = files["train.json"]["data"], files["heldout.json"]["data"]
    assert len(train) == 8 and len(held) == 2                                   # split by passage
    for article in train:
        p = article["paragraphs"][0]
        assert [len(qa["answers"]) for qa in p["qas"]] == [1, 1, 1]
        assert sum(qa["weight"] for qa in p["qas"]) == pytest.approx(1.0)       # each question totals 1
        for qa in p["qas"]:
            a = qa["answers"][0]
            assert p["context"][a["answer_start"]:a["answer_start"] + len(a["text"])] == a["text"]
    for article in held:
        qa = article["paragraphs"][0]["qas"][0]
        assert [a["text"] for a in qa["answers"]] == ["Paris", "Lyon", "Nice"]
    train_ctx = {a["paragraphs"][0]["context"] for a in train}
    assert not train_ctx & {a["paragraphs"][0]["context"] for a in held}
    assert files == M.build(recs, [], 30, heldout_frac=0.2, seed=0)             # deterministic


def test_answers_in_markup_or_too_long_are_dropped():
    context = "<P> one two three four </P> <Td> five </Td>"
    s, e = span(context, "<Td> five")
    rec = {"context": context, "qas": [{"qid": "x", "question": "?", "detected_answers": [
        {"text": "one", "char_spans": [span(context, "one")]},
        {"text": "two three four", "char_spans": [span(context, "two three four")]},
        {"text": "<Td> five", "char_spans": [[s, e]]}]}]}
    counts = M.build([rec], [], 2, heldout_frac=0.0)["summary.json"]
    assert counts["answers_in_markup"] == 1 and counts["answers_too_long"] == 1
    assert counts["questions_with_fewer_than_2_answers"] == 1


def test_passages_sharing_a_10_gram_with_dev_are_dropped():
    shared = "a b c d e f g h i j"
    recs = [record(f"<P> {shared} , Paris or Lyon . </P>", {"near": ["Paris", "Lyon"]}),
            record("<P> k l m n o p q r s t , Paris or Lyon . </P>", {"far": ["Paris", "Lyon"]})]
    files = M.build(recs, [f"Earlier text . {shared} and more ."], 30, heldout_frac=0.0)
    assert set(files["q.json"]) == {"far"}
    assert files["summary.json"]["passages_overlapping_dev"] == 1


def test_question_groups_counts_every_answer():
    context = "the cat sat"
    words = M.word_spans(context)
    counts = Counter()
    groups = M.question_groups(context, words, M.content_flags(context, words), list(range(len(context))),
                               {"detected_answers": [{"text": "cat", "char_spans": [[4, 6]]},
                                                     {"text": "the cat", "char_spans": [[0, 6]]}]}, 30, counts)
    assert groups == [(4, "cat")] and counts["answers"] == 2                   # one group of boundary variants
