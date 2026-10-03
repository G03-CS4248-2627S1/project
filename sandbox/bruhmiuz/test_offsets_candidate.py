"""Candidate for tests/reference/test_offsets.py (issue #19).

This file is deliberately outside tests/reference/: repository policy requires a human who
did not write squadqa/data.py to review the reference test line by line before placing it
there.  The test reads the raw SQuAD JSON independently and exercises the public mapping and
window functions rather than trusting featurize's bookkeeping.
"""

import json

from squadqa import data as D
from squadqa.config import load_config


def test_word_rule_and_deberta_leading_space():
    text = "U.S. $5,000 well-known Beyoncé's"
    assert [text[s:e] for s, e in D.word_spans(text)] == [
        "U", ".", "S", ".", "$", "5", ",", "000", "well", "-", "known",
        "Beyoncé", "'", "s",
    ]

    # DeBERTa-v3 includes the leading space in the raw token offset.  The adjusted
    # span must lie inside the second word rather than crossing its boundary.
    assert D.word_spans("Who won") == [(0, 3), (4, 7)]
    assert D.token_spans("Who won", [(0, 3), (3, 7)]) == [(0, 3), (4, 7)]


def test_all_training_offsets_round_trip_and_fit_a_window():
    cfg = load_config()
    tokenizer = D.load_tokenizer(cfg["model"])
    assert tokenizer(
        "Who won", add_special_tokens=False, return_offsets_mapping=True
    )["offset_mapping"] == [(0, 3), (3, 7)]
    layout = D.pair_layout(tokenizer)
    skipped = set(D.SKIPPED_IDS.read_text(encoding="utf-8").splitlines())

    with open(D.TRAIN, encoding="utf-8") as f:
        articles = json.load(f)["data"]

    misaligned: set[str] = set()
    overlong: set[str] = set()
    checked = 0

    for article in articles:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            ctx = D.context_tokens(context, tokenizer)
            qas = paragraph["qas"]
            question_ids = tokenizer(
                [qa["question"] for qa in qas], add_special_tokens=False
            )["input_ids"]

            for qa, q_ids in zip(qas, question_ids):
                # train-v1.1 has exactly one answer per question.
                assert len(qa["answers"]) == 1
                answer = qa["answers"][0]
                text = answer["text"]
                start = answer["answer_start"]
                assert context[start:start + len(text)] == text

                word_span = D.answer_word_span(ctx.words, start, text)
                if word_span is None:
                    misaligned.add(qa["id"])
                    continue

                i, j = word_span
                token_start = int(ctx.word_first[i])
                token_end = int(ctx.word_last[j])
                char_start = ctx.spans[token_start][0]
                char_end = ctx.spans[token_end][1]
                assert context[char_start:char_end] == text, qa["id"]

                if j - i + 1 > cfg["max_answer_words"]:
                    overlong.add(qa["id"])

                windows = D.make_windows(
                    q_ids, ctx, layout, cfg["max_seq_len"], cfg["doc_stride"]
                )
                for window in windows:
                    assert len(window.input_ids) <= cfg["max_seq_len"], qa["id"]
                containing = [
                    w for w in windows
                    if w.first_word <= i and j < w.first_word + w.n_words
                ]
                assert containing, qa["id"]

                # The answer's passage token IDs must survive unchanged in every window
                # that contains it, including answers near an overlap boundary.
                expected_ids = ctx.ids[token_start:token_end + 1]
                for window in containing:
                    local_i = i - window.first_word
                    local_j = j - window.first_word
                    actual_ids = window.input_ids[
                        window.first_tok[local_i]:window.last_tok[local_j] + 1
                    ].tolist()
                    assert actual_ids == expected_ids, qa["id"]
                checked += 1

    assert checked == 87_599 - 265
    assert len(misaligned) == 265
    assert len(overlong) == 78
    assert misaligned.isdisjoint(overlong)
    assert misaligned | overlong == skipped
