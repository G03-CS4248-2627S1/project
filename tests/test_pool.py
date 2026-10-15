"""Task pools (squadqa.pool, workflow.md §5.4): the issue block and the comment log."""

import json

import pytest

from squadqa import pool as P

POOL = {"pool": 1, "branch": "alice/grid", "commit": "a" * 40, "script": "slurm/train_reader.sbatch",
        "sbatch": ["--gpus=a100-40", "--time=00:45:00"],
        "tasks": {"t1": {"export": {"EXPERIMENT": "1b-joint", "TAG": "x"}, "args": ["--set", "length_bias=false"]},
                  "t2": {"export": {"EXPERIMENT": "1b-compound"}, "args": []},
                  "t3": {"args": []}}}


def comment(i, login, body):
    return {"id": i, "login": login, "created": f"2026-10-15T12:00:{i:02d}Z", "body": body}


def test_the_block_round_trips_and_replaces_itself():
    body = P.with_pool("Question: does it help?", POOL)
    assert body.startswith("Question: does it help?") and P.parse_pool(body) == POOL
    json.loads(P.BLOCK.search(body).group(1))                               # valid JSON, one task per line
    changed = {**POOL, "commit": "b" * 40}
    again = P.with_pool(body, changed)
    assert P.parse_pool(again) == changed and again.count("pool:begin") == 1


def test_the_body_gets_plain_newlines():
    """A body read back from GitHub can carry \\r\\n (or worse); every \\r is a line break there."""
    body = P.with_pool("| a | b |\r\r\n|---|---|\r\n| 1 | 2 |\r\n", POOL)
    assert "\r" not in body and "| a | b |\n|---|---|\n| 1 | 2 |" in body and P.parse_pool(body) == POOL


@pytest.mark.parametrize("bad",[{**POOL, "commit": "abc1234"}, {k: v for k, v in POOL.items() if k != "script"},
                                 {**POOL, "tasks": {"a b": {"args": []}}}, {**POOL, "tasks": {"t": {"argv": []}}}])
def test_malformed_pools_raise(bad):
    with pytest.raises(ValueError):
        P.check_pool(bad)
    with pytest.raises(ValueError, match="no pool block"):
        P.parse_pool("no block here")


def test_a_task_moves_through_claim_started_done():
    log = [comment(1, "bob", "pool: claim t1"), comment(2, "bob", "pool: started t1 job=123 gpu=a100-40"),
           comment(3, "bob", "pool: done t1 run=2026-10-15-1b-joint-x-s0-j123 commit=abc1234\nValidation EM / F1: ...")]
    s = P.fold(list(POOL["tasks"]), log, owner="alice")
    assert (s["t1"].state, s["t1"].holder, s["t1"].job, s["t1"].run) == ("done", "bob", "123", "2026-10-15-1b-joint-x-s0-j123")
    assert s["t2"].state == "open" and s["t3"].state == "open"


def test_the_earliest_claim_wins_a_race():
    log = [comment(5, "carol", "pool: claim t2"), comment(4, "bob", "pool: claim t2"),       # out of order on purpose
           comment(6, "carol", "pool: started t2 job=9 gpu=h100-96")]
    t = P.fold(list(POOL["tasks"]), log, owner="alice")["t2"]
    assert (t.state, t.holder, t.job) == ("claimed", "bob", None)                           # carol's start doesn't count


def test_release_reopens_and_only_the_holder_or_owner_may_release():
    log = [comment(1, "bob", "pool: claim t1"), comment(2, "carol", "pool: release t1 mine now"),
           comment(3, "carol", "pool: claim t1")]
    assert P.fold(["t1"], log, owner="alice")["t1"].holder == "bob"
    log += [comment(4, "bob", "pool: started t1 job=7 gpu=a100-40"), comment(5, "bob", "pool: release t1 job 7 ran out of memory")]
    t = P.fold(["t1"], log, owner="alice")["t1"]
    assert (t.state, t.holder, t.note) == ("open", None, "job 7 ran out of memory")
    log += [comment(6, "carol", "pool: claim t1"), comment(7, "carol", "pool: done t1 run=r commit=c"),
            comment(8, "alice", "pool: release t1 bad run, redo")]
    t = P.fold(["t1"], log, owner="alice")["t1"]
    assert (t.state, t.note) == ("open", "bad run, redo")                                  # the owner can reopen a done task


def test_other_comments_and_unknown_tasks_are_ignored():
    log = [comment(1, "bob", "Looks good! pool: claim t1"), comment(2, "bob", ""), comment(3, "bob", "pool: claim t9"),
           comment(4, "bob", "pool: started t1 job=1")]                                     # started without a claim
    assert all(t.state == "open" for t in P.fold(list(POOL["tasks"]), log, owner="alice").values())


def test_pick_named_or_next_open():
    s = P.fold(list(POOL["tasks"]), [comment(1, "bob", "pool: claim t1")], owner="alice")
    assert P.pick(s, [], 5) == ["t2", "t3"] and P.pick(s, [], 1) == ["t2"] and P.pick(s, ["t3"], 1) == ["t3"]
    with pytest.raises(SystemExit, match="claimed"):
        P.pick(s, ["t1"], 1)
    with pytest.raises(SystemExit, match="no task"):
        P.pick(s, ["t7"], 1)
