"""Task pools: share a grid of runs across the team through one GitHub issue (workflow.md §5.4).

The issue body holds the tasks, as JSON between <!-- pool:begin --> and <!-- pool:end -->:

    {"pool": 1, "branch": "<branch>", "commit": "<full sha>", "script": "slurm/train_reader.sbatch",
     "sbatch": ["--gpus=a100-40", "--time=00:45:00"],
     "tasks": {"<name>": {"export": {"EXPERIMENT": "...", "TAG": "..."}, "args": ["--set", "..."]}, ...}}

The issue's comments are the log. Every command posts one comment whose first line is one of

    pool: claim <task>
    pool: started <task> job=<Slurm job id> gpu=<type>
    pool: done <task> run=<run id> commit=<results commit>
    pool: release <task> <reason>

A task's state follows from its comments in order (fold): the earliest claim on an open task holds
it, a later claim loses, and only the holder's started / done / release count (the pool owner, who
opened the issue, may also release). Two people claiming at once is safe: both post, both re-read,
and the later one moves on. Don't delete pool comments.

    python -m squadqa.pool status <issue> [--jobs]       # every task's state; --jobs asks Slurm about running ones
    python -m squadqa.pool claim <issue> [task ...] [-n N] [--gpus TYPE]   # claim open tasks and submit them
    python -m squadqa.pool done <issue> <task>           # fetch results, push them to the pool branch, post done
    python -m squadqa.pool release <issue> <task> <reason ...>
    python -m squadqa.pool publish <issue> <pool.json>   # pool owner: write the tasks into the issue body

Run from the repo root on your own machine, with gh logged in and the soc.sh tunnel open. Agents
follow AGENTS.md, "Task pools": nothing until the user names the pool issue; finish this user's own
tasks first; ask before claim (it submits GPU jobs); record every claim in local/MEMORY.md.
Standard library only.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

SOC = ".claude/skills/soc-cluster/soc.sh"
BLOCK = re.compile(r"<!-- pool:begin -->\s*```json\s*(.*?)\s*```\s*<!-- pool:end -->", re.S)
EVENT = re.compile(r"^pool: (claim|started|done|release) (\S+)(.*)$")


# The pool and its log.

def parse_pool(body: str) -> dict:
    """The pool JSON in an issue body. Raises if it is missing or malformed."""
    m = BLOCK.search(body or "")
    if not m:
        raise ValueError("the issue has no pool block (<!-- pool:begin --> ... <!-- pool:end -->)")
    pool = json.loads(m.group(1))
    check_pool(pool)
    return pool


def check_pool(pool: dict) -> None:
    for key in ("pool", "branch", "commit", "script", "sbatch", "tasks"):
        if key not in pool:
            raise ValueError(f"the pool has no {key!r}")
    if not re.fullmatch(r"[0-9a-f]{40}", pool["commit"]):
        raise ValueError("the pool's commit must be a full 40-character sha")
    for name, spec in pool["tasks"].items():
        if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
            raise ValueError(f"task name {name!r} must be letters, digits, '.', '_' or '-'")
        if set(spec) - {"export", "args"} or not isinstance(spec.get("args", []), list):
            raise ValueError(f"task {name!r} must hold only 'export' (a mapping) and 'args' (a list)")


def render_pool(pool: dict) -> str:
    """The pool block for an issue body: valid JSON, one task per line."""
    check_pool(pool)
    head = json.dumps({k: v for k, v in pool.items() if k != "tasks"})
    tasks = ",\n".join(f"  {json.dumps(name)}: {json.dumps(spec)}" for name, spec in pool["tasks"].items())
    return f"<!-- pool:begin -->\n```json\n{head[:-1]}, \"tasks\": {{\n{tasks}\n}}}}\n```\n<!-- pool:end -->"


def with_pool(body: str, pool: dict) -> str:
    """The issue body with its pool block replaced, or appended if it has none. Line endings
    become \\n: GitHub reads every \\r as a line break, so \\r\\r\\n would add a blank line
    after each line and split tables."""
    block = render_pool(pool)
    body = (body or "").replace("\r\n", "\n").replace("\r", "\n")
    if BLOCK.search(body):
        return BLOCK.sub(lambda m: block, body)
    return f"{(body or '').rstrip()}\n\n### Task pool\n\nRun with `python -m squadqa.pool` (workflow.md §5.4); don't edit this block by hand.\n\n{block}\n"


@dataclass
class Task:
    name: str
    state: str = "open"             # open | claimed | running | done
    holder: str | None = None
    job: str | None = None
    run: str | None = None
    since: str | None = None        # time of the comment that set the state
    note: str = ""                  # the last release reason


def fold(tasks: list[str], comments: list[dict], owner: str) -> dict[str, Task]:
    """Each task's state from the issue's comments: dicts with id, login, created and body,
    taken in id order (ids grow with posting time; created alone can tie)."""
    states = {t: Task(t) for t in tasks}
    for c in sorted(comments, key=lambda c: c["id"]):
        m = EVENT.match((c["body"] or "").strip().splitlines()[0]) if (c["body"] or "").strip() else None
        if not m or m.group(2) not in states:
            continue
        kind, t, rest, who = m.group(1), states[m.group(2)], m.group(3).strip(), c["login"]
        fields = dict(re.findall(r"(\w+)=(\S+)", rest))
        if kind == "claim":
            if t.state == "open":
                t.state, t.holder, t.job, t.run, t.since = "claimed", who, None, None, c["created"]
        elif kind == "release":
            if t.state != "open" and who in (t.holder, owner):
                t.state, t.holder, t.job, t.run, t.since, t.note = "open", None, None, None, c["created"], rest
        elif who != t.holder:
            continue
        elif kind == "started" and t.state == "claimed":
            t.state, t.job, t.since = "running", fields.get("job"), c["created"]
        elif kind == "done" and t.state in ("claimed", "running"):
            t.state, t.run, t.since = "done", fields.get("run"), c["created"]
    return states


def pick(states: dict[str, Task], names: list[str], n: int) -> list[str]:
    """The named tasks (each must be open), or else the first n open ones in pool order."""
    if names:
        for name in names:
            if name not in states:
                raise SystemExit(f"no task {name!r} in the pool")
            if states[name].state != "open":
                raise SystemExit(f"task {name!r} is {states[name].state} ({states[name].holder})")
        return names
    return [name for name, t in states.items() if t.state == "open"][:n]


# GitHub, git and the cluster.

def sh(cmd: list[str], check: bool = True, input: str | None = None) -> str:
    """Run a command and return its stdout. Bytes in and out: text-mode pipes on Windows turn
    each \\n written to stdin into \\r\\n, which doubled the \\r of every line of an issue body."""
    r = subprocess.run(cmd, capture_output=True, input=None if input is None else input.encode("utf-8"))
    out, err = r.stdout.decode("utf-8", "replace"), r.stderr.decode("utf-8", "replace")
    if check and r.returncode != 0:
        raise SystemExit(f"{' '.join(cmd[:4])} ... failed:\n{out}{err}")
    return out


def bash() -> str:
    """Git's bash on Windows (plain "bash" can resolve to WSL's, which has other SSH keys)."""
    for path in (os.environ.get("POOL_BASH"), "C:/Program Files/Git/bin/bash.exe"):
        if path and Path(path).exists():
            return path
    found = shutil.which("bash")
    if not found:
        raise SystemExit("bash not found; set POOL_BASH to Git's bash.exe")
    return found


def soc(*args: str, check: bool = True) -> str:
    return sh([bash(), SOC, *args], check=check)


def issue(number: int) -> tuple[str, str, list[dict]]:
    """(body, author, comments) of an issue."""
    data = json.loads(sh(["gh", "issue", "view", str(number), "--json", "body,author"]))
    raw = json.loads(sh(["gh", "api", "--paginate", "--slurp", f"repos/{{owner}}/{{repo}}/issues/{number}/comments"]))
    comments = [{"id": c["id"], "login": c["user"]["login"], "created": c["created_at"], "body": c["body"]}
                for page in raw for c in page]
    return data["body"], data["author"]["login"], comments


def post(number: int, text: str) -> None:
    sh(["gh", "issue", "comment", str(number), "--body", text])


def me() -> str:
    return sh(["gh", "api", "user", "--jq", ".login"]).strip()


def load(number: int) -> tuple[dict, dict[str, Task], str]:
    body, owner, comments = issue(number)
    try:
        pool = parse_pool(body)
    except ValueError as e:
        raise SystemExit(f"#{number}: {e}")
    return pool, fold(list(pool["tasks"]), comments, owner), owner


def require_pool_commit(pool: dict) -> None:
    head = sh(["git", "rev-parse", "HEAD"]).strip()
    if head != pool["commit"]:
        c = pool["commit"]
        raise SystemExit(f"This checkout is at {head[:7]}, but every run in this pool comes from {c[:7]}. Run:\n"
                         f"  git fetch && git checkout --detach {c}\n"
                         f"  bash {SOC} run 'git fetch && git checkout --detach {c}'\n"
                         "(not while your own jobs are queued: they read the clone when they start)")


# Commands.

def cmd_status(args) -> None:
    pool, states, owner = load(args.issue)
    user = me()
    queue = {}
    running = [t.job for t in states.values() if t.state == "running" and t.job]
    if args.jobs and running:
        for line in soc("run", f"squeue -h -j {','.join(running)} -o '%i %T'", check=False).splitlines():
            if len(line.split()) == 2:
                queue[line.split()[0]] = line.split()[1]
    print(f"pool of #{args.issue} (owner {owner}): commit {pool['commit'][:7]} on {pool['branch']}")
    for t in states.values():
        extra = f"job {t.job}" if t.job else ""
        if args.jobs and t.state == "running":
            extra += f" {queue.get(t.job, 'not in the queue: finished or failed, so run done or release')}"
        if t.run:
            extra += f" run {t.run}"
        if t.state == "open" and t.note:
            extra += f" (released: {t.note})"
        mark = "*" if t.holder == user else " "
        print(f" {mark}{t.name:28} {t.state:8} {t.holder or '':14} {extra} {('since ' + t.since) if t.since else ''}".rstrip())
    counts = {s: sum(t.state == s for t in states.values()) for s in ("open", "claimed", "running", "done")}
    print("  " + ", ".join(f"{v} {k}" for k, v in counts.items()))
    mine = [t.name for t in states.values() if t.holder == user and t.state in ("claimed", "running")]
    print(f"  * = {user}'s; unfinished: {', '.join(mine) or 'none'} (finish these with done or release before claiming more)")
    if sh(["git", "rev-parse", "HEAD"]).strip() != pool["commit"]:
        c = pool["commit"]
        print(f"  This checkout is not at the pool commit {c[:7]}. Before claiming: git fetch && git checkout --detach {c}\n"
              f"  and bash {SOC} run 'git fetch && git checkout --detach {c}' (not while your jobs are queued)")


def cmd_claim(args) -> None:
    pool, states, _ = load(args.issue)
    require_pool_commit(pool)
    user = me()
    for name in pick(states, args.tasks, args.n):
        post(args.issue, f"pool: claim {name}")
        if load(args.issue)[1][name].holder != user:
            print(f"{name}: someone claimed it first; skipped")
            continue
        spec = pool["tasks"][name]
        opts = list(pool["sbatch"])
        if args.gpus:
            opts = [o for o in opts if not o.startswith("--gpus")] + [f"--gpus={args.gpus}"]
        export = ",".join(["ALL"] + [f"{k}={v}" for k, v in spec.get("export", {}).items()])
        out = soc("submit", *opts, f"--export={export}", pool["script"], *spec.get("args", []), check=False)
        job = re.search(r"Submitted batch job (\d+)", out)
        if not job:
            post(args.issue, f"pool: release {name} submit failed")
            raise SystemExit(f"{name}: submit failed, task released:\n{out}")
        gpu = next((o.split("=", 1)[1] for o in opts if o.startswith("--gpus=")), "default")
        post(args.issue, f"pool: started {name} job={job.group(1)} gpu={gpu}")
        print(f"{name}: job {job.group(1)} on {gpu}")


def cmd_done(args) -> None:
    pool, states, _ = load(args.issue)
    t, user = states.get(args.task), me()
    if t is None or t.state != "running" or t.holder != user:
        raise SystemExit(f"{args.task} is not running under {user}: {t}")
    runs = soc("run", f"ls -d results/*-j{t.job} 2>/dev/null", check=False).split()
    if len(runs) != 1 or not soc("run", f"test -f {runs[0]}/metrics.json && echo ok", check=False).strip():
        raise SystemExit(f"job {t.job} has no finished run in the clone's results/ yet; "
                         f"check bash {SOC} log train_reader-{t.job}.out, or release the task")
    run = runs[0].split("/", 1)[1]
    if not Path("results", run).exists():
        soc("fetch", run)
    metrics = json.loads(Path("results", run, "metrics.json").read_text(encoding="utf-8"))
    commit = commit_results(pool["branch"], run, f"Pool #{args.issue}: results of {args.task} ({run})")
    soc("run", f"set -a && . ./.env && set +a && mkdir -p \"$SQUADQA_SCRATCH/fetched\" && "
               f"mv results/{run} \"$SQUADQA_SCRATCH/fetched/\"")
    scores = "; ".join(f"{rule.upper()} {m['exact']:.2f} / {m['f1']:.2f}" for rule, m in metrics["validation"].items())
    post(args.issue, f"pool: done {args.task} run={run} commit={commit[:7]}\nValidation EM / F1: {scores}")
    print(f"{args.task}: done, {run} pushed as {commit[:7]}; {scores}")


def commit_results(branch: str, run: str, message: str) -> str:
    """Commit results/<run>/ (all but candidates_*.json, workflow.md §2.2) to the pool branch from
    a temporary worktree, so this checkout stays at the pool commit. Moves the files there; the
    candidates stay here. Returns the pushed commit."""
    sh(["git", "fetch", "origin", branch])
    tmp = tempfile.mkdtemp(prefix="pool-")
    sh(["git", "worktree", "add", "--detach", tmp, f"origin/{branch}"])
    try:
        dest = Path(tmp, "results", run)
        dest.mkdir(parents=True, exist_ok=False)
        for f in sorted(Path("results", run).rglob("*")):
            if f.is_file() and not f.name.startswith("candidates_"):
                (dest / f.relative_to(Path("results", run))).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), dest / f.relative_to(Path("results", run)))
        sh(["git", "-C", tmp, "add", f"results/{run}"])
        sh(["git", "-C", tmp, "commit", "-q", "-m", message])
        for attempt in range(3):
            if subprocess.run(["git", "-C", tmp, "push", "-q", "origin", f"HEAD:refs/heads/{branch}"]).returncode == 0:
                return sh(["git", "-C", tmp, "rev-parse", "HEAD"]).strip()
            sh(["git", "-C", tmp, "fetch", "-q", "origin", branch])     # someone pushed first: results only add folders
            sh(["git", "-C", tmp, "rebase", "-q", f"origin/{branch}"])
        raise SystemExit(f"could not push to {branch}; the results commit is in {tmp}")
    finally:
        sh(["git", "worktree", "remove", "--force", tmp], check=False)


def cmd_release(args) -> None:
    _, states, owner = load(args.issue)
    t, user = states.get(args.task), me()
    if t is None or t.state == "open" or user not in (t.holder, owner):
        raise SystemExit(f"only the holder or the pool owner can release {args.task}: {t}")
    post(args.issue, f"pool: release {args.task} {' '.join(args.reason)}")
    print(f"{args.task}: released")


def cmd_publish(args) -> None:
    pool = json.loads(Path(args.file).read_text(encoding="utf-8"))
    check_pool(pool)
    sh(["git", "fetch", "origin", pool["branch"]])
    if subprocess.run(["git", "merge-base", "--is-ancestor", pool["commit"], f"origin/{pool['branch']}"]).returncode:
        raise SystemExit(f"commit {pool['commit'][:7]} is not on origin/{pool['branch']}; push it first")
    body, owner, _ = issue(args.issue)
    if owner != me():
        print(f"note: #{args.issue} was opened by {owner}, who counts as the pool owner", file=sys.stderr)
    sh(["gh", "issue", "edit", str(args.issue), "--body-file", "-"], input=with_pool(body, pool))
    print(f"#{args.issue}: pool of {len(pool['tasks'])} tasks at {pool['commit'][:7]}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Share a grid of runs through a GitHub issue (workflow.md §5.4).")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("status")
    p.add_argument("issue", type=int)
    p.add_argument("--jobs", action="store_true", help="ask Slurm about running tasks' jobs")
    p = sub.add_parser("claim")
    p.add_argument("issue", type=int)
    p.add_argument("tasks", nargs="*")
    p.add_argument("-n", type=int, default=1, help="how many open tasks to claim when none are named")
    p.add_argument("--gpus", help="GPU type instead of the pool's, e.g. h100-96")
    p = sub.add_parser("done")
    p.add_argument("issue", type=int)
    p.add_argument("task")
    p = sub.add_parser("release")
    p.add_argument("issue", type=int)
    p.add_argument("task")
    p.add_argument("reason", nargs="+")
    p = sub.add_parser("publish")
    p.add_argument("issue", type=int)
    p.add_argument("file")
    args = parser.parse_args(argv)
    {"status": cmd_status, "claim": cmd_claim, "done": cmd_done, "release": cmd_release, "publish": cmd_publish}[args.command](args)


if __name__ == "__main__":
    main()
