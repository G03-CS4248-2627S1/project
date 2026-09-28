"""Quick sanity fine-tune: DeBERTa-v3-large with a standard token-level start/end head.

This is NOT the spec's E0 baseline, which predicts word spans (spec §3.2-3.4, §4).
It gives a first accuracy number and times one large run (plan W1).
- Train: train-v1.1 minus the validation articles (data/splits/val_articles.txt).
- Evaluate: the validation articles, with the official scorer. dev-v1.1 is not used.

Run on a GPU node from the repo root, with the repo on PYTHONPATH (the .sbatch does this):
    python sandbox/bruhmiuz/quick_finetune.py --run-id <id> [--per-device-batch 16]
        [--max-steps N] [--val-limit N] [--ckpt-every N]
Checkpoints go to $SQUADQA_SCRATCH/runs/<run-id>/. Re-running with the same run id resumes.
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
os.environ["HF_HUB_OFFLINE"] = "1"  # use the pinned files in $HF_HOME, never the network

import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, get_linear_schedule_with_warmup

from squadqa.download_model import MODELS
from squadqa.splits import load_val_titles

NAME = "microsoft/deberta-v3-large"
MAX_LEN, STRIDE = 384, 128
EFFECTIVE_BATCH, LR, WARMUP_FRAC, WEIGHT_DECAY, EPOCHS, SEED = 32, 2e-5, 0.1, 0.01, 1, 0
TOPK, MAX_ANSWER_TOKENS = 20, 64


def env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set; add it to .env")
    return value


def load_data(val_limit: int | None):
    with open("qa/train-v1.1.json", encoding="utf-8") as f:
        articles = json.load(f)["data"]
    val_titles = load_val_titles()
    train, val = [], []
    for a in articles:
        if a["title"] in val_titles:
            val.append(a)
            continue
        for p in a["paragraphs"]:
            for q in p["qas"]:
                ans = q["answers"][0]
                train.append((q["question"], p["context"], ans["answer_start"],
                              ans["answer_start"] + len(ans["text"])))
    if val_limit is not None:  # keep the first val_limit questions, same structure
        left, kept = val_limit, []
        for a in val:
            paras = []
            for p in a["paragraphs"]:
                if left <= 0:
                    break
                paras.append({**p, "qas": p["qas"][:left]})
                left -= len(paras[-1]["qas"])
            if paras:
                kept.append({**a, "paragraphs": paras})
        val = kept
    return train, val


def featurize_train(tok, qs):
    """(input_ids, start_token, end_token) per window; (0, 0) = CLS when the answer is not fully inside."""
    enc = tok([q[0] for q in qs], [q[1] for q in qs], truncation="only_second", max_length=MAX_LEN,
              stride=STRIDE, return_overflowing_tokens=True, return_offsets_mapping=True)
    feats = []
    for i in range(len(enc["input_ids"])):
        _, _, a0, a1 = qs[enc["overflow_to_sample_mapping"][i]]
        seq, offs = enc.sequence_ids(i), enc["offset_mapping"][i]
        ctx = [t for t, s in enumerate(seq) if s == 1]
        start = end = 0
        if offs[ctx[0]][0] <= a0 and offs[ctx[-1]][1] >= a1:
            # Offsets can include a word's leading space, so compare against span ends/starts.
            start = next(t for t in ctx if offs[t][1] > a0)
            end = next(t for t in reversed(ctx) if offs[t][0] < a1)
        feats.append((enc["input_ids"][i], start, end))
    return feats


def epoch_batches(feats, seed: int, epoch: int):
    """Shuffled batches of EFFECTIVE_BATCH windows, each of similar length (length bucketing)."""
    rng = random.Random(seed * 1000 + epoch)
    idx = list(range(len(feats)))
    rng.shuffle(idx)
    batches = []
    for s in range(0, len(idx), EFFECTIVE_BATCH * 50):
        part = sorted(idx[s:s + EFFECTIVE_BATCH * 50], key=lambda i: len(feats[i][0]))
        batches += [part[k:k + EFFECTIVE_BATCH] for k in range(0, len(part), EFFECTIVE_BATCH)]
    rng.shuffle(batches)
    return batches


def collate(tok, rows):
    batch = tok.pad([{"input_ids": ids} for ids, _, _ in rows], return_tensors="pt")
    batch["start_positions"] = torch.tensor([s for _, s, _ in rows])
    batch["end_positions"] = torch.tensor([e for _, _, e in rows])
    return {k: v.cuda(non_blocking=True) for k, v in batch.items()}


def predict(model, tok, articles, batch_size: int = 64) -> dict:
    qs = [(q["id"], q["question"], p["context"]) for a in articles for p in a["paragraphs"] for q in p["qas"]]
    enc = tok([q[1] for q in qs], [q[2] for q in qs], truncation="only_second", max_length=MAX_LEN,
              stride=STRIDE, return_overflowing_tokens=True, return_offsets_mapping=True)
    order = sorted(range(len(enc["input_ids"])), key=lambda i: len(enc["input_ids"][i]))
    best = {}  # question id -> (score, text), best non-empty span over all its windows
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            batch = tok.pad([{"input_ids": enc["input_ids"][i]} for i in idx], return_tensors="pt")
            out = model(**{k: v.cuda() for k, v in batch.items()})
            S, E = out.start_logits.float().cpu().numpy(), out.end_logits.float().cpu().numpy()
            for row, i in enumerate(idx):
                qid, _, ctx = qs[enc["overflow_to_sample_mapping"][i]]
                seq, offs = enc.sequence_ids(i), enc["offset_mapping"][i]
                ctx_tok = [t for t, sid in enumerate(seq) if sid == 1]
                lo, hi = ctx_tok[0], ctx_tok[-1]
                for a in np.argsort(S[row, lo:hi + 1])[::-1][:TOPK] + lo:
                    for b in np.argsort(E[row, lo:hi + 1])[::-1][:TOPK] + lo:
                        if b < a or b - a + 1 > MAX_ANSWER_TOKENS:
                            continue
                        score = S[row, a] + E[row, b]
                        if qid in best and score <= best[qid][0]:
                            continue
                        text = ctx[offs[a][0]:offs[b][1]].strip()  # offsets include the leading space
                        if text:
                            best[qid] = (score, text)
    missing = [q[0] for q in qs if q[0] not in best]
    if missing:
        raise RuntimeError(f"{len(missing)} questions got no span, e.g. {missing[:3]}")
    return {q[0]: best[q[0]][1] for q in qs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--per-device-batch", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=None, help="stop early (smoke tests)")
    ap.add_argument("--val-limit", type=int, default=None, help="evaluate only the first N questions")
    ap.add_argument("--ckpt-every", type=int, default=500)
    args = ap.parse_args()
    if EFFECTIVE_BATCH % args.per_device_batch:
        raise ValueError(f"--per-device-batch must divide {EFFECTIVE_BATCH}")
    run_dir = Path(env("SQUADQA_SCRATCH")) / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "checkpoint.pt"
    torch.manual_seed(SEED)
    rev = MODELS[NAME]["revision"]
    print(f"run {args.run_id} on {torch.cuda.get_device_name()}, dir {run_dir}", flush=True)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(NAME, revision=rev)
    train_qs, val_articles = load_data(args.val_limit)
    feats = featurize_train(tok, train_qs)
    n_batches = len(epoch_batches(feats, SEED, 0))
    total_steps = args.max_steps or n_batches * EPOCHS
    print(f"{len(train_qs)} train questions -> {len(feats)} windows, {n_batches} steps/epoch, "
          f"{total_steps} steps; featurized in {time.time() - t0:.0f}s", flush=True)

    # fp32 master weights, bf16 autocast. dtype must be explicit: transformers 5 loads the stored
    # dtype by default, and this checkpoint is stored in fp16.
    model = AutoModelForQuestionAnswering.from_pretrained(NAME, revision=rev, dtype=torch.float32).cuda()
    if {p.dtype for p in model.parameters()} != {torch.float32}:
        raise RuntimeError(f"expected fp32 weights, got {({p.dtype for p in model.parameters()})}")
    no_decay = lambda n: n.endswith("bias") or "LayerNorm" in n
    opt = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if not no_decay(n)], "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if no_decay(n)], "weight_decay": 0.0},
    ], lr=LR, fused=True)
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP_FRAC * total_steps), total_steps)
    step = 0
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cuda", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        step = ck["step"]
        print(f"resumed from step {step}", flush=True)

    def save_checkpoint():
        dtypes = {p.dtype for p in model.parameters()} | {
            v.dtype for s in opt.state.values() for v in s.values() if torch.is_tensor(v) and v.dim() > 0}
        if dtypes != {torch.float32}:
            raise RuntimeError(f"refusing to save non-fp32 weights or optimizer state: {dtypes}")
        tmp = ckpt_path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "step": step}, tmp)
        tmp.replace(ckpt_path)

    accum = EFFECTIVE_BATCH // args.per_device_batch
    model.train()
    t_train, windows_done, loss_sum, loss_n = time.time(), 0, 0.0, 0
    batches, batches_epoch = None, -1
    while step < total_steps:
        epoch, pos = divmod(step, n_batches)
        if epoch != batches_epoch:
            batches, batches_epoch = epoch_batches(feats, SEED, epoch), epoch
        batch = batches[pos]
        for k in range(accum):
            rows = [feats[i] for i in batch[k * args.per_device_batch:(k + 1) * args.per_device_batch]]
            if not rows:
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**collate(tok, rows)).loss
            (loss * len(rows) / len(batch)).backward()
            loss_sum, loss_n = loss_sum + loss.item() * len(rows), loss_n + len(rows)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        windows_done += len(batch)
        if step % 50 == 0 or step == total_steps:
            el = time.time() - t_train
            print(f"step {step}/{total_steps} loss {loss_sum / loss_n:.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{windows_done / el:.1f} windows/s, {el / 60:.1f} min", flush=True)
            loss_sum, loss_n = 0.0, 0
        if step % args.ckpt_every == 0 or step == total_steps:
            save_checkpoint()
    train_minutes = (time.time() - t_train) / 60

    t_pred = time.time()
    preds = predict(model, tok, val_articles)
    data_path, pred_path = run_dir / "val_data.json", run_dir / "val_predictions.json"
    json.dump({"version": "1.1", "data": val_articles}, open(data_path, "w"))  # ASCII, as the scorer needs
    json.dump(preds, open(pred_path, "w"))
    scores = json.loads(subprocess.run([sys.executable, "qa/evaluate-v2.0.py", str(data_path), str(pred_path)],
                                       capture_output=True, text=True, check=True).stdout)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    metrics = {"run_id": args.run_id, "commit": commit, "model": NAME, "revision": rev, "seed": SEED,
               "val_questions": scores["total"], "exact": scores["exact"], "f1": scores["f1"],
               "steps": step, "train_minutes_this_job": round(train_minutes, 1),
               "predict_seconds": round(time.time() - t_pred), "gpu": torch.cuda.get_device_name(),
               "config": {"lr": LR, "effective_batch": EFFECTIVE_BATCH, "per_device_batch": args.per_device_batch,
                          "epochs": EPOCHS, "warmup_frac": WARMUP_FRAC, "weight_decay": WEIGHT_DECAY,
                          "max_len": MAX_LEN, "stride": STRIDE, "max_answer_tokens": MAX_ANSWER_TOKENS}}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print("RESULT", json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
