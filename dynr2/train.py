"""Distributed response-only LoRA adaptation with validation checkpoint selection."""

import argparse
from contextlib import nullcontext
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .core import pack_history
from .data import history_counts, load_counts, load_rows
from .losses import sid_supports, sid_target_ids
from .paths import external_checkpoint_path
from .runtime import build_model, load_config, make_decoder, save_response_adapter
from .schedule import schedule


@torch.no_grad()
def validate(model, tokenizer, catalog, rows, counts, config, rank, world):
    model.eval()
    # Checkpoint selection averages E/L at all four nonzero depths.
    decoder = make_decoder(
        catalog, tokenizer, {**config, "step_weights": [1.0] * 8, "step_temperatures": [1.0] * 8}
    )
    total = torch.zeros(2, device=model.device, dtype=torch.float64)
    for row in rows[rank::world]:
        ids, _ = pack_history(row["description"], tokenizer, 4000)
        for depth in (1, 2, 3, 4):
            for action in ("E", "L"):
                try:
                    candidates, _ = model.recommend(ids, decoder, action * depth)
                except ValueError as error:
                    if "No valid SID survived" not in str(error):
                        raise
                    candidates = []
                scores = catalog.score(row, candidates, counts)
                total[0] += scores.get("item", scores["sid"])["ndcg@10"]
                total[1] += 1
    if world > 1:
        dist.all_reduce(total)
    if int(total[1]) != len(rows) * 8:
        raise RuntimeError("Incomplete checkpoint-selection coverage")
    return float(total[0] / total[1])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--history-counts", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=1427)
    p.add_argument("--schedule", choices=("anytime", "fixed"), default="anytime")
    a = p.parse_args()
    a.output = external_checkpoint_path(a.output)
    rank, local, world = (
        int(os.environ.get(k, 0 if k != "WORLD_SIZE" else 1))
        for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    )
    if a.epochs < 1 or world not in (1, 2, 4, 8):
        p.error("Use positive epochs and 1, 2, 4, or 8 workers for effective batch size eight")
    if not torch.cuda.is_available():
        p.error("Response adaptation requires CUDA")
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    random.seed(a.seed)
    config = load_config(a.config)
    if (config["block_size"], config["max_blocks"], config["latent_topk"]) != (4, 4, 8):
        p.error(
            "Response-adaptation schedules require block size four, four blocks, and top-eight inputs"
        )
    rows, validation = load_rows(a.train), load_rows(a.validation)
    if Path(a.train).resolve() == Path(a.validation).resolve():
        p.error("Training and checkpoint-selection files must differ")
    if rank == 0:
        a.output.mkdir(parents=True, exist_ok=False)
    if world > 1:
        dist.barrier()
    model, tokenizer, catalog = build_model(
        a.model, a.catalog, config, device=f"cuda:{local}", training=True
    )
    counts = (
        load_counts(a.history_counts, catalog)
        if a.history_counts
        else history_counts(rows, catalog)
        if all("history_items" in r for r in rows)
        else None
    )
    if any(r.get("target_item") is not None for r in validation) and counts is None:
        p.error("Exact-item checkpoint selection requires training-history counts")
    supports = sid_supports(tokenizer)
    trainable = [v for v in model.parameters() if v.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-5, weight_decay=0.01)
    steps = math.ceil(len(rows) / 8)
    updates = steps * a.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda i: min(1.0, (i + 1) / max(1, updates // 10)) * max(0.1, 1 - i / updates)
    )
    ddp = (
        DistributedDataParallel(
            model, device_ids=[local], broadcast_buffers=False, find_unused_parameters=False
        )
        if world > 1
        else model
    )
    best, selected, records = -math.inf, 0, []
    for epoch in range(a.epochs + 1):
        if epoch:
            order = list(range(len(rows)))
            random.Random(a.seed + epoch).shuffle(order)
            ddp.train()
            for step in range(steps):
                active_n = min(8, len(rows) - step * 8)
                losses = torch.zeros((), device=model.device)
                for micro in range(8 // world):
                    index = step * 8 + micro * world + rank
                    active = index < len(rows)
                    row = rows[order[index if active else 0]]
                    prompt, _ = pack_history(row["description"], tokenizer, 4000)
                    target = sid_target_ids(catalog.target_sid(row), tokenizer)
                    history_hash = hashlib.sha256(row["description"].encode()).hexdigest()
                    layout = schedule(row["user_id"], history_hash, epoch - 1, a.seed, a.schedule)
                    scope = ddp.no_sync() if world > 1 and micro + 1 < 8 // world else nullcontext()
                    with scope, torch.autocast("cuda", dtype=torch.bfloat16):
                        loss, _, _ = ddp(prompt, target, layout, supports, 0.2)
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite response-adaptation loss")
                        (loss * (world / active_n if active else 0.0)).backward()
                    if active:
                        losses += loss.detach()
                norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                if world > 1:
                    dist.all_reduce(losses)
                if rank == 0 and (step == 0 or (step + 1) % 25 == 0 or step + 1 == steps):
                    print(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "step": step + 1,
                                "loss": float(losses) / active_n,
                                "grad_norm": float(norm),
                            }
                        ),
                        flush=True,
                    )
        score = validate(model, tokenizer, catalog, validation, counts, config, rank, world)
        if score > best + 1e-12:
            best, selected = score, epoch
        if rank == 0:
            directory = a.output / f"epoch_{epoch}"
            save_response_adapter(model, directory)
            records.append({"epoch": epoch, "validation_ndcg@10": score})
            (a.output / "training.json").write_text(
                json.dumps(
                    {
                        "seed": a.seed,
                        "schedule": a.schedule,
                        "selected_epoch": selected,
                        "selected_adapter": f"epoch_{selected}",
                        "validation": records,
                        "selection": "Mean NDCG@10 over E/L at depths 1,2,3,4; earliest ties",
                        "validation_metric": "item"
                        if all("target_item" in r for r in validation)
                        else "sid",
                        "global_batch": 8,
                        "lr": 1e-5,
                        "anchor_weight": 0.2,
                        "lora_rank": 16,
                        "lora_alpha": 32,
                        "world_size": world,
                        "model_selection_users": sorted({r["user_id"] for r in validation}),
                    },
                    indent=2,
                )
                + "\n"
            )
            (a.output / "model_selection_users.json").write_text(
                json.dumps(sorted({r["user_id"] for r in validation}), indent=2) + "\n"
            )
            print(
                json.dumps(
                    {"epoch": epoch, "validation_ndcg@10": score, "selected_epoch": selected}
                ),
                flush=True,
            )
        if world > 1:
            dist.barrier()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
