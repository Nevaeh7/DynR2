"""Target-blind inference, SID/item evaluation, and complete-request timing."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .core import pack_history
from .data import history_counts, load_counts, load_rows
from .runtime import artifact_identity, build_model, counts_identity, load_config, make_decoder


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def request(model, tokenizer, catalog, decoder, row, config, policy, artifact, counts):
    sync(model.device)
    start = time.perf_counter()
    ids, truncated = pack_history(row["description"], tokenizer, 4000)
    error = None
    try:
        candidates, detail = model.recommend_policy(
            ids,
            decoder,
            policy,
            config["tau_u"],
            config["tau_b"],
            prefix_counts=catalog.prefix_counts,
            predictor=artifact["model"] if artifact else None,
            costs=artifact["incremental_cost_ms"] if artifact else None,
            cost_weight=artifact.get("cost_weight", 0.0) if artifact else 0.0,
        )
    except ValueError as exc:
        if "No valid SID survived" not in str(exc):
            raise
        error, candidates, detail = str(exc), [], {}
    items = catalog.expand(candidates, counts) if counts is not None else None
    sync(model.device)
    elapsed = (time.perf_counter() - start) * 1000
    # Labels are accessed only after prediction and the request timer have finished.
    scores = (
        catalog.score(row, candidates, counts)
        if row.get("groundtruth") is not None or row.get("target_item") is not None
        else None
    )
    return {
        "sample_id": row["sample_id"],
        "user_id": row["user_id"],
        "policy": policy,
        "candidates": candidates,
        "candidate_items": items,
        "scores": scores,
        "end_to_end_ms": elapsed,
        "truncated": truncated,
        "invalid_reason": error,
        **detail,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--policy", choices=("direct", "E", "L", "rule", "value"), default="rule")
    p.add_argument("--value-model", type=Path)
    p.add_argument("--history-counts", type=Path)
    p.add_argument("--prior-train", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--warmup", type=int, default=1)
    a = p.parse_args()
    if a.repeats < 1 or a.warmup < 0 or (a.history_counts and a.prior_train):
        p.error("Invalid repeats/warmup or conflicting prior inputs")
    if a.policy == "value" and a.value_model is None:
        p.error("Value inference requires --value-model")
    if a.policy != "direct" and a.adapter is None:
        p.error("Continued-response policies require a trained response adapter")
    config = load_config(a.config)
    artifact = json.loads(a.value_model.read_text()) if a.value_model else None
    if artifact and (artifact.get("format") != "dynr2-value-v1" or artifact["config"] != config):
        p.error("The frozen Value predictor does not match this inference configuration")
    a.output.mkdir(parents=True, exist_ok=False)
    model, tokenizer, catalog = build_model(
        a.model, a.catalog, config, a.adapter, a.device, a.dtype
    )
    rows = load_rows(a.data)
    counts = (
        load_counts(a.history_counts, catalog)
        if a.history_counts
        else history_counts(load_rows(a.prior_train), catalog)
        if a.prior_train
        else None
    )
    if any(row.get("target_item") is not None for row in rows) and counts is None:
        p.error("Exact-item metrics require --history-counts or --prior-train")
    if artifact:
        if counts is None or artifact["counts_sha256"] != counts_identity(counts):
            p.error("Value inference requires the same frozen training-history counts as fitting")
        if artifact["artifact_identity"] != artifact_identity(a.model, a.adapter, catalog):
            p.error("Value predictor model/tokenizer/catalog differs from inference artifacts")
        overlap = {r["user_id"] for r in rows} & set(
            artifact["fitting_users"]
            + artifact["selection_users"]
            + artifact["model_selection_users"]
        )
        if overlap:
            p.error("Evaluation users overlap controller fitting/selection users")
    decoder = make_decoder(catalog, tokenizer, config)
    for i in range(a.warmup):
        request(
            model,
            tokenizer,
            catalog,
            decoder,
            rows[i % len(rows)],
            config,
            a.policy,
            artifact,
            counts,
        )
    results = []
    with (a.output / "predictions.jsonl").open("x") as stream:
        for row in rows:
            repeats = [
                request(model, tokenizer, catalog, decoder, row, config, a.policy, artifact, counts)
                for _ in range(a.repeats)
            ]
            result = repeats[0]
            if any(r["candidates"] != result["candidates"] for r in repeats[1:]):
                raise RuntimeError("Repeated deterministic inference changed its ranking")
            result["end_to_end_ms"] = float(np.mean([r["end_to_end_ms"] for r in repeats]))
            stream.write(json.dumps(result, allow_nan=False) + "\n")
            results.append(result)
    summary = {
        "policy": a.policy,
        "n": len(results),
        "config": config,
        "invalid": sum(r["invalid_reason"] is not None for r in results),
        "mean_ms": float(np.mean([r["end_to_end_ms"] for r in results])),
        "p95_ms": float(np.percentile([r["end_to_end_ms"] for r in results], 95)),
        "timing_scope": "Tokenization, history encoding, queries, control, continuation, SID decoding, item expansion",
        "repeats": a.repeats,
        "warmup": a.warmup,
    }
    labeled = [r for r in results if r["scores"] is not None]
    if labeled:
        for scope in ("sid", "item"):
            scored = [r["scores"][scope] for r in labeled if scope in r["scores"]]
            if scored:
                summary[scope] = {
                    key: float(np.mean([s[key] for s in scored])) for key in scored[0]
                }
                summary[scope + "_n"] = len(scored)
    for key in ("blocks", "answer_queries"):
        measured = [r[key] for r in results if key in r]
        summary["mean_" + key] = float(np.mean(measured)) if measured else None
    (a.output / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
