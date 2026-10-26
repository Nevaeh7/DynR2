"""Fit two signed continuation-gain Ridge heads on user-disjoint state banks."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .core import pack_history
from .data import disjoint_users, history_counts, load_counts, load_rows
from .evaluate import request, sync
from .features import belief_features
from .runtime import artifact_identity, build_model, counts_identity, load_config, make_decoder
from .value import FEATURE_GROUPS, FORMAT, design_vector, predict, vector

ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)


def ridge(x, y, alpha):
    names = FEATURE_GROUPS["full_belief"]
    x = np.asarray([design_vector(dict(zip(names, row)), names, "onehot_0_3") for row in x])
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-12] = 1.0
    z = (x - mean) / scale
    intercept = y.mean(0)
    coef = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y - intercept))
    return {
        "format": FORMAT,
        "feature_group": "full_belief",
        "feature_names": list(names),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "alpha": alpha,
        "depth_encoding": "onehot_0_3",
        "heads": {
            a: {"intercept": float(intercept[i]), "coef": coef[:, i].tolist()}
            for i, a in enumerate(("E", "L"))
        },
    }


def fit_banks(train, validation):
    disjoint_users(train, validation)
    names = FEATURE_GROUPS["full_belief"]

    def arrays(rows):
        x, y = [], []
        for row in rows:
            if len(row["states"]) != 3:
                raise ValueError("Each user must contribute three target-independent states")
            for state in row["states"]:
                x.append(vector(state["features"], names))
                y.append([state["rewards"][a] - state["rewards"]["D"] for a in ("E", "L")])
        x, y = np.asarray(x), np.asarray(y)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Nonfinite bank features/rewards")
        return x, y

    x, y = arrays(train)
    vx, vy = arrays(validation)
    models = [ridge(x, y, alpha) for alpha in ALPHAS]
    errors = [
        float(
            np.mean(
                (
                    np.asarray(
                        [
                            [predict(model, dict(zip(names, row)))[a] for a in ("E", "L")]
                            for row in vx
                        ]
                    )
                    - vy
                )
                ** 2
            )
        )
        for model in models
    ]
    winner = min(range(len(models)), key=lambda i: (errors[i], ALPHAS[i]))
    costs = {
        a: max(
            0.0,
            float(
                np.mean(
                    [
                        s["costs_ms"][a] - s["costs_ms"]["D"]
                        for row in validation
                        for s in row["states"]
                    ]
                )
            ),
        )
        for a in ("E", "L")
    }
    rate = float(np.abs(vy).mean()) / max(1.0, max(costs.values()))
    return {
        "model": models[winner],
        "incremental_cost_ms": costs,
        "lambda_candidates": [0.0, rate / 4, rate],
        "cost_weight": 0.0,
        "alpha_candidates": list(ALPHAS),
        "validation_mse": errors,
    }


def state_plan(domain, user):
    raw = hashlib.sha256(f"dynr2|{domain}|{user}|state-bank".encode()).digest()
    family = raw[0] % 3
    layout = (
        "EEE"
        if family == 0
        else "LLL"
        if family == 1
        else "".join("E" if bit & 1 else "L" for bit in raw[1:4])
    )
    depths = sorted(sorted((1, 2, 3), key=lambda d: raw[4 + d])[:2])
    return layout, [0, *depths]


def collect(model, tokenizer, catalog, decoder, rows, counts, domain):
    bank = []
    for row in rows:
        layout, depths = state_plan(domain, row["user_id"])
        prompt, _ = pack_history(row["description"], tokenizer, 4000)
        parent, previous, last = model.start(prompt), None, None
        records = []
        for depth in range(max(depths) + 1):
            q, detail = model.observe(parent)
            if depth in depths:
                features = belief_features(
                    q, previous, depth, last, model.first_ids, catalog.prefix_counts, detail
                )
                outcomes, costs = {}, {}
                for action in ("D", "E", "L"):
                    sync(model.device)
                    start = time.perf_counter()
                    branch = parent.fork()
                    if action != "D":
                        branch, _ = model.advance(branch, action)
                    try:
                        outcomes[action] = model.decode_state(branch, decoder)
                    except ValueError as error:
                        if "No valid SID survived" not in str(error):
                            raise
                        outcomes[action] = []
                    catalog.expand(outcomes[action], counts)
                    sync(model.device)
                    costs[action] = (time.perf_counter() - start) * 1000
                records.append(
                    {"depth": depth, "features": features, "outcomes": outcomes, "costs_ms": costs}
                )
            if depth < max(depths):
                previous, last = q, layout[depth]
                parent, _ = model.advance(parent, last)
        # Finish every counterfactual before reading any target for this user.
        for state in records:
            state["rewards"] = {
                a: catalog.score(row, cands, counts)["item"]["ndcg@10"]
                for a, cands in state["outcomes"].items()
            }
        bank.append(
            {
                "sample_id": row["sample_id"],
                "user_id": row["user_id"],
                "layout": layout,
                "states": records,
            }
        )
    return bank


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument(
        "--model-selection-users",
        type=Path,
        required=True,
        help="JSON list of users used for response-checkpoint selection",
    )
    p.add_argument("--history-counts", type=Path)
    p.add_argument("--prior-train", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if bool(a.history_counts) == bool(a.prior_train):
        p.error("Supply one of --history-counts and --prior-train")
    train, validation = load_rows(a.train), load_rows(a.validation)
    disjoint_users(train, validation)
    excluded = set(map(str, json.loads(a.model_selection_users.read_text())))
    for rows in (train, validation):
        users = [r["user_id"] for r in rows]
        if len(set(users)) != len(users) or set(users) & excluded:
            p.error("Use one row per user and exclude response-checkpoint-selection users")
        if any(r.get("target_item") is None for r in rows):
            p.error("Continuation-value fitting requires exact-item targets")
    config = load_config(a.config)
    if (config["block_size"], config["max_blocks"], config["latent_topk"]) != (4, 4, 8):
        p.error("Value fitting requires four four-position blocks and top-eight inputs")
    a.output.mkdir(parents=True, exist_ok=False)
    model, tokenizer, catalog = build_model(a.model, a.catalog, config, a.adapter)
    counts = (
        load_counts(a.history_counts, catalog)
        if a.history_counts
        else history_counts(load_rows(a.prior_train), catalog)
    )
    decoder = make_decoder(catalog, tokenizer, config)
    # Warm-up calls do not enter bank timings or fitting labels.
    prompt, _ = pack_history(train[0]["description"], tokenizer, 4000)
    model.recommend_policy(prompt, decoder, "direct")
    banks = [
        collect(model, tokenizer, catalog, decoder, rows, counts, config["domain"])
        for rows in (train, validation)
    ]
    artifact = {
        "format": "dynr2-value-v1",
        **fit_banks(*banks),
        "config": config,
        "artifact_identity": artifact_identity(a.model, a.adapter, catalog),
        "counts_sha256": counts_identity(counts),
        "fitting_users": [r["user_id"] for r in train],
        "selection_users": [r["user_id"] for r in validation],
        "model_selection_users": sorted(excluded),
        "cost_scope": "Mean validation branch(E/L)-branch(D); current observation excluded",
        "test_labels_used": False,
    }
    scores = []
    for weight in artifact["lambda_candidates"]:
        candidate = {**artifact, "cost_weight": weight}
        results = [
            request(model, tokenizer, catalog, decoder, row, config, "value", candidate, counts)
            for row in validation
        ]
        scores.append(
            {
                "cost_weight": weight,
                "ndcg@10": float(np.mean([r["scores"]["item"]["ndcg@10"] for r in results])),
                "hr@10": float(np.mean([r["scores"]["item"]["hr@10"] for r in results])),
                "mean_ms": float(np.mean([r["end_to_end_ms"] for r in results])),
            }
        )
    winner = min(
        range(len(scores)),
        key=lambda i: (
            -round(scores[i]["ndcg@10"], 12),
            -round(scores[i]["hr@10"], 12),
            scores[i]["mean_ms"],
            i,
        ),
    )
    artifact["cost_weight"] = artifact["lambda_candidates"][winner]
    artifact["validation_policy_scores"] = scores
    for split, bank in zip(("train", "validation"), banks):
        with (a.output / f"{split}_bank.jsonl").open("x") as stream:
            for row in bank:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
    (a.output / "value_model.json").write_text(
        json.dumps(artifact, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "alpha": artifact["model"]["alpha"],
                "incremental_cost_ms": artifact["incremental_cost_ms"],
                "lambda_candidates": artifact["lambda_candidates"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
