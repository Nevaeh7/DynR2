"""Download Amazon 2014 data, fit semantic codes, and prepare training inputs."""

import argparse
import ast
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import shutil
import urllib.request

import numpy as np

DATASETS = {
    "Beauty": ("Beauty", 22363, 12101, 198502),
    "Sports": ("Sports_and_Outdoors", 35598, 18357, 296337),
    "Toys": ("Toys_and_Games", 19412, 11924, 167597),
}
SOURCE = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles"


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def read_records(path):
    with gzip.open(path, "rt", encoding="utf-8", errors="strict") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                try:
                    yield ast.literal_eval(line)
                except (ValueError, SyntaxError) as error:
                    raise ValueError(f"Invalid record at {path}:{number}") from error


def download(category, raw_root):
    slug = DATASETS[category][0]
    folder = Path(raw_root) / category
    folder.mkdir(parents=True, exist_ok=True)
    for name in (f"reviews_{slug}_5.json.gz", f"meta_{slug}.json.gz"):
        target = folder / name
        if target.exists():
            if target.stat().st_size == 0:
                raise ValueError(f"Empty download: {target}")
            continue
        temporary = target.with_suffix(target.suffix + ".part")
        with (
            urllib.request.urlopen(f"{SOURCE}/{name}", timeout=120) as response,
            temporary.open("wb") as stream,
        ):
            shutil.copyfileobj(response, stream)
        if temporary.stat().st_size == 0:
            raise ValueError(f"Empty download: {name}")
        temporary.replace(target)


def construct_sids(category, args):
    from . import prep_kernels as kernels

    slug, users, items, interactions = DATASETS[category]
    raw = args.raw_root / category
    output = args.processed_root / category
    output.mkdir(parents=True, exist_ok=True)
    sequences = defaultdict(list)
    frequency = Counter()
    for record in read_records(raw / f"reviews_{slug}_5.json.gz"):
        user, item = record.get("reviewerID"), record.get("asin")
        if not user or not item:
            continue
        sequences[str(user)].append((str(item), int(record.get("unixReviewTime", 0))))
        frequency[str(item)] += 1
    sequences = {
        user: [item for item, _ in sorted(events, key=lambda pair: pair[1])]
        for user, events in sequences.items()
    }
    if (
        not sequences
        or min(map(len, sequences.values())) < args.k_core
        or min(frequency.values()) < args.k_core
    ):
        raise ValueError("Input does not satisfy the requested k-core protocol")
    actual = (len(sequences), len(frequency), sum(map(len, sequences.values())))
    if not args.skip_benchmark_check and actual != (users, items, interactions):
        raise ValueError(
            f"{category} benchmark counts changed: {actual} != {(users, items, interactions)}"
        )
    item_ids = list(dict.fromkeys(item for seq in sequences.values() for item in seq))
    user_ids = list(sequences)
    metadata = {item: {"title": "", "description": "", "categories": ""} for item in item_ids}
    for record in read_records(raw / f"meta_{slug}.json.gz"):
        item = record.get("asin")
        if item in metadata:
            metadata[item] = {
                "title": kernels._normalize_scalar_text(record.get("title")),
                "description": kernels._normalize_description(record.get("description")),
                "categories": kernels._normalize_categories(record.get("categories")),
            }
    texts = [
        " ".join(value for value in metadata[item].values() if value).strip()
        or "No embedding text available."
        for item in item_ids
    ]
    if args.embeddings:
        embeddings = np.load(args.embeddings, allow_pickle=False)
    elif args.skip_embedding:
        embeddings = np.load(output / "item_embeddings.npy", allow_pickle=False)
    else:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(args.embedding_model, device=args.device)
        encoder.max_seq_length = 512
        if str(encoder.device).startswith("cuda"):
            encoder.half()
        embeddings = encoder.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        del encoder
    if (
        embeddings.ndim != 2
        or len(embeddings) != len(item_ids)
        or not np.isfinite(embeddings).all()
    ):
        raise ValueError("Embedding rows must align with the first-appearance item ordering")
    if len(embeddings) < args.codebook_size:
        raise ValueError("More items than codebook entries are required")
    np.save(output / "item_embeddings.npy", embeddings)
    kernels.CODEBOOK_SIZE, kernels.RQ_N_ITER = args.codebook_size, args.rq_iterations
    codes, codebooks, _ = kernels._run_rq_kmeans(embeddings)
    np.save(output / "sid_codes.npy", codes)
    np.savez(output / "codebooks.npz", **dict(zip(kernels.SID_LAYER_PREFIX, codebooks)))
    mapping = {
        item: {
            "int_id": i + 1,
            "sid": "<|sid_begin|>"
            + "".join(f"<s_{layer}_{int(code)}>" for layer, code in zip("abcd", codes[i]))
            + "<|sid_end|>",
        }
        for i, item in enumerate(item_ids)
    }
    save_json(output / "item_mapping.json", mapping)
    save_json(
        output / f"{category}.pretrain.json",
        {
            str(i + 1): {**metadata[item], "sid": mapping[item]["sid"]}
            for i, item in enumerate(item_ids)
        },
    )
    item_to_int = {item: i + 1 for i, item in enumerate(item_ids)}
    save_json(
        output / "id_mapping.json",
        {
            "user2id": {"[PAD]": 0, **{u: i + 1 for i, u in enumerate(user_ids)}},
            "item2id": {"[PAD]": 0, **item_to_int},
            "id2user": ["[PAD]", *user_ids],
            "id2item": ["[PAD]", *item_ids],
        },
    )
    with (output / f"{category}_sequential_data.txt").open("w") as stream:
        for index, seq in enumerate(sequences.values(), 1):
            stream.write(f"{index} " + " ".join(str(item_to_int[item]) for item in seq) + "\n")
    save_json(
        output / "preparation.json",
        {
            "category": category,
            "codebook_size": args.codebook_size,
            "rq_iterations": args.rq_iterations,
            "seed": 42,
            "counts": list(actual),
            "benchmark_checked": not args.skip_benchmark_check,
        },
    )


def prepare_splits(category, args):
    from . import prep_kernels as kernels

    source = args.processed_root / category
    items = json.loads((source / f"{category}.pretrain.json").read_text())
    mapping = json.loads((source / "item_mapping.json").read_text())
    sequences = [
        line.split()
        for line in (source / f"{category}_sequential_data.txt").read_text().splitlines()
    ]
    sequences = [(user, seq) for user, *seq in sequences]
    centroids = np.load(source / "codebooks.npz", allow_pickle=False)
    adjacency = {}
    for layer, prefix in enumerate(kernels.SID_LAYER_PREFIX):
        matrix = centroids[prefix].astype(np.float32)
        normed = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)
        similarities = normed @ normed.T
        np.fill_diagonal(similarities, 0.0)
        edges, neighbors = [], defaultdict(list)
        for src in range(len(matrix)):
            for dst in np.argsort(similarities[src])[::-1][:16]:
                weight = float(similarities[src, dst])
                if weight >= 0.15:
                    edge = {"src": int(src), "dst": int(dst), "weight": round(weight, 4)}
                    edges.append(edge)
                    neighbors[int(src)].append({"dst": int(dst), "weight": edge["weight"]})
        adjacency[layer] = dict(neighbors)
        save_json(source / f"edges_{prefix}.json", edges)
    sa = {key: {entry["dst"] for entry in value} for key, value in adjacency[0].items()}
    for prefix, splits in [
        ("training_align_data", kernels._split_align(items, sequences)),
        ("training_prediction_sid_data", kernels._split_sid_prediction(items, sequences)),
    ]:
        for split, frame in splits.items():
            if frame.empty:
                raise ValueError(f"Empty {prefix} {split} input")
            frame.to_parquet(source / f"{prefix}_{split}.parquet", index=False)
    histories = kernels._split_ra(items, sequences)
    reverse = {str(entry["int_id"]): item for item, entry in mapping.items()}
    by_user = dict(sequences)
    exported = {}
    for split, tail in [("train", 2), ("val", 1), ("test", 0)]:
        frame = histories[split].copy()
        if frame.empty:
            raise ValueError(f"Empty {split} histories")
        labels, sample_ids, targets, history_items = [], [], [], []
        for row in frame.itertuples(index=False):
            prior = kernels.extract_sid_strings(row.description)
            anchor = kernels._classify_sample(prior, row.groundtruth, sa)["anchor_sid"]
            path = kernels._construct_nav_path(anchor, row.groundtruth, adjacency)
            labels.append(kernels._build_cot_text(path)[0])
            sample_ids.append(
                kernels._make_sample_id(row.user_id, row.groundtruth, row.description)
            )
            visible = by_user[row.user_id][:-tail] if tail else by_user[row.user_id]
            targets.append(reverse[visible[-1]])
            history_items.append([reverse[item] for item in visible[:-1]])
        frame["sample_id"], frame["sid_routing_think"] = sample_ids, labels
        frame["target_item"], frame["history_items"] = targets, history_items
        frame.to_parquet(source / f"training_sid_routing_{split}.parquet", index=False)
        exported[split] = frame
    output = args.export_root / category
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / "item_mapping.json", output / "item_mapping.json")
    for split in ("train", "test"):
        exported[split].to_parquet(output / f"{split}.parquet", index=False)
    val = exported["val"]
    count = min(args.model_selection_users, len(val))
    if count < 1:
        raise ValueError("Checkpoint selection requires at least one validation user")
    selected = val.sample(n=count, random_state=1427)
    selected.to_parquet(output / "model_selection.parquet", index=False)
    counts = Counter(reverse[item] for _, seq in sequences for item in seq[:-3])
    save_json(output / "train_history_counts.json", dict(counts))
    save_json(
        output / "preparation.json",
        {
            "source_category": category,
            "model_selection_users": selected.user_id.astype(str).tolist(),
            "model_selection_seed": 1427,
            "split_sizes": {k: len(v) for k, v in exported.items()},
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--category", choices=[*DATASETS, "all"], required=True)
    p.add_argument("--phase", choices=["download", "sid", "hnsw", "all"], default="all")
    p.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    p.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    p.add_argument("--export-root", type=Path, default=Path("data"))
    p.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--device", default=None)
    p.add_argument("--embeddings", type=Path)
    p.add_argument("--skip-embedding", action="store_true")
    p.add_argument("--codebook-size", type=int, default=256)
    p.add_argument("--rq-iterations", type=int, default=30)
    p.add_argument("--model-selection-users", type=int, default=256)
    p.add_argument("--k-core", type=int, default=5)
    p.add_argument("--skip-benchmark-check", action="store_true")
    a = p.parse_args()
    if min(a.codebook_size, a.rq_iterations, a.k_core, a.model_selection_users) < 1:
        p.error("Codebook size, iterations, k-core, and selection size must be positive")
    for category in DATASETS if a.category == "all" else [a.category]:
        if a.phase in ("all", "download"):
            download(category, a.raw_root)
        if a.phase in ("all", "sid"):
            construct_sids(category, a)
        if a.phase in ("all", "hnsw"):
            prepare_splits(category, a)
        print(f"{category}: {a.phase} complete", flush=True)
