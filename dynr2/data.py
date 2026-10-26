"""Portable history, catalog, collision-prior, and ranking interfaces."""

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re

from .losses import sid_target_ids

SID_PATTERN = re.compile(r"<\|sid_begin\|><s_a_\d+><s_b_\d+><s_c_\d+><s_d_\d+><\|sid_end\|>")


def load_rows(path):
    path = Path(path)
    if path.suffix == ".parquet":
        import pandas as pd

        rows = pd.read_parquet(path).to_dict("records")
    elif path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    elif path.suffix == ".json":
        rows = json.loads(path.read_text())
    else:
        raise ValueError("Data must be JSONL, JSON, or Parquet")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty list of histories")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not {"sample_id", "user_id", "description"} <= row.keys():
            raise ValueError("Each history needs sample_id, user_id, and description")
        row["sample_id"], row["user_id"] = str(row["sample_id"]), str(row["user_id"])
        if row["sample_id"] in seen or not isinstance(row["description"], str):
            raise ValueError("Repeated sample ID or invalid history")
        seen.add(row["sample_id"])
    return rows


def disjoint_users(*partitions):
    seen = set()
    for rows in partitions:
        users = {row["user_id"] for row in rows}
        if seen & users:
            raise ValueError("Controller/model-selection partitions overlap by user")
        seen |= users


def rank_metrics(ranking, target):
    rank = ranking.index(target) + 1 if target in ranking else math.inf
    return {
        f"{metric}@{k}": (
            float(rank <= k) if metric == "hr" else 1 / math.log2(rank + 1) if rank <= k else 0.0
        )
        for k in (5, 10, 20)
        for metric in ("hr", "ndcg")
    }


class Catalog:
    def __init__(self, path, tokenizer):
        self.mapping = json.loads(Path(path).read_text())
        if not isinstance(self.mapping, dict) or not self.mapping:
            raise ValueError("Catalog must map item identifiers to SID metadata")
        self.groups = defaultdict(list)
        self.prefix_counts = Counter()
        self.trie = defaultdict(set)
        for item, entry in self.mapping.items():
            sid = entry["sid"]
            if not isinstance(sid, str) or not SID_PATTERN.fullmatch(sid):
                raise ValueError("Catalog requires four-layer SIDs with begin/end markers")
            ids = sid_target_ids(sid, tokenizer)
            if any(i == tokenizer.unk_token_id for i in ids):
                raise ValueError("SID vocabulary is missing from the checkpoint tokenizer")
            if tokenizer.decode(ids[:-1], skip_special_tokens=False) != sid:
                raise ValueError("SID tokens do not round-trip through the tokenizer")
            self.groups[sid].append(str(item))
            self.prefix_counts[ids[1]] += 1
            for i, token in enumerate(ids):
                self.trie[tuple(ids[:i])].add(token)
        self.trie = {prefix: tuple(sorted(tokens)) for prefix, tokens in self.trie.items()}
        self.valid = set(self.groups)
        if len(self.prefix_counts) < 2:
            raise ValueError("Belief observations require at least two first-code groups")

    def target_sid(self, row):
        if row.get("groundtruth") is not None:
            sid = str(row["groundtruth"])
        elif row.get("target_item") is not None:
            sid = self.mapping[str(row["target_item"])]["sid"]
        else:
            raise ValueError("Supervised rows require groundtruth or target_item")
        if sid not in self.valid:
            raise ValueError("Target SID is outside the catalog")
        if row.get("target_item") is not None:
            if self.mapping[str(row["target_item"])]["sid"] != sid:
                raise ValueError("Item and SID targets disagree")
        return sid

    def expand(self, candidates, counts):
        scores = {}
        for sid, score in candidates:
            if sid not in self.groups or not math.isfinite(float(score)):
                raise ValueError("Invalid catalog candidate")
            group = self.groups[sid]
            total = sum(counts.get(item, 0) + 1 for item in group)
            for item in group:
                value = float(score) + math.log((counts.get(item, 0) + 1) / total)
                scores[item] = max(value, scores.get(item, -math.inf))
        return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))

    def score(self, row, candidates, counts=None):
        metrics = {"sid": rank_metrics([sid for sid, _ in candidates], self.target_sid(row))}
        if row.get("target_item") is not None:
            if counts is None:
                raise ValueError("Exact-item evaluation requires training-history counts")
            items = self.expand(candidates, counts)
            metrics["item"] = rank_metrics([item for item, _ in items], str(row["target_item"]))
        return metrics


def history_counts(rows, catalog):
    """Count explicitly supplied visible training items, never held-out targets."""
    counts = Counter()
    for row in rows:
        items = row.get("history_items")
        if not isinstance(items, list):
            raise ValueError("Prior estimation requires history_items on every training row")
        for item in items:
            item = str(item)
            if item not in catalog.mapping:
                raise ValueError("History item is outside the catalog")
            counts[item] += 1
    return dict(counts)


def load_counts(path, catalog):
    counts = json.loads(Path(path).read_text())
    if not isinstance(counts, dict) or any(
        item not in catalog.mapping or type(n) is not int or n < 0 for item, n in counts.items()
    ):
        raise ValueError("Counts must map catalog items to nonnegative training-history counts")
    return counts
