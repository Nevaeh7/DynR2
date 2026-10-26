"""SID quantization, chronological splits, and transition kernels from the reference."""

from __future__ import annotations

from collections import deque
import hashlib
import html
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

NUM_SID_LAYERS = 4
CODEBOOK_SIZE = 256
RQ_N_ITER = 30
SID_LAYER_PREFIX = ["s_a", "s_b", "s_c", "s_d"]
HARD_SAMPLE_PREFIX_MATCH_LAYERS = 2
HARD_SAMPLE_GRAPH_DIST_THRESH = 2
SID_PATTERN = re.compile(
    r"(<\|sid_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)><s_d_(\d+)><\|sid_end\|>)"
)


def _elapsed(start):
    return time.time() - start


def _round_seconds(value):
    return round(value, 3)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    text = html.unescape(value)
    text = re.sub("<[^>]+>", " ", text)
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub("\\s+", " ", text).strip()
    return text


def _normalize_scalar_text(value: Any) -> str:
    return _clean_text(value)


def _normalize_description(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        parts = [_clean_text(x) for x in value]
        return " ".join([p for p in parts if p]).strip()
    return _clean_text(value)


def _normalize_categories(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        if len(value) == 0:
            return ""
        if isinstance(value[0], list):
            for path in value:
                cleaned = [_clean_text(x) for x in path if _clean_text(x)]
                if cleaned:
                    return " > ".join(cleaned).strip()
            return ""
        cleaned = [_clean_text(x) for x in value if _clean_text(x)]
        return " > ".join(cleaned).strip()
    return _clean_text(value)


def _run_rq_kmeans(embeddings: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[dict]]:
    import faiss

    n_items, dim = embeddings.shape
    codes = np.zeros((n_items, NUM_SID_LAYERS), dtype=np.int32)
    codebooks = []
    residuals = embeddings.astype(np.float32).copy()
    layer_stats = []
    for layer in range(NUM_SID_LAYERS):
        t_layer = time.time()
        kmeans = faiss.Kmeans(
            d=dim,
            k=CODEBOOK_SIZE,
            niter=RQ_N_ITER,
            verbose=False,
            gpu=False,
            spherical=False,
            seed=42,
        )
        kmeans.train(residuals)
        _, assignments = kmeans.index.search(residuals, 1)
        code = assignments.flatten()
        codes[:, layer] = code
        centroids = kmeans.centroids.copy()
        codebooks.append(centroids)
        residuals = residuals - centroids[code]
        unique_count = len(np.unique(code))
        layer_seconds = _round_seconds(_elapsed(t_layer))
        layer_stats.append(
            {
                "layer": int(layer + 1),
                "seconds": layer_seconds,
                "utilized_codes": int(unique_count),
                "utilization_rate": round(unique_count / CODEBOOK_SIZE, 6),
            }
        )
    return (codes, codebooks, layer_stats)


def parse_sid(sid_str: str) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(sid_str, str):
        return None
    m = SID_PATTERN.search(sid_str)
    if m:
        return (int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)))
    tokens = re.findall("<s_([a-d])_(\\d+)>", sid_str)
    if len(tokens) >= 4:
        layer_map = {"a": 0, "b": 1, "c": 2, "d": 3}
        codes = [0, 0, 0, 0]
        for ch, val in tokens[:4]:
            codes[layer_map[ch]] = int(val)
        return tuple(codes)
    return None


def extract_sid_strings(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return [m.group(1) for m in SID_PATTERN.finditer(text)]


def _make_sample_id(user_id: object, groundtruth: str, description: str) -> str:
    raw = f"{user_id}\t{groundtruth}\t{description}".encode("utf-8", errors="ignore")
    return hashlib.md5(raw).hexdigest()


def _split_align(items: dict, sequences: List[Tuple[str, List[str]]]) -> Dict[str, pd.DataFrame]:

    def _build_desc(item_ids: List[str]) -> List[str]:
        descs = []
        for iid in item_ids:
            info = items.get(iid)
            if info and info.get("sid") and info.get("title") and info.get("categories"):
                descs.append(
                    f'''{info["sid"]}, its title is "{info["title"]}", its categories are "{info["categories"]}"'''
                )
        return descs

    splits = {"train": [], "val": [], "test": []}
    for uid, iids in sequences:
        full = _build_desc(iids)
        if full:
            splits["test"].append(
                {
                    "user_id": uid,
                    "description": "The user has purchased the following items: "
                    + "; ".join(full)
                    + ";",
                }
            )
        if len(iids) > 1:
            val = _build_desc(iids[:-1])
            if val:
                splits["val"].append(
                    {
                        "user_id": uid,
                        "description": "The user has purchased the following items: "
                        + "; ".join(val)
                        + ";",
                    }
                )
        if len(iids) > 2:
            train = _build_desc(iids[:-2])
            if train:
                splits["train"].append(
                    {
                        "user_id": uid,
                        "description": "The user has purchased the following items: "
                        + "; ".join(train)
                        + ";",
                    }
                )
    return {k: pd.DataFrame(v) for k, v in splits.items()}


def _split_sid_prediction(
    items: dict, sequences: List[Tuple[str, List[str]]]
) -> Dict[str, pd.DataFrame]:

    def _extract_sids(item_ids: List[str]) -> List[str]:
        return [items[i]["sid"] for i in item_ids if items.get(i, {}).get("sid")]

    def _entry(sids: List[str], uid: str, tail_remove: int):
        cand = sids[: len(sids) - tail_remove] if tail_remove > 0 else sids
        if len(cand) < 2:
            return None
        return {
            "user_id": uid,
            "description": "The user has purchased the following items: "
            + "; ".join(cand[:-1])
            + ";",
            "groundtruth": cand[-1],
        }

    splits = {"train": [], "val": [], "test": []}
    for uid, iids in sequences:
        sids = _extract_sids(iids)
        if not sids:
            continue
        for name, tail in [("train", 2), ("val", 1), ("test", 0)]:
            e = _entry(sids, uid, tail)
            if e:
                splits[name].append(e)
    return {k: pd.DataFrame(v) for k, v in splits.items()}


def _split_ra(items: dict, sequences: List[Tuple[str, List[str]]]) -> Dict[str, pd.DataFrame]:

    def _extract_seq(item_ids: List[str]) -> List[dict]:
        return [
            {
                "sid": items[i]["sid"],
                "title": items[i].get("title", ""),
                "categories": items[i].get("categories", ""),
            }
            for i in item_ids
            if items.get(i, {}).get("sid")
        ]

    def _entry(seq: List[dict], uid: str, tail_remove: int):
        cand = seq[: len(seq) - tail_remove] if tail_remove > 0 else seq
        if len(cand) < 2:
            return None
        gt = cand[-1]
        history = cand[:-1]
        descs = [
            f'''{h["sid"]}, its title is "{h["title"]}", its categories are "{h["categories"]}"'''
            for h in history
        ]
        return {
            "user_id": uid,
            "description": "The user has purchased the following items: " + "; ".join(descs) + ";",
            "groundtruth": gt["sid"],
            "title": gt["title"],
            "categories": gt["categories"],
        }

    splits = {"train": [], "val": [], "test": []}
    for uid, iids in sequences:
        seq = _extract_seq(iids)
        if not seq:
            continue
        for name, tail in [("train", 2), ("val", 1), ("test", 0)]:
            e = _entry(seq, uid, tail)
            if e:
                splits[name].append(e)
    return {k: pd.DataFrame(v) for k, v in splits.items()}


def _bfs_dist(src: int, dst: int, adj: Dict[int, set[int]], max_d: int = 4) -> int:
    if src == dst:
        return 0
    q: deque[Tuple[int, int]] = deque([(src, 0)])
    seen = {src}
    while q:
        node, depth = q.popleft()
        if depth >= max_d:
            continue
        for nxt in adj.get(node, set()):
            if nxt == dst:
                return depth + 1
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, depth + 1))
    return max_d + 1


def _classify_sample(history_sids: List[str], target_sid: str, sa_adj: Dict[int, set[int]]) -> dict:
    tc = parse_sid(target_sid)
    if not tc:
        return {
            "difficulty": "error",
            "anchor_sid": None,
            "anchor_match_depth": 0,
            "s_a_min_distance": 999,
        }
    best_depth, best_anchor, min_dist = (0, None, 999)
    for h in history_sids:
        hc = parse_sid(h)
        if not hc:
            continue
        depth = 0
        for i in range(NUM_SID_LAYERS):
            if hc[i] == tc[i]:
                depth += 1
            else:
                break
        if depth > best_depth:
            best_depth, best_anchor = (depth, h)
        min_dist = min(min_dist, _bfs_dist(hc[0], tc[0], sa_adj))
    if best_anchor is None and history_sids:
        best_anchor = history_sids[-1]
    if best_depth >= HARD_SAMPLE_PREFIX_MATCH_LAYERS:
        difficulty = "easy"
    elif best_depth >= 1 or min_dist <= HARD_SAMPLE_GRAPH_DIST_THRESH:
        difficulty = "medium"
    else:
        difficulty = "hard"
    return {
        "difficulty": difficulty,
        "anchor_sid": best_anchor,
        "anchor_match_depth": int(best_depth),
        "s_a_min_distance": int(min_dist),
    }


def _construct_nav_path(
    anchor_sid: Optional[str], target_sid: str, layer_adjs: Dict[int, Dict[int, List[dict]]]
) -> Optional[dict]:
    if not anchor_sid:
        return None
    ac = parse_sid(anchor_sid)
    tc = parse_sid(target_sid)
    if not ac or not tc:
        return None
    steps: List[dict] = []
    total_cost = 0
    for idx in range(NUM_SID_LAYERS):
        prefix = SID_LAYER_PREFIX[idx]
        src, dst = (ac[idx], tc[idx])
        if src == dst:
            steps.append({"layer": prefix, "action": "match", "from": src, "to": dst, "cost": 0})
            continue
        edge_info = next((n for n in layer_adjs.get(idx, {}).get(src, []) if n["dst"] == dst), None)
        if edge_info:
            steps.append(
                {
                    "layer": prefix,
                    "action": "lateral_jump",
                    "from": src,
                    "to": dst,
                    "weight": float(edge_info.get("weight", 0.0)),
                    "cost": 1,
                }
            )
            total_cost += 1
        else:
            steps.append({"layer": prefix, "action": "explore", "from": src, "to": dst, "cost": 2})
            total_cost += 2
    return {"path_steps": steps, "total_cost": int(total_cost)}


def _format_step_text(step: dict) -> str:
    prefix = step["layer"]
    action = step["action"]
    src, dst = (step["from"], step["to"])
    if action == "match":
        return f"{prefix}: Match Code {dst}, direct match with history."
    if action == "lateral_jump":
        w = step.get("weight", 0.0)
        return f"{prefix}: Lateral jump from cluster {src} to {dst} via edge (similarity={w:.4f})."
    return f"{prefix}: Explore from cluster {src} to {dst} (no direct edge, inferred from user interest pattern)."


def _build_cot_text(navigation: Optional[dict]) -> Tuple[str, int]:
    if not navigation or "path_steps" not in navigation:
        return ("<think>\n\n</think>", 0)
    non_trivial = [s for s in navigation["path_steps"] if s["action"] != "match"]
    if not non_trivial:
        return ("<think>\n\n</think>", 0)
    return (
        "<think>\n" + "\n".join((_format_step_text(s) for s in non_trivial)) + "\n</think>",
        len(non_trivial),
    )
