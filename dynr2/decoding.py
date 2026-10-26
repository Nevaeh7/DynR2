"""Catalog-constrained beam search with full-vocabulary score normalization."""

import math
import torch
import torch.nn.functional as F


def select_cache(cache, indices):
    if hasattr(cache, "layers") and any(
        not hasattr(layer, "batch_select_indices") for layer in cache.layers
    ):
        # HF's hybrid cache exposes batch_select_indices, but its DeltaNet
        # layers only implement reorder_cache. Reorder ALL layers together.
        cache.reorder_cache(indices)
        for layer in cache.layers:
            if hasattr(layer, "batch_size") and hasattr(layer, "conv_states"):
                layer.batch_size = len(indices)
        return cache
    if hasattr(cache, "batch_select_indices"):
        cache.batch_select_indices(indices)
        return cache
    if isinstance(cache, tuple):
        return tuple(tuple(t.index_select(0, indices) for t in layer) for layer in cache)
    raise TypeError(f"Unsupported cache: {type(cache)}")


def cache_length(cache):
    return cache.get_seq_length() if hasattr(cache, "get_seq_length") else cache[0][0].shape[-2]


def constrained_log_probs(logits, *, temperature=1.0):
    """Normalize over the full vocabulary before masking catalog continuations."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Positive finite decoding temperature required")
    values = logits.float() if temperature == 1.0 else logits.float() / temperature
    return F.log_softmax(values, dim=-1)


@torch.no_grad()
def cached_beam(
    model,
    context,
    trie,
    tokenizer,
    valid,
    beams=20,
    max_tokens=8,
    *,
    step_temperatures=None,
    step_weights=None,
):
    if beams < 1:
        raise ValueError("Invalid beam count")
    for values, label, allow_zero in (
        (step_temperatures, "temperature", False),
        (step_weights, "score weight", True),
    ):
        if values is not None and (
            len(values) < max_tokens
            or any(not math.isfinite(v) or (v < 0 if allow_zero else v <= 0) for v in values)
        ):
            raise ValueError("Invalid per-position decoding " + label)
    cache, logits = context.past_key_values, context.logits[:, -1, :]
    hypotheses = [([], 0.0)]
    finished = []
    for position in range(max_tokens):
        proposals = []
        for index, (prefix, score) in enumerate(hypotheses):
            allowed = trie.get(tuple(prefix), ())
            if not allowed:
                continue
            log_probs = constrained_log_probs(
                logits[index],
                temperature=step_temperatures[position] if step_temperatures is not None else 1.0,
            )
            weight = step_weights[position] if step_weights is not None else 1.0
            for token in allowed:
                proposals.append(
                    (score + weight * float(log_probs[token]), index, prefix + [token])
                )
        if not proposals:
            break
        proposals.sort(key=lambda item: -item[0])
        chosen = proposals[:beams]
        finished.extend(
            (prefix[:-1], score)
            for score, _, prefix in chosen
            if prefix[-1] == tokenizer.eos_token_id
        )
        active = [
            (score, parent, prefix)
            for score, parent, prefix in chosen
            if prefix[-1] != tokenizer.eos_token_id
        ]
        if not active:
            break
        parents = torch.tensor(
            [parent for _, parent, _ in active], dtype=torch.long, device=logits.device
        )
        cache = select_cache(cache, parents)
        tokens = torch.tensor(
            [[prefix[-1]] for _, _, prefix in active], dtype=torch.long, device=logits.device
        )
        length = cache_length(cache) + 1
        out = model.backbone(
            input_ids=tokens,
            past_key_values=cache,
            use_cache=True,
            attention_mask=torch.ones(
                (len(active), length), dtype=torch.long, device=logits.device
            ),
        )
        cache, logits = out.past_key_values, out.logits[:, -1, :]
        hypotheses = [(prefix, score) for score, _, prefix in active]
    found = {}
    for prefix, score in finished:
        sid = tokenizer.decode(prefix, skip_special_tokens=False)
        if sid in valid:
            found[sid] = max(score, found.get(sid, -float("inf")))
    if not found:
        raise ValueError("No valid SID survived cached beam search")
    return sorted(found.items(), key=lambda item: -item[1])[:beams]
