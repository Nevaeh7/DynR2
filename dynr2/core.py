"""Shared Transformer operations for prediction-guided reasoning."""

import copy
import json
import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F


SYSTEM_MESSAGE = (
    "You are a professional recommendation expert who needs to recommend "
    "the next possible purchase for users based on their purchase history. "
    "Please predict the most likely next product that the user will purchase "
    "based on the user's historical purchase information. Express your "
    "prediction as a Semantic ID by navigating the SID hierarchy."
)
FIRST_CODE = re.compile(r"<\|sid_begin\|><s_a_(\d+)>")


def history_prompt(description):
    return (
        f"<|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>\n"
        f"<|im_start|>user\n{description}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    )


def pack_history(description, tokenizer, limit):
    """Truncate whole oldest interactions identically for every mode."""
    text = str(description)

    def encode(value):
        return tokenizer(history_prompt(value), add_special_tokens=False)["input_ids"]

    ids = encode(text)
    if len(ids) <= limit:
        return ids, False
    entries = text.split("; ")
    header = "The user has purchased the following items: "
    lo, hi = 0, len(entries)
    while lo < hi:
        mid = (lo + hi) // 2
        if len(encode(header + "; ".join(entries[mid:]))) <= limit:
            hi = mid
        else:
            lo = mid + 1
    if lo < len(entries):
        text = header + "; ".join(entries[lo:])
    else:
        budget = limit - len(encode(header)) - 16
        if budget <= 0:
            raise ValueError("Prompt budget cannot fit one interaction")
        latest = tokenizer(entries[-1], add_special_tokens=False)["input_ids"][-budget:]
        text = header + tokenizer.decode(latest, skip_special_tokens=False)
    ids = encode(text)
    if len(ids) > limit:
        raise ValueError("History still exceeds prompt budget")
    return ids, True


def first_code_token_ids(catalog_path, tokenizer):
    mapping = json.loads(catalog_path.read_text())
    codes = {
        int(match.group(1))
        for item in mapping.values()
        if (match := FIRST_CODE.match(str(item.get("sid", ""))))
    }
    if len(codes) < 2:
        raise ValueError("Static catalog has fewer than two first-layer codes")
    ids = [tokenizer.convert_tokens_to_ids(f"<s_a_{code}>") for code in sorted(codes)]
    if (
        any(not isinstance(token, int) or token < 0 for token in ids)
        or tokenizer.unk_token_id in ids
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("Catalog first-layer SID codes do not match tokenizer")
    return ids


def normalized_entropy(probs):
    if probs.shape[-1] < 2:
        raise ValueError("Entropy needs at least two catalog codes")
    return -(probs * probs.clamp_min(1e-12).log()).sum(-1) / math.log(probs.shape[-1])


def js_divergence(left, right):
    midpoint = 0.5 * (left + right)
    return 0.5 * (left * (left.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())).sum(
        -1
    ) + 0.5 * (right * (right.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())).sum(-1)


def decide_mode(current, previous, tau_u, tau_b):
    if not 0 <= tau_u <= 1 or not 0 <= tau_b <= math.log(2):
        raise ValueError("Invalid belief thresholds")
    uncertainty = float(normalized_entropy(current).item())
    transition = 0.0 if previous is None else float(js_divergence(current, previous).item())
    if not math.isfinite(uncertainty) or not math.isfinite(transition):
        raise FloatingPointError("Nonfinite belief statistics")
    uncertainty = min(1.0, max(0.0, uncertainty))
    transition = min(math.log(2), max(0.0, transition))
    if uncertainty > tau_u:
        return "L", uncertainty, transition
    if transition > tau_b:
        return "E", uncertainty, transition
    return "D", uncertainty, transition


class DynR2Base(nn.Module):
    """Runs one live Transformer cache through blockwise E/L/D reasoning."""

    def __init__(
        self,
        backbone,
        first_ids,
        suffix_ids,
        sid_begin_ids,
        forbidden_ids,
        block_size=4,
        max_blocks=4,
        latent_topk=8,
    ):
        super().__init__()
        if (
            len(first_ids) < 2
            or len(set(first_ids)) != len(first_ids)
            or not suffix_ids
            or not sid_begin_ids
            or block_size < 1
            or max_blocks < 1
            or latent_topk < 1
        ):
            raise ValueError("Invalid DynR2 configuration")
        self.backbone = backbone
        self.first_ids = tuple(first_ids)
        self.suffix_ids = tuple(suffix_ids)
        self.answer_probe_ids = tuple(suffix_ids) + tuple(sid_begin_ids)
        self.forbidden_ids = tuple(sorted(set(forbidden_ids)))
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.latent_topk = latent_topk

    @property
    def device(self):
        return next(self.backbone.parameters()).device

    def _forward(self, *, ids=None, embeds=None, cache=None, length=None):
        kwargs = {"use_cache": True}
        if ids is not None:
            kwargs["input_ids"] = torch.as_tensor(
                ids, dtype=torch.long, device=self.device
            ).reshape(1, -1)
        else:
            kwargs["inputs_embeds"] = embeds
        if cache is not None:
            kwargs["past_key_values"] = cache
            kwargs["attention_mask"] = torch.ones((1, length), dtype=torch.long, device=self.device)
        return self.backbone(**kwargs)

    def _reasoning_logits(self, logits):
        choices = logits.float().clone()
        choices[:, list(self.forbidden_ids)] = -float("inf")
        if not torch.isfinite(choices).any(dim=-1).all():
            raise ValueError("No legal reasoning tokens remain")
        return choices

    def _recommendation_belief(self, output, length):
        # The next token under <think> is reasoning text, not a SID code.
        # Query the hypothetical answer position without changing the live KV.
        original = output.past_key_values
        original_length = (
            original.get_seq_length()
            if hasattr(original, "get_seq_length")
            else getattr(original, "n", None)
        )
        shadow = copy.deepcopy(original)
        probed = self._forward(
            ids=self.answer_probe_ids, cache=shadow, length=length + len(self.answer_probe_ids)
        )
        if original_length is not None:
            after = (
                original.get_seq_length()
                if hasattr(original, "get_seq_length")
                else getattr(original, "n", None)
            )
            if after != original_length:
                raise RuntimeError("Answer probe changed the live reasoning cache")
        return F.softmax(probed.logits[:, -1, list(self.first_ids)].float(), dim=-1).squeeze(0)
