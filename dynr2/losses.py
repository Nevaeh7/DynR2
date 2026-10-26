"""Seven-token SID supervision and fixed-reference code KL."""

import re

import torch
import torch.nn.functional as F


def sid_supports(tokenizer):
    layers = []
    vocab = tokenizer.get_vocab()
    for layer in "abcd":
        ids = tuple(
            sorted(
                value
                for token, value in vocab.items()
                if re.fullmatch(r"<s_" + layer + r"_\d+>", token)
            )
        )
        if len(ids) < 2 or len(ids) != len(set(ids)):
            raise ValueError("Tokenizer lacks distinct legal SID codes")
        layers.append(ids)
    return tuple(layers)


def sid_target_ids(target, tokenizer):
    ids = tokenizer(str(target), add_special_tokens=False)["input_ids"]
    begin = tokenizer("<|sid_begin|>", add_special_tokens=False)["input_ids"]
    end = tokenizer("<|sid_end|>", add_special_tokens=False)["input_ids"]
    if (
        len(begin) != 1
        or len(end) != 1
        or len(ids) != 6
        or ids[0] != begin[0]
        or ids[-1] != end[0]
        or tokenizer.eos_token_id is None
    ):
        raise ValueError("Expected four SID codes with begin/end and an EOS target")
    return ids + [tokenizer.eos_token_id]


def legal_sid_kl(student, teacher, supports):
    if (
        student.shape != teacher.shape
        or student.ndim != 3
        or student.shape[1] != 7
        or len(supports) != 4
    ):
        raise ValueError("Anchor shape/support does not match four-layer SID targets")
    losses = []
    for position, ids in enumerate(supports, start=1):
        if (
            len(ids) < 2
            or len(set(ids)) != len(ids)
            or min(ids) < 0
            or max(ids) >= student.shape[-1]
        ):
            raise ValueError("Invalid legal SID support")
        a, b = (
            student[:, position, list(ids)].float(),
            teacher.detach()[:, position, list(ids)].float(),
        )
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("Non-finite SID anchor logits")
        losses.append(F.kl_div(a.log_softmax(-1), b.softmax(-1), reduction="batchmean"))
    return torch.stack(losses).mean()
