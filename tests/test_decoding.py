import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dynr2.data import Catalog
from dynr2.decoding import cached_beam, constrained_log_probs


def logits_for(prefix):
    grid = torch.arange(32, dtype=torch.float32)
    return torch.sin(grid * 0.7 + sum((i + 1) * token for i, token in enumerate(prefix)) * 0.13)


class PrefixCache:
    def __init__(self, prefixes):
        self.prefixes = prefixes

    def get_seq_length(self):
        return len(self.prefixes[0])

    def batch_select_indices(self, indices):
        self.prefixes = [list(self.prefixes[i]) for i in indices.tolist()]


class LogitTree:
    def __call__(self, input_ids, past_key_values, **kwargs):
        prefixes = [
            prefix + [int(token)]
            for prefix, token in zip(past_key_values.prefixes, input_ids[:, 0])
        ]
        return SimpleNamespace(
            logits=torch.stack([logits_for(p) for p in prefixes])[:, None],
            past_key_values=PrefixCache(prefixes),
        )


@pytest.mark.parametrize("domain", ["unit", "beauty", "sports", "toys"])
@pytest.mark.parametrize("beams", [1, 4, 20])
def test_weighted_scores_and_unused_eighth_slot(tmp_path, tokenizer, domain, beams):
    mapping = {}
    for index, codes in enumerate(itertools.product(range(2), repeat=4)):
        sid = (
            "<|sid_begin|>"
            + "".join(f"<s_{layer}_{code}>" for layer, code in zip("abcd", codes))
            + "<|sid_end|>"
        )
        mapping[str(index)] = {"sid": sid}
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(mapping))
    catalog = Catalog(path, tokenizer)
    root = Path(__file__).resolve().parents[1]
    config = (
        {"step_weights": [1.0] * 8, "step_temperatures": [1.0] * 8}
        if domain == "unit"
        else json.loads((root / "configs" / f"{domain}.json").read_text())
    )
    weights, temperatures = config["step_weights"], config["step_temperatures"]

    def run(w, t):
        context = SimpleNamespace(
            logits=logits_for([])[None, None], past_key_values=PrefixCache([[]])
        )
        return cached_beam(
            SimpleNamespace(backbone=LogitTree()),
            context,
            catalog.trie,
            tokenizer,
            catalog.valid,
            beams=beams,
            step_weights=w,
            step_temperatures=t,
        )

    candidates = run(weights, temperatures)
    changed_w, changed_t = list(weights), list(temperatures)
    changed_w[7], changed_t[7] = 100.0, 0.5
    assert candidates == run(changed_w, changed_t)
    assert len(candidates) == min(beams, 16)
    for sid, actual in candidates:
        ids = tokenizer(sid)["input_ids"] + [tokenizer.eos_token_id]
        terms = [
            weights[i] * float(torch.log_softmax(logits_for(ids[:i]) / temperatures[i], -1)[token])
            for i, token in enumerate(ids)
        ]
        assert actual == pytest.approx(sum(terms), abs=2e-6)
        # The raw first-code belief contributes with coefficient w2/T2.
        raw = logits_for([7])
        support = [20, 24]
        q = raw[support].softmax(-1)
        rho = raw.div(temperatures[1]).softmax(-1)[support].sum()
        constant = (
            terms[0]
            + weights[1] * float(rho.log())
            - weights[1] * float(q.pow(1 / temperatures[1]).sum().log())
        )
        factorized = (
            constant
            + weights[1] / temperatures[1] * float(q[support.index(ids[1])].log())
            + sum(terms[2:])
        )
        assert actual == pytest.approx(factorized, abs=3e-6)


def test_normalization_precedes_catalog_mask():
    logits = torch.tensor([0.0, 1.0, 20.0])
    result = constrained_log_probs(logits, temperature=1.2)
    assert torch.equal(result, torch.log_softmax(logits / 1.2, -1))
    assert result[0] < -10  # Illegal-token mass is not removed from the denominator.
