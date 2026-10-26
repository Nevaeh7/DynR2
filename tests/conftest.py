from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from dynr2.model import DynR2

torch.set_num_threads(1)


class TinyCache:
    def __init__(self, n=0, last=None):
        self.n, self.last = n, last

    def get_seq_length(self):
        return self.n

    def batch_select_indices(self, indices):
        self.last = self.last.index_select(0, indices)


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 8)
        self.head = torch.nn.Linear(8, 32)
        self.requires_grad_(False)
        self.adapter = torch.nn.Parameter(torch.zeros(8))
        self.adapter_enabled = True
        self.calls = []

    @contextmanager
    def disable_adapter(self):
        previous = self.adapter_enabled
        self.adapter_enabled = False
        try:
            yield
        finally:
            self.adapter_enabled = previous

    def get_input_embeddings(self):
        return self.embedding

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        past_key_values=None,
        attention_mask=None,
        use_cache=True,
    ):
        hidden = self.embedding(input_ids) if input_ids is not None else inputs_embeds
        if self.adapter_enabled:
            hidden = hidden + self.adapter
        cache = past_key_values if past_key_values is not None else TinyCache()
        if cache.last is not None:
            hidden = hidden + cache.last
        hidden = hidden.cumsum(1)
        cache.n += hidden.shape[1]
        cache.last = hidden[:, -1:]
        if attention_mask is not None:
            assert attention_mask.shape[-1] == cache.n
        self.calls.append({"adapted": self.adapter_enabled, "soft": inputs_embeds is not None})
        return SimpleNamespace(logits=self.head(hidden), past_key_values=cache)


class TinyTokenizer:
    eos_token_id = 9
    unk_token_id = 31

    def __init__(self):
        self.vocab = {"<|sid_begin|>": 7, "<|sid_end|>": 8, "<eos>": 9}
        for i, layer in enumerate("abcd"):
            for code in range(2):
                self.vocab[f"<s_{layer}_{code}>"] = 20 + 4 * code + i
        self.inverse = {v: k for k, v in self.vocab.items()}

    def get_vocab(self):
        return self.vocab

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def __call__(self, text, add_special_tokens=False):
        import re

        tokens = re.findall(r"<[^>]+>", text)
        return {"input_ids": [self.vocab.get(token, 31) for token in tokens]}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.inverse[int(i)] for i in ids)


@pytest.fixture
def tokenizer():
    return TinyTokenizer()


@pytest.fixture
def model():
    torch.manual_seed(42)
    return DynR2(
        TinyBackbone(),
        [20, 24],
        [4, 5],
        [7],
        [7, 8, 9, *range(20, 28), 31],
        block_size=2,
        max_blocks=4,
        latent_topk=4,
    )
