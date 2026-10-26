"""Real PEFT and mutable Transformer-cache checks without pretrained weights."""

import copy

import pytest
import torch

peft = pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

from dynr2.model import DynR2  # noqa: E402


def tiny_model():
    torch.manual_seed(17)
    base = transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=128,
            bos_token_id=1,
            eos_token_id=9,
            attention_dropout=0.0,
            tie_word_embeddings=False,
        )
    )
    backbone = peft.get_peft_model(
        base,
        peft.LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            bias="none",
            task_type=peft.TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj"],
        ),
    )
    return DynR2(
        backbone,
        [20, 24],
        [4, 5],
        [7],
        [7, 8, 9, *range(20, 28)],
        block_size=2,
        max_blocks=4,
        latent_topk=4,
    )


def decode(model, context):
    return context.logits.detach().clone()


def test_real_qwen_peft_training_cache_and_direct_identity():
    model = tiny_model().eval()
    before, _ = model.recommend_policy([1, 2, 3], decode, "direct")
    state = model.start([1, 2, 3])
    cache = copy.deepcopy(state.output.past_key_values)
    model.observe(state)
    assert state.output.past_key_values.get_seq_length() == cache.get_seq_length()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    original = {k: p.detach().clone() for k, p in model.named_parameters() if not p.requires_grad}
    for layout in ("E", "LL", "ELE"):
        model.train()
        loss, _, _ = model(
            [1, 2, 3], [7, 20, 21, 22, 23, 8, 9], layout, ((20, 24), (21, 25), (22, 26), (23, 27))
        )
        loss.backward()
        assert torch.isfinite(loss)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        after, _ = model.recommend_policy([1, 2, 3], decode, "rule", tau_u=1.0)
        assert torch.equal(before, after)
    assert all(torch.equal(p, dict(model.named_parameters())[k]) for k, p in original.items())
    state = peft.get_peft_model_state_dict(model.backbone)
    rebuilt = tiny_model().eval()
    peft.set_peft_model_state_dict(rebuilt.backbone, state)
    a, _ = model.recommend_policy([1, 2, 3], decode, "L")
    b, _ = rebuilt.recommend_policy([1, 2, 3], decode, "L")
    assert torch.equal(a, b)


def test_real_qwen_top_one_and_forced_path_equivalence():
    model = tiny_model().eval()
    model.latent_topk = 1
    e, _ = model.recommend_policy([1, 2, 3], decode, "E")
    latent, _ = model.recommend_policy([1, 2, 3], decode, "L")
    assert torch.equal(e, latent)
    for action in ("E", "L"):
        old, _ = model.recommend([1, 2, 3], decode, action * 4)
        new, _ = model.recommend_policy([1, 2, 3], decode, action)
        assert torch.equal(old, new)


def test_offline_model_loader_adapter_export_and_evaluation_cli(tmp_path, monkeypatch):
    import json
    import sys
    from pathlib import Path

    from tokenizers import Tokenizer, decoders, models
    from transformers import PreTrainedTokenizerFast

    from dynr2.evaluate import main
    from dynr2.runtime import build_model, load_config, save_response_adapter

    initial = tmp_path / "initial"
    vocab = {f"token_{i}": i for i in range(32)}
    for token, i in {
        "<think>": 4,
        "</think>": 5,
        "<|sid_begin|>": 7,
        "<|sid_end|>": 8,
        "<eos>": 9,
        "<unk>": 31,
    }.items():
        del vocab[f"token_{i}"]
        vocab[token] = i
    for code in range(2):
        for i, layer in enumerate("abcd"):
            index = 20 + 4 * code + i
            del vocab[f"token_{index}"]
            vocab[f"<s_{layer}_{code}>"] = index
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.decoder = decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, eos_token="<eos>", unk_token="<unk>"
    )
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                token
                for token in vocab
                if token.startswith("<") and token not in ("<eos>", "<unk>")
            ]
        }
    )
    base = tiny_model().backbone.get_base_model()
    # Export a fresh ordinary base, not the PEFT-injected module.
    base.config.vocab_size = 64
    transformers.Qwen3ForCausalLM(base.config).save_pretrained(initial)
    tokenizer.save_pretrained(initial)
    a = "<|sid_begin|><s_a_0><s_b_0><s_c_0><s_d_0><|sid_end|>"
    b = "<|sid_begin|><s_a_1><s_b_0><s_c_0><s_d_0><|sid_end|>"
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"a": {"sid": a}, "b": {"sid": b}}))
    config_path = Path(__file__).resolve().parents[1] / "configs" / "beauty.json"
    config = load_config(config_path)
    model, _, _ = build_model(
        initial, catalog, config, device="cpu", dtype="float32", training=True
    )
    assert set(range(32, 64)) <= set(model.forbidden_ids)
    adapter = tmp_path / "adapter"
    save_response_adapter(model, adapter)
    assert (
        json.loads((adapter / "adapter_config.json").read_text())["base_model_name_or_path"] == ""
    )
    data = tmp_path / "test.jsonl"
    data.write_text(
        json.dumps({"sample_id": "1", "user_id": "1", "description": "history", "target_item": "a"})
        + "\n"
    )
    counts = tmp_path / "counts.json"
    counts.write_text(json.dumps({"a": 1, "b": 0}))
    for policy in ("direct", "rule"):
        output = tmp_path / policy
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "dynr2.evaluate",
                "--model",
                str(initial),
                "--adapter",
                str(adapter),
                "--catalog",
                str(catalog),
                "--config",
                str(config_path),
                "--data",
                str(data),
                "--history-counts",
                str(counts),
                "--policy",
                policy,
                "--device",
                "cpu",
                "--dtype",
                "float32",
                "--warmup",
                "0",
                "--output",
                str(output),
            ],
        )
        main()
        metrics = json.loads((output / "metrics.json").read_text())
        assert metrics["n"] == 1 and metrics["invalid"] == 0
        assert metrics["item_n"] == 1 and metrics["sid_n"] == 1
        assert metrics["mean_ms"] > 0


def test_complete_initial_model_training_pipeline(tmp_path, monkeypatch):
    import json
    import sys

    import pandas as pd
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    from dynr2.initialize import main, selected_checkpoint
    from dynr2.runtime import build_model

    storage = tmp_path / "artifacts"
    base_path = storage / "model/Qwen3-1-7B"
    vocab = {f"token_{i}": i for i in range(32)}
    for token, i in {
        "<|im_start|>": 2,
        "<|im_end|>": 3,
        "<think>": 4,
        "</think>": 5,
        "user": 6,
        "<eos>": 9,
        "<unk>": 31,
    }.items():
        del vocab[f"token_{i}"]
        vocab[token] = i
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.decoder = decoders.Fuse()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>", unk_token="<unk>")
    tok.add_special_tokens(
        {"additional_special_tokens": ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]}
    )
    transformers.Qwen3ForCausalLM(tiny_model().backbone.get_base_model().config).save_pretrained(
        base_path
    )
    tok.save_pretrained(base_path)
    source = tmp_path / "data/processed/Beauty"
    source.mkdir(parents=True)

    def sid(code):
        return "<|sid_begin|>" + "".join(f"<s_{layer}_{code}>" for layer in "abcd") + "<|sid_end|>"

    rows = [
        {
            "description": "history " + sid(0),
            "groundtruth": sid(1),
            "sid_routing_think": "<think>\ns_a: Explore\n</think>",
        }
    ] * 2
    for prefix in ("training_align_data", "training_prediction_sid_data", "training_sid_routing"):
        for split in ("train", "val"):
            pd.DataFrame(rows).to_parquet(source / f"{prefix}_{split}.parquet", index=False)
    for stage in ("expand", "align", "merge", "rec", "sid_routing"):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "initialize",
                "--stage",
                stage,
                "--category",
                "Beauty",
                "--root",
                str(tmp_path),
                "--artifact-root",
                str(storage),
                "--device",
                "cpu",
                "--codebook-size",
                "2",
                "--epochs",
                "1",
                "--max-steps",
                "1",
            ],
        )
        main()
        if stage in ("align", "rec", "sid_routing"):
            assert selected_checkpoint(storage / "outputs/beauty_initial" / stage).is_dir()
    initial = storage / "models/Beauty/initial"
    assert initial.is_symlink()
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"a": {"sid": sid(0)}, "b": {"sid": sid(1)}}))
    config = {"block_size": 4, "max_blocks": 4, "latent_topk": 8, "adapter_strength": 1.0}
    model, tokenizer, _ = build_model(initial, catalog, config, device="cpu", dtype="float32")
    assert model.backbone.config.vocab_size >= len(tokenizer)
