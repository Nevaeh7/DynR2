"""Model loading and paper inference profiles; all artifact paths are explicit."""

import json
import hashlib
import math
from pathlib import Path

import torch

from .core import first_code_token_ids
from .data import Catalog
from .decoding import cached_beam
from .model import DynR2
from .paths import external_checkpoint_path

LORA_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def counts_identity(counts):
    return hashlib.sha256(json.dumps(counts, sort_keys=True).encode()).hexdigest()


def save_response_adapter(model, directory):
    directory = external_checkpoint_path(directory)
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file

    directory.mkdir(parents=True, exist_ok=False)
    adapter = model.backbone.peft_config["default"].to_dict()
    adapter["base_model_name_or_path"] = ""
    adapter["inference_mode"] = True
    (directory / "adapter_config.json").write_text(
        json.dumps(adapter, default=list, indent=2) + "\n"
    )
    weights = {
        key: value.detach().cpu().contiguous()
        for key, value in get_peft_model_state_dict(
            model.backbone, save_embedding_layers=False
        ).items()
    }
    save_file(weights, str(directory / "adapter_model.safetensors"), metadata={"format": "pt"})


def artifact_identity(model_path, adapter_path, catalog):
    """Bind fitted controllers to exact model/tokenizer weights without recording paths."""
    result = {}
    for role, directory in (("base", model_path), ("response", adapter_path)):
        for path in sorted(Path(directory).iterdir()):
            if path.is_file() and (
                path.suffix in (".json", ".safetensors", ".bin", ".model")
                or path.name in ("merges.txt", "vocab.txt")
            ):
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                result[role + "/" + path.name] = digest.hexdigest()
    mapping = {item: entry["sid"] for item, entry in catalog.mapping.items()}
    result["catalog"] = hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()
    return result


def load_config(path):
    config = json.loads(Path(path).read_text())
    for name in ("block_size", "max_blocks", "latent_topk", "beams"):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError("Positive integer required: " + name)
    if config["max_blocks"] > 4:
        raise ValueError("The released Value feature encoding supports at most four blocks")
    for name, high in (("tau_u", 1.0), ("tau_b", math.log(2)), ("adapter_strength", 1.0)):
        value = config[name]
        if not math.isfinite(value) or not 0 <= value <= high:
            raise ValueError("Invalid " + name)
    for name in ("step_weights", "step_temperatures"):
        values = config[name]
        if len(values) != 8 or any(
            not math.isfinite(v) or v < 0 or (name == "step_temperatures" and v == 0)
            for v in values
        ):
            raise ValueError("Invalid decoder configuration: " + name)
    return config


def build_model(
    model_path,
    catalog_path,
    config,
    adapter_path=None,
    device="cuda",
    dtype="bfloat16",
    training=False,
):
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not Path(model_path).is_dir():
        raise ValueError("Supply a local SID-trained model directory")
    if training and adapter_path is not None:
        raise ValueError("Start response adaptation from a fresh LoRA adapter")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    catalog = Catalog(catalog_path, tokenizer)
    base = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=getattr(torch, dtype), local_files_only=True, low_cpu_mem_usage=True
    )
    if base.config.vocab_size < len(tokenizer):
        raise ValueError("Model vocabulary does not cover the SID tokenizer")
    if adapter_path is None:
        backbone = get_peft_model(
            base,
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.0,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=LORA_MODULES,
            ),
        )
    else:
        backbone = PeftModel.from_pretrained(
            base, adapter_path, is_trainable=False, local_files_only=True
        )
    for adapter in backbone.peft_config.values():
        if (
            getattr(adapter.peft_type, "value", adapter.peft_type) != "LORA"
            or adapter.bias != "none"
            or adapter.modules_to_save
            or any(
                getattr(adapter, field, False)
                for field in ("use_dora", "use_rslora", "lora_bias", "alora_invocation_tokens")
            )
        ):
            raise ValueError("Only unmerged vanilla LoRA with a protected base is supported")
    if not training:
        from peft.tuners.lora.layer import LoraLayer

        for module in backbone.modules():
            if isinstance(module, LoraLayer):
                if module.merged:
                    raise ValueError("Response adapters must remain unmerged")
                for name in module.scaling:
                    module.scaling[name] *= config["adapter_strength"]
    vocab = tokenizer.get_vocab()
    forbidden = set(tokenizer.all_special_ids)
    forbidden.update(range(len(tokenizer), base.config.vocab_size))
    # SID tokens must never become intermediate reasoning inputs.
    forbidden.update(
        token_id
        for token, token_id in vocab.items()
        if token.startswith("<s_") or token.startswith("<|sid_")
    )
    forbidden.update(
        token_id for token, token_id in vocab.items() if token in ("<think>", "</think>")
    )
    model = DynR2(
        backbone,
        first_code_token_ids(Path(catalog_path), tokenizer),
        tokenizer("\n</think>\n", add_special_tokens=False)["input_ids"],
        tokenizer("<|sid_begin|>", add_special_tokens=False)["input_ids"],
        forbidden,
        block_size=config["block_size"],
        max_blocks=config["max_blocks"],
        latent_topk=config["latent_topk"],
    ).to(device)
    if len(forbidden) + model.latent_topk > base.config.vocab_size:
        raise ValueError("Insufficient legal vocabulary for latent inputs")
    model.train(training)
    return model, tokenizer, catalog


def make_decoder(catalog, tokenizer, config):
    def decode(model, context):
        return cached_beam(
            model,
            context,
            catalog.trie,
            tokenizer,
            catalog.valid,
            beams=config["beams"],
            step_weights=config["step_weights"],
            step_temperatures=config["step_temperatures"],
        )

    return decode
