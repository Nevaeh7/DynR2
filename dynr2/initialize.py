"""Vocabulary expansion, SID alignment, and supervised initial-recommender training."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from .core import SYSTEM_MESSAGE
from .paths import artifact_root, external_checkpoint_path
from .preprocess import save_json


def sid_tokens(size=256):
    return [
        "<|sid_begin|>",
        "<|sid_end|>",
        *[f"<s_{layer}_{code}>" for layer in "abcd" for code in range(size)],
    ]


def expand_vocabulary(source, output, size=256, dtype="bfloat16"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output = external_checkpoint_path(output)
    if output.exists():
        raise FileExistsError(f"Vocabulary output already exists: {output}")
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        source, local_files_only=True, torch_dtype=getattr(torch, dtype)
    )
    core_tokens = {
        getattr(tokenizer, name, None)
        for name in (
            "bos_token",
            "eos_token",
            "unk_token",
            "pad_token",
            "sep_token",
            "cls_token",
            "mask_token",
        )
    }
    existing = [token for token in tokenizer.all_special_tokens if token not in core_tokens]
    tokenizer.add_special_tokens(
        {"additional_special_tokens": list(dict.fromkeys([*existing, *sid_tokens(size)]))}
    )
    model.resize_token_embeddings(math.ceil(len(tokenizer) / 256) * 256)
    sample = "<|sid_begin|>" + "".join(f"<s_{layer}_0>" for layer in "abcd") + "<|sid_end|>"
    if len(tokenizer.encode(sample, add_special_tokens=False)) != 6:
        raise ValueError("The expanded tokenizer does not encode SID tokens atomically")
    output.mkdir(parents=True)
    tokenizer.save_pretrained(output)
    model.save_pretrained(output)


def selected_checkpoint(output):
    output = Path(output)
    summary = json.loads((output / "selection.json").read_text())
    selected = output / summary["selected_checkpoint"]
    if not selected.is_dir():
        raise FileNotFoundError(f"Selected checkpoint is missing: {selected}")
    return selected


def link_checkpoint(source, destination):
    source, destination = Path(source).resolve(), external_checkpoint_path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Export already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(
        os.path.relpath(source, destination.parent.resolve()), target_is_directory=True
    )


def merge_alignment(base, training, output, dtype="bfloat16"):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output = external_checkpoint_path(output)
    if output.exists():
        raise FileExistsError(f"Merged alignment output already exists: {output}")
    checkpoint = selected_checkpoint(training)
    base_model = AutoModelForCausalLM.from_pretrained(
        base, local_files_only=True, torch_dtype=getattr(torch, dtype)
    )
    model = PeftModel.from_pretrained(
        base_model, checkpoint, local_files_only=True
    ).merge_and_unload()
    output.mkdir(parents=True)
    model.save_pretrained(output)
    AutoTokenizer.from_pretrained(checkpoint, local_files_only=True).save_pretrained(output)


def format_training_text(row, stage):
    if stage == "align":
        return row["description"]
    if stage == "rec":
        system = (
            "You are a professional recommendation expert who needs to recommend "
            "the next possible purchase for users based on their purchase history. "
            "Please predict the most likely next product that the user will purchase "
            "based on the user's historical purchase information."
        )
        thought = "<think>\n\n</think>"
    else:
        system = SYSTEM_MESSAGE
        value = row.get("sid_routing_think")
        thought = (
            str(value).strip()
            if value is not None and str(value).strip()
            else "<think>\n\n</think>"
        )
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{row['description']}<|im_end|>\n"
        f"<|im_start|>assistant\n{thought}\n{row['groundtruth']}<|im_end|>\n"
    )


class InitialCollator:
    """Preserve the reference initial-model loss mask starting at the user turn."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.user = tokenizer.encode("<|im_start|>user", add_special_tokens=False)

    def __call__(self, features):
        length = max(len(row["input_ids"]) for row in features)
        result = {"input_ids": [], "attention_mask": [], "labels": []}
        for row in features:
            ids = list(row["input_ids"])
            padding = length - len(ids)
            padded = ids + [self.tokenizer.pad_token_id] * padding
            labels = list(padded)
            for index in range(len(ids) - len(self.user) + 1):
                if ids[index : index + len(self.user)] == self.user:
                    labels[:index] = [-100] * index
                    break
            else:
                raise ValueError("Initial-model input has no recognizable user turn")
            result["input_ids"].append(padded)
            result["attention_mask"].append(list(row["attention_mask"]) + [0] * padding)
            result["labels"].append(labels)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}


class EncodedRows(torch.utils.data.Dataset):
    def __init__(self, frame, tokenizer, stage):
        self.rows = [
            tokenizer(
                format_training_text(row, stage),
                add_special_tokens=True,
                truncation=True,
                max_length=4096,
                return_attention_mask=True,
            )
            for row in frame.to_dict("records")
        ]
        if not self.rows:
            raise ValueError("An initial-model training split is empty")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def train_initial(args, source, output):
    output = external_checkpoint_path(output)
    import pandas as pd
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    device_cpu = args.device == "cpu"
    if not device_cpu and not torch.cuda.is_available():
        raise RuntimeError("Use a CUDA environment for initial-model training")
    if args.stage == "align":
        global_batch, per_device, epochs, lr, warmup, decay = 64, 4, 15, 1e-4, 0.0, 0.0
        prefix = "training_align_data"
    else:
        global_batch, per_device, epochs, lr, warmup, decay = (
            16,
            2,
            6 if args.stage == "rec" else 3,
            1e-5,
            0.1,
            0.01,
        )
        prefix = "training_prediction_sid_data" if args.stage == "rec" else "training_sid_routing"
    if global_batch % (world * per_device):
        raise ValueError("Worker count cannot preserve the registered effective batch size")
    if rank == 0 and Path(output).exists():
        raise FileExistsError(f"Initial training output already exists: {output}")
    folder = args.root / "data/processed" / args.category
    train_file, val_file = (folder / f"{prefix}_{split}.parquet" for split in ("train", "val"))
    train, val = pd.read_parquet(train_file), pd.read_parquet(val_file)
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        source, local_files_only=True, torch_dtype=torch.float32 if device_cpu else torch.bfloat16
    )
    model.config.use_cache = False
    if args.stage == "align":
        from peft import TrainableTokensConfig, get_peft_model

        ids = tokenizer.convert_tokens_to_ids(sid_tokens(args.codebook_size))
        if tokenizer.unk_token_id in ids or len(ids) != len(set(ids)):
            raise ValueError("Expanded SID tokens are missing or aliased")
        model = get_peft_model(
            model,
            TrainableTokensConfig(
                token_indices=ids, target_modules=["embed_tokens"], init_weights=True
            ),
        )
    if not device_cpu:
        model.enable_input_require_grads()
    training = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=args.epochs or epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=per_device,
        per_device_eval_batch_size=per_device,
        gradient_accumulation_steps=global_batch // (world * per_device),
        learning_rate=lr,
        warmup_ratio=warmup,
        weight_decay=decay,
        lr_scheduler_type="linear" if args.stage == "align" else "constant_with_warmup",
        optim="adamw_torch",
        eval_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=False,
        bf16=not device_cpu,
        use_cpu=device_cpu,
        gradient_checkpointing=not device_cpu,
        seed=42,
        data_seed=42,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        logging_steps=1 if args.max_steps > 0 else 25,
        save_only_model=True,
        deepspeed=str(args.deepspeed) if args.deepspeed else None,
    )
    collator = (
        DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        if args.stage == "align"
        else InitialCollator(tokenizer)
    )
    trainer = Trainer(
        model=model,
        args=training,
        processing_class=tokenizer,
        train_dataset=EncodedRows(train, tokenizer, args.stage),
        eval_dataset=EncodedRows(val, tokenizer, args.stage),
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        if args.stage == "align"
        else [],
    )
    trainer.train()
    if rank == 0:
        records = [row for row in trainer.state.log_history if "eval_loss" in row]
        if not records:
            raise RuntimeError("Initial-model training produced no validation measurements")
        winner = min(records, key=lambda row: (row["eval_loss"], row["step"]))
        checkpoint = Path(output) / f"checkpoint-{winner['step']}"
        if not checkpoint.is_dir():
            raise RuntimeError("Validation-selected checkpoint was not retained")
        save_json(
            Path(output) / "selection.json",
            {
                "stage": args.stage,
                "selected_checkpoint": checkpoint.name,
                "validation_loss": winner["eval_loss"],
                "validation": records,
                "seed": 42,
                "global_batch": global_batch,
                "train_sha256": hashlib.sha256(train_file.read_bytes()).hexdigest(),
                "validation_sha256": hashlib.sha256(val_file.read_bytes()).hexdigest(),
            },
        )
        if args.stage == "sid_routing":
            link_checkpoint(checkpoint, args.artifact_root / "models" / args.category / "initial")
    if world > 1:
        torch.distributed.barrier()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stage", choices=["expand", "align", "merge", "rec", "sid_routing"], required=True
    )
    p.add_argument("--category", choices=["Beauty", "Sports", "Toys"], required=True)
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--artifact-root", type=Path)
    p.add_argument("--base-model", type=Path)
    p.add_argument("--codebook-size", type=int, default=256)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--epochs", type=int)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--deepspeed", type=Path)
    a = p.parse_args()
    if a.codebook_size < 1 or (a.epochs is not None and a.epochs < 1):
        p.error("Codebook size and epochs must be positive")
    a.artifact_root = (
        external_checkpoint_path(a.artifact_root) if a.artifact_root else artifact_root(a.root)
    )
    base = a.base_model or a.artifact_root / "model/Qwen3-1-7B"
    expanded = a.artifact_root / "model/Qwen3-1-7B-expand"
    outputs = a.artifact_root / "outputs" / f"{a.category.lower()}_initial"
    aligned = a.artifact_root / "models" / a.category / "aligned"
    if a.stage == "expand":
        expand_vocabulary(
            base, expanded, a.codebook_size, "float32" if a.device == "cpu" else "bfloat16"
        )
    elif a.stage == "merge":
        merge_alignment(
            expanded, outputs / "align", aligned, "float32" if a.device == "cpu" else "bfloat16"
        )
    else:
        source = (
            expanded
            if a.stage == "align"
            else aligned
            if a.stage == "rec"
            else selected_checkpoint(outputs / "rec")
        )
        train_initial(a, source, outputs / a.stage)


if __name__ == "__main__":
    main()
