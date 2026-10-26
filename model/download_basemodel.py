#!/usr/bin/env python3
"""Download the base model and tokenizer into an explicit local directory."""

import argparse
from pathlib import Path
import sys


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dynr2.paths import artifact_root, external_checkpoint_path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output", type=Path, default=artifact_root() / "model/Qwen3-1-7B")
    args = parser.parse_args()
    args.output = external_checkpoint_path(args.output)
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=args.model_id, local_dir=str(args.output))


if __name__ == "__main__":
    main()
