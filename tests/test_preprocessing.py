from argparse import Namespace
from collections import Counter
import gzip
import json

import numpy as np
import pandas as pd
import pytest

from dynr2 import prep_kernels as kernels
from dynr2.initialize import format_training_text
from dynr2.preprocess import construct_sids, prepare_splits, read_records, save_json


def raw_fixture(root):
    folder = root / "raw" / "Beauty"
    folder.mkdir(parents=True)
    with gzip.open(folder / "reviews_Beauty_5.json.gz", "wt") as stream:
        for user in ("u0", "u1"):
            for time, item in enumerate(("a", "b", "c", "d", "a", "b")):
                stream.write(
                    json.dumps({"reviewerID": user, "asin": item, "unixReviewTime": time}) + "\n"
                )
    with gzip.open(folder / "meta_Beauty.json.gz", "wt") as stream:
        for item in ("a", "b", "c", "d"):
            stream.write(
                repr(
                    {
                        "asin": item,
                        "title": "title " + item,
                        "description": ["description"],
                        "categories": [["category"]],
                    }
                )
                + "\n"
            )
    return Namespace(
        raw_root=root / "raw",
        processed_root=root / "processed",
        export_root=root / "export",
        k_core=1,
        skip_benchmark_check=True,
        embeddings=root / "embeddings.npy",
        skip_embedding=False,
        embedding_model="unused",
        device="cpu",
        codebook_size=2,
        rq_iterations=2,
        model_selection_users=1,
    )


def test_json_and_python_metadata_records_are_read_without_text_mutation(tmp_path):
    path = tmp_path / "metadata.gz"
    with gzip.open(path, "wt") as stream:
        stream.write(json.dumps({"title": "true null false", "flag": True}) + "\n")
        stream.write(repr({"title": "literal", "flag": None}) + "\n")
    assert list(read_records(path)) == [
        {"title": "true null false", "flag": True},
        {"title": "literal", "flag": None},
    ]


def test_chronological_export_connects_to_response_training(tmp_path):
    args = raw_fixture(tmp_path)
    source = args.processed_root / "Beauty"
    source.mkdir(parents=True)
    mapping = {}
    items = {}
    for index, item in enumerate("abcd", 1):
        sid = (
            "<|sid_begin|>"
            + "".join(f"<s_{layer}_{(index - 1) % 2}>" for layer in "abcd")
            + "<|sid_end|>"
        )
        mapping[item] = {"int_id": index, "sid": sid}
        items[str(index)] = {"sid": sid, "title": item, "categories": "category"}
    save_json(source / "item_mapping.json", mapping)
    save_json(source / "Beauty.pretrain.json", items)
    (source / "Beauty_sequential_data.txt").write_text("1 1 2 3 4 1 2\n2 1 2 3 4 1 2\n")
    np.savez(source / "codebooks.npz", **{f"s_{layer}": np.eye(2) for layer in "abcd"})
    prepare_splits("Beauty", args)
    target = args.export_root / "Beauty"
    train, val, test = [
        pd.read_parquet(target / f"{name}.parquet") for name in ("train", "model_selection", "test")
    ]
    assert set(train.target_item) == {"d"}
    assert set(val.target_item) == {"a"}
    assert set(test.target_item) == {"b"}
    assert Counter(train.iloc[0].history_items) == Counter("abc")
    assert json.loads((target / "train_history_counts.json").read_text()) == {
        "a": 2,
        "b": 2,
        "c": 2,
    }
    assert len(val) == 1
    initial = pd.read_parquet(source / "training_sid_routing_train.parquet")
    assert len(initial) == len(train)
    assert all(initial.sid_routing_think.str.startswith("<think>"))
    expected = kernels._split_sid_prediction(items, [("1", list("123412")), ("2", list("123412"))])
    actual = pd.read_parquet(source / "training_prediction_sid_data_train.parquet")
    pd.testing.assert_frame_equal(actual, expected["train"])


def test_actual_faiss_sid_construction(tmp_path):
    pytest.importorskip("faiss")
    args = raw_fixture(tmp_path)
    np.save(args.embeddings, np.random.default_rng(42).normal(size=(4, 8)).astype(np.float32))
    try:
        construct_sids("Beauty", args)
        source = args.processed_root / "Beauty"
        codes = np.load(source / "sid_codes.npy")
        assert codes.shape == (4, 4)
        assert codes.min() >= 0 and codes.max() < 2
        prepare_splits("Beauty", args)
        assert (args.export_root / "Beauty/train.parquet").is_file()
    finally:
        kernels.CODEBOOK_SIZE, kernels.RQ_N_ITER = 256, 30


def test_initial_model_chat_format_matches_reference():
    row = {
        "description": "history",
        "groundtruth": "sid",
        "sid_routing_think": "<think>\ns_a: Explore\n</think>",
    }
    assert format_training_text(row, "align") == "history"
    direct = format_training_text(row, "rec")
    assert "<|im_start|>assistant\n<think>\n\n</think>\nsid<|im_end|>" in direct
    transition = format_training_text(row, "sid_routing")
    assert row["sid_routing_think"] + "\nsid" in transition
