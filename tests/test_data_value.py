import json
from pathlib import Path

import numpy as np
import pytest

from dynr2.data import Catalog, disjoint_users, history_counts, load_rows, rank_metrics
from dynr2.features import belief_features
from dynr2.fit_value import fit_banks, ridge, state_plan
from dynr2.runtime import load_config
from dynr2.value import FEATURE_GROUPS, choose_value, predict


def test_catalog_collisions_and_history_only_prior(tmp_path, tokenizer):
    a = "<|sid_begin|><s_a_0><s_b_0><s_c_0><s_d_0><|sid_end|>"
    b = "<|sid_begin|><s_a_1><s_b_0><s_c_0><s_d_0><|sid_end|>"
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"a": {"sid": a}, "b": {"sid": a}, "c": {"sid": b}}))
    catalog = Catalog(path, tokenizer)
    counts = history_counts([{"history_items": ["b", "b"], "target_item": "a"}], catalog)
    assert counts == {"b": 2}
    expanded = catalog.expand([(a, 0.0)], counts)
    assert [item for item, _ in expanded] == ["b", "a"]
    assert expanded[0][1] == pytest.approx(np.log(0.75))
    assert catalog.prefix_counts == {20: 2, 24: 1}
    assert catalog.score({"target_item": "a"}, [], counts)["item"]["hr@10"] == 0
    with pytest.raises(ValueError):
        catalog.target_sid({"target_item": "a", "groundtruth": b})


def test_invalid_outputs_remain_zero_and_rank_metrics():
    assert rank_metrics([], "a")["ndcg@10"] == 0
    assert rank_metrics(["b", "a"], "a")["ndcg@10"] == pytest.approx(1 / np.log2(3))


def test_user_disjoint_loading(tmp_path):
    path = tmp_path / "rows.jsonl"
    row = {"sample_id": 1, "user_id": 1, "description": "history"}
    path.write_text(json.dumps(row) + "\n")
    rows = load_rows(path)
    assert rows[0]["user_id"] == "1"
    with pytest.raises(ValueError, match="overlap"):
        disjoint_users(rows, rows)
    path.write_text((json.dumps(row) + "\n") * 2)
    with pytest.raises(ValueError, match="Repeated"):
        load_rows(path)


def features(depth=0):
    return belief_features(
        [0.7, 0.3],
        None if depth == 0 else [0.6, 0.4],
        depth,
        None if depth == 0 else "L",
        [20, 24],
        {20: 3, 24: 4},
        {"valid_first_code_mass": 0.8, "reasoning_entropy": 0.0},
    )


def test_ridge_ties_and_cost_penalty():
    names = FEATURE_GROUPS["full_belief"]
    f = features()
    x = np.asarray([[f[n] for n in names]] * 3)
    model = ridge(x, np.zeros((3, 2)), 0.1)
    assert choose_value(model, f, {"E": 1.0, "L": 1.0})[0] == "D"
    model["heads"]["E"]["intercept"] = 0.2
    model["heads"]["L"]["intercept"] = 0.2
    assert choose_value(model, f, {"E": 1.0, "L": 1.0})[0] == "E"
    assert choose_value(model, f, {"E": 1.0, "L": 1.0}, 0.3)[0] == "D"
    model["scale"][0] = 0
    with pytest.raises(ValueError, match="scales"):
        predict(model, f)


def test_value_fit_and_split_guard():
    def row(user):
        return {
            "user_id": user,
            "states": [
                {
                    "features": features(depth),
                    "rewards": {"D": 0.1, "E": 0.2, "L": 0.05},
                    "costs_ms": {"D": 2.0, "E": 4.0, "L": 3.0},
                }
                for depth in (0, 1, 2)
            ],
        }

    result = fit_banks([row("a")], [row("b")])
    assert result["incremental_cost_ms"] == {"E": 2.0, "L": 1.0}
    assert result["lambda_candidates"][0] == 0
    assert predict(result["model"], features())["E"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="overlap"):
        fit_banks([row("a")], [row("a")])


def test_state_plans_and_paper_profiles():
    layout, depths = state_plan("Beauty", "a")
    assert len(layout) == 3 and set(layout) <= {"E", "L"}
    assert len(depths) == 3 and depths[0] == 0 and depths == sorted(set(depths))
    root = Path(__file__).resolve().parents[1]
    for domain, u, strength in [("beauty", 0.55, 1.0), ("sports", 0.55, 1.0), ("toys", 0.75, 0.5)]:
        config = load_config(root / "configs" / f"{domain}.json")
        assert config["tau_u"] == u and config["adapter_strength"] == strength
