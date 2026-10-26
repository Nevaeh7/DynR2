import copy
import math

import pytest
import torch

from dynr2.core import decide_mode, js_divergence, normalized_entropy
from dynr2.losses import legal_sid_kl, sid_target_ids
from dynr2.schedule import schedule


def decode_logits(model, context):
    return context.logits.clone()


def test_belief_statistics_and_rule():
    concentrated = torch.tensor([1.0, 0.0])
    diffuse = torch.tensor([0.5, 0.5])
    assert normalized_entropy(diffuse) == 1
    assert normalized_entropy(concentrated) == 0
    assert float(js_divergence(concentrated, concentrated.flip(0))) == pytest.approx(math.log(2))
    assert decide_mode(diffuse, None, 0.7, 0.1)[0] == "L"
    assert decide_mode(concentrated, None, 0.7, 0.1)[0] == "D"
    assert decide_mode(concentrated, concentrated.flip(0), 0.7, 0.1)[0] == "E"


def test_query_isolation_with_mutable_cache(model):
    model.eval()
    state = model.start([1, 2, 3])
    before = copy.deepcopy(state)
    q, detail = model.observe(state)
    assert state.output.past_key_values.n == before.output.past_key_values.n
    assert torch.equal(state.output.past_key_values.last, before.output.past_key_values.last)
    assert torch.equal(state.output.logits, before.output.logits)
    assert q.sum().item() == pytest.approx(1.0)
    assert 0 <= detail["valid_first_code_mass"] <= 1
    assert not model.backbone.calls[-1]["adapted"]
    state, _ = model.advance(state, "L")
    model.observe(state)
    assert model.backbone.calls[-1]["adapted"]


def test_direct_preserved_after_real_adapter_update(model):
    model.eval()
    before, _ = model.recommend([1, 2, 3], decode_logits, "direct")
    frozen = {k: v.detach().clone() for k, v in model.backbone.named_parameters() if k != "adapter"}
    optimizer = torch.optim.AdamW([model.backbone.adapter], lr=0.01)
    for layout in ("E", "LL", "ELE", "LLLL"):
        model.train()
        loss, _, _ = model(
            [1, 2, 3], [7, 20, 21, 22, 23, 8, 9], layout, ((20, 24), (21, 25), (22, 26), (23, 27))
        )
        loss.backward()
        assert torch.isfinite(model.backbone.adapter.grad).all()
        optimizer.step()
        optimizer.zero_grad()
        model.eval()
        actual, _ = model.recommend([1, 2, 3], decode_logits, "direct")
        assert torch.equal(before, actual)
        assert all(
            p.grad is None for name, p in model.backbone.named_parameters() if name != "adapter"
        )
    assert model.backbone.adapter.abs().sum() > 0
    assert all(
        torch.equal(v, dict(model.backbone.named_parameters())[k]) for k, v in frozen.items()
    )


def test_immediate_stop_matches_direct_and_query_counts(model):
    model.eval()
    direct, d = model.recommend_policy([1, 2, 3], decode_logits, "direct")
    stopped, r = model.recommend_policy([1, 2, 3], decode_logits, "rule", tau_u=1.0)
    assert torch.equal(direct, stopped)
    assert d["answer_queries"] == 0
    assert r["answer_queries"] == 1 and r["blocks"] == 0
    _, horizon = model.recommend_policy([1, 2, 3], decode_logits, "rule", tau_u=0.0)
    assert horizon["blocks"] == model.max_blocks
    assert horizon["answer_queries"] == model.max_blocks
    assert horizon["trace"][-1]["forced_budget"]


@pytest.mark.parametrize("action", ["E", "L"])
def test_portable_state_machine_matches_training_path(model, action):
    model.eval()
    with torch.no_grad():
        model.backbone.adapter.add_(0.1)
    expected, _ = model.recommend([1, 2, 3], decode_logits, action * 4)
    actual, detail = model.recommend_policy([1, 2, 3], decode_logits, action)
    assert torch.equal(expected, actual)
    assert detail["blocks"] == 4 and detail["answer_queries"] == 0


def test_top_one_latent_matches_explicit(model):
    model.eval()
    model.latent_topk = 1
    e, _ = model.recommend_policy([1, 2, 3], decode_logits, "E")
    latent, _ = model.recommend_policy([1, 2, 3], decode_logits, "L")
    assert torch.equal(e, latent)


def test_answer_targets_and_kl_support(tokenizer):
    sid = "<|sid_begin|><s_a_0><s_b_0><s_c_0><s_d_0><|sid_end|>"
    assert sid_target_ids(sid, tokenizer) == [7, 20, 21, 22, 23, 8, 9]
    student = torch.randn(1, 7, 32, requires_grad=True)
    teacher = torch.randn(1, 7, 32, requires_grad=True)
    supports = ((20, 24), (21, 25), (22, 26), (23, 27))
    loss = legal_sid_kl(student, teacher, supports)
    loss.backward()
    assert teacher.grad is None
    assert student.grad[:, 1:5].abs().sum() > 0
    assert student.grad[:, [0, 5, 6]].abs().sum() == 0
    assert student.grad[:, :, 0].abs().sum() == 0


def test_schedule_has_target_free_matched_prefixes():
    lengths = set()
    for i in range(100):
        anytime = schedule(str(i), "history", 0, 1427, "anytime")
        fixed = schedule(str(i), "history", 0, 1427, "fixed")
        assert anytime == fixed[: len(anytime)]
        lengths.add(len(anytime))
    assert lengths == {1, 2, 3, 4}


def test_latent_weights_are_differentiable_after_initial_input(model):
    # Observe nonzero response gradients on a multi-position soft-input path.
    loss, _, _ = model(
        [1, 2, 3], [7, 20, 21, 22, 23, 8, 9], "LL", ((20, 24), (21, 25), (22, 26), (23, 27))
    )
    loss.backward()
    assert model.backbone.adapter.grad.abs().sum() > 0
