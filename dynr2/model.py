"""Response adaptation with a frozen history encoder and immediate predictor."""

import copy
from contextlib import nullcontext
from dataclasses import dataclass
import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .core import DynR2Base, decide_mode
from .features import belief_features
from .losses import legal_sid_kl
from .value import choose_value


@dataclass
class ReasoningState:
    output: object
    length: int
    depth: int = 0

    def fork(self):
        return ReasoningState(
            SimpleNamespace(
                logits=self.output.logits.clone(),
                past_key_values=copy.deepcopy(self.output.past_key_values),
            ),
            self.length,
            self.depth,
        )


class DynR2(DynR2Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not callable(getattr(self.backbone, "disable_adapter", None)):
            raise ValueError("A switchable adapter backbone is required")

    def adapter_scope(self, enabled):
        return nullcontext() if enabled else self.backbone.disable_adapter()

    def prefill(self, prompt_ids):
        if not prompt_ids:
            raise ValueError("Nonempty history required")
        # Original representation is protected in every training/inference arm.
        # Only response-side adapter operations receive gradients.
        with self.backbone.disable_adapter(), torch.no_grad():
            output = self._forward(ids=prompt_ids)
        return SimpleNamespace(
            logits=output.logits[:, -1:, :].detach(), past_key_values=output.past_key_values
        )

    def _answer(self, prefix, prompt_length, layout, adapt_answer=False, tau_u=0.7, tau_b=0.1):
        dynamic = layout == "dynamic"
        if (
            not dynamic
            and layout != "direct"
            and (not layout or set(layout) - {"E", "L"} or len(layout) > self.max_blocks)
        ):
            raise ValueError("A valid trajectory within the declared budget is required")
        if not 0 <= tau_u <= 1 or not 0 <= tau_b <= math.log(2):
            raise ValueError("Invalid fixed gate thresholds")
        output = SimpleNamespace(
            logits=prefix.logits, past_key_values=copy.deepcopy(prefix.past_key_values)
        )
        length, trace, previous = prompt_length, [], None
        if layout != "direct":
            for block in range(self.max_blocks if dynamic else len(layout)):
                if dynamic:
                    # At t=0, probe the unmodified Direct state; after thought
                    # execution, probe the adapted state without rewinding it.
                    with self.adapter_scope(bool(trace)):
                        belief = self._recommendation_belief(output, length)
                    mode, uncertainty, transition = decide_mode(belief, previous, tau_u, tau_b)
                    previous = belief
                    if mode == "D":
                        break
                else:
                    mode = layout[block]
                    uncertainty = transition = None
                record = {
                    "block": block,
                    "mode": mode,
                    "uncertainty": uncertainty,
                    "transition": transition,
                    "explicit_token_ids": [],
                }
                for _ in range(self.block_size):
                    allowed = self._reasoning_logits(output.logits[:, -1, :])
                    length += 1
                    if mode == "E":
                        token = int(allowed.argmax(-1).item())
                        record["explicit_token_ids"].append(token)
                        output = self._forward(
                            ids=[token], cache=output.past_key_values, length=length
                        )
                    else:
                        values, ids = allowed.topk(self.latent_topk, dim=-1)
                        weights = F.softmax(values, -1).to(
                            self.backbone.get_input_embeddings().weight.dtype
                        )
                        soft = (
                            (weights.unsqueeze(-1) * self.backbone.get_input_embeddings()(ids))
                            .sum(1)
                            .unsqueeze(1)
                        )
                        output = self._forward(
                            embeds=soft, cache=output.past_key_values, length=length
                        )
                trace.append(record)
        adapted = bool(trace) or adapt_answer
        with self.adapter_scope(adapted):
            context = self._forward(
                ids=self.suffix_ids,
                cache=output.past_key_values,
                length=length + len(self.suffix_ids),
            )
        return context, trace, adapted, length + len(self.suffix_ids)

    def _target_logits(self, prefix, prompt_length, target_ids, layout, adapt_answer=False):
        context, trace, adapted, length = self._answer(prefix, prompt_length, layout, adapt_answer)
        target = torch.as_tensor(target_ids, dtype=torch.long, device=self.device).reshape(1, -1)
        if target.shape[1] != 7:
            raise ValueError("Expected begin, four codes, end and EOS")
        with self.adapter_scope(adapted):
            follow = self.backbone(
                input_ids=target[:, :-1],
                past_key_values=context.past_key_values,
                attention_mask=torch.ones(
                    (1, length + target.shape[1] - 1), dtype=torch.long, device=self.device
                ),
                use_cache=False,
            )
        logits = torch.cat((context.logits[:, -1:, :], follow.logits), 1).float()
        return logits, trace

    def forward(
        self, prompt_ids, target_ids, layout, supports, anchor_weight=0.2, adapt_answer=False
    ):
        if layout == "dynamic" or (layout == "direct" and not adapt_answer):
            raise ValueError("Train forced adapted paths; original Direct is immutable")
        if not math.isfinite(anchor_weight) or anchor_weight < 0:
            raise ValueError("Invalid anchor weight")
        prefix = self.prefill(prompt_ids)
        with torch.no_grad():
            teacher, _ = self._target_logits(prefix, len(prompt_ids), target_ids, "direct")
        student, trace = self._target_logits(
            prefix, len(prompt_ids), target_ids, layout, adapt_answer
        )
        target = torch.as_tensor(target_ids, device=self.device, dtype=torch.long)
        sid = F.cross_entropy(student.flatten(0, 1), target)
        anchor = legal_sid_kl(student, teacher, supports)
        return sid + anchor_weight * anchor, sid.detach(), anchor.detach()

    @torch.no_grad()
    def recommend(
        self, prompt_ids, decoder, layout="dynamic", tau_u=0.7, tau_b=0.1, adapt_answer=False
    ):
        """decoder(model, context) is called exactly once after the final stop."""
        if self.training:
            raise ValueError("Recommendation requires eval mode")
        prefix = self.prefill(prompt_ids)
        context, trace, adapted, _ = self._answer(
            prefix, len(prompt_ids), layout, adapt_answer, tau_u, tau_b
        )
        with self.adapter_scope(adapted):
            result = decoder(self, context)
        return result, trace

    @torch.no_grad()
    def start(self, prompt_ids):
        if self.training:
            raise ValueError("Inference requires eval mode")
        return ReasoningState(self.prefill(prompt_ids), len(prompt_ids))

    @torch.no_grad()
    def observe(self, state):
        """Raw first-code belief in a disposable answer-context cache."""
        with self.adapter_scope(state.depth > 0):
            out = self._forward(
                ids=self.answer_probe_ids,
                cache=copy.deepcopy(state.output.past_key_values),
                length=state.length + len(self.answer_probe_ids),
            )
        logits = out.logits[0, -1].float()
        selected = logits[list(self.first_ids)]
        q = F.softmax(selected, -1)
        rho = float((torch.logsumexp(selected, 0) - torch.logsumexp(logits, 0)).exp())
        if not torch.isfinite(q).all() or not math.isfinite(rho):
            raise FloatingPointError("Nonfinite answer belief")
        return q, {"valid_first_code_mass": min(1.0, max(0.0, rho)), "reasoning_entropy": 0.0}

    @torch.no_grad()
    def advance(self, state, action):
        if action not in ("E", "L") or state.depth >= self.max_blocks:
            raise ValueError("An E/L block within the reasoning horizon is required")
        output, length, tokens = state.output, state.length, []
        for _ in range(self.block_size):
            allowed = self._reasoning_logits(output.logits[:, -1, :])
            length += 1
            if action == "E":
                token = int(allowed.argmax(-1).item())
                tokens.append(token)
                output = self._forward(ids=[token], cache=output.past_key_values, length=length)
            else:
                if self.latent_topk == 1:
                    ids = allowed.argmax(-1, keepdim=True)
                    values = allowed.gather(-1, ids)
                else:
                    values, ids = allowed.topk(self.latent_topk, dim=-1)
                weights = F.softmax(values, -1).to(
                    self.backbone.get_input_embeddings().weight.dtype
                )
                soft = (
                    (weights.unsqueeze(-1) * self.backbone.get_input_embeddings()(ids))
                    .sum(1)
                    .unsqueeze(1)
                )
                output = self._forward(embeds=soft, cache=output.past_key_values, length=length)
        return ReasoningState(output, length, state.depth + 1), tokens

    @torch.no_grad()
    def decode_state(self, state, decoder):
        with self.adapter_scope(state.depth > 0):
            context = self._forward(
                ids=self.suffix_ids,
                cache=state.output.past_key_values,
                length=state.length + len(self.suffix_ids),
            )
            return decoder(self, context)

    @torch.no_grad()
    def recommend_policy(
        self,
        prompt_ids,
        decoder,
        policy="rule",
        tau_u=0.7,
        tau_b=0.1,
        prefix_counts=None,
        predictor=None,
        costs=None,
        cost_weight=0.0,
    ):
        if policy not in ("direct", "E", "L", "rule", "value"):
            raise ValueError("Unknown inference policy")
        if policy == "value" and (prefix_counts is None or predictor is None or costs is None):
            raise ValueError("Value inference requires a frozen predictor, catalog, and costs")
        state = self.start(prompt_ids)
        previous, last, trace, queries = None, None, [], 0
        while state.depth < self.max_blocks and policy != "direct":
            q, features, gains, scores = None, None, None, None
            if policy in ("E", "L"):
                action = policy
            else:
                q, detail = self.observe(state)
                queries += 1
                if policy == "rule":
                    action, u, b = decide_mode(q, previous, tau_u, tau_b)
                    features = {"U": u, "B": b}
                else:
                    features = belief_features(
                        q, previous, state.depth, last, self.first_ids, prefix_counts, detail
                    )
                    action, gains, scores = choose_value(predictor, features, costs, cost_weight)
            trace.append(
                {
                    "depth": state.depth,
                    "action": action,
                    "features": features,
                    "predicted_gain": gains,
                    "action_scores": scores,
                }
            )
            if action == "D":
                break
            previous, last = q, action
            state, tokens = self.advance(state, action)
            trace[-1]["explicit_token_ids"] = tokens
        if not trace or trace[-1]["action"] != "D":
            trace.append({"depth": state.depth, "action": "D", "forced_budget": policy != "direct"})
        candidates = self.decode_state(state, decoder)
        return candidates, {
            "trace": trace,
            "blocks": state.depth,
            "answer_queries": queries,
            "intermediate_positions": state.depth * self.block_size,
        }
