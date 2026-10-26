"""Small frozen two-head continuation-value predictor; target-free inference.

This module intentionally has no Torch, model, dataset or target dependencies.
All standardization and costs are serialized from controller train/validation.
"""

import math


FORMAT = "dynr2-value-ridge-v1"
FEATURE_GROUPS = {
    "full_belief": (
        "U",
        "B",
        "rho",
        "margin",
        "top1_probability",
        "expected_log_prefix_size",
        "log_top1_prefix_size",
        "depth",
        "has_prev_belief",
        "last_action_E",
        "last_action_L",
    ),
}
ACTIONS = ("D", "E", "L")


def vector(features, names):
    values = [float(features[k]) for k in names]
    if not all(math.isfinite(v) for v in values):
        raise FloatingPointError("Nonfinite online feature")
    return values


def design_vector(features, names, depth_encoding=None):
    values = vector(features, names)
    if depth_encoding is None:
        return values
    if depth_encoding != "onehot_0_3" or "depth" not in names:
        raise ValueError("Unknown frozen depth encoding")
    index = names.index("depth")
    depth = values[index]
    if depth != int(depth) or not 0 <= depth < 4:
        raise ValueError("Value decisions require a registered nonterminal depth")
    return values[:index] + [float(depth == d) for d in range(4)] + values[index + 1 :]


def predict(model, features):
    if model.get("format") != FORMAT:
        raise ValueError("Unknown value predictor format")
    names = tuple(model["feature_names"])
    if names != FEATURE_GROUPS[model["feature_group"]]:
        raise ValueError("Predictor feature contract changed")
    x = design_vector(features, names, model.get("depth_encoding"))
    if len(model["mean"]) != len(x) or len(model["scale"]) != len(x):
        raise ValueError("Predictor standardization width mismatch")
    if any(not math.isfinite(float(sd)) or float(sd) <= 0 for sd in model["scale"]):
        raise ValueError("Predictor scales must be finite and positive")
    z = [(v - float(mu)) / float(sd) for v, mu, sd in zip(x, model["mean"], model["scale"])]
    result = {}
    for action in ("E", "L"):
        head = model["heads"][action]
        if len(head["coef"]) != len(z):
            raise ValueError("Predictor head width mismatch")
        result[action] = float(head["intercept"]) + sum(
            float(w) * v for w, v in zip(head["coef"], z)
        )
    if not all(math.isfinite(v) for v in result.values()):
        raise FloatingPointError("Nonfinite continuation-value prediction")
    return result


def choose_value(model, features, cost_ms, cost_weight=0.0):
    """D has gain zero; deterministic D/E/L ties, no target consulted."""
    weight = float(cost_weight)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("Invalid frozen cost weight")
    gain = predict(model, features)
    costs = {a: float(cost_ms[a]) for a in ("E", "L")}
    if any(not math.isfinite(v) or v < 0 for v in costs.values()):
        raise ValueError("Invalid validation-calibrated incremental cost")
    scores = {"D": 0.0, **{a: gain[a] - weight * costs[a] for a in ("E", "L")}}
    action = max(ACTIONS, key=lambda a: scores[a])
    return action, gain, scores
