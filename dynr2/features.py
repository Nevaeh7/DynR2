"""Target-free belief features for the continuation-value controller."""

import math


def _values(q):
    return q.detach().float().cpu().tolist() if hasattr(q, "detach") else list(q)


def belief_features(q, previous, depth, last_action, first_ids, counts, detail):
    """A closed feature interface: no target, rank, reward or oracle argument."""
    q = [float(x) for x in _values(q)]
    previous = None if previous is None else [float(x) for x in _values(previous)]
    if (
        len(q) < 2
        or len(q) != len(first_ids)
        or len(set(first_ids)) != len(first_ids)
        or any(not math.isfinite(x) or x < 0 for x in q)
        or not math.isclose(sum(q), 1.0, abs_tol=2e-5)
    ):
        raise ValueError("Invalid belief support/probabilities")
    if type(depth) is not int or not 0 <= depth <= 4:
        raise ValueError("Depth outside registered block budget")
    if (depth == 0 and (previous is not None or last_action is not None)) or (
        depth > 0 and (previous is None or last_action not in ("E", "L"))
    ):
        raise ValueError("Previous-belief/last-action semantics do not match depth")
    if previous is not None and (
        len(previous) != len(q)
        or any(not math.isfinite(x) or x < 0 for x in previous)
        or not math.isclose(sum(previous), 1.0, abs_tol=2e-5)
    ):
        raise ValueError("Previous belief has incompatible support")
    rho = float(detail["valid_first_code_mass"])
    entropy = float(detail["reasoning_entropy"])
    if (
        not math.isfinite(rho)
        or not 0 <= rho <= 1
        or not math.isfinite(entropy)
        or not 0 <= entropy <= 1
    ):
        raise ValueError("Invalid observation mass or language uncertainty")
    sizes = []
    for token in first_ids:
        value = counts.get(token, counts.get(str(token)))
        if type(value) is not int or value <= 0:
            raise ValueError("Missing frozen catalog prefix size")
        sizes.append(value)
    order = sorted(range(len(q)), key=lambda index: (-q[index], index))
    u = -sum(p * math.log(max(p, 1e-12)) for p in q) / math.log(len(q))
    b = 0.0
    if previous is not None:
        mixture = [(p + old) / 2 for p, old in zip(q, previous)]
        b = 0.5 * sum(
            p * math.log(max(p, 1e-12) / max(m, 1e-12))
            + old * math.log(max(old, 1e-12) / max(m, 1e-12))
            for p, old, m in zip(q, previous, mixture)
        )
    return {
        "U": min(1.0, max(0.0, u)),
        "B": min(math.log(2), max(0.0, b)),
        "rho": rho,
        "margin": q[order[0]] - q[order[1]],
        "top1_probability": q[order[0]],
        "expected_log_prefix_size": sum(p * math.log(size) for p, size in zip(q, sizes)),
        "log_top1_prefix_size": math.log(sizes[order[0]]),
        "depth": float(depth),
        "has_prev_belief": float(previous is not None),
        "last_action_E": float(last_action == "E"),
        "last_action_L": float(last_action == "L"),
        "reasoning_entropy": entropy,
    }
