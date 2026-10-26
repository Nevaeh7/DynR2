"""Target-independent explicit, latent, and mixed training schedules."""

import hashlib


def schedule(user_id, history_hash, epoch, seed, arm):
    if arm not in ("fixed", "anytime"):
        raise ValueError("Unknown registered training arm")
    draw = hashlib.sha256(f"{seed}|{epoch}|{user_id}|{history_hash}".encode()).digest()
    family = draw[0] % 4
    layout = (
        "EEEE"
        if family == 0
        else "LLLL"
        if family == 1
        else "".join("E" if (draw[1] >> i) & 1 else "L" for i in range(4))
    )
    return layout if arm == "fixed" else layout[: 1 + draw[2] % 4]
