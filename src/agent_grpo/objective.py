"""GRPO math; torch imported only by the differentiable implementation."""
import math
import statistics


def group_advantages(rewards, epsilon=1e-6):
    if len(rewards) < 2:
        raise ValueError("GRPO needs at least two trajectories per question")
    mean = statistics.fmean(rewards)
    std = statistics.pstdev(rewards)
    return [(value - mean) / (std + epsilon) for value in rewards]


def scalar_loss(current, old, reference, mask, advantage, beta=0.04, clip=0.2):
    """Independent scalar oracle for tests; averages only generated actions."""
    if not (len(current) == len(old) == len(reference) == len(mask)) or not any(mask):
        raise ValueError("aligned nonempty generated-token mask required")
    terms = []
    for cur, prev, ref, selected in zip(current, old, reference, mask):
        if selected:
            ratio = math.exp(cur - prev)
            kl = math.expm1(ref - cur) - (ref - cur)
            terms.append(-min(ratio * advantage, min(1 + clip, max(1 - clip, ratio)) * advantage) + beta * kl)
    return statistics.fmean(terms)


def grpo_loss(current, old, reference, mask, advantage, beta=0.04, clip=0.2):
    """One trajectory's loss. Caller averages losses over the sampled group."""
    import torch
    mask = mask.bool()
    if not mask.any():
        raise ValueError("trajectory has no generated tokens")
    cur, prev, ref = current[mask], old.detach()[mask], reference.detach()[mask]
    ratio = torch.exp(cur - prev)
    surrogate = torch.minimum(ratio * advantage, torch.clamp(ratio, 1 - clip, 1 + clip) * advantage)
    delta = ref - cur
    kl = torch.expm1(delta) - delta
    loss = (-surrogate + beta * kl).mean()
    return loss, {"kl": float(kl.detach().mean()), "clip_fraction": float(((ratio < 1 - clip) | (ratio > 1 + clip)).float().mean().detach())}
