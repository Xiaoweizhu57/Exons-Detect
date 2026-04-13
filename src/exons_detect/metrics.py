"""Numerical scoring utilities for Exons-Detect."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def weighted_entropy(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pad_token_id: int,
    delta: torch.Tensor,
) -> torch.Tensor:
    """Compute weighted soft cross-entropy across the non-padding sequence."""
    del input_ids
    del pad_token_id

    p_scores = p_logits[..., :-1, :]
    q_scores = q_logits[..., :-1, :]
    valid_mask = attention_mask[..., 1:].to(dtype=p_scores.dtype)
    p_probs = F.softmax(p_scores, dim=-1)
    q_log_probs = F.log_softmax(q_scores, dim=-1)

    token_entropy = -(p_probs * q_log_probs).sum(dim=-1)
    weights = (1.0 + delta) * valid_mask
    denom = weights.sum(dim=1).clamp_min(1e-12)
    return (token_entropy * weights).sum(dim=1) / denom


def weighted_sum_perplexity_dual(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    performer_logits: torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:
    """Compute weighted token perplexity using gold tokens and greedy performer targets."""
    shifted_logits = performer_logits[..., :-1, :]
    labels_std = input_ids[..., 1:]
    labels_max = shifted_logits.argmax(dim=-1)
    valid_mask = attention_mask[..., 1:].to(dtype=shifted_logits.dtype)

    logits_t = shifted_logits.transpose(1, 2)
    ce_std = F.cross_entropy(logits_t, labels_std, reduction="none")
    ce_max = F.cross_entropy(logits_t, labels_max, reduction="none")

    weights = (1.0 + delta) * valid_mask
    denom = weights.sum(dim=1).clamp_min(1e-12)
    ppl_std = (ce_std * weights).sum(dim=1) / denom
    ppl_max = (ce_max * weights).sum(dim=1) / denom
    return ppl_std + ppl_max


def compute_hidden_delta(
    observer_hidden_states: tuple[torch.Tensor, ...],
    performer_hidden_states: tuple[torch.Tensor, ...],
    tau: float,
    alpha: float,
    hidden_num: int,
) -> torch.Tensor:
    """Aggregate last-layer hidden-state cosine divergence into token weights."""
    available_layers = min(len(observer_hidden_states), len(performer_hidden_states))
    if available_layers == 0:
        raise ValueError("Hidden states are empty.")
    if hidden_num <= 0:
        raise ValueError("hidden_num must be positive.")

    start = max(0, available_layers - hidden_num)
    deltas = []

    for layer_idx in range(start, available_layers):
        obs = F.normalize(observer_hidden_states[layer_idx], p=2, dim=-1)
        perf = F.normalize(performer_hidden_states[layer_idx], p=2, dim=-1)
        cosine_similarity = (obs * perf).sum(dim=-1)
        deltas.append(1.0 - cosine_similarity)

    delta = torch.stack(deltas, dim=0).mean(dim=0)
    delta = torch.clamp(delta - tau, min=0.0)
    delta = 1.0 - torch.exp(-alpha * delta)
    return delta[..., 1:]


def weighted_score(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    performer_logits: torch.Tensor,
    observer_logits: torch.Tensor,
    pad_token_id: int,
    observer_hidden_states: tuple[torch.Tensor, ...],
    performer_hidden_states: tuple[torch.Tensor, ...],
    tau: float = 0.15,
    alpha: float = 10.0,
    hidden_num: int = 32,
) -> torch.Tensor:
    """Compute the final weighted detector score."""
    delta = compute_hidden_delta(
        observer_hidden_states=observer_hidden_states,
        performer_hidden_states=performer_hidden_states,
        tau=tau,
        alpha=alpha,
        hidden_num=hidden_num,
    )

    numerator = weighted_sum_perplexity_dual(
        input_ids=input_ids,
        attention_mask=attention_mask,
        performer_logits=performer_logits,
        delta=delta,
    )
    denominator = weighted_entropy(
        p_logits=observer_logits,
        q_logits=performer_logits,
        input_ids=input_ids,
        attention_mask=attention_mask,
        pad_token_id=pad_token_id,
        delta=delta,
    )
    return numerator / denominator.clamp_min(1e-12)
