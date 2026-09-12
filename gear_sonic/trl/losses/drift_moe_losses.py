"""Auxiliary losses for the Drift-MoE replacement decoder."""

from __future__ import annotations

from typing import Any

from drift_moe.losses import cross_expert_margin_loss, load_balance_loss, window_compactness_loss


def compute_drift_moe_losses(output: Any, config: dict | None = None, mask=None) -> dict:
    config = config or {}
    if mask is not None:
        mask = mask.bool()
    return {
        "drift_balance": load_balance_loss(output.router_probs),
        "drift_intra": window_compactness_loss(
            output.router_probs,
            window_size=config.get("window_size", 25),
            mask=mask,
        ),
        "drift_cross": cross_expert_margin_loss(
            output.expert_candidates,
            margin=config.get("margin", 1.0),
        ),
    }
