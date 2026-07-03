"""
networks/__init__.py
Dynamic model registry and factory to retrieve feedforward Q-network architectures.
"""
from __future__ import annotations

from typing import Any
from torch import nn

from .feedforward import FeedForwardNetwork

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "mlp_small": {"hidden_sizes": (128, 128)},
    "mlp_medium": {"hidden_sizes": (256, 128, 64)},
    "mlp_large": {"hidden_sizes": (512, 256, 128, 64)},
}


def get_model(
        name: str,
        *,
        state_dim: int,
        action_dim: int,
        **kwargs: Any,
) -> nn.Module:
    """
    Factory function to instantiate and return a neural network model.

    Args:
        name (str): Name of the model architecture to load (e.g. mlp_small).
        state_dim (int): Input state space dimension.
        action_dim (int): Output discrete action space dimension.
        **kwargs: Additional model parameters (e.g. dropout).

    Returns:
        nn.Module: The initialized PyTorch model.
    """
    key = name.lower()

    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )

    defaults = MODEL_REGISTRY[key]
    cfg = {"state_dim": state_dim, "action_dim": action_dim, **defaults, **kwargs}

    return FeedForwardNetwork(**cfg)
