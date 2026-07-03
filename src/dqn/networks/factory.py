"""
networks/factory.py
Modular registry for Q-Network architectures.
"""
from __future__ import annotations

from typing import Any, Dict
import torch.nn as nn

from .feedforward import FeedForwardNetwork

class NetworkFactory:
    """
    Registry-based factory for neural network architectures.
    Allows easy expansion and swapping of model sizes and types.
    """
    _REGISTRY: Dict[str, Dict[str, Any]] = {
        "mlp_small": {"hidden_sizes": (128, 128)},
        "mlp_medium": {"hidden_sizes": (256, 128, 64)},
        "mlp_large": {"hidden_sizes": (512, 256, 128, 64)},
    }

    @classmethod
    def create(
        cls, 
        name: str, 
        input_dim: int, 
        output_dim: int, 
        **kwargs: Any
    ) -> nn.Module:
        """
        Instantiate a network from the registry.
        """
        name = name.lower()
        if name not in cls._REGISTRY:
            available = list(cls._REGISTRY.keys())
            raise ValueError(f"Model '{name}' not found. Available: {available}")
        
        config = cls._REGISTRY[name].copy()
        config.update(kwargs)
        
        return FeedForwardNetwork(
            state_dim=input_dim, 
            action_dim=output_dim, 
            **config
        )

    @classmethod
    def register(cls, name: str, config: Dict[str, Any]) -> None:
        """
        Register a new architecture configuration.
        """
        cls._REGISTRY[name.lower()] = config
