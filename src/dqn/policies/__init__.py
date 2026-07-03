"""
policies/__init__.py
Modular registry for action selection policies.
"""
from __future__ import annotations

import torch.nn as nn
import torch
from typing import Any, Dict, Type

from .base import BasePolicy
from .epsilon_greedy import EpsilonGreedyPolicy
from .greedy import GreedyPolicy
from .boltzmann import BoltzmannPolicy

class PolicyFactory:
    """
    Registry for various RL policies.
    """
    _REGISTRY: Dict[str, Type[BasePolicy]] = {
        "epsilon_greedy": EpsilonGreedyPolicy,
        "greedy": GreedyPolicy,
        "boltzmann": BoltzmannPolicy,
    }

    @classmethod
    def create(
        cls, 
        name: str, 
        network: nn.Module, 
        device: torch.device, 
        **kwargs: Any
    ) -> BasePolicy:
        """
        Instantiate a policy and inject the network 'brain'.
        """
        name = name.lower()
        if name not in cls._REGISTRY:
            available = list(cls._REGISTRY.keys())
            raise ValueError(f"Policy '{name}' not found. Available: {available}")
        
        return cls._REGISTRY[name](network=network, device=device, **kwargs)

    @classmethod
    def register(cls, name: str, policy_class: Type[BasePolicy]) -> None:
        """
        Register a new policy class.
        """
        cls._REGISTRY[name.lower()] = policy_class
