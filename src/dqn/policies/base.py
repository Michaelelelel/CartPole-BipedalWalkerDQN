"""
policies/base.py
Abstract Base Class for Reinforcement Learning Policies.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import numpy as np


class BasePolicy(nn.Module, ABC):
    """
    Abstract Base Policy that integrates a neural network for action selection.
    This replaces the 'OnlinePredictor' pattern by having the policy own its 'brain'.
    """

    def __init__(self, network: nn.Module, device: torch.device):
        super().__init__()
        self.network = network.to(device)
        self.device = device
        self._dev_states: torch.Tensor | None = None

    def _get_q_values(self, state: np.ndarray) -> torch.Tensor:
        """
        Fast forward pass for one state or a vectorized batch of states.
        The reusable tensor avoids repeated allocations during environment collection.
        """
        if state.ndim == 1:
            state = state[np.newaxis, ...]
            
        batch_size = state.shape[0]
        state_shape = state.shape[1:]
        
        if self._dev_states is None or self._dev_states.shape[0] != batch_size or self._dev_states.shape[1:] != state_shape:
            self._dev_states = torch.empty((batch_size, *state_shape), dtype=torch.float32, device=self.device)
            
        self._dev_states.copy_(torch.from_numpy(state), non_blocking=True)
        with torch.no_grad():
            return self.network(self._dev_states)

    @abstractmethod
    def get_action(self, state: np.ndarray, **kwargs) -> np.ndarray:
        """
        Select an action based on the state.
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expose the underlying network's forward pass.
        """
        return self.network(x)
