"""
policies/boltzmann.py
Boltzmann (Softmax) exploration policy.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Any

from .base import BasePolicy
from ..logic import linear_anneal


class BoltzmannPolicy(BasePolicy):
    """
    Selects actions by sampling from a softmax distribution over Q-values.
    The 'temperature' parameter controls the degree of exploration.
    """

    def __init__(
        self,
        network: nn.Module,
        device: torch.device,
        temperature_start: float = 1.0,
        temperature_end: float = 0.1,
        seed: Optional[int] = None,
    ):
        super().__init__(network, device)
        self.temp_start = temperature_start
        self.temp_end = temperature_end
        self.rng = np.random.default_rng(seed)

    def get_action(
        self, 
        state: np.ndarray, 
        current_step: Optional[int] = None,
        max_steps: Optional[int] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Choose action(s) using Boltzmann sampling.
        """
        temp = linear_anneal(
            self.temp_start,
            self.temp_end,
            current_step,
            max_steps
        )

        # Fast non-blocking inference
        q_values = self._get_q_values(state)
            
        # Convert temperature-scaled Q-values into action probabilities.
        # PyTorch's softmax implementation handles the usual numerical-stability details.
        probs = torch.softmax(q_values / max(temp, 1e-8), dim=1).cpu().numpy()
        
        batch_size = probs.shape[0]
        actions = np.zeros(batch_size, dtype=np.int64)
        
        for i in range(batch_size):
            actions[i] = self.rng.choice(len(probs[i]), p=probs[i])

        return actions
