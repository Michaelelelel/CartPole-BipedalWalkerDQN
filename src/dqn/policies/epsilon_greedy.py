"""
policies/epsilon_greedy.py
Epsilon-greedy exploration policy.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Optional

from .base import BasePolicy
from ..logic import linear_anneal


class EpsilonGreedyPolicy(BasePolicy):
    """
    Epsilon-greedy policy that owns its Q-network.
    """

    def __init__(
        self,
        network: nn.Module,
        device: torch.device,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        seed: Optional[int] = None,
    ):
        super().__init__(network, device)
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.rng = np.random.default_rng(seed)

    def get_action(
        self, 
        state: np.ndarray, 
        current_step: Optional[int] = None,
        max_steps: Optional[int] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Choose action(s) using epsilon-greedy exploration.
        """
        epsilon = linear_anneal(
            self.epsilon_start,
            self.epsilon_end,
            current_step,
            max_steps
        )

        # Fast non-blocking inference
        q_values = self._get_q_values(state)
            
        batch_size, num_actions = q_values.shape
        # Perform argmax on GPU before moving to CPU to minimize PCIe bus transfer size
        actions = q_values.argmax(dim=1).cpu().numpy()

        # Exploration
        if epsilon > 0:
            explore_mask = self.rng.random(batch_size) < epsilon
            if explore_mask.any():
                explore_indices = np.where(explore_mask)[0]
                random_actions = self.rng.integers(0, num_actions, size=len(explore_indices))
                actions[explore_indices] = random_actions

        return actions
